from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency guard
    tqdm = None


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a spectra-token transformer from scratch with a continuous redshift head."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hw/final_project/data/processed/aion_spectrum_tokens.pt"),
        help="Input tokenized bundle produced by process_aion_codec_tokens.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hw/final_project/outputs/aion_codec_baseline"),
        help="Directory for baseline outputs.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split fraction inside the training pool.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Mask probability for MLM training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Validation/test batch size.")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer width.")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on the redshift loss term.")
    parser.add_argument("--mask-seed", type=int, default=42, help="Seed used for deterministic validation/test masks.")
    return parser.parse_args()


def split_arrays(
    *arrays: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    if not arrays:
        raise ValueError("split_arrays requires at least one array")
    n = len(arrays[0])
    if any(len(arr) != n for arr in arrays):
        raise ValueError("All arrays must have the same length")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    test_count = int(round(n * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    split_outputs: list[np.ndarray] = []
    for arr in arrays:
        split_outputs.extend([arr[train_idx], arr[test_idx]])
    return tuple(split_outputs)


def decode_spectrum_tokens(codec_manager, token_ids: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    import torch
    from aion.modalities import DESISpectrum

    device = getattr(codec_manager, "device", "cpu")
    token_tensor = torch.from_numpy(token_ids.astype(np.int32)).to(device)
    wavelength_tensor = torch.from_numpy(wavelength.astype(np.float32)).to(device)
    decoded = codec_manager.decode({"tok_spectrum_desi": token_tensor}, DESISpectrum, wavelength=wavelength_tensor)
    return decoded.flux.detach().cpu().numpy()


def token_mask_to_flux_mask(token_mask: np.ndarray, flux_length: int) -> np.ndarray:
    """Approximate masked spectral regions in flux space from codec-token masks.

    AION's spectrum token sequence stores a normalization token first, followed by
    compressed spectrum tokens. For reconstruction metrics, only the spectrum-token
    positions correspond to local spectral regions, so the normalization token is
    ignored here.
    """

    token_mask = np.asarray(token_mask, dtype=bool)
    if token_mask.ndim != 2:
        raise ValueError(f"token_mask must have shape (batch, tokens), got {token_mask.shape}")
    if flux_length <= 0:
        raise ValueError("flux_length must be positive")

    spectrum_token_mask = token_mask[:, 1:] if token_mask.shape[1] > 1 else token_mask
    batch_size, num_spectrum_tokens = spectrum_token_mask.shape
    flux_mask = np.zeros((batch_size, flux_length), dtype=bool)
    if num_spectrum_tokens == 0:
        return flux_mask

    boundaries = np.linspace(0, flux_length, num_spectrum_tokens + 1, dtype=int)
    for token_idx in range(num_spectrum_tokens):
        start = boundaries[token_idx]
        end = boundaries[token_idx + 1]
        if end > start:
            flux_mask[:, start:end] = spectrum_token_mask[:, token_idx, None]
    return flux_mask


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("train_aion_codec_baseline")

    import torch
    from torch import nn

    from data_pipeline import make_token_mask
    from neural_baseline import AIONSpectrumMaskedTransformerWithRedshiftHead
    from project_paths import AION_REPO_DIR, ensure_project_dirs
    from redshift_diagnostics import (
        decode_redshift_target,
        encode_redshift_target,
        masked_normalized_reconstruction_metrics,
        plot_flux_reconstruction,
        plot_predicted_vs_actual_redshift,
        redshift_metrics,
        redshift_sample_weights,
        spectrum_reconstruction_metrics,
    )

    ensure_project_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)

    if args.input.suffix != ".pt":
        raise ValueError("Spectrum-token baseline expects a tokenized .pt bundle from process_aion_codec_tokens.py")

    logger.info("Loading tokenized bundle: %s", args.input)
    bundle = torch.load(args.input, map_location="cpu", weights_only=True)
    input_ids = bundle["input_ids"].long().cpu().numpy()
    z_values = bundle["redshift_values"].float().cpu().numpy().reshape(-1)
    flux_raw = bundle["flux_raw"].float().cpu().numpy()
    wavelength = bundle["wavelength"].float().cpu().numpy()
    flux_mask = bundle["mask"].bool().cpu().numpy()

    seq_len = int(bundle["seq_len"])
    mask_token_id = int(bundle.get("mask_token_id", 1024))
    input_vocab_size = int(bundle.get("input_vocab_size", 1025))
    output_vocab_size = int(bundle.get("output_vocab_size", 1024))

    (
        X_train_pool,
        X_test,
        y_train_pool,
        y_test,
        flux_train_pool,
        flux_test,
        wavelength_train_pool,
        wavelength_test,
        flux_mask_train_pool,
        flux_mask_test,
    ) = split_arrays(
        input_ids,
        z_values,
        flux_raw,
        wavelength,
        flux_mask,
        test_size=args.test_size,
        random_state=42,
    )

    val_fraction_of_train = args.val_size / max(1.0 - args.test_size, 1e-6)
    (
        X_train,
        X_val,
        y_train,
        y_val,
        flux_train,
        flux_val,
        wavelength_train,
        wavelength_val,
        flux_mask_train,
        flux_mask_val,
    ) = split_arrays(
        X_train_pool,
        y_train_pool,
        flux_train_pool,
        wavelength_train_pool,
        flux_mask_train_pool,
        test_size=val_fraction_of_train,
        random_state=42,
    )

    X_train_t = torch.from_numpy(X_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).long().to(device)
    X_test_t = torch.from_numpy(X_test).long().to(device)
    y_train_raw = y_train.astype(np.float32)
    y_val_raw = y_val.astype(np.float32)
    y_test_raw = y_test.astype(np.float32)
    y_train_target = torch.from_numpy(encode_redshift_target(y_train_raw)).float().to(device)
    y_val_target = torch.from_numpy(encode_redshift_target(y_val_raw)).float().to(device)
    y_test_target = torch.from_numpy(encode_redshift_target(y_test_raw)).float().to(device)
    y_train_weights = torch.from_numpy(redshift_sample_weights(y_train_raw)).float().to(device)
    y_val_weights = torch.from_numpy(redshift_sample_weights(y_val_raw)).float().to(device)
    flux_test_t = torch.from_numpy(flux_test).float()
    wavelength_test_t = torch.from_numpy(wavelength_test).float()

    val_mask = make_token_mask(
        (len(X_val_t), seq_len),
        mask_prob=args.mask_prob,
        seed=args.mask_seed,
        always_mask_last=False,
    )
    test_mask = make_token_mask(
        (len(X_test_t), seq_len),
        mask_prob=args.mask_prob,
        seed=args.mask_seed + 1,
        always_mask_last=False,
    )
    torch.save(
        {"val_mask": val_mask, "test_mask": test_mask},
        args.output_dir / "eval_masks.pt",
    )

    sys.path.insert(0, str(AION_REPO_DIR))
    from aion.codecs.manager import CodecManager

    codec_manager = CodecManager(device=device)

    model = AIONSpectrumMaskedTransformerWithRedshiftHead(
        seq_len=seq_len,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    redshift_loss_fn = nn.SmoothL1Loss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    batch_size = min(args.batch_size, len(X_train_t))
    eval_batch_size = min(args.eval_batch_size, max(len(X_val_t), 1), max(len(X_test_t), 1))

    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        redshift_pred: torch.Tensor,
        redshift_target: torch.Tensor,
        redshift_weight: torch.Tensor,
    ) -> torch.Tensor:
        token_losses = recon_loss_fn(logits.reshape(-1, output_vocab_size), labels.reshape(-1)).view(labels.shape)
        mask = labels != -100
        recon_loss = token_losses[mask].mean() if bool(mask.any().item()) else token_losses.mean() * 0.0
        z_loss = redshift_loss_fn(redshift_pred, redshift_target)
        z_loss = (z_loss * redshift_weight).mean()
        return recon_loss + args.alpha * z_loss

    logger.info(
        "Loaded %d tokenized spectra | train=%d val=%d test=%d | seq_len=%d",
        len(input_ids),
        len(X_train_t),
        len(X_val_t),
        len(X_test_t),
        seq_len,
    )
    logger.info("Starting training for %d epochs (batch_size=%d, patience=%d)", args.epochs, batch_size, args.patience)

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        perm = torch.randperm(len(X_train_t), device=device)
        batch_starts = range(0, len(X_train_t), batch_size)
        if tqdm is not None:
            batch_starts = tqdm(batch_starts, desc=f"Epoch {epoch + 1:03d}/{args.epochs}", leave=False)

        for start in batch_starts:
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx]
            yb = y_train_target[idx]
            wb = y_train_weights[idx]

            rand_mask = torch.rand_like(xb.float()) < args.mask_prob
            labels = xb.clone()
            labels[~rand_mask] = -100
            masked_xb = xb.clone()
            masked_xb[rand_mask] = mask_token_id

            logits, redshift_pred = model(masked_xb)
            loss = compute_loss(logits, labels, redshift_pred, yb, wb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            num_train_batches += 1

        train_duration = time.perf_counter() - epoch_start

        model.eval()
        val_start = time.perf_counter()
        val_loss_sum = 0.0
        num_val_batches = 0
        val_starts = range(0, len(X_val_t), eval_batch_size)
        if tqdm is not None:
            val_starts = tqdm(val_starts, desc=f"Val {epoch + 1:03d}/{args.epochs}", leave=False)

        with torch.no_grad():
            for start in val_starts:
                xb = X_val_t[start : start + eval_batch_size]
                yb = y_val_target[start : start + eval_batch_size]
                wb = y_val_weights[start : start + eval_batch_size]
                eval_mask = val_mask[start : start + eval_batch_size].to(device)
                labels = xb.clone()
                labels[~eval_mask] = -100
                masked_xb = xb.clone()
                masked_xb[eval_mask] = mask_token_id

                logits, redshift_pred = model(masked_xb)
                batch_loss = compute_loss(logits, labels, redshift_pred, yb, wb)
                val_loss_sum += float(batch_loss.item())
                num_val_batches += 1

        val_loss = val_loss_sum / max(num_val_batches, 1)
        val_duration = time.perf_counter() - val_start

        logger.info(
            "Epoch %03d/%d | loss=%.6f | val_loss=%.6f | train_time=%.1fs | val_time=%.1fs | epoch_time=%.1fs",
            epoch + 1,
            args.epochs,
            train_loss_sum / max(num_train_batches, 1),
            val_loss,
            train_duration,
            val_duration,
            time.perf_counter() - epoch_start,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            logger.info("Early stopping triggered after %d epochs", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("Running final test evaluation")
    model.eval()
    predicted_z_chunks = []
    actual_z_chunks = []
    token_accuracy_numer = 0.0
    token_accuracy_denom = 0
    predicted_flux_chunks = []
    actual_flux_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_test_t), eval_batch_size):
            xb = X_test_t[start : start + eval_batch_size]
            yb = y_test_raw[start : start + eval_batch_size]
            eval_mask = test_mask[start : start + eval_batch_size].to(device)
            labels = xb.clone()
            labels[~eval_mask] = -100
            masked_xb = xb.clone()
            masked_xb[eval_mask] = mask_token_id

            logits, redshift_pred = model(masked_xb)
            pred_ids = logits.argmax(dim=-1)
            filled_ids = xb.clone()
            filled_ids[eval_mask] = pred_ids[eval_mask]

            masked_positions = eval_mask
            if bool(masked_positions.any().item()):
                token_accuracy_numer += float((pred_ids[masked_positions] == xb[masked_positions]).sum().item())
                token_accuracy_denom += int(masked_positions.sum().item())

            predicted_z_chunks.append(decode_redshift_target(redshift_pred.detach().cpu().numpy().reshape(-1)))
            actual_z_chunks.append(yb.reshape(-1))

            batch_pred_flux = decode_spectrum_tokens(
                codec_manager,
                filled_ids.detach().cpu().numpy(),
                wavelength_test_t[start : start + eval_batch_size].numpy(),
            )
            predicted_flux_chunks.append(batch_pred_flux)
            actual_flux_chunks.append(flux_test_t[start : start + eval_batch_size].numpy())

    predicted_z = np.concatenate(predicted_z_chunks, axis=0)
    actual_z = np.concatenate(actual_z_chunks, axis=0)
    token_accuracy = token_accuracy_numer / token_accuracy_denom if token_accuracy_denom else float("nan")
    predicted_flux = np.concatenate(predicted_flux_chunks, axis=0)
    actual_flux = np.concatenate(actual_flux_chunks, axis=0)
    flux_eval_mask = token_mask_to_flux_mask(test_mask.cpu().numpy(), actual_flux.shape[-1])
    metrics = {
        "redshift": redshift_metrics(actual_z, predicted_z),
        "reconstruction": masked_normalized_reconstruction_metrics(actual_flux, predicted_flux, flux_eval_mask),
        "reconstruction_raw": spectrum_reconstruction_metrics(actual_flux, predicted_flux),
        "masked_token_accuracy": token_accuracy,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "redshift_predictions.json"
    plot_path = args.output_dir / "redshift_pred_vs_actual.png"
    recon_plot_path = args.output_dir / "spectrum_reconstruction.png"
    metrics_path = args.output_dir / "redshift_metrics.json"
    best_model_path = args.output_dir / "best_model.pt"

    payload = {
        "actual_redshift": actual_z.tolist(),
        "predicted_redshift": predicted_z.tolist(),
        "metrics": metrics,
        "n_train": int(len(X_train_t)),
        "n_val": int(len(X_val_t)),
        "n_test": int(len(X_test_t)),
        "source": str(args.input),
        "best_val_loss": best_val_loss,
        "seq_len": seq_len,
        "mask_seed": args.mask_seed,
    }

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if best_state is not None:
        torch.save(best_state, best_model_path)

    logger.info("Saving predictions, metrics, checkpoint, and plots")
    plot_predicted_vs_actual_redshift(actual_z, predicted_z, plot_path, title="Spectrum-token baseline")
    plot_flux_reconstruction(actual_flux[0], predicted_flux[0], recon_plot_path, title="Spectrum reconstruction example")

    logger.info("Loaded records: %d", len(input_ids))
    logger.info("Train/val/test sizes: %d / %d / %d", len(X_train_t), len(X_val_t), len(X_test_t))
    logger.info("Metrics: %s", metrics)
    logger.info("Saved predictions to: %s", predictions_path)
    logger.info("Saved plot to: %s", plot_path)
    logger.info("Saved reconstruction plot to: %s", recon_plot_path)
    logger.info("Saved metrics to: %s", metrics_path)
    logger.info("Saved best model to: %s", best_model_path)


if __name__ == "__main__":
    main()
