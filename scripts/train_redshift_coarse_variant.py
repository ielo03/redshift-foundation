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
    parser = argparse.ArgumentParser(description="Train a coarse-to-fine redshift variant on raw spectra.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hw/final_project/data/processed/desi_sv3_primary.pt"),
        help="Input dataset file (.pt processed bundle or raw HDF5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hw/final_project/outputs/redshift_coarse_variant"),
        help="Directory for variant outputs.",
    )
    parser.add_argument("--survey", type=str, default="sv3", help="Survey filter to apply.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test split fraction.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split fraction inside the training pool.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on the redshift losses.")
    parser.add_argument("--coarse-weight", type=float, default=1.0, help="Weight on the coarse bin loss.")
    parser.add_argument("--fine-weight", type=float, default=1.0, help="Weight on the fine residual loss.")
    parser.add_argument("--num-bins", type=int, default=10, help="Number of redshift bins.")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Mask probability for reconstruction training.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for the neural baseline.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation loss.")
    parser.add_argument("--mask-seed", type=int, default=42, help="Seed for deterministic validation/test masks.")
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


def build_quantile_bins(values: np.ndarray, num_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles)
    if np.unique(edges).size < edges.size:
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1e-3
        edges = np.linspace(lo, hi, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_widths = np.clip(0.5 * (edges[1:] - edges[:-1]), 1e-6, None)
    return edges, centers, half_widths


def encode_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def decode_coarse_fine(
    coarse_logits: np.ndarray,
    fine_residual: np.ndarray,
    centers: np.ndarray,
    half_widths: np.ndarray,
) -> np.ndarray:
    pred_bins = np.asarray(np.argmax(coarse_logits, axis=-1), dtype=np.int64)
    pred_centers = centers[pred_bins]
    pred_widths = half_widths[pred_bins]
    pred_residual = np.tanh(fine_residual)
    pred = pred_centers + pred_residual * pred_widths
    return pred


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("train_redshift_coarse_variant")

    from data_pipeline import collate_records, load_hdf5, make_token_mask
    from neural_baseline import SpectraTransformerWithCoarseRedshiftHead
    from redshift_diagnostics import (
        masked_normalized_reconstruction_metrics,
        plot_flux_reconstruction,
        plot_predicted_vs_actual_redshift,
        redshift_metrics,
        redshift_sample_weights,
        spectrum_reconstruction_metrics,
    )

    from sklearn.model_selection import train_test_split
    import torch
    from torch import nn

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)

    if args.input.suffix == ".pt":
        logger.info("Loading cached processed bundle: %s", args.input)
        bundle = torch.load(args.input, map_location="cpu", weights_only=True)
        batch = bundle.get("batch", bundle)
        X = batch["flux"].numpy()
        wavelength = batch.get("wavelength")
        wavelength = wavelength.numpy() if wavelength is not None else None
        y = batch["z"].numpy().reshape(-1)
        source_path = str(bundle.get("source", args.input))
    else:
        logger.info("Loading raw HDF5 file: %s", args.input)
        records = load_hdf5(
            args.input,
            survey=args.survey,
            primary_only=True,
            max_items=None,
        )
        if not records:
            raise RuntimeError("No records loaded for coarse-to-fine training")
        batch = collate_records(records, target_length=records[0].flux.numel())
        X = batch["flux"].numpy()
        wavelength = batch["wavelength"].numpy()
        y = batch["z"].numpy().reshape(-1)
        source_path = str(args.input)

    if wavelength is None:
        raise RuntimeError("Wavelengths are required for coarse-to-fine training")

    X_train_pool, X_test, wavelength_train_pool, wavelength_test, y_train_pool, y_test = train_test_split(
        X,
        wavelength,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    val_fraction_of_train = args.val_size / max(1.0 - args.test_size, 1e-6)
    X_train, X_val, wavelength_train, wavelength_val, y_train, y_val = train_test_split(
        X_train_pool,
        wavelength_train_pool,
        y_train_pool,
        test_size=val_fraction_of_train,
        random_state=42,
    )

    edges, centers, half_widths = build_quantile_bins(y_train.astype(np.float32), args.num_bins)
    y_train_bins = encode_bins(y_train.astype(np.float32), edges)
    y_val_bins = encode_bins(y_val.astype(np.float32), edges)
    y_test_bins = encode_bins(y_test.astype(np.float32), edges)

    y_train_centers = centers[y_train_bins]
    y_val_centers = centers[y_val_bins]
    y_test_centers = centers[y_test_bins]
    y_train_half_widths = half_widths[y_train_bins]
    y_val_half_widths = half_widths[y_val_bins]
    y_test_half_widths = half_widths[y_test_bins]

    y_train_resid = np.clip((y_train - y_train_centers) / y_train_half_widths, -1.0, 1.0).astype(np.float32)
    y_val_resid = np.clip((y_val - y_val_centers) / y_val_half_widths, -1.0, 1.0).astype(np.float32)
    y_test_resid = np.clip((y_test - y_test_centers) / y_test_half_widths, -1.0, 1.0).astype(np.float32)

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    wavelength_train_t = torch.from_numpy(wavelength_train).float()
    wavelength_val_t = torch.from_numpy(wavelength_val).float()
    wavelength_test_t = torch.from_numpy(wavelength_test).float()
    y_train_raw_t = torch.from_numpy(y_train.astype(np.float32)).float()
    y_val_raw_t = torch.from_numpy(y_val.astype(np.float32)).float()
    y_test_raw_t = torch.from_numpy(y_test.astype(np.float32)).float()
    y_train_bins_t = torch.from_numpy(y_train_bins).long()
    y_val_bins_t = torch.from_numpy(y_val_bins).long()
    y_test_bins_t = torch.from_numpy(y_test_bins).long()
    y_train_resid_t = torch.from_numpy(y_train_resid).float()
    y_val_resid_t = torch.from_numpy(y_val_resid).float()
    y_test_resid_t = torch.from_numpy(y_test_resid).float()
    y_train_weights = torch.from_numpy(redshift_sample_weights(y_train.astype(np.float32))).float()
    y_val_weights = torch.from_numpy(redshift_sample_weights(y_val.astype(np.float32))).float()

    x_mean = X_train_t.mean(dim=0, keepdim=True)
    x_std = X_train_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    X_train_t = (X_train_t - x_mean) / x_std
    X_val_t = (X_val_t - x_mean) / x_std
    X_test_t = (X_test_t - x_mean) / x_std

    val_mask = make_token_mask(X_val_t.shape, mask_prob=args.mask_prob, seed=args.mask_seed)
    test_mask = make_token_mask(X_test_t.shape, mask_prob=args.mask_prob, seed=args.mask_seed + 1)

    logger.info(
        "Loaded %d spectra | train=%d val=%d test=%d | bins=%d",
        len(X),
        len(X_train_t),
        len(X_val_t),
        len(X_test_t),
        args.num_bins,
    )

    X_train_t = X_train_t.to(device)
    X_val_t = X_val_t.to(device)
    X_test_t = X_test_t.to(device)
    wavelength_train_t = wavelength_train_t.to(device)
    wavelength_val_t = wavelength_val_t.to(device)
    wavelength_test_t = wavelength_test_t.to(device)
    y_train_raw_t = y_train_raw_t.to(device)
    y_val_raw_t = y_val_raw_t.to(device)
    y_test_raw_t = y_test_raw_t.to(device)
    y_train_bins_t = y_train_bins_t.to(device)
    y_val_bins_t = y_val_bins_t.to(device)
    y_test_bins_t = y_test_bins_t.to(device)
    y_train_resid_t = y_train_resid_t.to(device)
    y_val_resid_t = y_val_resid_t.to(device)
    y_test_resid_t = y_test_resid_t.to(device)
    y_train_weights = y_train_weights.to(device)
    y_val_weights = y_val_weights.to(device)

    model = SpectraTransformerWithCoarseRedshiftHead(
        input_length=X_train_t.shape[1],
        num_bins=len(centers),
        patch_size=61,
        d_model=256,
        nhead=8,
        num_layers=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    recon_loss_fn = nn.MSELoss()
    coarse_loss_fn = nn.CrossEntropyLoss(reduction="none")
    fine_loss_fn = nn.SmoothL1Loss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    batch_size = min(32, len(X_train_t))
    num_epochs = args.epochs
    logger.info("Starting training for %d epochs (batch_size=%d, patience=%d)", num_epochs, batch_size, args.patience)

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_coarse_loss = 0.0
        epoch_fine_loss = 0.0
        num_batches = 0

        perm = torch.randperm(len(X_train_t), device=device)
        batch_starts = range(0, len(X_train_t), batch_size)
        if tqdm is not None:
            batch_starts = tqdm(batch_starts, desc=f"Epoch {epoch + 1:03d}/{num_epochs}", leave=False)

        for start in batch_starts:
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx]
            yb_bins = y_train_bins_t[idx]
            yb_resid = y_train_resid_t[idx]
            yb_weight = y_train_weights[idx]

            mask = (torch.rand_like(xb) < args.mask_prob).float()
            masked_xb = xb * (1.0 - mask)

            recon, coarse_logits, fine_residual = model(masked_xb)
            mask_bool = mask.bool()
            recon_loss = recon_loss_fn(recon[mask_bool], xb[mask_bool]) if bool(mask_bool.any().item()) else recon.sum() * 0.0
            coarse_loss = (coarse_loss_fn(coarse_logits, yb_bins) * yb_weight).mean()
            fine_loss = (fine_loss_fn(fine_residual, yb_resid) * yb_weight).mean()
            loss = recon_loss + args.coarse_weight * coarse_loss + args.fine_weight * fine_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_recon_loss += float(recon_loss.item())
            epoch_coarse_loss += float(coarse_loss.item())
            epoch_fine_loss += float(fine_loss.item())
            num_batches += 1

        train_duration = time.perf_counter() - epoch_start
        model.eval()
        val_start = time.perf_counter()
        val_loss = 0.0
        val_batches = 0
        val_starts = range(0, len(X_val_t), batch_size)
        if tqdm is not None:
            val_starts = tqdm(val_starts, desc=f"Val {epoch + 1:03d}/{num_epochs}", leave=False)
        with torch.no_grad():
            for start in val_starts:
                xb = X_val_t[start : start + batch_size]
                yb_bins = y_val_bins_t[start : start + batch_size]
                yb_resid = y_val_resid_t[start : start + batch_size]
                yb_weight = y_val_weights[start : start + batch_size]
                eval_mask = val_mask[start : start + batch_size].to(device).float()
                masked_xb = xb * (1.0 - eval_mask)
                val_recon, val_coarse_logits, val_fine_residual = model(masked_xb)
                eval_mask_bool = eval_mask.bool()
                val_recon_loss = (
                    recon_loss_fn(val_recon[eval_mask_bool], xb[eval_mask_bool])
                    if bool(eval_mask_bool.any().item())
                    else val_recon.sum() * 0.0
                )
                val_coarse_loss = (coarse_loss_fn(val_coarse_logits, yb_bins) * yb_weight).mean()
                val_fine_loss = (fine_loss_fn(val_fine_residual, yb_resid) * yb_weight).mean()
                batch_val_loss = float(
                    val_recon_loss.item()
                    + args.coarse_weight * val_coarse_loss.item()
                    + args.fine_weight * val_fine_loss.item()
                )
                val_loss += batch_val_loss
                val_batches += 1
        val_loss = val_loss / max(val_batches, 1)
        val_duration = time.perf_counter() - val_start
        logger.info(
            "Epoch %03d/%d | loss=%.6f | recon=%.6f | coarse=%.6f | fine=%.6f | val_loss=%.6f | train_time=%.1fs | val_time=%.1fs | epoch_time=%.1fs",
            epoch + 1,
            num_epochs,
            epoch_loss / max(num_batches, 1),
            epoch_recon_loss / max(num_batches, 1),
            epoch_coarse_loss / max(num_batches, 1),
            epoch_fine_loss / max(num_batches, 1),
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
    predicted_bins_chunks = []
    actual_bins_chunks = []
    predicted_flux_chunks = []
    actual_flux_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_test_t), batch_size):
            xb = X_test_t[start : start + batch_size]
            eval_mask = test_mask[start : start + batch_size].to(device).float()
            masked_xb = xb * (1.0 - eval_mask)
            test_recon, test_coarse_logits, test_fine_residual = model(masked_xb)
            pred_z = decode_coarse_fine(
                test_coarse_logits.detach().cpu().numpy(),
                test_fine_residual.detach().cpu().numpy(),
                centers,
                half_widths,
            )
            predicted_z_chunks.append(pred_z)
            actual_z_chunks.append(y_test_raw_t[start : start + batch_size].detach().cpu().numpy())
            predicted_bins_chunks.append(test_coarse_logits.argmax(dim=-1).detach().cpu().numpy())
            actual_bins_chunks.append(y_test_bins_t[start : start + batch_size].detach().cpu().numpy())
            predicted_flux_chunks.append(test_recon.detach().cpu().numpy())
            actual_flux_chunks.append(X_test[start : start + batch_size])

    predicted_z = np.concatenate(predicted_z_chunks, axis=0)
    actual_z = np.concatenate(actual_z_chunks, axis=0)
    predicted_bins = np.concatenate(predicted_bins_chunks, axis=0)
    actual_bins = np.concatenate(actual_bins_chunks, axis=0)
    predicted_flux = np.concatenate(predicted_flux_chunks, axis=0)
    actual_flux = np.concatenate(actual_flux_chunks, axis=0)

    metrics = {
        "redshift": redshift_metrics(actual_z, predicted_z),
        "coarse_accuracy": float(np.mean(predicted_bins == actual_bins)),
        "reconstruction": masked_normalized_reconstruction_metrics(actual_flux, predicted_flux, test_mask.cpu().numpy()),
        "reconstruction_raw": spectrum_reconstruction_metrics(actual_flux, predicted_flux),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "redshift_predictions.json"
    plot_path = args.output_dir / "redshift_pred_vs_actual.png"
    recon_plot_path = args.output_dir / "spectrum_reconstruction.png"
    metrics_path = args.output_dir / "redshift_metrics.json"
    best_model_path = args.output_dir / "best_model.pt"

    payload = {
        "actual": actual_z.tolist(),
        "predicted": predicted_z.tolist(),
        "actual_bins": actual_bins.tolist(),
        "predicted_bins": predicted_bins.tolist(),
        "metrics": metrics,
        "n_train": int(len(X_train_t)),
        "n_val": int(len(X_val_t)),
        "n_test": int(len(X_test_t)),
        "source": source_path,
        "best_val_loss": best_val_loss,
        "num_bins": int(len(centers)),
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
    }

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if best_state is not None:
        torch.save(best_state, best_model_path)

    logger.info("Saving predictions, metrics, checkpoint, and plots")
    plot_predicted_vs_actual_redshift(actual_z, predicted_z, plot_path, title="Coarse-to-fine redshift variant")
    plot_flux_reconstruction(actual_flux[0], predicted_flux[0], recon_plot_path, title="Raw-spectrum reconstruction example")

    logger.info("Loaded records: %d", len(X))
    logger.info("Train/val/test sizes: %d / %d / %d", len(X_train_t), len(X_val_t), len(X_test_t))
    logger.info("Metrics: %s", metrics)
    logger.info("Saved predictions to: %s", predictions_path)
    logger.info("Saved plot to: %s", plot_path)
    logger.info("Saved reconstruction plot to: %s", recon_plot_path)
    logger.info("Saved metrics to: %s", metrics_path)
    logger.info("Saved best model to: %s", best_model_path)


if __name__ == "__main__":
    main()
