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
except Exception:  # pragma: no cover
    tqdm = None


def add_src_to_path(project_root: Path) -> None:
    sys.path.insert(0, str(project_root / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train redshift-token foundation model with bin classification.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/mmu_desi_hf_edr_sv3.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/redshift_token_bins_hf_100k"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=61)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--bin-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", choices=["tail", "uniform"], default="tail")
    parser.add_argument("--disable-sample-weights", action="store_true")
    parser.add_argument("--tail-power", type=float, default=1.75)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--mask-seed", type=int, default=42)
    return parser.parse_args()


def build_quantile_bins(values: np.ndarray, num_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles)
    if np.unique(edges).size < edges.size:
        edges = np.linspace(float(values.min()), float(values.max()) + 1e-6, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_widths = np.clip(0.5 * (edges[1:] - edges[:-1]), 1e-6, None)
    return edges.astype(np.float32), centers.astype(np.float32), half_widths.astype(np.float32)


def encode_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2).astype(np.int64)


def decode_bins_residual(
    logits: np.ndarray,
    residual: np.ndarray,
    centers: np.ndarray,
    half_widths: np.ndarray,
) -> np.ndarray:
    bins = np.argmax(logits, axis=-1).astype(np.int64)
    return centers[bins] + np.tanh(residual) * half_widths[bins]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("train_redshift_token_bins")

    import torch
    from torch import nn
    from sklearn.model_selection import train_test_split

    from data_pipeline import make_token_mask
    from neural_baseline import SpectraTransformerWithRedshiftTokenBins
    from redshift_diagnostics import (
        masked_normalized_reconstruction_metrics,
        plot_flux_reconstruction,
        plot_predicted_vs_actual_redshift,
        redshift_sample_weights,
        spectrum_reconstruction_metrics,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    bundle = torch.load(args.input, map_location="cpu", weights_only=True)
    batch = bundle.get("batch", bundle)
    X = batch["flux"].numpy()
    y = batch["z"].numpy().reshape(-1).astype(np.float32)
    if args.max_items is not None:
        X = X[: args.max_items]
        y = y[: args.max_items]

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    val_fraction = args.val_size / max(1.0 - args.test_size, 1e-6)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=val_fraction,
        random_state=42,
    )

    edges, centers, half_widths = build_quantile_bins(y_train, args.num_bins)
    y_train_bins = encode_bins(y_train, edges)
    y_val_bins = encode_bins(y_val, edges)
    y_test_bins = encode_bins(y_test, edges)
    y_train_residual = np.clip((y_train - centers[y_train_bins]) / half_widths[y_train_bins], -1.0, 1.0).astype(np.float32)
    y_val_residual = np.clip((y_val - centers[y_val_bins]) / half_widths[y_val_bins], -1.0, 1.0).astype(np.float32)
    y_test_residual = np.clip((y_test - centers[y_test_bins]) / half_widths[y_test_bins], -1.0, 1.0).astype(np.float32)

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_train_bins_t = torch.from_numpy(y_train_bins).long()
    y_val_bins_t = torch.from_numpy(y_val_bins).long()
    y_test_bins_t = torch.from_numpy(y_test_bins).long()
    y_train_residual_t = torch.from_numpy(y_train_residual).float()
    y_val_residual_t = torch.from_numpy(y_val_residual).float()
    if args.disable_sample_weights:
        y_train_weights = torch.ones(len(y_train), dtype=torch.float32)
        y_val_weights = torch.ones(len(y_val), dtype=torch.float32)
    else:
        y_train_weights = torch.from_numpy(redshift_sample_weights(y_train) ** args.tail_power).float()
        y_val_weights = torch.from_numpy(redshift_sample_weights(y_val) ** args.tail_power).float()

    val_mask = make_token_mask(X_val_t.shape, mask_prob=args.mask_prob, seed=args.mask_seed)
    test_mask = make_token_mask(X_test_t.shape, mask_prob=args.mask_prob, seed=args.mask_seed + 1)

    model = SpectraTransformerWithRedshiftTokenBins(
        input_length=X_train_t.shape[1],
        num_bins=args.num_bins,
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    recon_loss_fn = nn.MSELoss()
    bin_loss_fn = nn.CrossEntropyLoss(reduction="none")
    residual_loss_fn = nn.SmoothL1Loss(reduction="none")

    batch_size = min(args.batch_size, len(X_train_t))
    accumulation = max(1, args.gradient_accumulation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loaded %d spectra | train=%d val=%d test=%d | bins=%d | batch=%d | accumulation=%d",
        len(X),
        len(X_train_t),
        len(X_val_t),
        len(X_test_t),
        args.num_bins,
        batch_size,
        accumulation,
    )

    best_mae = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_weights_cpu = torch.clamp(y_train_weights, min=1e-6)

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        if args.sampling_strategy == "tail":
            sampled_idx = torch.multinomial(train_weights_cpu, len(X_train_t), replacement=True)
        else:
            sampled_idx = torch.randperm(len(X_train_t))
        batch_starts = range(0, len(X_train_t), batch_size)
        if tqdm is not None:
            batch_starts = tqdm(batch_starts, desc=f"epoch {epoch + 1:03d}/{args.epochs}", leave=False)

        optimizer.zero_grad(set_to_none=True)
        train_loss = train_recon = train_bin = train_resid = 0.0
        train_batches = 0
        for step, start in enumerate(batch_starts, start=1):
            idx = sampled_idx[start : start + batch_size]
            xb = X_train_t[idx].to(device)
            yb_bins = y_train_bins_t[idx].to(device)
            yb_residual = y_train_residual_t[idx].to(device)
            yb_weights = y_train_weights[idx].to(device)
            mask = (torch.rand_like(xb) < args.mask_prob).float()
            masked_xb = xb * (1.0 - mask)

            recon, logits, residual = model(masked_xb)
            mask_bool = mask.bool()
            recon_loss = recon_loss_fn(recon[mask_bool], xb[mask_bool]) if bool(mask_bool.any().item()) else recon.sum() * 0.0
            bin_loss = (bin_loss_fn(logits, yb_bins) * yb_weights).mean()
            residual_loss = (residual_loss_fn(residual, yb_residual) * yb_weights).mean()
            loss = (
                args.reconstruction_weight * recon_loss
                + args.bin_weight * bin_loss
                + args.residual_weight * residual_loss
            )
            (loss / accumulation).backward()
            if step % accumulation == 0 or start + batch_size >= len(X_train_t):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += float(loss.item())
            train_recon += float(recon_loss.item())
            train_bin += float(bin_loss.item())
            train_resid += float(residual_loss.item())
            train_batches += 1

        model.eval()
        val_logits_chunks = []
        val_resid_chunks = []
        val_loss = val_recon = val_bin = val_resid = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(X_val_t), batch_size):
                xb = X_val_t[start : start + batch_size].to(device)
                yb_bins = y_val_bins_t[start : start + batch_size].to(device)
                yb_residual = y_val_residual_t[start : start + batch_size].to(device)
                yb_weights = y_val_weights[start : start + batch_size].to(device)
                mask = val_mask[start : start + batch_size].to(device).float()
                recon, logits, residual = model(xb * (1.0 - mask))
                mask_bool = mask.bool()
                recon_loss = recon_loss_fn(recon[mask_bool], xb[mask_bool]) if bool(mask_bool.any().item()) else recon.sum() * 0.0
                bin_loss = (bin_loss_fn(logits, yb_bins) * yb_weights).mean()
                residual_loss = (residual_loss_fn(residual, yb_residual) * yb_weights).mean()
                loss = (
                    args.reconstruction_weight * recon_loss
                    + args.bin_weight * bin_loss
                    + args.residual_weight * residual_loss
                )
                val_loss += float(loss.item())
                val_recon += float(recon_loss.item())
                val_bin += float(bin_loss.item())
                val_resid += float(residual_loss.item())
                val_batches += 1
                val_logits_chunks.append(logits.detach().cpu())
                val_resid_chunks.append(residual.detach().cpu())

        val_logits = torch.cat(val_logits_chunks, dim=0).numpy()
        val_residual_pred = torch.cat(val_resid_chunks, dim=0).numpy()
        val_pred = decode_bins_residual(val_logits, val_residual_pred, centers, half_widths)
        val_metrics = metrics(y_val, val_pred)
        val_acc = float(np.mean(np.argmax(val_logits, axis=-1) == y_val_bins))
        epoch_time = time.perf_counter() - epoch_start
        logger.info(
            "epoch %03d/%d | loss=%.6f | recon=%.6f | bin=%.6f | resid=%.6f | val_loss=%.6f | val_recon=%.6f | val_bin=%.6f | val_resid=%.6f | val_acc=%.4f | val_mae=%.6f | val_rmse=%.6f | val_bias=%.6f | time=%.1fs",
            epoch + 1,
            args.epochs,
            train_loss / max(train_batches, 1),
            train_recon / max(train_batches, 1),
            train_bin / max(train_batches, 1),
            train_resid / max(train_batches, 1),
            val_loss / max(val_batches, 1),
            val_recon / max(val_batches, 1),
            val_bin / max(val_batches, 1),
            val_resid / max(val_batches, 1),
            val_acc,
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["bias"],
            epoch_time,
        )

        payload = {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_state": best_state,
            "best_mae": best_mae,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "args": vars(args),
            "bin_edges": edges.tolist(),
            "bin_centers": centers.tolist(),
            "bin_half_widths": half_widths.tolist(),
            "val_metrics": val_metrics,
            "val_acc": val_acc,
        }

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            payload["best_state"] = best_state
            payload["best_mae"] = best_mae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
        torch.save(payload, checkpoint_dir / "latest.pt")
        if not args.disable_early_stopping and epochs_without_improvement >= args.patience:
            logger.info("Early stopping after %d epochs", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    recon_chunks = []
    logits_chunks = []
    residual_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_test_t), batch_size):
            xb = X_test_t[start : start + batch_size].to(device)
            mask = test_mask[start : start + batch_size].to(device).float()
            recon, logits, residual = model(xb * (1.0 - mask))
            recon_chunks.append(recon.detach().cpu())
            logits_chunks.append(logits.detach().cpu())
            residual_chunks.append(residual.detach().cpu())

    recon_pred = torch.cat(recon_chunks, dim=0).numpy()
    test_logits = torch.cat(logits_chunks, dim=0).numpy()
    test_residual_pred = torch.cat(residual_chunks, dim=0).numpy()
    test_pred = decode_bins_residual(test_logits, test_residual_pred, centers, half_widths)
    test_bins_pred = np.argmax(test_logits, axis=-1)
    test_metrics = {
        "redshift": metrics(y_test, test_pred),
        "bin_accuracy": float(np.mean(test_bins_pred == y_test_bins)),
        "reconstruction": masked_normalized_reconstruction_metrics(X_test, recon_pred, test_mask.numpy()),
        "reconstruction_raw": spectrum_reconstruction_metrics(X_test, recon_pred),
    }

    predictions_path = args.output_dir / "redshift_predictions.json"
    metrics_path = args.output_dir / "redshift_metrics.json"
    plot_path = args.output_dir / "redshift_pred_vs_actual.png"
    recon_plot_path = args.output_dir / "spectrum_reconstruction.png"
    best_model_path = args.output_dir / "best_model.pt"

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "actual": y_test.tolist(),
                "predicted": test_pred.tolist(),
                "actual_bins": y_test_bins.tolist(),
                "predicted_bins": test_bins_pred.tolist(),
                "metrics": test_metrics,
                "n_train": int(len(X_train_t)),
                "n_val": int(len(X_val_t)),
                "n_test": int(len(X_test_t)),
                "bin_edges": edges.tolist(),
                "bin_centers": centers.tolist(),
                "best_mae": best_mae,
            },
            f,
            indent=2,
        )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    if best_state is not None:
        torch.save(best_state, best_model_path)
    plot_predicted_vs_actual_redshift(y_test, test_pred, plot_path, title="Redshift-token bin prediction")
    plot_flux_reconstruction(X_test[0], recon_pred[0], recon_plot_path, title="Redshift-token bin reconstruction")

    logger.info("Metrics: %s", test_metrics)
    logger.info("Saved metrics to: %s", metrics_path)


if __name__ == "__main__":
    main()
