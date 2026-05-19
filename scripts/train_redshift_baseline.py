from __future__ import annotations

# This file is the shared raw-spectrum training implementation.
# The official frozen baseline entrypoint is
# `scripts/train_redshift_main_baseline.py`.
# Experimental changes should go through separate variant wrappers.

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
    parser = argparse.ArgumentParser(description="Train a real-data redshift baseline with best-checkpoint saving.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hw/final_project/data/processed/desi_sv3_primary.pt"),
        help="Input dataset file (.pt processed bundle or raw HDF5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hw/final_project/outputs/redshift_baseline"),
        help="Directory for baseline outputs.",
    )
    parser.add_argument("--survey", type=str, default="sv3", help="Survey filter to apply.")
    parser.add_argument(
        "--no-primary-only",
        action="store_true",
        help="Do not filter to SV_PRIMARY=True.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test split fraction.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split fraction inside the training pool.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on the redshift loss term.")
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=1.0,
        help="Weight on the masked spectrum reconstruction loss term.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["val_loss", "val_mae", "val_rmse"],
        default="val_loss",
        help="Metric used to select best_state and drive early stopping.",
    )
    parser.add_argument(
        "--redshift-loss",
        choices=["mse", "smoothl1"],
        default="mse",
        help="Loss used for the redshift target.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["uniform", "tail"],
        default="uniform",
        help="How to sample training examples each epoch.",
    )
    parser.add_argument(
        "--tail-power",
        type=float,
        default=1.0,
        help="Exponent applied to redshift-based sample weights for tail-focused sampling.",
    )
    parser.add_argument(
        "--mask-strategy",
        choices=["random", "domain"],
        default="random",
        help="Training/evaluation masking strategy.",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.15,
        help="Base mask probability.",
    )
    parser.add_argument(
        "--domain-line-sigma",
        type=float,
        default=25.0,
        help="Gaussian width in Angstrom for informative lines in domain masking.",
    )
    parser.add_argument(
        "--domain-line-boost",
        type=float,
        default=2.5,
        help="Weight multiplier for informative lines in domain masking.",
    )
    parser.add_argument(
        "--domain-break-sigma",
        type=float,
        default=40.0,
        help="Gaussian width in Angstrom for the 4000A break in domain masking.",
    )
    parser.add_argument(
        "--domain-break-boost",
        type=float,
        default=2.0,
        help="Weight multiplier for the 4000A break in domain masking.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=42,
        help="Seed for deterministic masking.",
    )
    parser.add_argument(
        "--use-ivar-channel",
        action="store_true",
        help="Add log-scaled inverse variance as an extra input channel.",
    )
    parser.add_argument(
        "--use-validity-channel",
        action="store_true",
        help="Add the observed-pixel validity mask as an extra input channel.",
    )
    parser.add_argument(
        "--redshift-objective",
        choices=["encoded", "relative"],
        default="encoded",
        help="Optimize either encoded z or relative redshift error in physical z space.",
    )
    parser.add_argument(
        "--architecture",
        choices=["cls_head", "redshift_token"],
        default="cls_head",
        help="Raw-spectrum transformer architecture variant.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=0,
        help="Optional second-stage fine-tuning epochs with stronger redshift emphasis.",
    )
    parser.add_argument(
        "--stage2-alpha",
        type=float,
        default=2.0,
        help="Redshift loss weight during the optional second stage.",
    )
    parser.add_argument(
        "--stage2-mask-prob",
        type=float,
        default=0.05,
        help="Mask probability during the optional second training stage.",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=3e-4,
        help="Learning rate during the optional second training stage.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate during the first training stage.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs for the neural baseline.",
    )
    parser.add_argument("--d-model", type=int, default=256, help="Transformer hidden width.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer encoder layers.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of transformer attention heads.")
    parser.add_argument("--patch-size", type=int, default=61, help="Raw-spectrum patch size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-step batch size.")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--throttle-sleep-ms",
        type=float,
        default=0.0,
        help="Optional pause after each optimizer step to leave GPU headroom for other work.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on the number of spectra to load.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an epoch checkpoint saved by this trainer.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run all requested epochs without patience-based early stopping.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience on validation loss.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("train_redshift_baseline")

    from data_pipeline import collate_records, load_hdf5
    from data_pipeline import make_domain_informed_mask, make_token_mask
    from neural_baseline import SpectraTransformerWithRedshiftHead, SpectraTransformerWithRedshiftToken
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
        import torch

        logger.info("Loading cached processed bundle: %s", args.input)
        bundle = torch.load(args.input, map_location="cpu", weights_only=True)
        batch = bundle.get("batch", bundle)
        X = batch["flux"].numpy()
        X_raw = batch.get("flux_raw", batch["flux"]).numpy()
        ivar = batch["ivar"].numpy()
        observed_valid = (~batch["mask"]).numpy().astype(np.float32)
        wavelength = batch.get("wavelength")
        wavelength = wavelength.numpy() if wavelength is not None else None
        y = batch["z"].numpy().reshape(-1)
        source_path = str(bundle.get("source", args.input))
    else:
        logger.info("Loading raw HDF5 file: %s", args.input)
        records = load_hdf5(
            args.input,
            survey=args.survey,
            primary_only=not args.no_primary_only,
            max_items=args.max_items,
        )
        if not records:
            raise RuntimeError("No records loaded for baseline training")

        batch = collate_records(records, target_length=records[0].flux.numel())
        X = batch["flux"].numpy()
        X_raw = batch["flux_raw"].numpy()
        ivar = batch["ivar"].numpy()
        observed_valid = (~batch["mask"]).numpy().astype(np.float32)
        wavelength = batch["wavelength"].numpy()
        y = batch["z"].numpy().reshape(-1)
        source_path = str(args.input)

    if args.max_items is not None:
        max_items = min(args.max_items, len(X))
        X = X[:max_items]
        X_raw = X_raw[:max_items]
        ivar = ivar[:max_items]
        observed_valid = observed_valid[:max_items]
        wavelength = wavelength[:max_items]
        y = y[:max_items]

    if wavelength is None:
        raise RuntimeError("Wavelengths are required for the raw-spectrum baseline")

    (
        X_train_pool,
        X_test,
        ivar_train_pool,
        ivar_test,
        valid_train_pool,
        valid_test,
        wavelength_train_pool,
        wavelength_test,
        y_train_pool,
        y_test,
    ) = train_test_split(
        X,
        ivar,
        observed_valid,
        wavelength,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    val_fraction_of_train = args.val_size / max(1.0 - args.test_size, 1e-6)
    (
        X_train,
        X_val,
        ivar_train,
        ivar_val,
        valid_train,
        valid_val,
        wavelength_train,
        wavelength_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_train_pool,
        ivar_train_pool,
        valid_train_pool,
        wavelength_train_pool,
        y_train_pool,
        test_size=val_fraction_of_train,
        random_state=42,
    )

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    ivar_train_t = torch.from_numpy(np.log1p(np.clip(ivar_train, 0.0, None))).float()
    ivar_val_t = torch.from_numpy(np.log1p(np.clip(ivar_val, 0.0, None))).float()
    ivar_test_t = torch.from_numpy(np.log1p(np.clip(ivar_test, 0.0, None))).float()
    valid_train_t = torch.from_numpy(valid_train).float()
    valid_val_t = torch.from_numpy(valid_val).float()
    valid_test_t = torch.from_numpy(valid_test).float()
    wavelength_train_t = torch.from_numpy(wavelength_train).float()
    wavelength_val_t = torch.from_numpy(wavelength_val).float()
    wavelength_test_t = torch.from_numpy(wavelength_test).float()
    y_train_raw = y_train.astype(np.float32)
    y_val_raw = y_val.astype(np.float32)
    y_test_raw = y_test.astype(np.float32)
    y_train_t = torch.from_numpy(encode_redshift_target(y_train_raw)).float()
    y_val_t = torch.from_numpy(encode_redshift_target(y_val_raw)).float()
    y_test_t = torch.from_numpy(encode_redshift_target(y_test_raw)).float()
    y_train_weights = torch.from_numpy(redshift_sample_weights(y_train_raw)).float()
    y_val_weights = torch.from_numpy(redshift_sample_weights(y_val_raw)).float()

    x_mean = X_train_t.mean(dim=0, keepdim=True)
    x_std = X_train_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    X_train_t = (X_train_t - x_mean) / x_std
    X_val_t = (X_val_t - x_mean) / x_std
    X_test_t = (X_test_t - x_mean) / x_std

    ivar_mean = ivar_train_t.mean(dim=0, keepdim=True)
    ivar_std = ivar_train_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    ivar_train_t = (ivar_train_t - ivar_mean) / ivar_std
    ivar_val_t = (ivar_val_t - ivar_mean) / ivar_std
    ivar_test_t = (ivar_test_t - ivar_mean) / ivar_std

    def build_input_tensor(
        flux_tensor: torch.Tensor,
        ivar_tensor: torch.Tensor,
        valid_tensor: torch.Tensor,
    ) -> torch.Tensor:
        channels = [flux_tensor]
        if args.use_ivar_channel:
            channels.append(ivar_tensor)
        if args.use_validity_channel:
            channels.append(valid_tensor)
        return torch.stack(channels, dim=1)

    val_mask = make_token_mask(X_val_t.shape, mask_prob=args.mask_prob, seed=42)
    test_mask = make_token_mask(X_test_t.shape, mask_prob=args.mask_prob, seed=43)

    logger.info(
        "Loaded %d spectra | train=%d val=%d test=%d",
        len(X),
        len(X_train_t),
        len(X_val_t),
        len(X_test_t),
    )

    X_train_input_t = build_input_tensor(X_train_t, ivar_train_t, valid_train_t)
    X_val_input_t = build_input_tensor(X_val_t, ivar_val_t, valid_val_t)
    X_test_input_t = build_input_tensor(X_test_t, ivar_test_t, valid_test_t)

    y_train_raw_t = torch.from_numpy(y_train_raw).float()
    y_val_raw_t = torch.from_numpy(y_val_raw).float()
    y_test_raw_t = torch.from_numpy(y_test_raw).float()

    input_channels = X_train_input_t.shape[1]
    model_class = SpectraTransformerWithRedshiftToken if args.architecture == "redshift_token" else SpectraTransformerWithRedshiftHead
    model = model_class(
        input_length=X_train_t.shape[1],
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        input_channels=input_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    recon_loss_fn = nn.MSELoss()
    if args.redshift_loss == "smoothl1":
        redshift_loss_fn = nn.SmoothL1Loss(reduction="none")
    else:
        redshift_loss_fn = nn.MSELoss(reduction="none")

    def decode_redshift_target_torch(encoded_redshift: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(1.0 - encoded_redshift, min=1e-6)
        return encoded_redshift / denom

    def compute_redshift_loss(
        pred_encoded: torch.Tensor,
        target_encoded: torch.Tensor,
        target_raw: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if args.redshift_objective == "relative":
            pred_raw = decode_redshift_target_torch(pred_encoded)
            residual = (pred_raw - target_raw) / torch.clamp(1.0 + target_raw, min=1e-6)
            if args.redshift_loss == "smoothl1":
                per_item = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
            else:
                per_item = residual.square()
        else:
            per_item = redshift_loss_fn(pred_encoded, target_encoded)
        return (per_item * sample_weights).mean()
    best_val_loss = float("inf")
    best_score = float("inf")
    best_state = None
    epochs_without_improvement = 0

    batch_size = min(args.batch_size, len(X_train_t))
    gradient_accumulation = max(1, args.gradient_accumulation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = None
    resume_stage_name = None
    resume_next_epoch = 0
    stage_order = {"stage1": 1, "stage2": 2}
    if args.resume is not None:
        logger.info("Resuming checkpoint: %s", args.resume)
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state"])
        best_state = resume_checkpoint.get("best_state")
        best_val_loss = float(resume_checkpoint.get("best_val_loss", best_val_loss))
        best_score = float(resume_checkpoint.get("best_score", best_val_loss))
        epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))
        resume_stage_name = resume_checkpoint.get("stage_name")
        resume_next_epoch = int(resume_checkpoint.get("next_epoch", 0))
        if best_state is not None:
            best_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}

    def build_mask(
        wavelength_batch: torch.Tensor,
        redshift_batch: torch.Tensor,
        *,
        seed: int,
        mask_prob: float,
    ) -> torch.Tensor:
        if args.mask_strategy == "domain":
            return make_domain_informed_mask(
                wavelength_batch,
                redshift_batch,
                mask_prob=mask_prob,
                seed=seed,
                line_sigma_angstrom=args.domain_line_sigma,
                line_boost=args.domain_line_boost,
                break_sigma_angstrom=args.domain_break_sigma,
                break_boost=args.domain_break_boost,
            ).to(device)
        return (torch.rand(wavelength_batch.shape, device=device) < mask_prob)

    def apply_input_mask(input_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return input_batch * (1.0 - mask.unsqueeze(1))

    def run_stage(
        *,
        stage_name: str,
        num_epochs: int,
        alpha: float,
        mask_prob: float,
        learning_rate: float,
    ) -> None:
        nonlocal best_val_loss, best_score, best_state, epochs_without_improvement, optimizer
        if num_epochs <= 0:
            return
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        if (
            resume_checkpoint is not None
            and resume_stage_name == stage_name
            and "optimizer_state" in resume_checkpoint
        ):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if resume_stage_name is not None and stage_order.get(resume_stage_name, 0) > stage_order[stage_name]:
            logger.info("Skipping completed %s because resume checkpoint is from %s", stage_name, resume_stage_name)
            return
        start_epoch = resume_next_epoch if resume_stage_name == stage_name else 0
        if start_epoch >= num_epochs:
            logger.info("Skipping completed %s (%d/%d epochs)", stage_name, start_epoch, num_epochs)
            return
        logger.info(
            "Starting %s for epochs %d-%d (batch_size=%d, accumulation=%d, effective_batch=%d, patience=%d, alpha=%.2f, recon_weight=%.2f, best_metric=%s, mask_prob=%.2f, channels=%d, d_model=%d, layers=%d, nhead=%d, patch_size=%d)",
            stage_name,
            start_epoch + 1,
            num_epochs,
            batch_size,
            gradient_accumulation,
            batch_size * gradient_accumulation,
            args.patience,
            alpha,
            args.reconstruction_weight,
            args.best_metric,
            mask_prob,
            input_channels,
            args.d_model,
            args.num_layers,
            args.nhead,
            args.patch_size,
        )
        model.train()
        if resume_stage_name != stage_name:
            epochs_without_improvement = 0

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_z_loss = 0.0
            num_batches = 0
            accumulation_step = 0

            if args.sampling_strategy == "tail":
                sampling_weights = torch.clamp(y_train_weights.detach().cpu(), min=1e-6) ** args.tail_power
                sampled_idx = torch.multinomial(sampling_weights, len(X_train_input_t), replacement=True)
                perm = sampled_idx
            else:
                perm = torch.randperm(len(X_train_input_t))

            batch_starts = range(0, len(X_train_input_t), batch_size)
            if tqdm is not None:
                batch_starts = tqdm(batch_starts, desc=f"{stage_name} {epoch + 1:03d}/{num_epochs}", leave=False)

            optimizer.zero_grad(set_to_none=True)
            for start in batch_starts:
                idx = perm[start : start + batch_size]
                xb_input = X_train_input_t[idx].to(device)
                xb_flux = X_train_t[idx].to(device)
                yb = y_train_t[idx].to(device)
                wavelength_b = wavelength_train_t[idx].to(device)
                yb_raw = y_train_raw_t[idx].to(device)
                yb_weights = y_train_weights[idx].to(device)

                mask = build_mask(
                    wavelength_b,
                    yb_raw,
                    seed=args.mask_seed + epoch * 1000 + int(start),
                    mask_prob=mask_prob,
                ).float()
                masked_xb = apply_input_mask(xb_input, mask)

                recon, redshift = model(masked_xb)
                mask_bool = mask.bool()
                recon_loss = (
                    recon_loss_fn(recon[mask_bool], xb_flux[mask_bool]) if bool(mask_bool.any().item()) else recon.sum() * 0.0
                )
                z_loss = compute_redshift_loss(redshift, yb, yb_raw, yb_weights)
                loss = args.reconstruction_weight * recon_loss + alpha * z_loss

                (loss / gradient_accumulation).backward()
                accumulation_step += 1
                if accumulation_step % gradient_accumulation == 0 or start + batch_size >= len(X_train_input_t):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if args.throttle_sleep_ms > 0:
                        time.sleep(args.throttle_sleep_ms / 1000.0)

                epoch_loss += float(loss.item())
                epoch_recon_loss += float(recon_loss.item())
                epoch_z_loss += float(z_loss.item())
                num_batches += 1

            train_duration = time.perf_counter() - epoch_start
            model.eval()
            val_start = time.perf_counter()
            val_loss = 0.0
            val_recon_total = 0.0
            val_z_total = 0.0
            val_batches = 0
            val_pred_chunks = []
            val_actual_chunks = []
            val_starts = range(0, len(X_val_input_t), batch_size)
            if tqdm is not None:
                val_starts = tqdm(val_starts, desc=f"{stage_name} val {epoch + 1:03d}/{num_epochs}", leave=False)
            with torch.no_grad():
                for start in val_starts:
                    xb_input = X_val_input_t[start : start + batch_size].to(device)
                    xb_flux = X_val_t[start : start + batch_size].to(device)
                    yb = y_val_t[start : start + batch_size].to(device)
                    wavelength_b = wavelength_val_t[start : start + batch_size].to(device)
                    yb_raw = y_val_raw_t[start : start + batch_size].to(device)
                    yb_weights = y_val_weights[start : start + batch_size].to(device)
                    if args.mask_strategy == "domain":
                        eval_mask = build_mask(
                            wavelength_b,
                            yb_raw,
                            seed=args.mask_seed + int(start),
                            mask_prob=mask_prob,
                        ).float()
                    else:
                        eval_mask = val_mask[start : start + batch_size].to(device).float()
                    masked_X_val = apply_input_mask(xb_input, eval_mask)
                    val_recon, val_redshift = model(masked_X_val)
                    eval_mask_bool = eval_mask.bool()
                    val_recon_loss = (
                        recon_loss_fn(val_recon[eval_mask_bool], xb_flux[eval_mask_bool])
                        if bool(eval_mask_bool.any().item())
                        else val_recon.sum() * 0.0
                    )
                    val_z_loss = compute_redshift_loss(
                        val_redshift,
                        yb,
                        yb_raw,
                        yb_weights,
                    )
                    batch_val_loss = float(args.reconstruction_weight * val_recon_loss.item() + alpha * val_z_loss.item())
                    val_loss += batch_val_loss
                    val_recon_total += float(val_recon_loss.item())
                    val_z_total += float(val_z_loss.item())
                    val_batches += 1
                    val_pred_chunks.append(val_redshift.detach().cpu())
                    val_actual_chunks.append(yb_raw.detach().cpu())
            val_loss = val_loss / max(val_batches, 1)
            val_recon_mean = val_recon_total / max(val_batches, 1)
            val_z_mean = val_z_total / max(val_batches, 1)
            val_pred = decode_redshift_target(torch.cat(val_pred_chunks, dim=0).numpy())
            val_actual = torch.cat(val_actual_chunks, dim=0).numpy()
            val_error = val_pred - val_actual
            val_mae = float(np.mean(np.abs(val_error)))
            val_rmse = float(np.sqrt(np.mean(val_error**2)))
            val_bias = float(np.mean(val_error))
            metric_value = {
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }[args.best_metric]
            val_duration = time.perf_counter() - val_start
            logger.info(
                "%s epoch %03d/%d | loss=%.6f | recon=%.6f | z=%.6f | val_loss=%.6f | val_recon=%.6f | val_z=%.6f | val_mae=%.6f | val_rmse=%.6f | val_bias=%.6f | best_metric=%s | train_time=%.1fs | val_time=%.1fs | epoch_time=%.1fs",
                stage_name,
                epoch + 1,
                num_epochs,
                epoch_loss / max(num_batches, 1),
                epoch_recon_loss / max(num_batches, 1),
                epoch_z_loss / max(num_batches, 1),
                val_loss,
                val_recon_mean,
                val_z_mean,
                val_mae,
                val_rmse,
                val_bias,
                args.best_metric,
                train_duration,
                val_duration,
                time.perf_counter() - epoch_start,
            )

            if metric_value < best_score:
                best_val_loss = val_loss
                best_score = metric_value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            checkpoint_payload = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_val_loss": best_val_loss,
                "best_score": best_score,
                "best_metric": args.best_metric,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_bias": val_bias,
                "epochs_without_improvement": epochs_without_improvement,
                "stage_name": stage_name,
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "args": vars(args),
                "input_channels": input_channels,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "source": source_path,
            }
            epoch_checkpoint_path = checkpoint_dir / f"{stage_name}_epoch_{epoch + 1:03d}.pt"
            latest_checkpoint_path = checkpoint_dir / "latest.pt"
            torch.save(checkpoint_payload, epoch_checkpoint_path)
            torch.save(checkpoint_payload, latest_checkpoint_path)
            logger.info("Saved epoch checkpoint to: %s", epoch_checkpoint_path)

            if not args.disable_early_stopping and epochs_without_improvement >= args.patience:
                logger.info("%s early stopping triggered after %d epochs", stage_name, epoch + 1)
                break

            model.train()

    run_stage(
        stage_name="stage1",
        num_epochs=args.epochs,
        alpha=args.alpha,
        mask_prob=args.mask_prob,
        learning_rate=args.lr,
    )
    run_stage(
        stage_name="stage2",
        num_epochs=args.stage2_epochs,
        alpha=args.stage2_alpha,
        mask_prob=args.stage2_mask_prob,
        learning_rate=args.stage2_lr,
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    logger.info("Running final test evaluation")
    recon_chunks = []
    pred_chunks = []
    mask_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_test_input_t), batch_size):
            xb_input = X_test_input_t[start : start + batch_size].to(device)
            wavelength_b = wavelength_test_t[start : start + batch_size].to(device)
            yb_raw = y_test_raw_t[start : start + batch_size].to(device)
            if args.mask_strategy == "domain":
                eval_test_mask = build_mask(
                    wavelength_b,
                    yb_raw,
                    seed=args.mask_seed + 999 + int(start),
                    mask_prob=args.mask_prob,
                ).float()
            else:
                eval_test_mask = test_mask[start : start + batch_size].to(device).float()
            masked_X_test = apply_input_mask(xb_input, eval_test_mask)
            test_recon, y_pred_t = model(masked_X_test)
            recon_chunks.append(test_recon.detach().cpu())
            pred_chunks.append(y_pred_t.detach().cpu())
            mask_chunks.append(eval_test_mask.detach().cpu())
        recon_pred = torch.cat(recon_chunks, dim=0).numpy()
        y_pred = decode_redshift_target(torch.cat(pred_chunks, dim=0).numpy())
        test_mask_np = torch.cat(mask_chunks, dim=0).numpy().astype(bool)

    metrics = redshift_metrics(y_test, y_pred)
    metrics["reconstruction"] = masked_normalized_reconstruction_metrics(X_test, recon_pred, test_mask_np)
    metrics["reconstruction_raw"] = spectrum_reconstruction_metrics(X_test, recon_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "redshift_predictions.json"
    plot_path = args.output_dir / "redshift_pred_vs_actual.png"
    recon_plot_path = args.output_dir / "spectrum_reconstruction.png"
    metrics_path = args.output_dir / "redshift_metrics.json"
    best_model_path = args.output_dir / "best_model.pt"

    payload = {
        "actual": y_test.tolist(),
        "predicted": y_pred.tolist(),
        "metrics": metrics,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "source": source_path,
        "best_val_loss": best_val_loss,
        "mask_strategy": args.mask_strategy,
        "redshift_loss": args.redshift_loss,
        "redshift_objective": args.redshift_objective,
        "sampling_strategy": args.sampling_strategy,
        "tail_power": args.tail_power,
        "use_ivar_channel": args.use_ivar_channel,
        "use_validity_channel": args.use_validity_channel,
        "stage2_epochs": args.stage2_epochs,
    }

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if best_state is not None:
        torch.save(best_state, best_model_path)

    logger.info("Saving predictions, metrics, checkpoint, and plot")
    plot_predicted_vs_actual_redshift(y_test, y_pred, plot_path, title="Real-data redshift baseline")
    plot_flux_reconstruction(X_test[0], recon_pred[0], recon_plot_path, title="Raw-spectrum reconstruction example")

    logger.info("Loaded records: %d", len(X))
    logger.info("Train/val/test sizes: %d / %d / %d", len(y_train), len(y_val), len(y_test))
    logger.info("Metrics: %s", metrics)
    logger.info("Saved predictions to: %s", predictions_path)
    logger.info("Saved plot to: %s", plot_path)
    logger.info("Saved reconstruction plot to: %s", recon_plot_path)
    logger.info("Saved metrics to: %s", metrics_path)
    logger.info("Saved best model to: %s", best_model_path)


if __name__ == "__main__":
    main()
