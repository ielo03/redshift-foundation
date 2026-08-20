#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator
from queue import Full, Queue
from threading import Event, Thread

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROJECT_SRC = PROJECT_ROOT / "final-project" / "src"
sys.path.insert(0, str(FINAL_PROJECT_SRC))

from data_pipeline import materialized_shard_paths, normalize_flux_batch  # noqa: E402
from build_split_manifest import assignment  # noqa: E402
from train_experiments import (  # noqa: E402
    DynamicSpectraTransformerWithRedshiftToken,
)
from redshift_metrics import redshift_metrics_numpy  # noqa: E402


TRAIN_FRACTION = 0.85
VALIDATION_FRACTION = 0.10
TRAINING_SPLITS = {"train", "validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream materialized DESI shards for redshift-token training.")
    parser.add_argument("--input", type=Path, default=Path("data/materialized_training_v1"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="PSCRATCH-backed directory for latest checkpoints, metrics, and history.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Home/project directory for only the run-best resumable checkpoint and metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Deprecated alias for --checkpoint-dir.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-spectra-per-epoch", type=int, default=100_000)
    parser.add_argument("--max-val-spectra", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument(
        "--mask-mode",
        choices=["contiguous", "independent"],
        default="contiguous",
        help="Contiguous wavelength-span reconstruction masks by default.",
    )
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.25, help="Weight for GALAXY/QSO cross-entropy.")
    parser.add_argument("--target-label-weight", type=float, default=0.25, help="Weight for BGS/LRG/ELG/QSO-target BCE loss.")
    parser.add_argument("--tail-power", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=61)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variable-min-fraction",
        type=float,
        default=1.0,
        help="Sample a contiguous crop from this fraction through 100%% of each training spectrum.",
    )
    parser.add_argument(
        "--random-transforms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply gain, tilt, offset, and noise to training crops only.",
    )
    parser.add_argument(
        "--augmentation-policy",
        choices=["combined", "exclusive"],
        default="combined",
        help="Apply configured crop/transforms together, or choose clean/crop/transform exclusively per batch.",
    )
    parser.add_argument("--exclusive-clean-prob", type=float, default=0.25)
    parser.add_argument("--exclusive-crop-prob", type=float, default=0.375)
    parser.add_argument("--exclusive-transform-prob", type=float, default=0.375)
    parser.add_argument("--transform-gain", type=float, default=0.10)
    parser.add_argument("--transform-tilt", type=float, default=0.10)
    parser.add_argument("--transform-offset", type=float, default=0.05)
    parser.add_argument("--transform-noise", type=float, default=0.02)
    parser.add_argument("--use-ivar-channel", action="store_true")
    parser.add_argument("--use-validity-channel", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bfloat16 autocast when a GPU is available.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CPU batch tensors before non-blocking GPU transfer.",
    )
    parser.add_argument("--prefetch-batches", type=int, default=2, help="CPU batches prepared ahead of GPU consumption.")
    parser.add_argument(
        "--profile-sync",
        action="store_true",
        help="Synchronize CUDA every batch for precise profiling; disables normal CPU/GPU overlap.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Opt into torch.compile(dynamic=True) after a dedicated throughput smoke test.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without validation selection-metric improvement; omit to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation selection-metric decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["z_sigma_nmad", "z_mae"],
        default="z_sigma_nmad",
        help="Validation redshift metric minimized for best-checkpoint and early-stopping decisions.",
    )
    parser.add_argument("--num-shards", type=int, default=None, help="Optional cap for debugging.")
    return parser.parse_args()


def encode_redshift(z: torch.Tensor) -> torch.Tensor:
    return z / (1.0 + z)


def decode_redshift(encoded: torch.Tensor) -> torch.Tensor:
    return encoded / torch.clamp(1.0 - encoded, min=1e-6)


def sigma_nmad(normalized_residual: torch.Tensor) -> torch.Tensor:
    """Robust normalized redshift scatter: 1.4826 * median absolute deviation."""

    normalized_residual = normalized_residual.float()
    center = torch.quantile(normalized_residual, 0.5)
    return 1.4826 * torch.quantile((normalized_residual - center).abs(), 0.5)


def load_shard(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = ["flux", "ivar", "valid", "wavelength", "z", "spectype", "targetid", "records"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    valid = payload["valid"].bool()
    invalid = ~valid
    flux_raw = payload["flux"].float()
    flux = normalize_flux_batch(flux_raw, invalid)
    valid_float = valid.to(flux_raw.dtype)
    counts = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (flux_raw * valid_float).sum(dim=1, keepdim=True) / counts
    variance = ((flux_raw - mean).square() * valid_float).sum(dim=1, keepdim=True) / counts
    flux_scale = variance.sqrt().clamp_min(1e-6)
    raw_ivar = torch.clamp(payload["ivar"].float(), min=0.0)
    normalized_ivar = raw_ivar * flux_scale.square()
    log_ivar = torch.log1p(normalized_ivar).masked_fill(~valid, 0.0)
    split_rows: list[str] = []
    for record in payload["records"]:
        record_split = record.get("split") or assignment(record, TRAIN_FRACTION, VALIDATION_FRACTION)
        if record_split not in {"train", "validation", "benchmark"}:
            raise RuntimeError(f"Unexpected split {record_split!r} in {path}")
        split_rows.extend([record_split] * int(record["n_spectra"]))
    if len(split_rows) != int(flux.shape[0]):
        raise RuntimeError(f"Shard record metadata does not match its spectra count: {path}")
    if "benchmark" in split_rows:
        raise RuntimeError(
            f"{path} contains benchmark spectra from the pre-split materialization. "
            "Training requires split-clean shards; materialize data/preprocessed/split_manifest_v1.jsonl "
            "into a new output directory."
        )
    return {
        "flux": flux,
        "ivar": log_ivar,
        "valid": valid.float(),
        "z": payload["z"].float().view(-1),
        "spectype": payload["spectype"].long().view(-1),
        "targetid": payload["targetid"].long().view(-1),
        "wavelength": payload["wavelength"].float(),
        "split": np.asarray(split_rows),
    } | ({"target_labels": payload["target_labels"].float()} if "target_labels" in payload else {})


def iter_batches(
    shard_paths: list[Path],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    max_spectra: int | None,
    shuffle_shards: bool,
    shuffle_rows: bool,
    split: str,
    pin_memory: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    if split not in TRAINING_SPLITS:
        raise ValueError(f"Training may only request train or validation rows, got {split!r}")
    rng = random.Random(seed + epoch * 10_000)
    paths = list(shard_paths)
    if shuffle_shards:
        rng.shuffle(paths)

    yielded = 0
    for shard_i, shard_path in enumerate(paths):
        shard = load_shard(shard_path)
        allowed = torch.from_numpy(np.flatnonzero(shard.pop("split") == split))
        n = int(allowed.numel())
        if n == 0:
            continue
        if shuffle_rows:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch * 1_000_003 + shard_i)
            order = allowed[torch.randperm(n, generator=generator)]
        else:
            order = allowed

        for start in range(0, n, batch_size):
            if max_spectra is not None and yielded >= max_spectra:
                return
            idx = order[start : start + batch_size]
            if max_spectra is not None:
                remaining = max_spectra - yielded
                idx = idx[:remaining]
            if idx.numel() == 0:
                continue
            yielded += int(idx.numel())
            yielded_batch = {
                key: value if key == "wavelength" else value[idx]
                for key, value in shard.items()
            }
            if pin_memory:
                yielded_batch = {key: value.pin_memory() for key, value in yielded_batch.items()}
            yield yielded_batch


def prefetch_batches(batches: Iterator[dict[str, torch.Tensor]], depth: int) -> Iterator[dict[str, torch.Tensor]]:
    """Prepare shard batches on one background thread while CUDA consumes prior work."""

    if depth <= 0:
        yield from batches
        return
    queue: Queue[object] = Queue(maxsize=depth)
    sentinel = object()
    stop = Event()

    def put(item: object) -> bool:
        """Enqueue without trapping the producer during early consumer exit."""

        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def producer() -> None:
        try:
            for item in batches:
                if not put(item):
                    return
        except BaseException as exc:  # propagate loader failures to the main thread
            put(exc)
        finally:
            put(sentinel)

    thread = Thread(target=producer, daemon=True)
    thread.start()
    try:
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        # PyTorch's pinned-memory allocations occur in the producer.  Joining
        # it before interpreter teardown avoids leaving a live C++ allocator
        # thread, which can otherwise abort Python after a successful epoch.
        stop.set()
        thread.join()


def build_input_tensor(batch: dict[str, torch.Tensor], *, use_ivar_channel: bool, use_validity_channel: bool) -> torch.Tensor:
    channels = [batch["flux"]]
    if use_ivar_channel:
        channels.append(batch["ivar"])
    if use_validity_channel:
        channels.append(batch["valid"])
    return torch.stack(channels, dim=1)


def full_length_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return materialized tensors by reference for the control/validation path.

    This deliberately performs no per-spectrum copying, crop selection,
    repadding, or transform work. The shared wavelength grid is expanded only
    after transfer to the device, where ``expand`` is a view rather than a
    B-by-L host allocation.
    """

    if batch["wavelength"].ndim != 1:
        raise ValueError(f"Expected one shared wavelength grid, got {batch['wavelength'].shape}")
    return {
        "input_flux": batch["flux"],
        "target_flux": batch["flux"],
        "ivar": batch["ivar"],
        "valid": batch["valid"],
        "wavelength": batch["wavelength"],
    }


def transform_log_ivar(
    log_ivar: torch.Tensor,
    flux_multiplier: torch.Tensor,
    added_noise_std: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Propagate normalized IVAR through multiplicative and noise transforms."""

    normalized_ivar = torch.expm1(log_ivar).clamp_min(0.0)
    input_variance = torch.where(
        normalized_ivar > 0,
        flux_multiplier.square() / normalized_ivar.clamp_min(1e-20) + added_noise_std.square(),
        torch.full_like(normalized_ivar, float("inf")),
    )
    transformed_ivar = torch.where(
        torch.isfinite(input_variance) & (input_variance > 0),
        input_variance.reciprocal(),
        torch.zeros_like(input_variance),
    )
    return torch.log1p(transformed_ivar).masked_fill(~valid.bool(), 0.0)


def contiguous_reconstruction_mask(
    valid: torch.Tensor,
    padding: torch.Tensor,
    *,
    mask_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Vectorized contiguous valid-span masking on the current device.

    A row selects one valid wavelength run large enough for its target span.
    Fragmented rows with no such run fall back to a contiguous interval in
    valid-pixel order, preserving the requested number of masked targets.
    """

    usable = valid.bool() & ~padding
    batch_size, length = usable.shape
    count = usable.sum(dim=1)
    target = torch.round(count.float() * mask_prob).long()
    if not bool((target >= 0).all()) or not 0.0 <= mask_prob <= 1.0:
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    left = torch.nn.functional.pad(usable[:, :-1], (1, 0), value=False)
    run_start = usable & ~left
    run_id = run_start.cumsum(dim=1) - 1
    safe_id = run_id.clamp_min(0)
    run_lengths = torch.zeros((batch_size, length), dtype=torch.long, device=valid.device)
    run_lengths.scatter_add_(1, safe_id, usable.long())
    eligible = run_lengths >= target.unsqueeze(1)
    scores = torch.rand((batch_size, length), device=valid.device, generator=generator)
    chosen_run = scores.masked_fill(~eligible, -1.0).argmax(dim=1)
    has_eligible = eligible.any(dim=1) & (target > 0)
    chosen_length = run_lengths.gather(1, chosen_run.unsqueeze(1)).squeeze(1)
    offset = torch.floor(
        torch.rand(batch_size, device=valid.device, generator=generator)
        * (chosen_length - target + 1).clamp_min(1)
    ).long()
    valid_rank = usable.long().cumsum(dim=1) - 1
    start_rank = torch.where(run_start, valid_rank, torch.zeros_like(valid_rank)).cummax(dim=1).values
    within_run = valid_rank - start_rank
    one_run_mask = usable & (safe_id == chosen_run.unsqueeze(1)) & (within_run >= offset.unsqueeze(1)) & (within_run < (offset + target).unsqueeze(1))

    # Rare fragmented fallback: a contiguous interval in valid-pixel order.
    fallback_start = torch.floor(
        torch.rand(batch_size, device=valid.device, generator=generator)
        * (count - target + 1).clamp_min(1)
    ).long()
    fallback = usable & (valid_rank >= fallback_start.unsqueeze(1)) & (valid_rank < (fallback_start + target).unsqueeze(1))
    return torch.where(has_eligible.unsqueeze(1), one_run_mask, fallback)


def gpu_crop_and_transform_batch(
    flux: torch.Tensor,
    valid: torch.Tensor,
    ivar: torch.Tensor,
    wavelength: torch.Tensor,
    *,
    args: argparse.Namespace,
    epoch: int,
    batch_index: int,
    crop_length: int,
    min_fraction: float,
    apply_transforms: bool,
) -> dict[str, torch.Tensor]:
    """Create variable crops and corruptions with batched device operations."""

    batch_size, full_length = flux.shape
    min_length = max(64, int(full_length * min_fraction))
    generator = torch.Generator(device=flux.device)
    generator.manual_seed(args.seed + epoch * 1_000_003 + batch_index)
    # One sampled length per batch removes padding waste in self-attention;
    # each spectrum still independently samples its wavelength-window start.
    crop_length = max(min_length, min(full_length, crop_length))
    lengths = torch.full((batch_size,), crop_length, device=flux.device, dtype=torch.long)
    positions = torch.arange(crop_length, device=flux.device)
    starts = torch.floor(torch.rand(batch_size, device=flux.device, generator=generator) * (full_length - crop_length + 1)).long()
    padding = torch.zeros((batch_size, crop_length), dtype=torch.bool, device=flux.device)
    indices = (starts.unsqueeze(1) + positions.unsqueeze(0)).clamp_max(full_length - 1)
    target = flux.gather(1, indices).masked_fill(padding, 0.0)
    cropped_ivar = ivar.gather(1, indices).masked_fill(padding, 0.0)
    cropped_valid = valid.gather(1, indices).masked_fill(padding, 0.0)
    cropped_wave = wavelength.unsqueeze(0).expand(batch_size, -1).gather(1, indices).masked_fill(padding, 0.0)

    if apply_transforms:
        relative_position = 2.0 * positions.unsqueeze(0) / (lengths.unsqueeze(1) - 1).clamp_min(1) - 1.0
        gain = 1.0 + (2.0 * torch.rand(batch_size, 1, device=flux.device, generator=generator) - 1.0) * args.transform_gain
        tilt = (2.0 * torch.rand(batch_size, 1, device=flux.device, generator=generator) - 1.0) * args.transform_tilt
        offset = (2.0 * torch.rand(batch_size, 1, device=flux.device, generator=generator) - 1.0) * args.transform_offset
        noise_std = torch.rand(batch_size, 1, device=flux.device, generator=generator) * args.transform_noise
        transformed = target * (gain + tilt * relative_position) + offset
        transformed = transformed + torch.randn(target.shape, device=flux.device, dtype=target.dtype, generator=generator) * noise_std
        input_flux = torch.where(cropped_valid.bool(), transformed, target)
        # The input confidence must describe the transformed input.  Gain and
        # tilt scale the original standard deviation; injected independent
        # Gaussian noise adds variance in quadrature. Offset adds no variance.
        flux_multiplier = gain + tilt * relative_position
        cropped_ivar = transform_log_ivar(cropped_ivar, flux_multiplier, noise_std, cropped_valid)
    else:
        input_flux = target
    return {"input_flux": input_flux, "target_flux": target, "ivar": cropped_ivar, "valid": cropped_valid, "wavelength": cropped_wave, "padding": padding, "generator": generator}


def redshift_weights(z: torch.Tensor, tail_power: float) -> torch.Tensor:
    weights = 1.0 + torch.clamp(z, min=0.0, max=4.0) / 4.0
    return weights.pow(tail_power)


def run_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    shard_paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    max_spectra: int | None,
    steps_limit: int | None,
    split: str,
    prediction_output: Path | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch.

    Training gives all heads the same randomly masked input.  Validation keeps
    redshift and SPECTYPE evaluation on the clean full spectrum, while scoring
    reconstruction on a separately masked copy with a fixed per-batch seed.
    This makes checkpoint selection stable and measures the deployed inference
    inputs for redshift/classification rather than an arbitrary missing span.
    """
    training = optimizer is not None
    model.train(training)
    metric_sums = {key: torch.zeros((), device=device) for key in ("loss", "recon", "z", "class", "class_accuracy", "target", "target_accuracy", "mae")}
    predicted_redshifts: list[torch.Tensor] = []
    actual_redshifts: list[torch.Tensor] = []
    targetids: list[torch.Tensor] = []
    actual_spectypes: list[torch.Tensor] = []
    predicted_spectypes: list[torch.Tensor] = []
    actual_target_labels: list[torch.Tensor] = []
    predicted_target_labels: list[torch.Tensor] = []
    spectra_total = 0
    loader_seconds = 0.0
    input_seconds = 0.0
    compute_submit_seconds = 0.0
    batches = 0
    crop_rng = random.Random(args.seed + epoch * 1_000_003)
    mode_rng = random.Random(args.seed + epoch * 1_000_003 + 97_531)
    augmentation_batch_counts = {"clean": 0, "crop": 0, "transform": 0, "combined": 0}
    previous_batch_end = time.perf_counter()
    epoch_start = previous_batch_end

    raw_batches = iter_batches(
        shard_paths, batch_size=args.batch_size, seed=args.seed, epoch=epoch,
        max_spectra=max_spectra, shuffle_shards=training, shuffle_rows=training,
        split=split, pin_memory=args.pin_memory and device.type == "cuda",
    )
    for batch in prefetch_batches(raw_batches, args.prefetch_batches):
        batch_arrival = time.perf_counter()
        loader_seconds += batch_arrival - previous_batch_end
        data_start = time.perf_counter()
        augmentation_mode = "clean"
        if training and args.augmentation_policy == "exclusive":
            draw = mode_rng.random()
            if draw < args.exclusive_clean_prob:
                augmentation_mode = "clean"
            elif draw < args.exclusive_clean_prob + args.exclusive_crop_prob:
                augmentation_mode = "crop"
            else:
                augmentation_mode = "transform"
        elif training and (args.variable_min_fraction < 1.0 or args.random_transforms):
            augmentation_mode = "combined"
        requires_augmentation = augmentation_mode != "clean"
        augmentation_batch_counts[augmentation_mode] += 1
        if requires_augmentation:
            full_flux = batch["flux"].to(device, non_blocking=True)
            full_valid = batch["valid"].to(device, non_blocking=True)
            full_ivar = batch["ivar"].to(device, non_blocking=True)
            shared_wavelength = batch["wavelength"].to(device, non_blocking=True)
            min_fraction = args.variable_min_fraction if augmentation_mode in ("crop", "combined") else 1.0
            prepared = gpu_crop_and_transform_batch(
                full_flux,
                full_valid,
                full_ivar,
                shared_wavelength,
                args=args,
                epoch=epoch,
                batch_index=batches,
                crop_length=crop_rng.randint(max(64, int(full_flux.shape[1] * min_fraction)), full_flux.shape[1]),
                min_fraction=min_fraction,
                apply_transforms=args.random_transforms and augmentation_mode in ("transform", "combined"),
            )
        else:
            prepared = full_length_batch(batch)
        flux = prepared["target_flux"] if requires_augmentation else prepared["target_flux"].to(device, non_blocking=True)
        valid = prepared["valid"] if requires_augmentation else prepared["valid"].to(device, non_blocking=True)
        if requires_augmentation:
            padding = prepared["padding"].to(device, non_blocking=True)
            wavelength = prepared["wavelength"].to(device, non_blocking=True)
        else:
            padding = torch.zeros_like(valid, dtype=torch.bool)
            wavelength = prepared["wavelength"].to(device, non_blocking=True).unsqueeze(0).expand(flux.shape[0], -1)
        z = batch["z"].to(device, non_blocking=True)
        spectype = batch["spectype"].to(device, non_blocking=True)
        target_labels = batch.get("target_labels")
        if target_labels is not None:
            target_labels = target_labels.to(device, non_blocking=True)
        input_batch = build_input_tensor(
            {
                "flux": flux if not requires_augmentation else prepared["input_flux"].to(device, non_blocking=True),
                "ivar": prepared["ivar"].to(device, non_blocking=True) if args.use_ivar_channel else flux,
                "valid": valid,
            },
            use_ivar_channel=args.use_ivar_channel,
            use_validity_channel=args.use_validity_channel,
        )
        input_seconds += time.perf_counter() - data_start

        compute_start = time.perf_counter()
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.mixed_precision and device.type == "cuda" else nullcontext()
        with autocast:
            # Validation masking is deliberately fixed across epochs.  The
            # batch order is fixed too, so each row is evaluated with the same
            # reconstruction gap every time.
            mask_generator = prepared.get("generator")
            if not training:
                mask_generator = torch.Generator(device=device)
                mask_generator.manual_seed(args.seed + 97_000_003 + batches)
            if args.mask_mode == "independent":
                mask = (
                    torch.rand(flux.shape, device=device, generator=mask_generator) < args.mask_prob
                ) & valid.bool() & ~padding
            else:
                mask = contiguous_reconstruction_mask(valid, padding, mask_prob=args.mask_prob, generator=mask_generator)
            masked_input = input_batch.clone()
            masked_input[:, 0].masked_fill_(mask, 0.0)
            if training:
                recon, z_pred, class_logits, target_logits = model(masked_input, wavelength, padding)
            else:
                # Redshift/classification deployment uses complete spectra.
                _, z_pred, class_logits, target_logits = model(input_batch, wavelength, padding)
                # Reconstruction alone is scored from a fixed masked view.
                recon, _, _, _ = model(masked_input, wavelength, padding)
            mask_weight = mask.to(recon.dtype)
            recon_loss = ((recon - flux).square() * mask_weight).sum() / mask_weight.sum().clamp_min(1.0)
            z_target = encode_redshift(z)
            per_item_z = torch.nn.functional.smooth_l1_loss(z_pred, z_target, reduction="none")
            z_loss = (per_item_z * redshift_weights(z, args.tail_power)).mean()
            class_loss = torch.nn.functional.cross_entropy(class_logits, spectype)
            if target_logits is not None and target_labels is not None:
                target_loss = torch.nn.functional.binary_cross_entropy_with_logits(target_logits, target_labels)
            else:
                target_loss = torch.zeros((), device=device, dtype=recon_loss.dtype)
            loss = args.reconstruction_weight * recon_loss + args.alpha * z_loss + args.classification_weight * class_loss + args.target_label_weight * target_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if args.profile_sync and device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_submit_seconds += time.perf_counter() - compute_start
        previous_batch_end = time.perf_counter()

        with torch.no_grad():
            decoded_z = decode_redshift(z_pred)
            mae = torch.mean(torch.abs(decoded_z - z))
            class_accuracy = (class_logits.argmax(dim=1) == spectype).float().mean()
            target_accuracy = ((target_logits.sigmoid() >= 0.5) == target_labels.bool()).float().mean() if target_logits is not None and target_labels is not None else torch.zeros((), device=device)
        n_spectra = int(flux.shape[0])
        metric_sums["loss"] += loss.detach()
        metric_sums["recon"] += recon_loss.detach()
        metric_sums["z"] += z_loss.detach()
        metric_sums["class"] += class_loss.detach()
        metric_sums["class_accuracy"] += class_accuracy.detach()
        metric_sums["target"] += target_loss.detach()
        metric_sums["target_accuracy"] += target_accuracy.detach()
        metric_sums["mae"] += mae.detach()
        predicted_redshifts.append(decoded_z.detach().float().cpu())
        actual_redshifts.append(z.detach().float().cpu())
        if prediction_output is not None:
            targetids.append(batch["targetid"].detach().cpu())
            actual_spectypes.append(spectype.detach().cpu())
            predicted_spectypes.append(class_logits.argmax(dim=1).detach().cpu())
            if target_logits is not None and target_labels is not None:
                actual_target_labels.append(target_labels.detach().cpu())
                predicted_target_labels.append((target_logits.sigmoid() >= 0.5).detach().cpu())
        spectra_total += n_spectra
        batches += 1
        if steps_limit is not None and batches >= steps_limit:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - epoch_start
    metrics = {key: float(value.item()) / max(batches, 1) for key, value in metric_sums.items()}
    if predicted_redshifts:
        predicted_np = torch.cat(predicted_redshifts).numpy()
        actual_np = torch.cat(actual_redshifts).numpy()
        metrics.update(redshift_metrics_numpy(predicted_np, actual_np))
        metrics.update(
            {
                "z_true_min": float(actual_np.min()),
                "z_true_max": float(actual_np.max()),
                "z_true_median": float(np.median(actual_np)),
                "z_pred_min": float(predicted_np.min()),
                "z_pred_max": float(predicted_np.max()),
                "z_pred_median": float(np.median(predicted_np)),
                "z_invalid_excluded": 0.0,
            }
        )
    else:
        metrics.update({key: float("nan") for key in ("z_mae", "z_rmse", "z_mae_norm", "z_bias", "z_sigma_nmad", "z_eta_0033", "z_eta_005", "z_r2", "sigma_nmad")})
    if prediction_output is not None and targetids:
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        pred = np.asarray(predicted_np, dtype=np.float32)
        true = np.asarray(actual_np, dtype=np.float32)
        prediction_payload = dict(
            targetid=torch.cat(targetids).numpy(),
            z_true=true,
            z_pred=pred,
            dz=(pred - true),
            dz_norm=(pred - true) / (1.0 + true),
            abs_dz_norm=np.abs((pred - true) / (1.0 + true)),
            spectype_true=torch.cat(actual_spectypes).numpy(),
            spectype_pred=torch.cat(predicted_spectypes).numpy(),
            spectype_classes=np.asarray(["GALAXY", "QSO"]),
        )
        if actual_target_labels:
            prediction_payload.update(
                target_labels_true=torch.cat(actual_target_labels).numpy(),
                target_labels_pred=torch.cat(predicted_target_labels).numpy(),
                target_label_names=np.asarray(["BGS", "LRG", "ELG", "QSO"]),
            )
        np.savez_compressed(prediction_output, **prediction_payload)
        if actual_target_labels:
            labels_np = torch.cat(actual_target_labels).numpy().astype(bool)
            by_target = {}
            for label_index, label_name in enumerate(("BGS", "LRG", "ELG", "QSO")):
                chosen = labels_np[:, label_index]
                if chosen.any():
                    by_target[label_name] = {"n": int(chosen.sum()), **redshift_metrics_numpy(predicted_np[chosen], actual_np[chosen])}
            metrics["redshift_by_target_selection"] = by_target
    return metrics | {
        "spectra": float(spectra_total),
        "batches": float(batches),
        "loader_seconds": loader_seconds,
        "input_seconds": input_seconds,
        "compute_submit_seconds": compute_submit_seconds,
        "wall_seconds": wall_seconds,
        "spectra_per_second": spectra_total / max(wall_seconds, 1e-6),
        "augmentation_batches": augmentation_batch_counts,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    best_eta_0033: float,
    epochs_without_improvement: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_eta_0033": best_eta_0033,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        },
        path,
    )


def resolve_run_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint_dir = args.checkpoint_dir or args.output_dir or Path("data/checkpoints/materialized_streaming")
    model_dir = args.model_dir or Path("models/materialized_streaming")
    return checkpoint_dir, model_dir


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    shard_paths = materialized_shard_paths(args.input)
    if args.num_shards is not None:
        shard_paths = shard_paths[: args.num_shards]
    if not shard_paths:
        raise RuntimeError(f"No materialized shards found under {args.input}")
    first = torch.load(shard_paths[0], map_location="cpu", weights_only=True)
    input_length = int(first["flux"].shape[1])
    has_target_labels = "target_labels" in first
    if args.target_label_weight > 0.0 and not has_target_labels:
        raise RuntimeError(
            "Materialized shards do not yet contain target_labels. Run scripts/enrich_materialized_target_labels_cpu.slurm "
            "or set --target-label-weight 0 for a legacy no-target-head run."
        )
    input_channels = 1 + int(args.use_ivar_channel) + int(args.use_validity_channel)
    del first

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not 0.0 < args.variable_min_fraction <= 1.0:
        raise ValueError(f"variable_min_fraction must be in (0, 1], got {args.variable_min_fraction}")
    if args.augmentation_policy == "exclusive":
        probabilities = (
            args.exclusive_clean_prob,
            args.exclusive_crop_prob,
            args.exclusive_transform_prob,
        )
        if any(value < 0.0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError(f"Exclusive augmentation probabilities must be nonnegative and sum to 1, got {probabilities}")
        if args.variable_min_fraction >= 1.0 or not args.random_transforms:
            raise ValueError("Exclusive augmentation requires variable_min_fraction < 1 and random_transforms enabled")
    model = DynamicSpectraTransformerWithRedshiftToken(
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        input_channels=input_channels,
        num_classes=2,
        num_target_labels=4 if has_target_labels and args.target_label_weight > 0.0 else 0,
    ).to(device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires a PyTorch build with torch.compile")
        model = torch.compile(model, dynamic=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    start_epoch = 1
    best_score = float("inf")
    best_eta_0033 = float("inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        best_eta_0033 = float(checkpoint.get("best_eta_0033", best_eta_0033))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    checkpoint_dir, model_dir = resolve_run_dirs(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "n_shards_total": len(shard_paths),
            "split_policy": "fixed desi-dr1-healpix-split-v1; benchmark rows are excluded",
            "input_length": input_length,
            "architecture": "DynamicSpectraTransformerWithRedshiftToken",
            "input_channels": input_channels,
            "has_target_labels": has_target_labels,
            "device": str(device),
        }
    )
    (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_path = checkpoint_dir / "metrics.jsonl"
    history_path = checkpoint_dir / "history.jsonl"
    summary_path = checkpoint_dir / "history_summary.json"

    print(
        f"[setup] device={device} shards={len(shard_paths)} fixed_splits=train,validation "
            f"length={input_length} channels={input_channels} crop_fraction={args.variable_min_fraction} "
            f"transforms={args.random_transforms} augmentation_policy={args.augmentation_policy} "
            f"mask={args.mask_mode} selection={args.selection_metric} classification=GALAXY,QSO targets=BGS,LRG,ELG,QSO"
    )
    print(
        f"[setup] streaming max_train_spectra_per_epoch={args.max_train_spectra_per_epoch} "
        f"max_val_spectra={args.max_val_spectra} batch_size={args.batch_size}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model=model,
            optimizer=optimizer,
            shard_paths=shard_paths,
            args=args,
            device=device,
            epoch=epoch,
            max_spectra=args.max_train_spectra_per_epoch,
            steps_limit=args.steps_per_epoch,
            split="train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                optimizer=None,
                shard_paths=shard_paths,
                args=args,
                device=device,
                epoch=epoch,
                max_spectra=args.max_val_spectra,
                steps_limit=None,
                split="validation",
                prediction_output=checkpoint_dir / f"validation_predictions_epoch_{epoch:04d}.npz",
            )

        selection_score = val_metrics[args.selection_metric]
        best_eta_0033 = min(best_eta_0033, val_metrics["z_eta_0033"])
        is_best = selection_score < best_score - args.early_stopping_min_delta
        if is_best:
            best_score = selection_score
            epochs_without_improvement = 0
            save_checkpoint(
                model_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                best_eta_0033=best_eta_0033,
                epochs_without_improvement=epochs_without_improvement,
                args=args,
            )
            (model_dir / "best_metadata.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "selection_metric": args.selection_metric,
                        "best_val_score": best_score,
                        "best_val_eta_0033": best_eta_0033,
                        "checkpoint_dir": str(checkpoint_dir),
                        "args": config,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            epochs_without_improvement += 1
        if epoch % max(1, args.save_every_epochs) == 0:
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                best_eta_0033=best_eta_0033,
                epochs_without_improvement=epochs_without_improvement,
                args=args,
            )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_metric": args.selection_metric,
            "best_val_score": best_score,
            "best_val_eta_0033": best_eta_0033,
            "is_best": is_best,
            "epochs_without_improvement": epochs_without_improvement,
            "elapsed_seconds": time.time() - t0,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        summary_path.write_text(
            json.dumps(
                {
                    "latest_epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "best_val_score": best_score,
                    "best_val_eta_0033": best_eta_0033,
                    "latest_train_metrics": train_metrics,
                    "latest_val_metrics": val_metrics,
                    "epochs_without_improvement": epochs_without_improvement,
                    "checkpoint_dir": str(checkpoint_dir),
                    "model_dir": str(model_dir),
                    "args": config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[ep {epoch:04d}] "
            f"tr σ={train_metrics['z_sigma_nmad']:.5f} mae={train_metrics['z_mae']:.5f} η33={train_metrics['z_eta_0033']:.3f} "
            f"cls={train_metrics['class_accuracy']:.4f} rate={train_metrics['spectra_per_second']:.0f}/s | "
            f"va σ={val_metrics['z_sigma_nmad']:.5f} mae={val_metrics['z_mae']:.5f} nmae={val_metrics['z_mae_norm']:.5f} "
            f"bias={val_metrics['z_bias']:+.5f} r2={val_metrics['z_r2']:.4f} η33={val_metrics['z_eta_0033']:.3f} η05={val_metrics['z_eta_005']:.3f} "
            f"cls={val_metrics['class_accuracy']:.4f} z=[{val_metrics['z_true_min']:.3f},{val_metrics['z_true_max']:.3f}]→[{val_metrics['z_pred_min']:.3f},{val_metrics['z_pred_max']:.3f}] med={val_metrics['z_true_median']:.3f}→{val_metrics['z_pred_median']:.3f} | "
            f"bestσ={best_score:.5f} bestη33={best_eta_0033:.3f} wait={epochs_without_improvement}"
        )
        if (
            args.early_stopping_patience is not None
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"[early-stop] validation MAE did not improve by at least {args.early_stopping_min_delta:g} "
                f"for {epochs_without_improvement} consecutive epochs"
            )
            break


if __name__ == "__main__":
    main()
