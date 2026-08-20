#!/usr/bin/env python3
"""DEPRECATED: legacy short FITS-streaming trainer.

Do not use this as a new experiment launcher. It remains at this path only
while benchmark readers reuse its model and spectrum-preparation helpers.
New work starts from an isolated directory under ``experiments/``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROJECT_SRC = PROJECT_ROOT / "final-project" / "src"
sys.path.insert(0, str(FINAL_PROJECT_SRC))

from neural_baseline import SpectraTransformerWithRedshiftToken  # noqa: E402


DEFAULT_MANIFEST = Path("data/preprocessed/split_manifest_v1.jsonl")
TRAINING_SPLITS = {"train", "validation"}


@dataclass
class Example:
    flux: np.ndarray
    ivar: np.ndarray
    valid: np.ndarray
    wavelength: np.ndarray
    z: float
    spectype: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short DR1 redshift + reconstruction experiments.")
    parser.add_argument("--experiment", choices=["fixed", "variable"], required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Scratch-backed directory for resumable checkpoints and logs. Defaults to data/checkpoints/EXPERIMENT.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Home/project directory for the best model artifact. Defaults to models/EXPERIMENT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --checkpoint-dir.",
    )
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.15,
        help="Fraction of valid spectral pixels withheld for the reconstruction objective.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["contiguous", "independent"],
        default="contiguous",
        help="Mask contiguous wavelength spans by default; 'independent' reproduces pixel-wise masking.",
    )
    parser.add_argument("--redshift-weight", type=float, default=1.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument(
        "--classification-weight", type=float, default=0.25,
        help="Weight for the GALAXY/QSO cross-entropy auxiliary objective.",
    )
    parser.add_argument("--patch-size", type=int, default=61)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--best-metric",
        choices=["val_loss", "val_mae", "val_recon", "val_z"],
        default="val_loss",
        help="Validation metric used to update MODEL_DIR/best.pt.",
    )
    parser.add_argument(
        "--variable-min-fraction",
        type=float,
        default=0.65,
        help="For --experiment variable, randomly crop each spectrum to this fraction or longer.",
    )
    parser.add_argument(
        "--random-transforms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply conservative flux calibration/noise transforms to variable-training crops.",
    )
    parser.add_argument("--transform-gain", type=float, default=0.10, help="Maximum multiplicative flux gain deviation.")
    parser.add_argument("--transform-tilt", type=float, default=0.10, help="Maximum linear wavelength-dependent gain tilt.")
    parser.add_argument("--transform-offset", type=float, default=0.05, help="Maximum additive normalized-flux offset.")
    parser.add_argument("--transform-noise", type=float, default=0.02, help="Maximum Gaussian noise standard deviation in normalized flux units.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint produced by this script, usually CHECKPOINT_DIR/latest.pt.",
    )
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def validate_resume_config(args: argparse.Namespace, checkpoint_args: dict[str, Any]) -> None:
    current = jsonable_args(args)
    ignored = {"epochs", "resume", "num_workers", "checkpoint_dir", "model_dir", "output_dir"}
    mismatches = []
    for key, old_value in checkpoint_args.items():
        if key in ignored or key not in current:
            continue
        if current[key] != old_value:
            mismatches.append((key, old_value, current[key]))
    if mismatches:
        details = "\n".join(f"  {key}: checkpoint={old!r}, current={new!r}" for key, old, new in mismatches)
        raise RuntimeError(f"Refusing to resume with mismatched metaparameters:\n{details}")


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_metric: str,
    best_score: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": jsonable_args(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_metric": best_metric,
        "best_score": best_score,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }


def restore_rng_state(payload: dict[str, Any]) -> None:
    rng_state = payload.get("rng_state")
    if not rng_state:
        return
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])


def resolve_run_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint_dir = args.checkpoint_dir or args.output_dir or (Path("data/checkpoints") / args.experiment)
    model_dir = args.model_dir or (Path("models") / args.experiment)
    return checkpoint_dir, model_dir


def metric_score(best_metric: str, val_metrics: dict[str, float]) -> float:
    metric_name = best_metric.removeprefix("val_")
    return float(val_metrics[metric_name])


def best_model_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_metric: str,
    best_score: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": jsonable_args(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_metric": best_metric,
        "best_score": best_score,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }


def encode_redshift(z: torch.Tensor) -> torch.Tensor:
    return z / (1.0 + z)


def decode_redshift(encoded: torch.Tensor) -> torch.Tensor:
    return encoded / torch.clamp(1.0 - encoded, min=1e-6)


def robust_flux_location_scale(flux: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """Return the location and scale used by benchmark flux normalization."""

    good = valid.astype(bool) & np.isfinite(flux)
    if not good.any():
        return 0.0, 1.0
    center = np.median(flux[good])
    scale = np.percentile(flux[good], 95) - np.percentile(flux[good], 5)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(flux[good])
    scale = max(float(scale), 1e-6)
    return float(center), scale


def robust_normalize(flux: np.ndarray, valid: np.ndarray) -> np.ndarray:
    center, scale = robust_flux_location_scale(flux, valid)
    normalized = ((flux - center) / scale).astype(np.float32)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def robust_normalize_with_log_ivar(
    flux: np.ndarray,
    ivar: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize flux and express IVAR in the same dimensionless units.

    If ``f_normalized = (f - center) / scale``, its inverse variance is
    ``ivar_normalized = ivar * scale**2``.  ``log1p`` keeps this confidence
    channel finite and numerically manageable while preserving its ordering.
    Invalid pixels receive zero confidence.
    """

    center, scale = robust_flux_location_scale(flux, valid)
    normalized = ((flux - center) / scale).astype(np.float32)
    normalized[~np.isfinite(normalized)] = 0.0
    good = valid.astype(bool) & np.isfinite(ivar) & (ivar > 0)
    normalized_ivar = np.zeros_like(normalized, dtype=np.float32)
    normalized_ivar[good] = np.asarray(ivar[good], dtype=np.float32) * np.float32(scale * scale)
    log_ivar = np.log1p(np.maximum(normalized_ivar, 0.0)).astype(np.float32)
    log_ivar[~good] = 0.0
    return normalized, log_ivar


def stitch_bands(
    wavelengths: list[np.ndarray],
    fluxes: list[np.ndarray],
    ivars: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    all_wave = np.concatenate(wavelengths)
    all_flux = np.concatenate(fluxes)
    all_ivar = np.concatenate(ivars)
    all_mask = np.concatenate(masks)
    order = np.argsort(all_wave)
    all_wave = all_wave[order]
    all_flux = all_flux[order]
    all_ivar = all_ivar[order]
    all_mask = all_mask[order]

    unique_waves: list[float] = []
    weighted_flux: list[float] = []
    total_ivar: list[float] = []
    combined_mask: list[bool] = []
    i = 0
    while i < len(all_wave):
        wave = all_wave[i]
        j = i
        while j < len(all_wave) and abs(float(all_wave[j] - wave)) < 0.1:
            j += 1
        flux_chunk = all_flux[i:j]
        ivar_chunk = all_ivar[i:j]
        mask_chunk = all_mask[i:j]
        good = ~mask_chunk
        if good.any():
            ivar_sum = float(ivar_chunk[good].sum())
            avg_flux = float((flux_chunk[good] * ivar_chunk[good]).sum() / ivar_sum) if ivar_sum > 0 else float(flux_chunk[good].mean())
            avg_mask = False
        else:
            ivar_sum = 0.0
            avg_flux = float(flux_chunk.mean())
            avg_mask = True
        unique_waves.append(float(wave))
        weighted_flux.append(avg_flux)
        total_ivar.append(ivar_sum)
        combined_mask.append(avg_mask)
        i = j

    return {
        "wavelength": np.asarray(unique_waves, dtype=np.float32),
        "flux": np.asarray(weighted_flux, dtype=np.float32),
        "ivar": np.asarray(total_ivar, dtype=np.float32),
        "mask": np.asarray(combined_mask, dtype=bool),
    }


def iter_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_examples(manifest: Path, max_examples: int) -> dict[str, list[Example]]:
    records = iter_manifest_rows(manifest)
    examples: dict[str, list[Example]] = {split: [] for split in TRAINING_SPLITS}
    loaded = 0
    for record in records:
        split = record.get("split")
        if split is None:
            raise RuntimeError(
                f"{manifest} has no fixed split assignment. Create data/preprocessed/split_manifest_v1.jsonl "
                "with scripts/build_split_manifest.py before training."
            )
        if split == "benchmark":
            continue
        if split not in TRAINING_SPLITS:
            raise RuntimeError(f"Unexpected split {split!r} in {manifest}")
        rows = record["rows"]
        zs = record["z"]
        with fits.open(record["coadd"], memmap=True) as coadd:
            waves = [coadd[f"{band}_WAVELENGTH"].data for band in ("B", "R", "Z")]
            for row, z in zip(rows, zs, strict=True):
                stitched = stitch_bands(
                    waves,
                    [coadd[f"{band}_FLUX"].data[int(row), :] for band in ("B", "R", "Z")],
                    [coadd[f"{band}_IVAR"].data[int(row), :] for band in ("B", "R", "Z")],
                    [coadd[f"{band}_MASK"].data[int(row), :] != 0 for band in ("B", "R", "Z")],
                )
                valid = (~stitched["mask"]) & (stitched["ivar"] > 0)
                flux = robust_normalize(stitched["flux"], valid)
                examples[split].append(
                Example(
                        flux=flux,
                        ivar=stitched["ivar"].astype(np.float32),
                        valid=valid.astype(np.float32),
                        wavelength=stitched["wavelength"].astype(np.float32),
                    z=float(z),
                    spectype=0 if str(clean_spectype) == "GALAXY" else 1,
                        split=split,
                    )
                )
                loaded += 1
                if loaded >= max_examples:
                    return examples
    return examples


class SpectraDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        *,
        variable: bool,
        min_fraction: float,
        seed: int,
        random_transforms: bool,
        transform_gain: float,
        transform_tilt: float,
        transform_offset: float,
        transform_noise: float,
    ):
        self.examples = examples
        self.variable = variable
        self.min_fraction = min_fraction
        self.seed = seed
        self.epoch = 0
        self.random_transforms = random_transforms
        self.transform_gain = transform_gain
        self.transform_tilt = transform_tilt
        self.transform_offset = transform_offset
        self.transform_noise = transform_noise

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def apply_random_transforms(self, flux: np.ndarray, valid: np.ndarray, rng: random.Random) -> np.ndarray:
        if not self.random_transforms:
            return flux
        good = valid.astype(bool)
        if not good.any():
            return flux

        transformed = flux.copy()
        position = np.linspace(-1.0, 1.0, num=len(flux), dtype=np.float32)
        gain = 1.0 + rng.uniform(-self.transform_gain, self.transform_gain)
        tilt = rng.uniform(-self.transform_tilt, self.transform_tilt)
        offset = rng.uniform(-self.transform_offset, self.transform_offset)
        calibration = gain + tilt * position
        transformed[good] = transformed[good] * calibration[good] + offset

        noise_std = rng.uniform(0.0, self.transform_noise)
        if noise_std > 0.0:
            noise_rng = np.random.default_rng(rng.randrange(2**63))
            transformed[good] += noise_rng.normal(0.0, noise_std, size=int(good.sum())).astype(np.float32)
        return transformed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        target_flux = ex.flux
        valid = ex.valid
        wavelength = ex.wavelength
        if self.variable:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + idx)
            min_len = max(64, int(len(target_flux) * self.min_fraction))
            length = rng.randint(min_len, len(target_flux))
            start = rng.randint(0, len(target_flux) - length)
            stop = start + length
            target_flux = target_flux[start:stop]
            valid = valid[start:stop]
            wavelength = wavelength[start:stop]
            input_flux = self.apply_random_transforms(target_flux, valid, rng)
        else:
            input_flux = target_flux
        return {
            "input_flux": torch.from_numpy(input_flux.astype(np.float32)),
            "flux": torch.from_numpy(target_flux.astype(np.float32)),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "wavelength": torch.from_numpy(wavelength.astype(np.float32)),
            "z": torch.tensor(float(ex.z), dtype=torch.float32),
            "spectype": torch.tensor(ex.spectype, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["flux"].numel() for item in batch)
    out = {
        "input_flux": torch.zeros(len(batch), max_len),
        "flux": torch.zeros(len(batch), max_len),
        "valid": torch.zeros(len(batch), max_len),
        "wavelength": torch.zeros(len(batch), max_len),
        "padding": torch.ones(len(batch), max_len, dtype=torch.bool),
        "z": torch.stack([item["z"] for item in batch]),
        "spectype": torch.stack([item["spectype"] for item in batch]),
    }
    for i, item in enumerate(batch):
        length = item["flux"].numel()
        out["input_flux"][i, :length] = item["input_flux"]
        out["flux"][i, :length] = item["flux"]
        out["valid"][i, :length] = item["valid"]
        out["wavelength"][i, :length] = item["wavelength"]
        out["padding"][i, :length] = False
    return out


def sinusoidal_positions(num_tokens: int, d_model: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(num_tokens, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def reconstruction_mask(
    valid: torch.Tensor,
    padding: torch.Tensor,
    *,
    mask_prob: float,
    mode: str,
) -> torch.Tensor:
    """Sample pixels withheld from the input and scored by reconstruction loss.

    Contiguous mode masks one wavelength interval whenever possible.  If a
    spectrum's valid coverage is split into smaller intervals, it masks as few
    intervals as necessary to reach the requested coverage without ever
    treating invalid or padded pixels as targets.
    """

    usable = (valid > 0) & (~padding)
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    if mode == "independent":
        return (torch.rand_like(valid) < mask_prob) & usable
    if mode != "contiguous":
        raise ValueError(f"Unknown mask mode: {mode}")

    mask = torch.zeros_like(usable)
    for row in range(usable.shape[0]):
        usable_indices = torch.nonzero(usable[row], as_tuple=False).flatten()
        target_count = int(round(mask_prob * usable_indices.numel()))
        if target_count == 0:
            continue

        # Split valid pixels into physically contiguous wavelength intervals.
        breaks = torch.nonzero(usable_indices[1:] != usable_indices[:-1] + 1, as_tuple=False).flatten() + 1
        intervals = list(torch.tensor_split(usable_indices, breaks.tolist()))
        # Prefer one interval that can accommodate the whole masked span.
        eligible = [interval for interval in intervals if interval.numel() >= target_count]
        if eligible:
            interval = eligible[torch.randint(len(eligible), (), device=valid.device).item()]
            offset = torch.randint(interval.numel() - target_count + 1, (), device=valid.device).item()
            mask[row, interval[offset : offset + target_count]] = True
            continue

        # Fragmented coverage: use the fewest valid intervals required.
        remaining = target_count
        for interval in sorted(intervals, key=lambda item: item.numel(), reverse=True):
            count = min(remaining, interval.numel())
            offset = torch.randint(interval.numel() - count + 1, (), device=valid.device).item()
            mask[row, interval[offset : offset + count]] = True
            remaining -= count
            if remaining == 0:
                break
    return mask


class DynamicSpectraTransformerWithRedshiftToken(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        input_channels: int = 1,
        num_classes: int = 2,
        num_target_labels: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_target_labels = num_target_labels
        self.patch_embed = nn.Linear(patch_size * input_channels, d_model)
        self.wavelength_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.redshift_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.recon_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, patch_size))
        self.redshift_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.classification_head = (
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, num_classes))
            if num_classes > 0 else None
        )
        self.target_selection_head = (
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, num_target_labels))
            if num_target_labels > 0 else None
        )

    def forward(
        self,
        x: torch.Tensor,
        wavelength: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        length = x.shape[-1]
        if wavelength.shape != (x.shape[0], length):
            raise ValueError(
                f"Expected wavelength shape {(x.shape[0], length)}, got {tuple(wavelength.shape)}"
            )
        padded_length = ((length + self.patch_size - 1) // self.patch_size) * self.patch_size
        if padded_length > length:
            x = torch.nn.functional.pad(x, (0, padded_length - length))
            wavelength = torch.nn.functional.pad(wavelength, (0, padded_length - length))
            if padding_mask is not None:
                padding_mask = torch.nn.functional.pad(padding_mask, (0, padded_length - length), value=True)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], -1, self.input_channels * self.patch_size)
        tokens = self.patch_embed(patches)
        tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device)

        wavelength_patches = wavelength.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        if padding_mask is None:
            wavelength_weights = torch.ones_like(wavelength_patches)
        else:
            wavelength_weights = (~padding_mask).float().unfold(
                dimension=-1,
                size=self.patch_size,
                step=self.patch_size,
            )
        patch_wavelength = (wavelength_patches * wavelength_weights).sum(dim=-1)
        patch_wavelength = patch_wavelength / wavelength_weights.sum(dim=-1).clamp_min(1.0)
        log_min = math.log(3600.0)
        log_max = math.log(10000.0)
        normalized_log_wavelength = 2.0 * (torch.log(patch_wavelength.clamp_min(1.0)) - log_min) / (log_max - log_min) - 1.0
        tokens = tokens + self.wavelength_embed(normalized_log_wavelength.unsqueeze(-1))

        redshift_tokens = self.redshift_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([redshift_tokens, tokens], dim=1)

        src_key_padding_mask = None
        if padding_mask is not None:
            patch_padding = padding_mask.unfold(dimension=-1, size=self.patch_size, step=self.patch_size).all(dim=-1)
            redshift_padding = torch.zeros(patch_padding.shape[0], 1, dtype=torch.bool, device=patch_padding.device)
            src_key_padding_mask = torch.cat([redshift_padding, patch_padding], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        recon = self.recon_head(encoded[:, 1:, :]).reshape(x.shape[0], -1)[:, :length]
        redshift = self.redshift_head(encoded[:, 0, :]).squeeze(-1)
        class_logits = self.classification_head(encoded[:, 0, :]) if self.classification_head is not None else None
        target_logits = self.target_selection_head(encoded[:, 0, :]) if self.target_selection_head is not None else None
        return recon, redshift, class_logits, target_logits


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "recon": 0.0, "z": 0.0, "class": 0.0, "class_accuracy": 0.0, "mae": 0.0}
    batches = 0
    for batch in loader:
        input_flux = batch["input_flux"].to(device)
        target_flux = batch["flux"].to(device)
        valid = batch["valid"].to(device)
        padding = batch["padding"].to(device)
        wavelength = batch["wavelength"].to(device)
        z = batch["z"].to(device)
        spectype = batch["spectype"].to(device)
        mask = reconstruction_mask(
            valid,
            padding,
            mask_prob=args.mask_prob,
            mode=args.mask_mode,
        )
        masked_flux = input_flux.masked_fill(mask, 0.0)
        if args.experiment == "variable":
            recon, z_pred, class_logits, _ = model(masked_flux, wavelength, padding)
        else:
            recon, z_pred = model(masked_flux)
            class_logits = None
        z_target = encode_redshift(z)
        recon_loss = (
            torch.nn.functional.mse_loss(recon[mask], target_flux[mask])
            if bool(mask.any().item())
            else recon.sum() * 0.0
        )
        z_loss = torch.nn.functional.smooth_l1_loss(z_pred, z_target)
        class_loss = torch.nn.functional.cross_entropy(class_logits, spectype) if class_logits is not None else recon.sum() * 0.0
        loss = args.reconstruction_weight * recon_loss + args.redshift_weight * z_loss + args.classification_weight * class_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mae = torch.mean(torch.abs(decode_redshift(z_pred) - z))
            class_accuracy = (class_logits.argmax(dim=1) == spectype).float().mean() if class_logits is not None else torch.tensor(0.0, device=device)
        totals["loss"] += float(loss.item())
        totals["recon"] += float(recon_loss.item())
        totals["z"] += float(z_loss.item())
        totals["class"] += float(class_loss.item())
        totals["class_accuracy"] += float(class_accuracy.item())
        totals["mae"] += float(mae.item())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    examples = load_examples(args.manifest, args.max_examples)
    train_examples = examples["train"]
    val_examples = examples["validation"]
    if not train_examples or not val_examples:
        raise RuntimeError(
            f"Need at least one train and validation example from {args.manifest}; "
            f"found train={len(train_examples)} validation={len(val_examples)}. "
            "Increase --max-examples if the cap stopped before both fixed splits were loaded."
        )
    variable = args.experiment == "variable"
    transform_kwargs = {
        "transform_gain": args.transform_gain,
        "transform_tilt": args.transform_tilt,
        "transform_offset": args.transform_offset,
        "transform_noise": args.transform_noise,
    }
    train_ds = SpectraDataset(
        train_examples,
        variable=variable,
        min_fraction=args.variable_min_fraction,
        seed=args.seed,
        random_transforms=variable and args.random_transforms,
        **transform_kwargs,
    )
    val_ds = SpectraDataset(
        val_examples,
        variable=False,
        min_fraction=1.0,
        seed=args.seed + 1_000_000,
        random_transforms=False,
        **transform_kwargs,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    if args.experiment == "fixed":
        input_length = train_examples[0].flux.shape[0]
        model = SpectraTransformerWithRedshiftToken(
            input_length=input_length,
            patch_size=args.patch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
        )
    else:
        model = DynamicSpectraTransformerWithRedshiftToken(
            patch_size=args.patch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    checkpoint_dir, model_dir = resolve_run_dirs(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(jsonable_args(args), indent=2), encoding="utf-8")

    start_epoch = 1
    best_score = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        validate_resume_config(args, checkpoint_args)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng_state(checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        print(f"[resume] loaded {args.resume} at epoch {checkpoint['epoch']}; next epoch={start_epoch}")

    print(
        f"[setup] experiment={args.experiment} device={device} "
        f"examples={len(train_examples) + len(val_examples)} train={len(train_examples)} val={len(val_examples)} "
        f"benchmark=0 manifest={args.manifest}"
    )
    print(f"[setup] loaded in {time.time() - t0:.1f}s checkpoints={checkpoint_dir} best_models={model_dir}")
    metrics_path = checkpoint_dir / "metrics.jsonl"
    history_path = checkpoint_dir / "history.jsonl"
    summary_path = checkpoint_dir / "history_summary.json"
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(0)
        train_metrics = run_epoch(model, train_loader, optimizer, args, device)
        validation_cpu_rng = torch.get_rng_state()
        validation_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(args.seed + 9_000_000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 9_000_000)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, args, device)
        torch.set_rng_state(validation_cpu_rng)
        if validation_cuda_rng is not None:
            torch.cuda.set_rng_state_all(validation_cuda_rng)
        score = metric_score(args.best_metric, val_metrics)
        is_best = score < best_score
        if is_best:
            best_score = score
        metrics_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_metric": args.best_metric,
            "score": score,
            "is_best": is_best,
            "args": jsonable_args(args),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_record) + "\n")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_record) + "\n")
        summary_path.write_text(
            json.dumps(
                {
                    "latest_epoch": epoch,
                    "best_metric": args.best_metric,
                    "best_score": best_score,
                    "latest_score": score,
                    "latest_is_best": is_best,
                    "latest_train_metrics": train_metrics,
                    "latest_val_metrics": val_metrics,
                    "checkpoint_dir": str(checkpoint_dir),
                    "model_dir": str(model_dir),
                    "args": jsonable_args(args),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "[epoch "
            f"{epoch:04d}] train_loss={train_metrics['loss']:.5f} train_recon={train_metrics['recon']:.5f} "
            f"train_z={train_metrics['z']:.5f} train_mae={train_metrics['mae']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_recon={val_metrics['recon']:.5f} "
            f"val_z={val_metrics['z']:.5f} val_mae={val_metrics['mae']:.5f} "
            f"best_{args.best_metric}={best_score:.5f}"
        )
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_metric=args.best_metric,
                best_score=best_score,
            ),
            checkpoint_dir / "latest.pt",
        )
        if is_best:
            best_payload = best_model_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_metric=args.best_metric,
                best_score=best_score,
            )
            torch.save(best_payload, model_dir / "best.pt")
            (model_dir / "best_metadata.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "best_metric": args.best_metric,
                        "best_score": best_score,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "args": jsonable_args(args),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[best] saved {model_dir / 'best.pt'} ({args.best_metric}={best_score:.5f})")


if __name__ == "__main__":
    main()
