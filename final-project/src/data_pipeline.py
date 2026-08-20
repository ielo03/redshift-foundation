from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency guard
    h5py = None


@dataclass
class SpectrumRecord:
    """Single spectra-only example.

    This matches the AION DESI spectrum fields plus the target redshift.
    """

    flux: torch.Tensor
    ivar: torch.Tensor
    mask: torch.Tensor
    wavelength: torch.Tensor
    z: torch.Tensor


@dataclass
class MaskBundle:
    """Deterministic masks used for validation and testing."""

    spectrum_mask: torch.Tensor


def _ensure_1d_float_tensor(value: Any, field_name: str) -> torch.Tensor:
    value = _parse_array_like(value)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be 1D, got shape {array.shape}")
    return torch.from_numpy(array)


def _ensure_1d_bool_tensor(value: Any, field_name: str) -> torch.Tensor:
    value = _parse_array_like(value)
    value = _parse_bool_like(value)
    array = np.asarray(value, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be 1D, got shape {array.shape}")
    return torch.from_numpy(array)


def _parse_bool_like(value: Any) -> Any:
    def parse_one(item: Any) -> Any:
        if not isinstance(item, str):
            return item
        normalized = item.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
        return item

    if isinstance(value, (list, tuple)):
        return [parse_one(item) for item in value]
    return parse_one(value)


def _parse_array_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    if stripped[0] in "[(":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    if "," in stripped:
        return [item.strip() for item in stripped.split(",")]

    return value


def record_from_dict(row: dict[str, Any]) -> SpectrumRecord:
    """Build a SpectrumRecord from a dict-like row.

    Accepted keys:
    - flux
    - ivar
    - mask
    - wavelength
    - z or redshift
    """

    z_value = row.get("z", row.get("redshift"))
    if z_value is None:
        raise KeyError("row must contain 'z' or 'redshift'")

    return SpectrumRecord(
        flux=_ensure_1d_float_tensor(row["flux"], "flux"),
        ivar=_ensure_1d_float_tensor(row.get("ivar", np.ones_like(row["flux"])), "ivar"),
        mask=_ensure_1d_bool_tensor(row.get("mask", np.zeros_like(row["flux"], dtype=bool)), "mask"),
        wavelength=_ensure_1d_float_tensor(row["wavelength"], "wavelength"),
        z=torch.tensor([float(z_value)], dtype=torch.float32),
    )


def record_from_arrays(
    flux: np.ndarray | torch.Tensor,
    ivar: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    wavelength: np.ndarray | torch.Tensor,
    z_value: float,
) -> SpectrumRecord:
    """Build a SpectrumRecord from already-aligned arrays."""

    return SpectrumRecord(
        flux=torch.as_tensor(flux, dtype=torch.float32).reshape(-1),
        ivar=torch.as_tensor(ivar, dtype=torch.float32).reshape(-1),
        mask=torch.as_tensor(mask, dtype=torch.bool).reshape(-1),
        wavelength=torch.as_tensor(wavelength, dtype=torch.float32).reshape(-1),
        z=torch.tensor([float(z_value)], dtype=torch.float32),
    )


def load_jsonl(path: Path) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(record_from_dict(json.loads(line)))
    return records


def load_csv(path: Path) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(record_from_dict(row))
    return records


def load_npz(path: Path) -> list[SpectrumRecord]:
    data = np.load(path, allow_pickle=True)
    required = ["flux", "ivar", "mask", "wavelength", "z"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"NPZ file missing required keys: {missing}")

    flux = np.asarray(data["flux"])
    ivar = np.asarray(data["ivar"])
    mask = np.asarray(data["mask"])
    wavelength = np.asarray(data["wavelength"])
    z = np.asarray(data["z"])

    if flux.ndim == 1:
        flux = flux[None, :]
        ivar = ivar[None, :]
        mask = mask[None, :]
        wavelength = wavelength[None, :]
        z = z[None]

    records = []
    for i in range(flux.shape[0]):
        records.append(
            SpectrumRecord(
                flux=torch.from_numpy(flux[i].astype(np.float32)),
                ivar=torch.from_numpy(ivar[i].astype(np.float32)),
                mask=torch.from_numpy(mask[i].astype(bool)),
                wavelength=torch.from_numpy(wavelength[i].astype(np.float32)),
                z=torch.tensor([float(z[i])], dtype=torch.float32),
            )
        )
    return records


def _decode_bytes_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in {"S", "O"}:
        return values
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded)


def load_hdf5(
    path: Path,
    *,
    survey: str | None = "sv3",
    primary_only: bool = True,
    max_items: int | None = None,
) -> list[SpectrumRecord]:
    """Load DESI/MMU-style spectra from an HDF5 file.

    Expected dataset names in the file include:
    - spectrum_flux
    - spectrum_ivar
    - spectrum_mask
    - spectrum_lambda
    - Z

    Optional filtering fields:
    - SURVEY
    - SV_PRIMARY
    """

    if h5py is None:
        raise ImportError("h5py is required to load HDF5 spectra files")

    with h5py.File(path, "r") as f:
        flux = np.asarray(f["spectrum_flux"])
        ivar = np.asarray(f["spectrum_ivar"])
        mask = np.asarray(f["spectrum_mask"])
        wavelength = np.asarray(f["spectrum_lambda"])
        z = np.asarray(f["Z"])

        n = flux.shape[0]
        indices = np.arange(n)

        if survey is not None and "SURVEY" in f:
            survey_values = _decode_bytes_array(np.asarray(f["SURVEY"]))
            indices = indices[np.asarray([str(value).lower() == survey.lower() for value in survey_values])]

        if primary_only and "SV_PRIMARY" in f:
            primary_values = np.asarray(f["SV_PRIMARY"]).astype(bool)
            indices = indices[primary_values[indices]]

        if max_items is not None:
            indices = indices[:max_items]

        records: list[SpectrumRecord] = []
        for i in indices:
            records.append(
                record_from_arrays(
                    flux[i],
                    ivar[i],
                    mask[i],
                    wavelength[i],
                    float(z[i]),
                )
            )

    return records


def pad_or_trim_1d(tensor: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(tensor.shape)}")
    if tensor.numel() == length:
        return tensor
    if tensor.numel() > length:
        return tensor[:length]
    pad = torch.full((length - tensor.numel(),), pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def normalize_flux(flux: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = ~mask
    if valid.any():
        mean = flux[valid].mean()
        std = flux[valid].std(unbiased=False).clamp_min(1e-6)
        return (flux - mean) / std
    return flux


def normalize_flux_batch(flux: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize each spectrum independently using unmasked pixels."""

    if flux.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"flux and mask must be 2D, got {tuple(flux.shape)} and {tuple(mask.shape)}")
    valid = ~mask
    valid_float = valid.to(flux.dtype)
    counts = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (flux * valid_float).sum(dim=1, keepdim=True) / counts
    centered = flux - mean
    var = (centered.square() * valid_float).sum(dim=1, keepdim=True) / counts
    std = var.sqrt().clamp_min(1e-6)
    return centered / std


def materialized_shard_to_batch(payload: dict[str, Any], *, normalize_flux_values: bool = True) -> dict[str, torch.Tensor]:
    """Convert a materialize_shards.py payload to the trainer bundle contract.

    materialize_shards.py writes `valid` where True/1 means an observed usable
    pixel. The original final-project trainer uses `mask` where True means an
    unusable or padded pixel, so this function flips that convention.
    """

    required = ["flux", "ivar", "valid", "wavelength", "z"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Materialized shard missing required keys: {missing}")

    flux_raw = torch.as_tensor(payload["flux"], dtype=torch.float32)
    ivar = torch.as_tensor(payload["ivar"], dtype=torch.float32)
    valid = torch.as_tensor(payload["valid"]).bool()
    mask = ~valid
    wavelength = torch.as_tensor(payload["wavelength"], dtype=torch.float32)
    z = torch.as_tensor(payload["z"], dtype=torch.float32).view(-1, 1)

    if wavelength.ndim == 1:
        wavelength = wavelength.unsqueeze(0).expand(flux_raw.shape[0], -1).clone()
    if wavelength.shape != flux_raw.shape:
        raise ValueError(f"wavelength shape {tuple(wavelength.shape)} does not match flux shape {tuple(flux_raw.shape)}")

    flux = normalize_flux_batch(flux_raw, mask) if normalize_flux_values else flux_raw.clone()
    return {
        "flux": flux,
        "flux_raw": flux_raw,
        "ivar": ivar,
        "mask": mask,
        "wavelength": wavelength,
        "z": z,
    }


def _read_shards_manifest(path: Path) -> list[Path]:
    shard_paths: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            shard_paths.append(Path(row["path"]))
    return shard_paths


def materialized_shard_paths(path: Path) -> list[Path]:
    """Resolve a materialized shard file, shard manifest, or materialized dir."""

    if path.is_dir():
        manifest = path / "shards_manifest.jsonl"
        if manifest.exists():
            return _read_shards_manifest(manifest)
        return sorted((path / "shards").glob("*.pt")) if (path / "shards").exists() else sorted(path.glob("*.pt"))
    if path.name == "shards_manifest.jsonl" or path.suffix == ".jsonl":
        return _read_shards_manifest(path)
    return [path]


def load_materialized_shards_batch(
    path: Path,
    *,
    max_items: int | None = None,
    shard_shuffle_seed: int | None = None,
    normalize_flux_values: bool = True,
) -> dict[str, torch.Tensor]:
    """Load one or more materialized shards into the final-project batch format.

    This is intended for smoke tests and capped runs. Full DR1 training should
    use a streaming Dataset so it does not try to hold all spectra in memory.
    """

    shard_paths = materialized_shard_paths(path)
    if shard_shuffle_seed is not None:
        rng = np.random.default_rng(shard_shuffle_seed)
        order = rng.permutation(len(shard_paths))
        shard_paths = [shard_paths[int(idx)] for idx in order]

    batches: list[dict[str, torch.Tensor]] = []
    total = 0
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        batch = materialized_shard_to_batch(payload, normalize_flux_values=normalize_flux_values)
        if max_items is not None:
            remaining = max_items - total
            if remaining <= 0:
                break
            batch = {key: value[:remaining] for key, value in batch.items()}
        batches.append(batch)
        total += int(batch["flux"].shape[0])
        if max_items is not None and total >= max_items:
            break

    if not batches:
        raise RuntimeError(f"No materialized shard data loaded from {path}")

    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]}


def collate_records(
    records: Sequence[SpectrumRecord],
    target_length: int | None = None,
    *,
    normalize_flux_values: bool = True,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("records must not be empty")

    if target_length is None:
        target_length = max(record.flux.numel() for record in records)

    flux = []
    flux_raw = []
    ivar = []
    mask = []
    wavelength = []
    z = []

    for record in records:
        cur_flux = pad_or_trim_1d(record.flux, target_length, pad_value=0.0)
        cur_mask = pad_or_trim_1d(record.mask.to(torch.bool), target_length, pad_value=True)
        cur_ivar = pad_or_trim_1d(record.ivar, target_length, pad_value=0.0)
        cur_wavelength = pad_or_trim_1d(record.wavelength, target_length, pad_value=float(record.wavelength[-1]))
        cur_flux_raw = cur_flux.clone()
        if normalize_flux_values:
            cur_flux = normalize_flux(cur_flux, cur_mask)

        flux.append(cur_flux)
        flux_raw.append(cur_flux_raw)
        ivar.append(cur_ivar)
        mask.append(cur_mask)
        wavelength.append(cur_wavelength)
        z.append(record.z)

    return {
        "flux": torch.stack(flux, dim=0),
        "flux_raw": torch.stack(flux_raw, dim=0),
        "ivar": torch.stack(ivar, dim=0),
        "mask": torch.stack(mask, dim=0),
        "wavelength": torch.stack(wavelength, dim=0),
        "z": torch.stack(z, dim=0).view(len(records), -1),
    }


def make_token_mask(
    shape: tuple[int, int],
    *,
    mask_prob: float,
    seed: int,
    always_mask_last: bool = False,
) -> torch.Tensor:
    """Create a deterministic boolean mask for token-level masking."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.rand(shape, generator=generator) < mask_prob
    if always_mask_last and shape[1] > 0:
        mask[:, -1] = True
    return mask


def make_domain_informed_mask(
    wavelength: torch.Tensor,
    redshift: torch.Tensor,
    *,
    mask_prob: float,
    seed: int,
    line_sigma_angstrom: float = 25.0,
    line_boost: float = 2.5,
    break_sigma_angstrom: float = 40.0,
    break_boost: float = 2.0,
    always_mask_last: bool = False,
) -> torch.Tensor:
    """Create a deterministic mask biased toward informative spectral regions.

    The weights are centered on common rest-frame features shifted by the
    provided redshift values. This keeps the masking scheme grounded in the
    physics of the spectrum instead of purely random masking.
    """

    if wavelength.ndim == 1:
        wavelength = wavelength.unsqueeze(0)
    if redshift.ndim == 2 and redshift.shape[1] == 1:
        redshift = redshift.squeeze(1)
    if redshift.ndim != 1:
        raise ValueError(f"redshift must have shape (B,) or (B, 1), got {tuple(redshift.shape)}")
    if wavelength.shape[0] != redshift.shape[0]:
        raise ValueError(
            f"wavelength batch and redshift batch must match, got {tuple(wavelength.shape)} and {tuple(redshift.shape)}"
        )

    wavelength = wavelength.detach().cpu().float()
    redshift = redshift.detach().cpu().float().view(-1, 1)

    rest_lines = torch.tensor(
        [3727.0, 3934.0, 3969.0, 4102.0, 4341.0, 4861.0, 5007.0, 6563.0],
        dtype=torch.float32,
    ).view(1, -1, 1)
    observed_lines = rest_lines * (1.0 + redshift.view(-1, 1, 1))
    wavelength_expanded = wavelength.unsqueeze(1)
    line_dist = (wavelength_expanded - observed_lines) / max(line_sigma_angstrom, 1e-6)
    line_weights = torch.exp(-0.5 * line_dist**2).max(dim=1).values

    observed_break = 4000.0 * (1.0 + redshift)
    break_dist = (wavelength - observed_break) / max(break_sigma_angstrom, 1e-6)
    break_weights = torch.exp(-0.5 * break_dist**2)

    weights = 1.0 + line_boost * line_weights + break_boost * break_weights
    weights = weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    probs = torch.clamp(mask_prob * weights, 0.0, 0.95)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.rand(wavelength.shape, generator=generator) < probs
    if always_mask_last and mask.shape[1] > 0:
        mask[:, -1] = True
    return mask


def save_mask_bundle(path: Path, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spectrum_mask": mask.cpu()}, path)


def load_mask_bundle(path: Path) -> MaskBundle:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return MaskBundle(spectrum_mask=payload["spectrum_mask"].bool())


def summarize_records(records: Iterable[SpectrumRecord]) -> dict[str, Any]:
    records = list(records)
    if not records:
        return {"count": 0}

    lengths = [record.flux.numel() for record in records]
    redshifts = torch.cat([record.z.reshape(1) for record in records], dim=0)
    return {
        "count": len(records),
        "length_min": int(min(lengths)),
        "length_max": int(max(lengths)),
        "length_mean": float(np.mean(lengths)),
        "z_min": float(redshifts.min().item()),
        "z_max": float(redshifts.max().item()),
        "z_mean": float(redshifts.mean().item()),
    }


def synthetic_records(batch_size: int = 2, length: int = 7081) -> list[SpectrumRecord]:
    wavelength = torch.linspace(3600.0, 9800.0, length)
    records: list[SpectrumRecord] = []
    for idx in range(batch_size):
        flux = torch.sin(wavelength / 250.0 + idx * 0.15) + 0.05 * idx
        ivar = torch.ones(length, dtype=torch.float32)
        mask = torch.zeros(length, dtype=torch.bool)
        if idx % 2 == 0:
            mask[length // 4 : length // 4 + 20] = True
        records.append(
            SpectrumRecord(
                flux=flux.to(torch.float32),
                ivar=ivar,
                mask=mask,
                wavelength=wavelength.to(torch.float32),
                z=torch.tensor([0.05 + 0.1 * idx], dtype=torch.float32),
            )
        )
    return records
