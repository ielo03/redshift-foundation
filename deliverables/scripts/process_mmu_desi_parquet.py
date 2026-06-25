from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def add_src_to_path(project_root: Path) -> None:
    sys.path.insert(0, str(project_root / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Hugging Face MMU DESI parquet shards into trainer tensors.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=sorted(Path("data/mmu_desi_hf/edr_sv3").glob("*.parquet")),
        help="Input parquet shard(s) from MultimodalUniverse/desi edr_sv3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mmu_desi_hf_edr_sv3.pt"),
        help="Output .pt bundle path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/tables/mmu_desi_hf_edr_sv3_summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on rows to process.")
    parser.add_argument(
        "--require-zwarn-good",
        action="store_true",
        help="Keep only rows with HF ZWARN=True. In this release the bool appears to mean good/no warning.",
    )
    return parser.parse_args()


def normalize_flux_batch(flux: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = (~mask).astype(bool)
    safe = np.where(valid, flux, np.nan)
    mean = np.nanmean(safe, axis=1, keepdims=True)
    std = np.nanstd(safe, axis=1, keepdims=True)
    bad = ~np.isfinite(mean) | ~np.isfinite(std) | (std < 1e-6)
    mean = np.where(bad, 0.0, mean)
    std = np.where(bad, 1.0, std)
    normalized = (flux - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fixed_list_to_numpy(chunked_array) -> np.ndarray:
    combined = chunked_array.combine_chunks() if hasattr(chunked_array, "combine_chunks") else chunked_array
    values = combined.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(combined), combined.type.list_size)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    flux_chunks = []
    flux_raw_chunks = []
    ivar_chunks = []
    mask_chunks = []
    wavelength_chunks = []
    lsf_sigma_chunks = []
    z_chunks = []
    zerr_chunks = []
    zwarn_chunks = []
    ebv_chunks = []
    object_ids: list[str] = []

    remaining = args.max_items
    source_paths = []
    for path in args.input:
        if remaining is not None and remaining <= 0:
            break
        source_paths.append(str(path))
        table = pq.read_table(path, columns=["spectrum", "Z", "ZERR", "EBV", "ZWARN", "object_id"])
        if remaining is not None and table.num_rows > remaining:
            table = table.slice(0, remaining)

        spectrum = table["spectrum"].combine_chunks()
        flux_raw = fixed_list_to_numpy(spectrum.field("flux")).astype(np.float32)
        ivar = fixed_list_to_numpy(spectrum.field("ivar")).astype(np.float32)
        mask = fixed_list_to_numpy(spectrum.field("mask")).astype(bool)
        wavelength = fixed_list_to_numpy(spectrum.field("lambda")).astype(np.float32)
        lsf_sigma = fixed_list_to_numpy(spectrum.field("lsf_sigma")).astype(np.float32)
        z = table["Z"].to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 1)
        zerr = table["ZERR"].to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 1)
        ebv = table["EBV"].to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 1)
        zwarn = np.asarray(table["ZWARN"].to_pylist(), dtype=bool).reshape(-1, 1)
        ids = [str(value) for value in table["object_id"].to_pylist()]

        if args.require_zwarn_good:
            keep = zwarn.reshape(-1)
            flux_raw = flux_raw[keep]
            ivar = ivar[keep]
            mask = mask[keep]
            wavelength = wavelength[keep]
            lsf_sigma = lsf_sigma[keep]
            z = z[keep]
            zerr = zerr[keep]
            ebv = ebv[keep]
            zwarn = zwarn[keep]
            ids = [value for value, keep_value in zip(ids, keep) if bool(keep_value)]

        flux = normalize_flux_batch(flux_raw, mask)
        flux_chunks.append(torch.from_numpy(flux))
        flux_raw_chunks.append(torch.from_numpy(flux_raw))
        ivar_chunks.append(torch.from_numpy(ivar))
        mask_chunks.append(torch.from_numpy(mask))
        wavelength_chunks.append(torch.from_numpy(wavelength))
        lsf_sigma_chunks.append(torch.from_numpy(lsf_sigma))
        z_chunks.append(torch.from_numpy(z))
        zerr_chunks.append(torch.from_numpy(zerr))
        zwarn_chunks.append(torch.from_numpy(zwarn))
        ebv_chunks.append(torch.from_numpy(ebv))
        object_ids.extend(ids)

        if remaining is not None:
            remaining -= table.num_rows

    if not flux_chunks:
        raise RuntimeError("No parquet rows were processed")

    batch = {
        "flux": torch.cat(flux_chunks, dim=0),
        "flux_raw": torch.cat(flux_raw_chunks, dim=0),
        "ivar": torch.cat(ivar_chunks, dim=0),
        "mask": torch.cat(mask_chunks, dim=0),
        "wavelength": torch.cat(wavelength_chunks, dim=0),
        "z": torch.cat(z_chunks, dim=0),
        "zerr": torch.cat(zerr_chunks, dim=0),
        "zwarn": torch.cat(zwarn_chunks, dim=0),
        "ebv": torch.cat(ebv_chunks, dim=0),
        "lsf_sigma": torch.cat(lsf_sigma_chunks, dim=0),
    }
    z = batch["z"].reshape(-1)
    summary = {
        "count": int(len(z)),
        "length_min": int(batch["flux"].shape[1]),
        "length_max": int(batch["flux"].shape[1]),
        "length_mean": float(batch["flux"].shape[1]),
        "z_min": float(z.min().item()),
        "z_max": float(z.max().item()),
        "z_mean": float(z.mean().item()),
        "zwarn_true_count": int(batch["zwarn"].sum().item()),
        "zwarn_false_count": int((~batch["zwarn"]).sum().item()),
    }
    payload = {
        "source": source_paths,
        "summary": summary,
        "batch": batch,
        "metadata": {
            "object_id": object_ids,
            "hf_zwarn_true_interpreted_as_good": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Loaded records:", summary["count"])
    print("Saved processed batch to:", args.output)
    print("Saved summary to:", args.summary)


if __name__ == "__main__":
    main()
