from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))


def count_filtered_hdf5(path: Path, *, survey: str | None, primary_only: bool) -> int:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        n = f["spectrum_flux"].shape[0]
        indices = np.arange(n)
        if survey is not None and "SURVEY" in f:
            values = np.asarray(f["SURVEY"])
            decoded = np.asarray(
                [
                    value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
                    for value in values
                ]
            )
            indices = indices[np.char.lower(decoded.astype(str)) == survey.lower()]
        if primary_only and "SV_PRIMARY" in f:
            primary = np.asarray(f["SV_PRIMARY"]).astype(bool)
            indices = indices[primary[indices]]
    return int(len(indices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a raw DESI/MMU HDF5 file into tensors.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("hw/final_project/data/raw/001-of-002.hdf5"), Path("hw/final_project/data/raw/002-of-002.hdf5")],
        help="Path(s) to the raw HDF5 file(s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hw/final_project/data/processed/desi_sv3_primary.pt"),
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("hw/final_project/outputs/tables/desi_sv3_primary_summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--survey",
        type=str,
        default="sv3",
        help="Survey filter to apply when loading the HDF5 file.",
    )
    parser.add_argument(
        "--no-primary-only",
        action="store_true",
        help="Do not filter to SV_PRIMARY=True.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional total cap on the number of spectra to save across all input files.",
    )
    parser.add_argument(
        "--skip-items",
        type=int,
        default=0,
        help="Skip this many filtered spectra across all input files before saving records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    from data_pipeline import collate_records, load_hdf5, summarize_records
    from project_paths import ensure_project_dirs

    ensure_project_dirs()

    records = []
    skipped_remaining = max(0, args.skip_items)
    for input_path in args.input:
        if skipped_remaining > 0:
            available = count_filtered_hdf5(
                input_path,
                survey=args.survey,
                primary_only=not args.no_primary_only,
            )
            if skipped_remaining >= available:
                skipped_remaining -= available
                continue
            skip_for_file = skipped_remaining
            skipped_remaining = 0
        else:
            skip_for_file = 0
        remaining = None if args.max_items is None else args.max_items - len(records)
        if remaining is not None and remaining <= 0:
            break
        records.extend(
            load_hdf5(
                input_path,
                survey=args.survey,
                primary_only=not args.no_primary_only,
                max_items=remaining,
                skip_items=skip_for_file,
            )
        )
    if not records:
        raise RuntimeError("No records were loaded from the raw HDF5 file")

    summary = summarize_records(records)
    batch = collate_records(records, target_length=records[0].flux.numel())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch_payload = {
        "source": [str(path) for path in args.input],
        "summary": summary,
        "batch": batch,
    }

    import torch

    torch.save(torch_payload, args.output)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Loaded records:", summary["count"])
    print("Saved processed batch to:", args.output)
    print("Saved summary to:", args.summary)


if __name__ == "__main__":
    main()
