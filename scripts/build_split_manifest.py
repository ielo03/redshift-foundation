#!/usr/bin/env python3
"""Create the immutable DR1 train/validation/benchmark split manifest.

The assignment is at the HEALPix-coadd level.  This keeps rows from one
coadd together and makes the assignment stable when more DR1 data is added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("data/preprocessed/clean_rows.jsonl")
DEFAULT_OUTPUT = Path("data/preprocessed/split_manifest_v1.jsonl")
DEFAULT_SUMMARY = Path("data/preprocessed/split_manifest_v1_summary.json")
SPLIT_VERSION = "desi-dr1-healpix-split-v1"
SPLITS = ("train", "validation", "benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the fixed HEALPix-level DR1 train/validation/benchmark manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split manifest. This changes the split artifact only, never the clean manifest.",
    )
    return parser.parse_args()


def validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave a nonzero benchmark fraction")


def assignment(record: dict[str, Any], train_fraction: float, validation_fraction: float) -> str:
    key = f"{SPLIT_VERSION}\0{record['survey']}/{record['program']}/{record['healpix']}"
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + validation_fraction:
        return "validation"
    return "benchmark"


def main() -> None:
    args = parse_args()
    validate_fractions(args.train_fraction, args.validation_fraction)
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Clean manifest not found: {args.manifest}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Split manifest already exists: {args.output}. Use --overwrite only to replace it deliberately.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    records = Counter()
    seen_keys: set[str] = set()
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as output, args.manifest.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = f"{record['survey']}/{record['program']}/{record['healpix']}"
                if key in seen_keys:
                    raise RuntimeError(f"Duplicate HEALPix record in {args.manifest}: {key} (line {line_number})")
                seen_keys.add(key)
                split = assignment(record, args.train_fraction, args.validation_fraction)
                record["split"] = split
                record["split_version"] = SPLIT_VERSION
                record["split_key"] = key
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                records[split] += 1
                counts[split] += len(record["rows"])
        Path(temp_name).replace(args.output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    summary = {
        "split_version": SPLIT_VERSION,
        "source_manifest": str(args.manifest),
        "split_manifest": str(args.output),
        "assignment_unit": "survey/program/healpix",
        "assignment_method": "sha256 of split version and HEALPix key",
        "fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "benchmark": 1.0 - args.train_fraction - args.validation_fraction,
        },
        "records": {split: records[split] for split in SPLITS},
        "spectra": {split: counts[split] for split in SPLITS},
        "total_records": sum(records.values()),
        "total_spectra": sum(counts.values()),
        "policy": {
            "training_allowed_splits": ["train", "validation"],
            "benchmark_only_split": "benchmark",
            "benchmark_rule": "Training scripts must reject benchmark records; only benchmarking code may request them.",
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.output}")
    print(f"[done] wrote {args.summary}")
    for split in SPLITS:
        print(f"[done] {split}: {records[split]:,} HEALPix records, {counts[split]:,} spectra")


if __name__ == "__main__":
    main()
