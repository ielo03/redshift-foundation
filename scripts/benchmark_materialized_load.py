#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark loading and normalizing materialized spectrum shards.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def normalize_rows(flux: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(flux, dtype=np.float32)
    for index in range(len(flux)):
        good = valid[index].astype(bool) & np.isfinite(flux[index])
        if not good.any():
            continue
        values = flux[index, good]
        center = np.median(values)
        scale = np.percentile(values, 95) - np.percentile(values, 5)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = np.std(values)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized[index] = ((flux[index] - center) / scale).astype(np.float32)
        normalized[index, ~good] = 0.0
    return normalized


def main() -> None:
    args = parse_args()
    shard_paths = sorted((args.input / "shards").glob("shard-*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No PyTorch shards found under {args.input}")

    t0 = time.perf_counter()
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]
    load_seconds = time.perf_counter() - t0
    flux = torch.cat([payload["flux"] for payload in payloads]).numpy()
    valid = torch.cat([payload["valid"] for payload in payloads]).numpy()

    t1 = time.perf_counter()
    normalized = normalize_rows(flux, valid)
    normalize_seconds = time.perf_counter() - t1
    n_spectra = int(len(flux))
    total_seconds = load_seconds + normalize_seconds
    result = {
        "input": str(args.input),
        "n_shards": len(shard_paths),
        "n_spectra": n_spectra,
        "bytes_on_disk": sum(path.stat().st_size for path in shard_paths),
        "load_seconds": load_seconds,
        "normalize_seconds": normalize_seconds,
        "load_and_normalize_seconds": total_seconds,
        "spectra_per_second": n_spectra / total_seconds,
        "normalized_shape": list(normalized.shape),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
