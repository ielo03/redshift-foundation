#!/usr/bin/env python3
"""Add DESI BGS/LRG/ELG/QSO target labels to existing materialized shards."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "final-project" / "src"))

from data_pipeline import materialized_shard_paths
from target_labels import TARGET_LABEL_NAMES, target_selection_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true", help="Replace existing target_labels after recomputing them.")
    return p.parse_args()


def enrich(path: Path, overwrite: bool) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "target_labels" in payload and not overwrite:
        print(f"[skip] {path.name}: target_labels already present", flush=True)
        return
    labels: list[np.ndarray] = []
    offset = 0
    targetid = payload["targetid"].numpy()
    for record in payload["records"]:
        n = int(record["n_spectra"])
        with fits.open(record["coadd"], memmap=True) as coadd:
            fibermap = coadd["FIBERMAP"].data
            rows = payload["row"][offset : offset + n].numpy().astype(np.int64)
            selected = fibermap[rows]
            if not np.array_equal(np.asarray(selected["TARGETID"], dtype=np.int64), targetid[offset : offset + n]):
                raise RuntimeError(f"TARGETID mismatch in {path} record {record['key']}")
            labels.append(target_selection_labels(selected, str(record["survey"])))
        offset += n
    if offset != len(targetid):
        raise RuntimeError(f"Record count mismatch in {path}")
    payload["target_labels"] = torch.from_numpy(np.concatenate(labels, axis=0))
    payload["target_label_names"] = TARGET_LABEL_NAMES
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    print(f"[done] {path.name}: {len(targetid):,} spectra", flush=True)


def main() -> None:
    args = parse_args()
    paths = materialized_shard_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No materialized shards in {args.input}")
    for index, path in enumerate(paths, start=1):
        enrich(path, args.overwrite)
        if index % 10 == 0 or index == len(paths):
            print(f"[progress] {index}/{len(paths)} shards", flush=True)


if __name__ == "__main__":
    main()
