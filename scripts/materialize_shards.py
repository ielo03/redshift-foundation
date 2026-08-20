#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits

from preprocess import SPECTYPE_TO_CODE, log, stitch_bands
from target_labels import TARGET_LABEL_NAMES, target_selection_labels


DEFAULT_MANIFEST = Path("data/preprocessed/split_manifest_v1.jsonl")
DEFAULT_OUT_DIR = Path("data/materialized_training_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize fixed-split clean DESI spectra into larger "
            "contiguous shards for faster PyTorch training."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-size", type=int, default=4096, help="Target spectra per shard.")
    parser.add_argument("--max-spectra", type=int, default=None, help="Optional cap on spectra to materialize.")
    parser.add_argument("--max-healpix", type=int, default=None, help="Optional cap on HEALPix manifest records.")
    parser.add_argument("--surveys", nargs="+", default=None, help="Optional survey filter, e.g. main sv3.")
    parser.add_argument("--programs", nargs="+", default=None, help="Optional program filter, e.g. dark bright.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "benchmark"],
        default=["train", "validation"],
        help="Fixed dataset splits to materialize. Benchmark is excluded by default.",
    )
    parser.add_argument(
        "--format",
        choices=["pt", "npz"],
        default="pt",
        help="Shard format. PyTorch .pt is the default for training.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print and write progress every N completed HEALPix records.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Start from scratch instead of skipping records in materialized_processed_files.jsonl.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def record_key(rec: dict[str, Any]) -> str:
    return f"{rec['survey']}/{rec['program']}/{rec['healpix']}"


def load_processed(path: Path) -> tuple[set[str], dict[str, int]]:
    processed: set[str] = set()
    state = {"n_healpix_written": 0, "n_spectra_written": 0}
    if not path.exists():
        return processed, state
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if key in processed:
                continue
            processed.add(key)
            state["n_healpix_written"] += 1
            state["n_spectra_written"] += int(row.get("n_spectra", 0))
    return processed, state


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def next_shard_index(shards_manifest: Path) -> int:
    if not shards_manifest.exists():
        return 0
    max_idx = -1
    with shards_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            max_idx = max(max_idx, int(json.loads(line)["shard_index"]))
    return max_idx + 1


def filter_record(rec: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.surveys is not None and rec["survey"] not in set(args.surveys):
        return False
    if args.programs is not None and rec["program"] not in set(args.programs):
        return False
    split = rec.get("split")
    if split is None:
        raise RuntimeError(
            "Materialization requires the fixed split manifest. Run scripts/build_split_manifest.py first."
        )
    if split not in set(args.splits):
        return False
    return True


def materialize_record(rec: dict[str, Any]) -> dict[str, Any]:
    rows = np.asarray(rec["rows"], dtype=np.int64)
    z = np.asarray(rec["z"], dtype=np.float32)
    targetid = np.asarray(rec["targetid"], dtype=np.int64)
    spectype_code = np.asarray([SPECTYPE_TO_CODE[value] for value in rec["spectype"]], dtype=np.int8)

    flux_rows = []
    ivar_rows = []
    valid_rows = []
    wavelength = None

    with fits.open(rec["coadd"], memmap=True) as coadd:
        fibermap = coadd["FIBERMAP"].data
        if not np.array_equal(np.asarray(fibermap["TARGETID"][rows], dtype=np.int64), targetid):
            raise RuntimeError(f"TARGETID mismatch between manifest and FIBERMAP for {record_key(rec)}")
        target_labels = target_selection_labels(fibermap[rows], str(rec["survey"]))
        waves = [coadd[f"{band}_WAVELENGTH"].data for band in ("B", "R", "Z")]
        for row in rows:
            stitched = stitch_bands(
                waves,
                [coadd[f"{band}_FLUX"].data[int(row), :] for band in ("B", "R", "Z")],
                [coadd[f"{band}_IVAR"].data[int(row), :] for band in ("B", "R", "Z")],
                [coadd[f"{band}_MASK"].data[int(row), :] != 0 for band in ("B", "R", "Z")],
            )
            if wavelength is None:
                wavelength = stitched["wavelength"]
            elif len(wavelength) != len(stitched["wavelength"]) or not np.allclose(wavelength, stitched["wavelength"]):
                raise RuntimeError(f"Wavelength grid changed inside {record_key(rec)}")

            valid = (~stitched["mask"]) & (stitched["ivar"] > 0)
            flux_rows.append(stitched["flux"])
            ivar_rows.append(stitched["ivar"])
            valid_rows.append(valid.astype(np.uint8))

    if wavelength is None:
        raise RuntimeError(f"Manifest record has no rows: {record_key(rec)}")

    return {
        "flux": np.stack(flux_rows).astype(np.float32),
        "ivar": np.stack(ivar_rows).astype(np.float32),
        "valid": np.stack(valid_rows).astype(np.uint8),
        "z": z,
        "targetid": targetid,
        "row": rows.astype(np.int32),
        "healpix": np.full(len(rows), int(rec["healpix"]), dtype=np.int32),
        "spectype": spectype_code,
        "target_labels": target_labels,
        "wavelength": wavelength.astype(np.float32),
        "record": {
            "key": record_key(rec),
            "survey": rec["survey"],
            "program": rec["program"],
            "healpix": rec["healpix"],
            "coadd": rec["coadd"],
            "n_spectra": int(len(rows)),
            "split": rec["split"],
        },
    }


def empty_buffer() -> dict[str, Any]:
    return {
        "flux": [],
        "ivar": [],
        "valid": [],
        "z": [],
        "targetid": [],
        "row": [],
        "healpix": [],
        "spectype": [],
        "target_labels": [],
        "records": [],
        "wavelength": None,
        "n_spectra": 0,
    }


def add_to_buffer(buffer: dict[str, Any], materialized: dict[str, Any]) -> None:
    wavelength = materialized["wavelength"]
    if buffer["wavelength"] is None:
        buffer["wavelength"] = wavelength
    elif len(buffer["wavelength"]) != len(wavelength) or not np.allclose(buffer["wavelength"], wavelength):
        raise RuntimeError("Wavelength grid changed across records in one shard")

    for key in ("flux", "ivar", "valid", "z", "targetid", "row", "healpix", "spectype", "target_labels"):
        buffer[key].append(materialized[key])
    buffer["records"].append(materialized["record"])
    buffer["n_spectra"] += int(materialized["flux"].shape[0])


def finalize_buffer(buffer: dict[str, Any]) -> dict[str, Any]:
    return {
        "flux": torch.from_numpy(np.concatenate(buffer["flux"], axis=0).astype(np.float32)),
        "ivar": torch.from_numpy(np.concatenate(buffer["ivar"], axis=0).astype(np.float32)),
        "valid": torch.from_numpy(np.concatenate(buffer["valid"], axis=0).astype(np.uint8)),
        "z": torch.from_numpy(np.concatenate(buffer["z"], axis=0).astype(np.float32)),
        "targetid": torch.from_numpy(np.concatenate(buffer["targetid"], axis=0).astype(np.int64)),
        "row": torch.from_numpy(np.concatenate(buffer["row"], axis=0).astype(np.int32)),
        "healpix": torch.from_numpy(np.concatenate(buffer["healpix"], axis=0).astype(np.int32)),
        "spectype": torch.from_numpy(np.concatenate(buffer["spectype"], axis=0).astype(np.int8)),
        "target_labels": torch.from_numpy(np.concatenate(buffer["target_labels"], axis=0).astype(np.uint8)),
        "target_label_names": TARGET_LABEL_NAMES,
        "wavelength": torch.from_numpy(buffer["wavelength"].astype(np.float32)),
        "records": buffer["records"],
    }


def save_shard(payload: dict[str, Any], path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pt":
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return

    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        flux=payload["flux"].numpy(),
        ivar=payload["ivar"].numpy(),
        valid=payload["valid"].numpy(),
        z=payload["z"].numpy(),
        targetid=payload["targetid"].numpy(),
        row=payload["row"].numpy(),
        healpix=payload["healpix"].numpy(),
        spectype=payload["spectype"].numpy(),
        target_labels=payload["target_labels"].numpy(),
        wavelength=payload["wavelength"].numpy(),
        records=np.asarray(json.dumps(payload["records"])),
    )
    tmp.replace(path)


def write_progress(path: Path, progress: dict[str, Any], start_time: float) -> None:
    payload = dict(progress)
    payload["elapsed_seconds"] = time.time() - start_time
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processed_path = args.out_dir / "materialized_processed_files.jsonl"
    shards_manifest = args.out_dir / "shards_manifest.jsonl"
    progress_path = args.out_dir / "materialize_progress.json"
    shard_dir = args.out_dir / "shards"

    if not args.resume:
        log(f"[resume] disabled; resetting {processed_path} and {shards_manifest}")
        processed_path.write_text("", encoding="utf-8")
        shards_manifest.write_text("", encoding="utf-8")

    processed, prior_state = load_processed(processed_path) if args.resume else (
        set(),
        {"n_healpix_written": 0, "n_spectra_written": 0},
    )
    shard_index = next_shard_index(shards_manifest)
    if processed:
        log(f"[resume] loaded {len(processed):,} materialized HEALPix records from {processed_path}")

    start_time = time.time()
    progress = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "format": args.format,
        "shard_size": args.shard_size,
        "max_spectra": args.max_spectra,
        "max_healpix": args.max_healpix,
        "splits": args.splits,
        "n_healpix_written": prior_state["n_healpix_written"],
        "n_spectra_written": prior_state["n_spectra_written"],
        "next_shard_index": shard_index,
    }
    write_progress(progress_path, progress, start_time)
    log(
        "[start] "
        f"manifest={args.manifest} out_dir={args.out_dir} format={args.format} "
        f"shard_size={args.shard_size} max_spectra={args.max_spectra} max_healpix={args.max_healpix} splits={args.splits}"
    )

    buffer = empty_buffer()
    selected_healpix = 0

    def flush_buffer() -> None:
        nonlocal buffer, shard_index
        if buffer["n_spectra"] == 0:
            return
        suffix = "pt" if args.format == "pt" else "npz"
        shard_path = shard_dir / f"shard-{shard_index:06d}.{suffix}"
        payload = finalize_buffer(buffer)
        save_shard(payload, shard_path, args.format)
        shard_record = {
            "shard_index": shard_index,
            "path": str(shard_path),
            "format": args.format,
            "n_spectra": int(payload["flux"].shape[0]),
            "length": int(payload["flux"].shape[1]),
            "records": payload["records"],
        }
        append_jsonl(shards_manifest, shard_record)
        for record in payload["records"]:
            append_jsonl(processed_path, record)
            processed.add(record["key"])
        progress["n_healpix_written"] += len(payload["records"])
        progress["n_spectra_written"] += int(payload["flux"].shape[0])
        progress["next_shard_index"] = shard_index + 1
        write_progress(progress_path, progress, start_time)
        log(
            f"[shard] wrote {shard_path} spectra={payload['flux'].shape[0]:,} "
            f"records={len(payload['records']):,} total_spectra={progress['n_spectra_written']:,}"
        )
        shard_index += 1
        buffer = empty_buffer()

    for rec in read_jsonl(args.manifest):
        if not filter_record(rec, args):
            continue
        key = record_key(rec)
        if key in processed:
            continue
        if args.max_healpix is not None and progress["n_healpix_written"] + selected_healpix >= args.max_healpix:
            break
        if args.max_spectra is not None and progress["n_spectra_written"] >= args.max_spectra:
            break

        materialized = materialize_record(rec)
        selected_healpix += 1
        if args.max_spectra is not None:
            remaining = args.max_spectra - progress["n_spectra_written"] - buffer["n_spectra"]
            if remaining <= 0:
                flush_buffer()
                break
            n = int(materialized["flux"].shape[0])
            if n > remaining:
                for key_to_trim in ("flux", "ivar", "valid", "z", "targetid", "row", "healpix", "spectype", "target_labels"):
                    materialized[key_to_trim] = materialized[key_to_trim][:remaining]
                materialized["record"] = dict(materialized["record"])
                materialized["record"]["n_spectra"] = int(remaining)

        add_to_buffer(buffer, materialized)
        if buffer["n_spectra"] >= args.shard_size:
            flush_buffer()

        if progress["n_healpix_written"] % max(1, args.progress_interval) == 0 and progress["n_healpix_written"] > 0:
            write_progress(progress_path, progress, start_time)

    flush_buffer()
    write_progress(progress_path, progress, start_time)
    log(f"[done] spectra={progress['n_spectra_written']:,}")
    log(f"[done] wrote {shards_manifest}")
    log(f"[done] wrote {processed_path}")
    log(f"[done] wrote {progress_path}")


if __name__ == "__main__":
    main()
