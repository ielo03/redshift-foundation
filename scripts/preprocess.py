#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


DEFAULT_DR1_ROOT = Path("/global/cfs/cdirs/desi/public/dr1")
DEFAULT_HEALPIX_ROOT = DEFAULT_DR1_ROOT / "spectro/redux/iron/healpix"
DEFAULT_ZCAT_PATH = DEFAULT_DR1_ROOT / "spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits"
DEFAULT_OUT_DIR = Path("data/preprocessed")
KEEP_SPECTYPES = ("GALAXY", "QSO")
SPECTYPE_TO_CODE = {"GALAXY": 0, "QSO": 1}
CODE_TO_SPECTYPE = np.asarray(["GALAXY", "QSO"])
_WORKER_ALLOWED_TARGETS: AllowedTargets | None = None


class AllowedTargets:
    def __init__(self, targetid: np.ndarray, z: np.ndarray, spectype_code: np.ndarray):
        order = np.argsort(targetid)
        self.targetid = np.asarray(targetid[order], dtype=np.int64)
        self.z = np.asarray(z[order], dtype=np.float32)
        self.spectype_code = np.asarray(spectype_code[order], dtype=np.int8)

    def match(self, targetid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.targetid, targetid)
        valid = (idx < len(self.targetid)) & (self.targetid[idx.clip(max=max(len(self.targetid) - 1, 0))] == targetid)
        return valid, idx

    def values_for_indices(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.z[idx].astype(np.float32)
        spectype = CODE_TO_SPECTYPE[self.spectype_code[idx]]
        return z, spectype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean DESI DR1 row index using redshifty's training-time "
            "cuts plus no-stars and zcatalog-primary filtering."
        )
    )
    parser.add_argument("--healpix-root", type=Path, default=DEFAULT_HEALPIX_ROOT)
    parser.add_argument("--zcat", type=Path, default=DEFAULT_ZCAT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--allowed-cache",
        type=Path,
        default=None,
        help=(
            "Optional .npz cache of zcatalog-clean TARGETIDs. "
            "Defaults to OUT_DIR/allowed_targets_galaxy_qso_primary.npz."
        ),
    )
    parser.add_argument("--surveys", nargs="+", default=["sv3", "main"])
    parser.add_argument("--programs", nargs="+", default=["bright", "dark"])
    parser.add_argument("--max-healpix", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for independent HEALPix files. "
            "Use conservatively on shared filesystems; 4 is a reasonable first test."
        ),
    )
    parser.add_argument(
        "--no-objtype-tgt",
        action="store_true",
        help="Do not require OBJTYPE == TGT in the DR1 zcatalog.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help=(
            "Also save stitched flux/ivar/mask/wavelength arrays per healpix. "
            "This can be large; the default clean-row index is usually better."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing materialized .npz shards.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Write progress output every N processed HEALPix files.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Start from scratch instead of resuming from OUT_DIR/processed_files.jsonl.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def decode_string_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in {"S", "O", "U"}:
        return values.astype(str)
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="ignore").strip())
        else:
            out.append(str(value).strip())
    return np.asarray(out)


def iter_coadd_pairs(healpix_root: Path, surveys: list[str], programs: list[str]):
    for survey in surveys:
        for program in programs:
            base = healpix_root / survey / program
            if not base.is_dir():
                log(f"[scan] skip missing {base}")
                continue
            for group_dir in sorted(path for path in base.iterdir() if path.is_dir()):
                for hpix_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
                    healpix = int(hpix_dir.name)
                    coadd = hpix_dir / f"coadd-{survey}-{program}-{healpix}.fits"
                    redrock = hpix_dir / f"redrock-{survey}-{program}-{healpix}.fits"
                    if coadd.is_file() and redrock.is_file():
                        yield {
                            "survey": survey,
                            "program": program,
                            "healpix": healpix,
                            "coadd": coadd,
                            "redrock": redrock,
                        }


def load_allowed_targets(
    zcat_path: Path,
    *,
    require_objtype_tgt: bool,
) -> AllowedTargets:
    """Return TARGETID -> (z, spectype) for the desired training rows."""
    log(f"[zcat] loading allowed TARGETIDs from {zcat_path}")
    t0 = time.time()
    with fits.open(zcat_path, memmap=True) as hdul:
        table = hdul["ZCATALOG"].data
        spectype = decode_string_array(table["SPECTYPE"])
        keep = (
            (table["ZWARN"] == 0)
            & table["ZCAT_PRIMARY"].astype(bool)
            & np.isin(spectype, KEEP_SPECTYPES)
        )
        if require_objtype_tgt:
            objtype = decode_string_array(table["OBJTYPE"])
            keep &= objtype == "TGT"

        targetid = np.asarray(table["TARGETID"][keep], dtype=np.int64)
        z = np.asarray(table["Z"][keep], dtype=np.float32)
        kept_spectype = spectype[keep]

    spectype_code = np.asarray([SPECTYPE_TO_CODE[value] for value in kept_spectype], dtype=np.int8)
    allowed = AllowedTargets(targetid, z, spectype_code)
    log(f"[zcat] allowed targets: {len(allowed.targetid):,} ({time.time() - t0:.1f}s)")
    return allowed


def load_or_build_allowed_targets(
    zcat_path: Path,
    cache_path: Path,
    *,
    require_objtype_tgt: bool,
) -> AllowedTargets:
    if cache_path.exists():
        log(f"[zcat] loading allowed-target cache {cache_path}")
        t0 = time.time()
        payload = np.load(cache_path)
        allowed = AllowedTargets(payload["targetid"], payload["z"], payload["spectype_code"])
        log(f"[zcat] allowed targets: {len(allowed.targetid):,} ({time.time() - t0:.1f}s)")
        return allowed

    allowed = load_allowed_targets(zcat_path, require_objtype_tgt=require_objtype_tgt)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        targetid=allowed.targetid,
        z=allowed.z,
        spectype_code=allowed.spectype_code,
        require_objtype_tgt=np.asarray([require_objtype_tgt], dtype=bool),
    )
    log(f"[zcat] wrote allowed-target cache {cache_path}")
    return allowed


def nonzero_flux_mask(coadd_hdul: fits.HDUList) -> np.ndarray:
    total = (
        np.abs(coadd_hdul["B_FLUX"].data).sum(axis=1)
        + np.abs(coadd_hdul["R_FLUX"].data).sum(axis=1)
        + np.abs(coadd_hdul["Z_FLUX"].data).sum(axis=1)
    )
    return total > 0


def select_clean_rows(
    coadd_path: Path,
    redrock_path: Path,
    allowed_targets: AllowedTargets,
) -> dict[str, Any]:
    with fits.open(coadd_path, memmap=True) as coadd, fits.open(redrock_path, memmap=True) as redrock:
        fibermap = coadd["FIBERMAP"].data
        redshifts = redrock["REDSHIFTS"].data

        targetid = np.asarray(redshifts["TARGETID"], dtype=np.int64)
        spectype = decode_string_array(redshifts["SPECTYPE"])

        allowed_mask, allowed_idx = allowed_targets.match(targetid)
        redshifty_mask = (
            (redshifts["ZWARN"] == 0)
            & (fibermap["COADD_FIBERSTATUS"] == 0)
            & nonzero_flux_mask(coadd)
        )
        no_star_mask = np.isin(spectype, KEEP_SPECTYPES)
        clean = allowed_mask & redshifty_mask & no_star_mask
        rows = np.flatnonzero(clean).astype(np.int32)
        clean_allowed_idx = allowed_idx[rows]
        z, clean_spectype = allowed_targets.values_for_indices(clean_allowed_idx)

        return {
            "rows": rows,
            "targetid": targetid[rows].astype(np.int64),
            "z": z,
            "spectype": clean_spectype,
            "counts": {
                "rows_total": int(len(targetid)),
                "zcat_allowed": int(allowed_mask.sum()),
                "redshifty_good": int(redshifty_mask.sum()),
                "no_star_spectype": int(no_star_mask.sum()),
                "clean": int(len(rows)),
            },
        }


def init_worker(allowed_targets: AllowedTargets) -> None:
    global _WORKER_ALLOWED_TARGETS
    _WORKER_ALLOWED_TARGETS = allowed_targets


def select_clean_rows_worker(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if _WORKER_ALLOWED_TARGETS is None:
        raise RuntimeError("Worker allowed-target cache was not initialized")
    selection = select_clean_rows(rec["coadd"], rec["redrock"], _WORKER_ALLOWED_TARGETS)
    return rec, selection


def stitch_bands(
    wavelengths: list[np.ndarray],
    fluxes: list[np.ndarray],
    ivars: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Match redshifty's B/R/Z stitching with ivar-weighted overlaps."""
    all_wave = np.concatenate(wavelengths)
    all_flux = np.concatenate(fluxes)
    all_ivar = np.concatenate(ivars)
    all_mask = np.concatenate(masks)

    order = np.argsort(all_wave)
    all_wave = all_wave[order]
    all_flux = all_flux[order]
    all_ivar = all_ivar[order]
    all_mask = all_mask[order]

    unique_waves = []
    weighted_flux = []
    total_ivar = []
    combined_mask = []

    i = 0
    while i < len(all_wave):
        wave = all_wave[i]
        j = i
        while j < len(all_wave) and abs(all_wave[j] - wave) < 0.1:
            j += 1

        flux_chunk = all_flux[i:j]
        ivar_chunk = all_ivar[i:j]
        mask_chunk = all_mask[i:j]
        good = ~mask_chunk

        if good.any():
            ivar_sum = ivar_chunk[good].sum()
            if ivar_sum > 0:
                avg_flux = (flux_chunk[good] * ivar_chunk[good]).sum() / ivar_sum
            else:
                avg_flux = flux_chunk[good].mean()
            avg_mask = False
        else:
            avg_flux = flux_chunk.mean()
            ivar_sum = 0.0
            avg_mask = True

        unique_waves.append(wave)
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


def materialize_rows(
    rec: dict[str, Any],
    selection: dict[str, Any],
    out_path: Path,
    *,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        return

    rows = selection["rows"]
    flux_rows = []
    ivar_rows = []
    mask_rows = []
    wavelength = None

    with fits.open(rec["coadd"], memmap=True) as coadd:
        waves = [coadd[f"{band}_WAVELENGTH"].data for band in ("B", "R", "Z")]
        for row in rows:
            stitched = stitch_bands(
                waves,
                [coadd[f"{band}_FLUX"].data[row, :] for band in ("B", "R", "Z")],
                [coadd[f"{band}_IVAR"].data[row, :] for band in ("B", "R", "Z")],
                [coadd[f"{band}_MASK"].data[row, :] != 0 for band in ("B", "R", "Z")],
            )
            if wavelength is None:
                wavelength = stitched["wavelength"]
            flux_rows.append(stitched["flux"])
            ivar_rows.append(stitched["ivar"])
            mask_rows.append(stitched["mask"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        flux=np.stack(flux_rows).astype(np.float32),
        ivar=np.stack(ivar_rows).astype(np.float32),
        mask=np.stack(mask_rows).astype(bool),
        wavelength=np.asarray(wavelength, dtype=np.float32),
        z=selection["z"].astype(np.float32),
        targetid=selection["targetid"].astype(np.int64),
        row=rows.astype(np.int32),
        spectype=selection["spectype"].astype("U16"),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def write_progress(path: Path, summary: dict[str, Any]) -> None:
    payload = dict(summary)
    payload["elapsed_seconds"] = time.time() - float(summary["start_time"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_key(rec: dict[str, Any]) -> str:
    return f"{rec['survey']}/{rec['program']}/{rec['healpix']}"


def load_processed_state(path: Path) -> tuple[set[str], dict[str, Any]]:
    processed_keys: set[str] = set()
    state: dict[str, Any] = {
        "n_files_seen": 0,
        "n_files_kept": 0,
        "n_rows_total": 0,
        "n_rows_clean": 0,
        "counts": {},
    }
    if not path.exists():
        return processed_keys, state

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if key in processed_keys:
                continue
            processed_keys.add(key)
            state["n_files_seen"] += 1
            if row.get("kept", False):
                state["n_files_kept"] += 1
            state["n_rows_total"] += int(row.get("rows_total", 0))
            state["n_rows_clean"] += int(row.get("clean_rows", 0))
            for count_key, value in row.get("counts", {}).items():
                state["counts"][count_key] = state["counts"].get(count_key, 0) + int(value)
    return processed_keys, state


def rebuild_manifest_from_processed(processed_path: Path, manifest_path: Path) -> None:
    if not processed_path.exists():
        return
    seen: set[str] = set()
    manifest_rows = []
    with processed_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if key in seen:
                continue
            seen.add(key)
            manifest_row = row.get("manifest_row")
            if manifest_row is not None:
                manifest_rows.append(manifest_row)
    if not manifest_rows:
        log(f"[resume] processed log has no embedded manifest rows; leaving {manifest_path} unchanged")
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as dst:
        for manifest_row in manifest_rows:
            dst.write(json.dumps(manifest_row) + "\n")
    log(f"[resume] rebuilt {manifest_path} with {len(manifest_rows):,} kept HEALPix records from {processed_path}")


def iter_limited_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for rec in iter_coadd_pairs(args.healpix_root, args.surveys, args.programs):
        if args.max_healpix is not None and len(records) >= args.max_healpix:
            break
        records.append(rec)
    return records


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowed_cache = args.allowed_cache or (args.out_dir / "allowed_targets_galaxy_qso_primary.npz")
    manifest_path = args.out_dir / "clean_rows.jsonl"
    processed_path = args.out_dir / "processed_files.jsonl"
    summary_path = args.out_dir / "summary.json"
    progress_path = args.out_dir / "progress.json"

    if not args.resume:
        log(f"[resume] disabled; resetting {manifest_path} and {processed_path}")
        manifest_path.write_text("", encoding="utf-8")
        processed_path.write_text("", encoding="utf-8")

    processed_keys, previous_state = load_processed_state(processed_path) if args.resume else (
        set(),
        {
            "n_files_seen": 0,
            "n_files_kept": 0,
            "n_rows_total": 0,
            "n_rows_clean": 0,
            "counts": {},
        },
    )
    if processed_keys:
        log(
            f"[resume] loaded {len(processed_keys):,} processed HEALPix files "
            f"with {previous_state['n_rows_clean']:,} clean rows from {processed_path}"
        )
        rebuild_manifest_from_processed(processed_path, manifest_path)
    elif args.resume:
        log(f"[resume] no prior processed-file log at {processed_path}; starting fresh")

    allowed_targets = load_or_build_allowed_targets(
        args.zcat,
        allowed_cache,
        require_objtype_tgt=not args.no_objtype_tgt,
    )

    summary: dict[str, Any] = {
        "healpix_root": str(args.healpix_root),
        "zcat": str(args.zcat),
        "allowed_cache": str(allowed_cache),
        "out_dir": str(args.out_dir),
        "surveys": args.surveys,
        "programs": args.programs,
        "cuts": {
            "zcat_zwarn": "ZWARN == 0",
            "zcat_primary": "ZCAT_PRIMARY == True",
            "zcat_spectype": list(KEEP_SPECTYPES),
            "zcat_objtype": None if args.no_objtype_tgt else "OBJTYPE == TGT",
            "coadd_fiberstatus": "COADD_FIBERSTATUS == 0",
            "redrock_zwarn": "ZWARN == 0",
            "nonzero_flux": True,
        },
        "n_files_seen": previous_state["n_files_seen"],
        "n_files_kept": previous_state["n_files_kept"],
        "n_rows_total": previous_state["n_rows_total"],
        "n_rows_clean": previous_state["n_rows_clean"],
        "counts": previous_state["counts"],
        "start_time": time.time(),
        "resume": args.resume,
        "processed_log": str(processed_path),
    }

    t0 = time.time()
    summary["num_workers"] = max(1, int(args.num_workers))
    log(
        "[start] "
        f"out_dir={args.out_dir} allowed_cache={allowed_cache} "
        f"surveys={args.surveys} programs={args.programs} "
        f"max_healpix={args.max_healpix} max_rows={args.max_rows} "
        f"num_workers={args.num_workers} materialize={args.materialize} resume={args.resume}"
    )
    write_progress(progress_path, summary)

    def handle_selection(rec: dict[str, Any], selection: dict[str, Any]) -> bool:
        rec_key = record_key(rec)
        summary["n_files_seen"] += 1
        counts = selection["counts"]
        for count_key, value in counts.items():
            summary["counts"][count_key] = summary["counts"].get(count_key, 0) + int(value)
        summary["n_rows_total"] += counts["rows_total"]

        if len(selection["rows"]) == 0:
            append_jsonl(
                processed_path,
                {
                    "key": rec_key,
                    "survey": rec["survey"],
                    "program": rec["program"],
                    "healpix": rec["healpix"],
                    "coadd": str(rec["coadd"]),
                    "redrock": str(rec["redrock"]),
                    "kept": False,
                    "rows_total": int(counts["rows_total"]),
                    "clean_rows": 0,
                    "counts": {k: int(v) for k, v in counts.items()},
                },
            )
            processed_keys.add(rec_key)
            if summary["n_files_seen"] % max(1, args.progress_interval) == 0:
                dt = time.time() - t0
                log(
                    f"[scan] files={summary['n_files_seen']:,} kept={summary['n_files_kept']:,} "
                    f"clean_rows={summary['n_rows_clean']:,} elapsed={dt:.1f}s "
                    f"last={rec['survey']}/{rec['program']}/{rec['healpix']} clean=0"
                )
                write_progress(progress_path, summary)
            return False

        if args.max_rows is not None:
            remaining = args.max_rows - summary["n_rows_clean"]
            if remaining <= 0:
                return True
            for key in ("rows", "targetid", "z", "spectype"):
                selection[key] = selection[key][:remaining]

        shard_rel = Path("shards") / f"{rec['survey']}-{rec['program']}-{rec['healpix']}.npz"
        manifest_row = {
            "coadd": str(rec["coadd"]),
            "redrock": str(rec["redrock"]),
            "survey": rec["survey"],
            "program": rec["program"],
            "healpix": rec["healpix"],
            "rows": selection["rows"].astype(int).tolist(),
            "targetid": selection["targetid"].astype(int).tolist(),
            "z": selection["z"].astype(float).tolist(),
            "spectype": selection["spectype"].astype(str).tolist(),
            "materialized": str(args.out_dir / shard_rel) if args.materialize else None,
        }
        append_jsonl(manifest_path, manifest_row)
        summary["n_files_kept"] += 1
        summary["n_rows_clean"] += int(len(selection["rows"]))

        if args.materialize:
            materialize_rows(rec, selection, args.out_dir / shard_rel, overwrite=args.overwrite)

        append_jsonl(
            processed_path,
            {
                "key": rec_key,
                "survey": rec["survey"],
                "program": rec["program"],
                "healpix": rec["healpix"],
                "coadd": str(rec["coadd"]),
                "redrock": str(rec["redrock"]),
                "kept": True,
                "rows_total": int(counts["rows_total"]),
                "clean_rows": int(len(selection["rows"])),
                "counts": {k: int(v) for k, v in counts.items()},
                "manifest_row": manifest_row,
            },
        )
        processed_keys.add(rec_key)

        if summary["n_files_seen"] % max(1, args.progress_interval) == 0:
            dt = time.time() - t0
            log(
                f"[scan] files={summary['n_files_seen']:,} kept={summary['n_files_kept']:,} "
                f"clean_rows={summary['n_rows_clean']:,} elapsed={dt:.1f}s "
                f"last={rec['survey']}/{rec['program']}/{rec['healpix']} clean={len(selection['rows']):,}"
            )
            write_progress(progress_path, summary)
        return args.max_rows is not None and summary["n_rows_clean"] >= args.max_rows

    if args.num_workers <= 1:
        for seen, rec in enumerate(iter_coadd_pairs(args.healpix_root, args.surveys, args.programs), start=1):
            if args.max_healpix is not None and seen > args.max_healpix:
                break
            if args.max_rows is not None and summary["n_rows_clean"] >= args.max_rows:
                break
            if record_key(rec) in processed_keys:
                continue
            selection = select_clean_rows(rec["coadd"], rec["redrock"], allowed_targets)
            if handle_selection(rec, selection):
                break
    else:
        log("[scan] discovering HEALPix files before parallel processing")
        records = [rec for rec in iter_limited_records(args) if record_key(rec) not in processed_keys]
        log(f"[scan] discovered {len(records):,} HEALPix files")
        ctx = mp.get_context("fork")
        max_workers = max(1, int(args.num_workers))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(allowed_targets,),
        ) as executor:
            if args.max_rows is None:
                for rec, selection in executor.map(select_clean_rows_worker, records):
                    handle_selection(rec, selection)
            else:
                chunk_size = max_workers * 4
                for start in range(0, len(records), chunk_size):
                    if summary["n_rows_clean"] >= args.max_rows:
                        break
                    chunk = records[start : start + chunk_size]
                    for rec, selection in executor.map(select_clean_rows_worker, chunk):
                        if handle_selection(rec, selection):
                            break

    summary["elapsed_seconds"] = time.time() - t0
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress(progress_path, summary)

    log(f"[done] files seen: {summary['n_files_seen']:,}")
    log(f"[done] files kept: {summary['n_files_kept']:,}")
    log(f"[done] clean rows: {summary['n_rows_clean']:,}")
    log(f"[done] wrote {manifest_path}")
    log(f"[done] wrote {summary_path}")
    log(f"[done] wrote {progress_path}")


if __name__ == "__main__":
    main()
