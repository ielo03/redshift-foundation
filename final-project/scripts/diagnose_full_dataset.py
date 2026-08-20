from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_RAW_FILES = [
    Path("data/raw/001-of-004.hdf5"),
    Path("data/raw/002-of-004.hdf5"),
    Path("data/raw/003-of-004.hdf5"),
    Path("data/raw/004-of-004.hdf5"),
]
Z_BINS = [-0.01, 0.0, 0.2, 0.5, 1.0, 2.0, 10.0]


def decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in {"S", "O"}:
        return values
    return np.asarray(
        [value.decode("utf-8", errors="ignore").strip() if isinstance(value, bytes) else str(value).strip() for value in values]
    )


def pct(values: np.ndarray, qs: list[float]) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.percentile(values, q)) for q in qs}


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def bin_label(lo: float, hi: float) -> str:
    return f"[{lo:g},{hi:g})"


def summarize_predictions(predictions_path: Path, metadata_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not predictions_path.exists():
        return None
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    actual = np.asarray(payload["actual"], dtype=np.float64)
    predicted = np.asarray(payload["predicted"], dtype=np.float64)
    error = predicted - actual
    abs_error = np.abs(error)

    # Reproduce the trainer's held-out split at the row-index level.
    from sklearn.model_selection import train_test_split

    row_indices = np.arange(len(metadata_rows))
    _, test_indices = train_test_split(row_indices, test_size=0.25, random_state=42)
    if len(test_indices) != len(actual):
        return {"error": f"prediction count {len(actual)} does not match reproduced test count {len(test_indices)}"}

    test_meta = [metadata_rows[int(i)] for i in test_indices]

    by_spectype: dict[str, list[int]] = defaultdict(list)
    by_zwarn: dict[str, list[int]] = defaultdict(list)
    by_zbin: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(test_meta):
        by_spectype[str(row["SPECTYPE"])].append(idx)
        by_zwarn[str(row["ZWARN"])].append(idx)
        z = actual[idx]
        for lo, hi in zip(Z_BINS[:-1], Z_BINS[1:]):
            if lo <= z < hi:
                by_zbin[bin_label(lo, hi)].append(idx)
                break

    def group_stats(groups: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
        out = {}
        for key, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            idx = np.asarray(indices, dtype=np.int64)
            out[key] = {
                "n": int(idx.size),
                "mae": float(abs_error[idx].mean()),
                "rmse": float(np.sqrt(np.mean(error[idx] ** 2))),
                "bias": float(error[idx].mean()),
            }
        return out

    return {
        "n_test": int(len(actual)),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "abs_error_percentiles": pct(abs_error, [50, 68, 80, 90, 95, 99]),
        "by_spectype": group_stats(by_spectype),
        "by_zwarn": group_stats(by_zwarn),
        "by_zbin": group_stats(by_zbin),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the full DESI training dataset and saved predictions.")
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_RAW_FILES)
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/full_dataset_diagnostics.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/redshift_tail_large_full_eval_best_now/redshift_predictions.json"),
    )
    args = parser.parse_args()

    metadata_rows: list[dict[str, Any]] = []
    source_counts = {}
    survey_counts = Counter()
    sv_primary_counts = Counter()
    spectype_counts = Counter()
    objtype_counts = Counter()
    zwarn_counts = Counter()
    zcat_primary_counts = Counter()
    fiberstatus_counts = Counter()
    program_counts = Counter()
    spectype_z: dict[str, list[float]] = defaultdict(list)
    spectype_zwarn: dict[str, Counter] = defaultdict(Counter)
    z_values = []
    zerr_values = []
    deltachi2_values = []
    tsnr2_bgs_values = []
    tsnr2_elg_values = []
    tsnr2_lrg_values = []
    tsnr2_qso_values = []
    mask_fractions = []
    zero_ivar_fractions = []
    raw_mean_values = []
    raw_std_values = []
    raw_abs_p99_values = []
    normalized_abs_p99_values = []
    nonfinite_flux_count = 0
    nonfinite_ivar_count = 0
    fully_masked_count = 0
    near_zero_std_count = 0
    selected_total = 0

    for path in args.input:
        with h5py.File(path, "r") as f:
            n_file = int(f["Z"].shape[0])
            survey = decode_array(np.asarray(f["SURVEY"])) if "SURVEY" in f else np.asarray([""] * n_file)
            sv_primary = np.asarray(f["SV_PRIMARY"]).astype(bool) if "SV_PRIMARY" in f else np.ones(n_file, dtype=bool)
            selected = np.asarray([str(value).lower() == "sv3" for value in survey]) & sv_primary
            indices = np.flatnonzero(selected)
            source_counts[str(path)] = {"rows": n_file, "selected": int(indices.size)}
            selected_total += int(indices.size)

            z = np.asarray(f["Z"])[indices]
            zerr = np.asarray(f["ZERR"])[indices] if "ZERR" in f else np.full(indices.size, np.nan)
            zwarn = np.asarray(f["ZWARN"])[indices] if "ZWARN" in f else np.full(indices.size, -1)
            spectype = decode_array(np.asarray(f["SPECTYPE"])[indices]) if "SPECTYPE" in f else np.asarray([""] * indices.size)
            objtype = decode_array(np.asarray(f["OBJTYPE"])[indices]) if "OBJTYPE" in f else np.asarray([""] * indices.size)
            program = decode_array(np.asarray(f["PROGRAM"])[indices]) if "PROGRAM" in f else np.asarray([""] * indices.size)
            zcat_primary = np.asarray(f["ZCAT_PRIMARY"])[indices].astype(bool) if "ZCAT_PRIMARY" in f else np.full(indices.size, False)
            fiberstatus = np.asarray(f["COADD_FIBERSTATUS"])[indices] if "COADD_FIBERSTATUS" in f else np.full(indices.size, -1)
            deltachi2 = np.asarray(f["DELTACHI2"])[indices] if "DELTACHI2" in f else np.full(indices.size, np.nan)

            for counter, values in [
                (survey_counts, survey[indices]),
                (sv_primary_counts, sv_primary[indices]),
                (spectype_counts, spectype),
                (objtype_counts, objtype),
                (zwarn_counts, zwarn),
                (zcat_primary_counts, zcat_primary),
                (fiberstatus_counts, fiberstatus),
                (program_counts, program),
            ]:
                counter.update(values.tolist())

            for stype, zi, zwi in zip(spectype, z, zwarn):
                spectype_z[str(stype)].append(float(zi))
                spectype_zwarn[str(stype)].update([int(zwi)])

            z_values.append(z)
            zerr_values.append(zerr)
            deltachi2_values.append(deltachi2)
            for key, sink in [
                ("TSNR2_BGS", tsnr2_bgs_values),
                ("TSNR2_ELG", tsnr2_elg_values),
                ("TSNR2_LRG", tsnr2_lrg_values),
                ("TSNR2_QSO", tsnr2_qso_values),
            ]:
                sink.append(np.asarray(f[key])[indices] if key in f else np.full(indices.size, np.nan))

            for i, idx in enumerate(indices):
                metadata_rows.append(
                    {
                        "source": str(path),
                        "index": int(idx),
                        "Z": float(z[i]),
                        "ZERR": float(zerr[i]),
                        "ZWARN": int(zwarn[i]),
                        "SPECTYPE": str(spectype[i]),
                        "OBJTYPE": str(objtype[i]),
                        "PROGRAM": str(program[i]),
                        "ZCAT_PRIMARY": bool(zcat_primary[i]),
                        "COADD_FIBERSTATUS": int(fiberstatus[i]),
                    }
                )

            chunk_size = 2048
            for start in range(0, indices.size, chunk_size):
                chunk_indices = indices[start : start + chunk_size]
                flux = np.asarray(f["spectrum_flux"][chunk_indices], dtype=np.float32)
                ivar = np.asarray(f["spectrum_ivar"][chunk_indices], dtype=np.float32)
                mask = np.asarray(f["spectrum_mask"][chunk_indices]).astype(bool)
                nonfinite_flux_count += int((~np.isfinite(flux)).sum())
                nonfinite_ivar_count += int((~np.isfinite(ivar)).sum())
                valid = (~mask) & np.isfinite(flux)
                valid_counts = valid.sum(axis=1)
                fully_masked_count += int((valid_counts == 0).sum())
                mask_fractions.extend(mask.mean(axis=1).astype(np.float64).tolist())
                zero_ivar_fractions.extend((ivar <= 0).mean(axis=1).astype(np.float64).tolist())

                safe_flux = np.where(valid, flux, np.nan)
                mean = np.nanmean(safe_flux, axis=1)
                std = np.nanstd(safe_flux, axis=1)
                near_zero_std_count += int((np.isfinite(std) & (std <= 1e-6)).sum())
                raw_mean_values.extend(mean.astype(np.float64).tolist())
                raw_std_values.extend(std.astype(np.float64).tolist())
                raw_abs_p99_values.extend(np.nanpercentile(np.abs(safe_flux), 99, axis=1).astype(np.float64).tolist())
                norm = (safe_flux - mean[:, None]) / np.maximum(std[:, None], 1e-6)
                normalized_abs_p99_values.extend(np.nanpercentile(np.abs(norm), 99, axis=1).astype(np.float64).tolist())

    z_all = np.concatenate(z_values)
    zerr_all = np.concatenate(zerr_values)
    deltachi2_all = np.concatenate(deltachi2_values)

    zbin_counts = {}
    for lo, hi in zip(Z_BINS[:-1], Z_BINS[1:]):
        zbin_counts[bin_label(lo, hi)] = int(((z_all >= lo) & (z_all < hi)).sum())

    diagnostics = {
        "selected_total": selected_total,
        "sources": source_counts,
        "metadata_counts": {
            "survey": counter_to_dict(survey_counts),
            "sv_primary": counter_to_dict(sv_primary_counts),
            "spectype": counter_to_dict(spectype_counts),
            "objtype": counter_to_dict(objtype_counts),
            "zwarn": counter_to_dict(zwarn_counts),
            "zcat_primary": counter_to_dict(zcat_primary_counts),
            "coadd_fiberstatus": counter_to_dict(fiberstatus_counts),
            "program": counter_to_dict(program_counts),
        },
        "redshift": {
            "min": float(np.min(z_all)),
            "max": float(np.max(z_all)),
            "mean": float(np.mean(z_all)),
            "percentiles": pct(z_all, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
            "bins": zbin_counts,
            "negative_count": int((z_all < 0).sum()),
            "high_z_2_plus_count": int((z_all >= 2).sum()),
        },
        "redshift_by_spectype": {
            key: {
                "n": int(len(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "percentiles": pct(np.asarray(values), [1, 5, 25, 50, 75, 95, 99]),
            }
            for key, values in sorted(spectype_z.items())
        },
        "zwarn_by_spectype": {key: counter_to_dict(counter) for key, counter in sorted(spectype_zwarn.items())},
        "quality": {
            "zwarn_nonzero_count": int((np.asarray([row["ZWARN"] for row in metadata_rows]) != 0).sum()),
            "zcat_primary_false_count": int((np.asarray([row["ZCAT_PRIMARY"] for row in metadata_rows]) == False).sum()),
            "fiberstatus_nonzero_count": int((np.asarray([row["COADD_FIBERSTATUS"] for row in metadata_rows]) != 0).sum()),
            "zerr_percentiles": pct(zerr_all[np.isfinite(zerr_all)], [50, 90, 95, 99]),
            "deltachi2_percentiles": pct(deltachi2_all[np.isfinite(deltachi2_all)], [1, 5, 25, 50, 75, 95, 99]),
        },
        "spectrum_tensors": {
            "nonfinite_flux_count": nonfinite_flux_count,
            "nonfinite_ivar_count": nonfinite_ivar_count,
            "fully_masked_count": fully_masked_count,
            "near_zero_valid_flux_std_count": near_zero_std_count,
            "mask_fraction_percentiles": pct(np.asarray(mask_fractions), [0, 50, 90, 95, 99, 100]),
            "zero_ivar_fraction_percentiles": pct(np.asarray(zero_ivar_fractions), [0, 50, 90, 95, 99, 100]),
            "raw_flux_mean_percentiles": pct(np.asarray(raw_mean_values), [1, 5, 50, 95, 99]),
            "raw_flux_std_percentiles": pct(np.asarray(raw_std_values), [1, 5, 50, 95, 99]),
            "raw_abs_flux_p99_percentiles": pct(np.asarray(raw_abs_p99_values), [50, 90, 95, 99]),
            "normalized_abs_flux_p99_percentiles": pct(np.asarray(normalized_abs_p99_values), [50, 90, 95, 99]),
        },
        "prediction_diagnostics": summarize_predictions(args.predictions, metadata_rows),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
