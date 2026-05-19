from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import h5py
import pyarrow.compute as pc
import pyarrow.parquet as pq


HF_DIR = Path("data/mmu_desi_hf/edr_sv3")
FULL_DIAG = Path("outputs/tables/full_dataset_diagnostics.json")
OLD_DIAG = Path("outputs/tables/old_8k_dataset_diagnostics.json")
OUT = Path("outputs/tables/hf_vs_hdf5_dataset_comparison.json")
Z_BINS = [-0.01, 0.0, 0.2, 0.5, 1.0, 2.0, 10.0]


def pct(values: np.ndarray, qs: list[float]) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.percentile(values, q)) for q in qs}


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def scalar_list_array_to_matrix(values: Any) -> np.ndarray:
    # pyarrow ListArray -> Python list -> ndarray. Kept for small sampled rows.
    return np.asarray(values.to_pylist(), dtype=np.float32)


def summarize_hf() -> dict[str, Any]:
    files = sorted(HF_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {HF_DIR}")

    schemas = {}
    total_rows = 0
    row_groups = 0
    z_chunks = []
    zerr_chunks = []
    zwarn_counter = Counter()
    ebv_chunks = []
    object_ids: set[int] = set()
    file_rows = {}
    sample_spectrum_summary = None

    for path in files:
        pf = pq.ParquetFile(path)
        file_rows[str(path)] = pf.metadata.num_rows
        total_rows += pf.metadata.num_rows
        row_groups += pf.metadata.num_row_groups
        schemas[str(path)] = {
            "columns": pf.schema_arrow.names,
            "schema": str(pf.schema_arrow),
        }

        wanted = [name for name in ["Z", "ZERR", "ZWARN", "EBV", "object_id"] if name in pf.schema_arrow.names]
        table = pq.read_table(path, columns=wanted)
        if "Z" in table.column_names:
            z_chunks.append(table["Z"].to_numpy(zero_copy_only=False).astype(np.float64))
        if "ZERR" in table.column_names:
            zerr_chunks.append(table["ZERR"].to_numpy(zero_copy_only=False).astype(np.float64))
        if "ZWARN" in table.column_names:
            zwarn_counter.update(table["ZWARN"].to_pylist())
        if "EBV" in table.column_names:
            ebv_chunks.append(table["EBV"].to_numpy(zero_copy_only=False).astype(np.float64))
        if "object_id" in table.column_names:
            object_ids.update(int(x) for x in table["object_id"].to_pylist())

        if sample_spectrum_summary is None:
            first = pq.read_table(path, columns=["spectrum"], use_threads=True).slice(0, min(32, pf.metadata.num_rows))
            spectra = first["spectrum"].combine_chunks()
            first_item = spectra[0].as_py()
            spectrum_keys = sorted(first_item.keys()) if isinstance(first_item, dict) else []
            sample_spectrum_summary = {
                "spectrum_type": str(spectra.type),
                "spectrum_keys": spectrum_keys,
            }
            if isinstance(first_item, dict):
                for key, value in first_item.items():
                    try:
                        arr = np.asarray(value)
                        sample_spectrum_summary[f"{key}_shape"] = list(arr.shape)
                        sample_spectrum_summary[f"{key}_dtype"] = str(arr.dtype)
                    except Exception:
                        sample_spectrum_summary[f"{key}_repr"] = repr(value)[:200]

    z_all = np.concatenate(z_chunks) if z_chunks else np.asarray([])
    zerr_all = np.concatenate(zerr_chunks) if zerr_chunks else np.asarray([])
    ebv_all = np.concatenate(ebv_chunks) if ebv_chunks else np.asarray([])
    zbin_counts = {}
    for lo, hi in zip(Z_BINS[:-1], Z_BINS[1:]):
        zbin_counts[f"[{lo:g},{hi:g})"] = int(((z_all >= lo) & (z_all < hi)).sum())

    return {
        "files": file_rows,
        "total_rows": int(total_rows),
        "row_groups": int(row_groups),
        "columns": schemas[str(files[0])]["columns"],
        "schema_first_file": schemas[str(files[0])]["schema"],
        "sample_spectrum": sample_spectrum_summary,
        "redshift": {
            "min": float(np.min(z_all)),
            "max": float(np.max(z_all)),
            "mean": float(np.mean(z_all)),
            "percentiles": pct(z_all, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
            "bins": zbin_counts,
            "negative_count": int((z_all < 0).sum()),
            "high_z_2_plus_count": int((z_all >= 2).sum()),
        },
        "quality": {
            "zwarn_counts": counter_to_dict(zwarn_counter),
            "zwarn_nonzero_count": int(sum(v for k, v in zwarn_counter.items() if bool(k))),
            "zerr_percentiles": pct(zerr_all, [50, 90, 95, 99]),
            "ebv_percentiles": pct(ebv_all, [1, 5, 50, 95, 99]),
        },
        "object_id_unique_count": int(len(object_ids)),
    }


def local_hdf5_object_ids_and_zwarn() -> tuple[set[int], dict[int, int]]:
    object_ids: set[int] = set()
    zwarn_by_id: dict[int, int] = {}
    for path in sorted(Path("data/raw").glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            n = int(f["Z"].shape[0])
            survey = np.asarray(f["SURVEY"])
            survey = np.asarray([v.decode("utf-8", errors="ignore").strip().lower() if isinstance(v, bytes) else str(v).lower() for v in survey])
            primary = np.asarray(f["SV_PRIMARY"]).astype(bool)
            indices = np.flatnonzero((survey == "sv3") & primary)
            ids = np.asarray(f["object_id"])[indices] if "object_id" in f else np.asarray(f["TARGETID"])[indices]
            zwarn = np.asarray(f["ZWARN"])[indices]
            for object_id, zw in zip(ids, zwarn):
                oid = int(object_id)
                object_ids.add(oid)
                zwarn_by_id[oid] = int(zw)
    return object_ids, zwarn_by_id


def main() -> None:
    hf = summarize_hf()
    local_ids, local_zwarn_by_id = local_hdf5_object_ids_and_zwarn()
    hf_ids = set()
    hf_zwarn_by_id: dict[int, bool] = {}
    for path in sorted(HF_DIR.glob("*.parquet")):
        table = pq.read_table(path, columns=["object_id", "ZWARN"])
        for oid, zw in zip(table["object_id"].to_pylist(), table["ZWARN"].to_pylist()):
            hf_ids.add(int(oid))
            hf_zwarn_by_id[int(oid)] = bool(zw)
    overlap_ids = local_ids & hf_ids
    overlap_zwarn_pairs = Counter()
    for oid in overlap_ids:
        overlap_zwarn_pairs[(local_zwarn_by_id[oid], hf_zwarn_by_id[oid])] += 1
    full = json.loads(FULL_DIAG.read_text(encoding="utf-8")) if FULL_DIAG.exists() else None
    old = json.loads(OLD_DIAG.read_text(encoding="utf-8")) if OLD_DIAG.exists() else None
    comparison = {
        "hf_parquet": hf,
        "local_hdf5_full": full,
        "local_hdf5_old_8k": old,
        "headline": {
            "hf_rows": hf["total_rows"],
            "local_full_rows": full["selected_total"] if full else None,
            "local_old_rows": old["selected_total"] if old else None,
            "hf_columns": hf["columns"],
            "local_training_kept_columns": ["flux", "flux_raw", "ivar", "mask", "wavelength", "z"],
            "object_id_overlap": len(overlap_ids),
            "hf_unique_object_ids": len(hf_ids),
            "local_unique_object_ids": len(local_ids),
            "overlap_zwarn_pairs_local_int_to_hf_bool": {
                f"{key[0]}->{key[1]}": int(value) for key, value in overlap_zwarn_pairs.most_common(20)
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison["headline"], indent=2))
    print(json.dumps(hf["sample_spectrum"], indent=2))
    print(json.dumps(hf["redshift"], indent=2))
    print(json.dumps(hf["quality"], indent=2))


if __name__ == "__main__":
    main()
