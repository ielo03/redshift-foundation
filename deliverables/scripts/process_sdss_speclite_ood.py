from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SDSS speclite FITS files into a DESI-grid OOD bundle.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def _read_header(f) -> dict[str, str] | None:
    header = b""
    while True:
        block = f.read(2880)
        if not block:
            return None
        header += block
        if b"END" in block:
            break
    text = header.decode("ascii", errors="ignore")
    cards = [text[i : i + 80] for i in range(0, len(text), 80)]
    values: dict[str, str] = {}
    for card in cards:
        key = card[:8].strip()
        if not key:
            continue
        if len(card) > 10 and card[8:10] == "= ":
            values[key] = card[10:80].split("/", 1)[0].strip()
        else:
            values[key] = card[8:80].strip()
    return values


def _payload_size(header: dict[str, str]) -> int:
    naxis = int(header.get("NAXIS", "0").split()[0])
    if header.get("XTENSION", "").startswith("'BINTABLE'"):
        return int(header["NAXIS1"].split()[0]) * int(header["NAXIS2"].split()[0])
    if naxis == 0:
        return 0
    size = abs(int(header["BITPIX"].split()[0])) // 8
    for axis in range(1, naxis + 1):
        size *= int(header[f"NAXIS{axis}"].split()[0])
    return size


def _parse_tform(tform: str) -> tuple[int, str]:
    token = tform.strip().strip("'").strip()
    digits = ""
    for char in token:
        if char.isdigit():
            digits += char
        else:
            code = char
            break
    else:
        raise ValueError(f"Unsupported TFORM value: {tform}")
    return int(digits or "1"), code


def _column_layout(header: dict[str, str]) -> dict[str, tuple[int, int, str]]:
    widths = {"A": 1, "B": 1, "I": 2, "J": 4, "K": 8, "E": 4, "D": 8}
    offset = 0
    layout: dict[str, tuple[int, int, str]] = {}
    for idx in range(1, int(header["TFIELDS"].split()[0]) + 1):
        name = header[f"TTYPE{idx}"].strip().strip("'").strip()
        repeat, code = _parse_tform(header[f"TFORM{idx}"])
        layout[name] = (offset, repeat, code)
        offset += repeat * widths[code]
    return layout


def _read_column(row: bytes, layout: dict[str, tuple[int, int, str]], name: str):
    offset, repeat, code = layout[name]
    if code == "A":
        return row[offset : offset + repeat].decode("ascii", errors="ignore").strip()
    dtype = {
        "B": ">u1",
        "I": ">i2",
        "J": ">i4",
        "K": ">i8",
        "E": ">f4",
        "D": ">f8",
    }[code]
    values = np.frombuffer(row, dtype=np.dtype(dtype), count=repeat, offset=offset)
    if repeat == 1:
        return values[0].item()
    return values.astype(np.float64 if code == "D" else np.float32)


def read_sdss_speclite(path: Path) -> dict[str, object]:
    coadd = None
    spall = None
    with path.open("rb") as f:
        while True:
            header = _read_header(f)
            if header is None:
                break
            size = _payload_size(header)
            data = f.read(size)
            pad = (2880 - size % 2880) % 2880
            if pad:
                f.seek(pad, 1)
            if not header.get("XTENSION", "").startswith("'BINTABLE'"):
                continue
            extname = header.get("EXTNAME", "").strip().strip("'").strip()
            row_len = int(header["NAXIS1"].split()[0])
            n_rows = int(header["NAXIS2"].split()[0])
            layout = _column_layout(header)
            if extname == "COADD":
                flux = np.empty(n_rows, dtype=np.float32)
                loglam = np.empty(n_rows, dtype=np.float32)
                ivar = np.empty(n_rows, dtype=np.float32)
                for row_idx in range(n_rows):
                    row = data[row_idx * row_len : (row_idx + 1) * row_len]
                    flux[row_idx] = _read_column(row, layout, "flux")
                    loglam[row_idx] = _read_column(row, layout, "loglam")
                    ivar[row_idx] = _read_column(row, layout, "ivar")
                coadd = {"flux": flux, "wavelength": np.power(10.0, loglam), "ivar": ivar}
            elif extname == "SPALL":
                row = data[:row_len]
                spall = {
                    "z": float(_read_column(row, layout, "Z")),
                    "class": str(_read_column(row, layout, "CLASS")),
                    "zwarning": int(_read_column(row, layout, "ZWARNING")),
                    "fiberid": int(_read_column(row, layout, "FIBERID")),
                }
    if coadd is None or spall is None:
        raise RuntimeError(f"{path} did not contain COADD and SPALL tables")
    return {**coadd, **spall, "path": str(path)}


def normalize_flux(flux: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.any():
        mean = float(np.mean(flux[valid]))
        std = float(np.std(flux[valid]))
        if math.isfinite(std) and std >= 1e-6:
            return ((flux - mean) / std).astype(np.float32)
    return np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    reference = torch.load(args.reference_bundle, map_location="cpu", weights_only=True)
    ref_batch = reference.get("batch", reference)
    ref_wavelength = ref_batch["wavelength"]
    if ref_wavelength.ndim == 2:
        ref_wavelength = ref_wavelength[0]
    ref_wavelength_np = ref_wavelength.numpy().astype(np.float32)

    files = sorted(args.input_dir.glob("*.fits"))
    if args.max_files is not None:
        files = files[: args.max_files]
    flux_rows = []
    raw_rows = []
    ivar_rows = []
    mask_rows = []
    z_rows = []
    metadata = []

    for path in files:
        item = read_sdss_speclite(path)
        z = float(item["z"])
        if not math.isfinite(z) or z < 0:
            continue
        sdss_wave = np.asarray(item["wavelength"], dtype=np.float32)
        sdss_flux = np.asarray(item["flux"], dtype=np.float32)
        sdss_ivar = np.asarray(item["ivar"], dtype=np.float32)
        finite = np.isfinite(sdss_wave) & np.isfinite(sdss_flux) & np.isfinite(sdss_ivar) & (sdss_ivar > 0)
        if finite.sum() < 100:
            continue
        interp_flux = np.interp(ref_wavelength_np, sdss_wave[finite], sdss_flux[finite], left=np.nan, right=np.nan)
        interp_ivar = np.interp(ref_wavelength_np, sdss_wave[finite], sdss_ivar[finite], left=0.0, right=0.0)
        valid = np.isfinite(interp_flux) & (interp_ivar > 0)
        raw_flux = np.nan_to_num(interp_flux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        flux_rows.append(torch.from_numpy(normalize_flux(raw_flux, valid)))
        raw_rows.append(torch.from_numpy(raw_flux))
        ivar_rows.append(torch.from_numpy(interp_ivar.astype(np.float32)))
        mask_rows.append(torch.from_numpy(~valid))
        z_rows.append(torch.tensor([z], dtype=torch.float32))
        metadata.append(
            {
                "path": item["path"],
                "fiberid": item["fiberid"],
                "class": item["class"],
                "zwarning": item["zwarning"],
            }
        )

    if not flux_rows:
        raise RuntimeError("No usable SDSS spectra were converted")

    z_tensor = torch.stack(z_rows, dim=0)
    classes = sorted({str(item["class"]) for item in metadata})
    summary = {
        "count": len(flux_rows),
        "source": "SDSS DR17 speclite plate 4444 MJD 55538",
        "reference_bundle": str(args.reference_bundle),
        "length": int(ref_wavelength_np.shape[0]),
        "z_min": float(z_tensor.min().item()),
        "z_max": float(z_tensor.max().item()),
        "z_mean": float(z_tensor.mean().item()),
        "classes": classes,
        "zwarning_nonzero": int(sum(1 for item in metadata if int(item["zwarning"]) != 0)),
    }
    payload = {
        "source": [str(path) for path in files],
        "summary": summary,
        "metadata": metadata,
        "batch": {
            "flux": torch.stack(flux_rows, dim=0),
            "flux_raw": torch.stack(raw_rows, dim=0),
            "ivar": torch.stack(ivar_rows, dim=0),
            "mask": torch.stack(mask_rows, dim=0),
            "wavelength": ref_wavelength,
            "z": z_tensor,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
