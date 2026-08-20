#!/usr/bin/env python3
"""Build a labeled SDSS/eBOSS OOD bundle on the DESI or native wavelength grid."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plate-dir", type=Path, default=Path("/global/cfs/cdirs/sdss/eBOSS/testFiles"))
    p.add_argument("--redshift-root", type=Path, required=True, help="Tree containing matching spZall-PLATE-MJD.fits files.")
    p.add_argument("--reference-manifest", type=Path, default=Path("data/preprocessed/split_manifest_v1.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/ood/sdss_eboss_ood.pt"))
    p.add_argument("--max-plates", type=int, default=2)
    p.add_argument("--max-spectra", type=int, default=1000)
    p.add_argument(
        "--grid",
        choices=["desi", "native"],
        default="desi",
        help="Use DESI's shared grid or preserve each SDSS spectrum's native wavelength coverage.",
    )
    return p.parse_args()


def desi_wavelength(manifest: Path) -> np.ndarray:
    import json
    with manifest.open() as f:
        record = json.loads(next(line for line in f if line.strip()))
    with fits.open(record["coadd"], memmap=True) as hdus:
        return np.unique(np.concatenate([np.asarray(hdus[f"{band}_WAVELENGTH"].data) for band in ("B", "R", "Z")])).astype(np.float32)


def label_path(root: Path, plate: int, mjd: int) -> Path | None:
    """Find a matching redshift file without recursively walking the full SAS tree."""

    filename = f"spZall-{plate}-{mjd}.fits"
    # DR16 eBOSS places each result directly under PLATE/RUN2D/.  Searching
    # only this one plate directory avoids a multi-minute CFS tree walk per
    # requested plate.
    plate_dir = root / str(plate)
    direct = plate_dir / "v5_13_0" / filename
    if direct.is_file():
        return direct
    matches = sorted(plate_dir.glob(f"*/{filename}"))
    return matches[0] if matches else None


def interpolate_flux_and_ivar(
    output_wave: np.ndarray,
    input_wave: np.ndarray,
    input_flux: np.ndarray,
    input_ivar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample flux and propagate independent pixel variances."""

    insertion = np.searchsorted(input_wave, output_wave, side="left")
    right = np.clip(insertion, 1, len(input_wave) - 1)
    left = right - 1
    span = input_wave[right] - input_wave[left]
    upper_weight = np.divide(
        output_wave - input_wave[left],
        span,
        out=np.zeros_like(output_wave, dtype=np.float64),
        where=span > 0,
    )
    upper_weight = np.clip(upper_weight, 0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    inside = (output_wave >= input_wave[0]) & (output_wave <= input_wave[-1]) & (span > 0)
    output_flux = lower_weight * input_flux[left] + upper_weight * input_flux[right]
    output_variance = np.square(lower_weight) / input_ivar[left] + np.square(upper_weight) / input_ivar[right]
    output_ivar = np.divide(
        1.0,
        output_variance,
        out=np.zeros_like(output_variance, dtype=np.float64),
        where=inside & np.isfinite(output_variance) & (output_variance > 0),
    )
    output_flux = np.where(inside, output_flux, 0.0)
    valid = inside & np.isfinite(output_flux) & np.isfinite(output_ivar) & (output_ivar > 0)
    return output_flux.astype(np.float32), output_ivar.astype(np.float32), valid


def main() -> None:
    args = parse_args()
    ref_wave = desi_wavelength(args.reference_manifest) if args.grid == "desi" else None
    pattern = re.compile(r"spPlate-(\d+)-(\d+)\.fits$")
    fluxes: list[np.ndarray] = []; ivars: list[np.ndarray] = []; valids: list[np.ndarray] = []; wavelengths: list[np.ndarray] = []; redshifts: list[float] = []; sources: list[str] = []
    plate_files = sorted(args.plate_dir.glob("*/spPlate-*.fits"))[: args.max_plates]
    print(f"[start] plates={len(plate_files)} target_spectra={args.max_spectra}", flush=True)
    for plate_number, plate_file in enumerate(plate_files, start=1):
        match = pattern.search(plate_file.name)
        if not match:
            continue
        plate, mjd = map(int, match.groups())
        labels_file = label_path(args.redshift_root, plate, mjd)
        if labels_file is None:
            print(f"[skip] plate={plate}-{mjd} no matching spZall label file", flush=True)
            continue
        with fits.open(labels_file, memmap=True) as hdus:
            labels = hdus[1].data
            # spZall is ranked by fit; retain its first good fit for each fiber.
            label_by_fiber = {}
            for row in labels:
                fiber, z = int(row["FIBERID"]), float(row["Z"])
                if fiber not in label_by_fiber and int(row["ZWARNING"]) == 0 and np.isfinite(z) and z >= 0:
                    label_by_fiber[fiber] = z
        with fits.open(plate_file, memmap=True) as hdus:
            header, flux, ivar = hdus[0].header, np.asarray(hdus[0].data), np.asarray(hdus[1].data)
            wave = 10.0 ** (float(header["COEFF0"]) + float(header["COEFF1"]) * np.arange(flux.shape[1]))
            for fiber, z in label_by_fiber.items():
                index = fiber - 1
                if not 0 <= index < flux.shape[0]:
                    continue
                good = np.isfinite(flux[index]) & np.isfinite(ivar[index]) & (ivar[index] > 0)
                if good.sum() < 2:
                    continue
                if args.grid == "desi":
                    assert ref_wave is not None
                    output_wave = ref_wave
                    output_flux, output_ivar, valid = interpolate_flux_and_ivar(
                        ref_wave,
                        wave[good],
                        flux[index, good],
                        ivar[index, good],
                    )
                else:
                    output_wave = wave.astype(np.float32)
                    output_flux = np.asarray(flux[index], dtype=np.float32)
                    output_ivar = np.asarray(ivar[index], dtype=np.float32)
                    valid = np.isfinite(output_flux) & np.isfinite(ivar[index]) & (ivar[index] > 0)
                fluxes.append(np.nan_to_num(output_flux, nan=0.0)); ivars.append(np.where(valid, output_ivar, 0.0)); valids.append(valid); wavelengths.append(output_wave); redshifts.append(z); sources.append(f"{plate}-{mjd}-{fiber}")
                if len(fluxes) >= args.max_spectra:
                    break
        if len(fluxes) >= args.max_spectra:
            break
        print(
            f"[plate {plate_number}/{len(plate_files)}] plate={plate}-{mjd} "
            f"labeled_fibers={len(label_by_fiber)} accumulated={len(fluxes)}/{args.max_spectra}",
            flush=True,
        )
    if not fluxes:
        raise RuntimeError("No labeled SDSS spectra found. Check that the redshift root matches the selected local plates.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.grid == "desi":
        assert ref_wave is not None
        flux_array = np.stack(fluxes)
        ivar_array = np.stack(ivars)
        valid_array = np.stack(valids)
        wavelength_array = ref_wave
    else:
        max_length = max(len(wave) for wave in wavelengths)
        flux_array = np.zeros((len(fluxes), max_length), dtype=np.float32)
        ivar_array = np.zeros((len(fluxes), max_length), dtype=np.float32)
        valid_array = np.zeros((len(fluxes), max_length), dtype=np.uint8)
        wavelength_array = np.zeros((len(fluxes), max_length), dtype=np.float32)
        for index, (flux, ivar, valid, wave) in enumerate(zip(fluxes, ivars, valids, wavelengths, strict=True)):
            length = len(wave)
            flux_array[index, :length] = flux
            ivar_array[index, :length] = ivar
            valid_array[index, :length] = valid
            wavelength_array[index, :length] = wave
    torch.save({"flux": torch.from_numpy(flux_array), "ivar": torch.from_numpy(ivar_array), "valid": torch.from_numpy(valid_array), "wavelength": torch.from_numpy(wavelength_array), "z": torch.tensor(redshifts), "source": sources, "grid": args.grid}, output_tmp)
    output_tmp.replace(args.output)
    print(f"[done] wrote {args.output} with {len(fluxes)} labeled SDSS spectra", flush=True)


if __name__ == "__main__":
    main()
