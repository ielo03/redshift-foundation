from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def add_local_aion_to_path(project_root: Path) -> Path:
    aion_repo = project_root / "AION"
    if not aion_repo.exists():
        raise FileNotFoundError(f"AION repo not found at {aion_repo}")
    sys.path.insert(0, str(aion_repo))
    return aion_repo


def load_modalities_module(project_root: Path):
    modalities_path = project_root / "AION" / "aion" / "modalities.py"
    spec = importlib.util.spec_from_file_location("aion_modalities_local", modalities_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load modalities module from {modalities_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_synthetic_spectrum(batch_size: int = 2, length: int = 7081) -> tuple:
    wavelength = torch.linspace(3600.0, 9800.0, length).repeat(batch_size, 1)
    flux = torch.ones(batch_size, length, dtype=torch.float32)
    ivar = torch.ones(batch_size, length, dtype=torch.float32)
    mask = torch.zeros(batch_size, length, dtype=torch.bool)
    z = torch.tensor([[0.12], [0.34]], dtype=torch.float32)[:batch_size]
    return flux, ivar, mask, wavelength, z


def dry_run(project_root: Path) -> None:
    modalities = load_modalities_module(project_root)
    DESISpectrum = modalities.DESISpectrum
    Z = modalities.Z

    flux, ivar, mask, wavelength, z = build_synthetic_spectrum()
    spectrum = DESISpectrum(
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
    )
    redshift = Z(value=z)

    print("Local AION import: OK")
    print("DESISpectrum token key:", spectrum.token_key)
    print("DESISpectrum num tokens:", spectrum.num_tokens)
    print("Z token key:", redshift.token_key)
    print("Z num tokens:", redshift.num_tokens)
    print("Synthetic spectrum shape:", tuple(spectrum.flux.shape))
    print("Synthetic redshift shape:", tuple(redshift.value.shape))


def codec_run(project_root: Path) -> None:
    add_local_aion_to_path(project_root)

    from aion.codecs.manager import CodecManager
    from aion.modalities import DESISpectrum, Z

    flux, ivar, mask, wavelength, z = build_synthetic_spectrum()
    spectrum = DESISpectrum(
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wavelength,
    )
    redshift = Z(value=z)

    manager = CodecManager(device="cpu")
    tokens = manager.encode(spectrum, redshift)

    print("Codec manager encode: OK")
    for key, value in tokens.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for local AION import and spectra-only modality setup."
    )
    parser.add_argument(
        "--run-codecs",
        action="store_true",
        help="Attempt CodecManager.encode on synthetic DESI spectrum + redshift. "
        "This may trigger Hugging Face downloads for codec weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    try:
        dry_run(project_root)
    except ModuleNotFoundError as exc:
        print("Dry-run import failed because an AION dependency is missing.")
        print("Missing module:", exc.name)
        print("Install AION dependencies first, then re-run this script.")
        return

    if args.run_codecs:
        try:
            codec_run(project_root)
        except ModuleNotFoundError as exc:
            print("Codec run failed because an AION dependency is missing.")
            print("Missing module:", exc.name)
            print("Install AION dependencies first, then re-run with --run-codecs.")
    else:
        print("Codec test skipped. Re-run with --run-codecs to test tokenizer loading.")


if __name__ == "__main__":
    main()
