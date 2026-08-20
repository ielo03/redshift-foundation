from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize DESI spectra with the AION codec and cache the result.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("hw/final_project/data/raw/001-of-002.hdf5"), Path("hw/final_project/data/raw/002-of-002.hdf5")],
        help="Path(s) to raw DESI/MMU HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hw/final_project/data/processed/aion_spectrum_tokens.pt"),
        help="Output .pt bundle containing AION spectrum tokens.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("hw/final_project/outputs/tables/aion_spectrum_tokens_summary.json"),
        help="Summary JSON path.",
    )
    parser.add_argument("--survey", type=str, default="sv3", help="Survey filter to apply.")
    parser.add_argument(
        "--no-primary-only",
        action="store_true",
        help="Do not filter to SV_PRIMARY=True.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of spectra to tokenize per codec batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    import torch

    from data_pipeline import collate_records, load_hdf5, summarize_records
    from project_paths import AION_REPO_DIR, ensure_project_dirs

    ensure_project_dirs()

    # Install the local AION package path at runtime.
    sys.path.insert(0, str(AION_REPO_DIR))
    from aion.codecs.manager import CodecManager
    from aion.modalities import DESISpectrum

    records = []
    for input_path in args.input:
        records.extend(
            load_hdf5(
                input_path,
                survey=args.survey,
                primary_only=not args.no_primary_only,
            )
        )

    if not records:
        raise RuntimeError("No records loaded from the raw HDF5 files")

    summary = summarize_records(records)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    codec_manager = CodecManager(device=device)
    print(f"Using codec device: {device}")

    spectrum_tokens = []
    redshift_values = []
    flux_raw = []
    wavelength = []
    mask = []
    ivar = []

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch = collate_records(
            batch_records,
            target_length=batch_records[0].flux.numel(),
            normalize_flux_values=False,
        )
        batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
        spectrum = DESISpectrum(
            flux=batch["flux"],
            ivar=batch["ivar"],
            mask=batch["mask"],
            wavelength=batch["wavelength"],
        )

        tokens = codec_manager.encode(spectrum)
        spectrum_tokens.append(tokens["tok_spectrum_desi"].cpu())
        redshift_values.append(batch["z"].cpu())
        flux_raw.append(batch["flux_raw"].cpu())
        wavelength.append(batch["wavelength"].cpu())
        mask.append(batch["mask"].cpu())
        ivar.append(batch["ivar"].cpu())

    spectrum_tokens_t = torch.cat(spectrum_tokens, dim=0)
    redshift_values_t = torch.cat(redshift_values, dim=0)
    flux_raw_t = torch.cat(flux_raw, dim=0)
    wavelength_t = torch.cat(wavelength, dim=0)
    mask_t = torch.cat(mask, dim=0)
    ivar_t = torch.cat(ivar, dim=0)

    payload = {
        "source": [str(path) for path in args.input],
        "summary": summary,
        "input_ids": spectrum_tokens_t.long(),
        "spectrum_tokens": spectrum_tokens_t.long(),
        "redshift_values": redshift_values_t.float(),
        "flux_raw": flux_raw_t.float(),
        "wavelength": wavelength_t.float(),
        "mask": mask_t.bool(),
        "ivar": ivar_t.float(),
        "seq_len": int(spectrum_tokens_t.shape[1]),
        "spectrum_len": int(spectrum_tokens_t.shape[1]),
        "mask_token_id": 1024,
        "input_vocab_size": 1025,
        "output_vocab_size": 1024,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "seq_len": payload["seq_len"],
                "spectrum_len": payload["spectrum_len"],
                "mask_token_id": payload["mask_token_id"],
            },
            f,
            indent=2,
        )

    print("Loaded records:", summary["count"])
    print("Saved tokenized bundle to:", args.output)
    print("Saved summary to:", args.summary)


if __name__ == "__main__":
    main()
