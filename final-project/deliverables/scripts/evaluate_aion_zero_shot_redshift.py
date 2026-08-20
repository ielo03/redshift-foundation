from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot AION-base redshift prediction from spectrum tokens.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--aion-dir", type=Path, default=Path("models/aion-base"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--raw-flux-key", type=str, default="flux_raw")
    parser.add_argument("--spectrum-modality", choices=["desi", "sdss"], default="desi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from aion import AION
    from aion.codecs.scalar import GridScalarCodec
    from aion.codecs.spectrum import SpectrumCodec
    from aion.modalities import DESISpectrum, SDSSSpectrum, Z
    from redshift_diagnostics import plot_predicted_vs_actual_redshift, redshift_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    batch = payload.get("batch", payload)
    flux_key = args.raw_flux_key if args.raw_flux_key in batch else "flux"
    flux = batch[flux_key].float()
    ivar = batch.get("ivar", torch.ones_like(flux)).float()
    mask = batch.get("mask", torch.zeros_like(flux, dtype=torch.bool)).bool()
    wavelength = batch["wavelength"].float()
    if wavelength.ndim == 1:
        wavelength = wavelength.unsqueeze(0).expand(flux.shape[0], -1)
    z_true = batch["z"].float().reshape(-1)
    if args.max_items is not None:
        n = min(args.max_items, len(z_true))
        flux, ivar, mask, wavelength, z_true = flux[:n], ivar[:n], mask[:n], wavelength[:n], z_true[:n]

    spec_cfg = json.loads((args.aion_dir / "codecs" / "spectrum" / "config.json").read_text())
    spec_codec = SpectrumCodec(**spec_cfg).eval()
    spec_codec.load_state_dict(load_file(str(args.aion_dir / "codecs" / "spectrum" / "model.safetensors")))
    z_cfg = json.loads((args.aion_dir / "codecs" / "Z" / "config.json").read_text())
    z_codec = GridScalarCodec(**z_cfg).eval()
    z_codec.load_state_dict(load_file(str(args.aion_dir / "codecs" / "Z" / "model.safetensors")))
    model = AION.from_pretrained(str(args.aion_dir)).eval().to(device)
    spec_codec = spec_codec.to(device)
    z_codec = z_codec.to(device)

    pred_chunks = []
    spectrum_type = DESISpectrum if args.spectrum_modality == "desi" else SDSSSpectrum
    token_key = spectrum_type.token_key
    with torch.no_grad():
        for start in range(0, len(z_true), args.batch_size):
            end = min(start + args.batch_size, len(z_true))
            spec = spectrum_type(
                flux=flux[start:end].to(device),
                ivar=ivar[start:end].to(device),
                mask=mask[start:end].to(device),
                wavelength=wavelength[start:end].to(device),
            )
            tokens = {token_key: spec_codec.encode(spec).long()}
            logits = model(tokens, target_modality=Z)["tok_z"]
            pred_token = logits[..., : z_codec.quantizer.codebook_size].argmax(dim=-1)
            pred_chunks.append(z_codec.decode(pred_token).value.reshape(-1).detach().cpu())

    actual = z_true.numpy()
    predicted = torch.cat(pred_chunks, dim=0).numpy()
    metrics = redshift_metrics(actual, predicted)
    metrics["n"] = int(len(actual))
    metrics["source_summary"] = payload.get("summary", {})
    metrics["note"] = f"AION-base zero-shot tok_z prediction from {token_key}; no redshift-head fine-tuning."

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "redshift_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "redshift_predictions.json").write_text(
        json.dumps({"actual": actual.tolist(), "predicted": predicted.tolist(), "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    plot_predicted_vs_actual_redshift(
        actual,
        predicted,
        args.output_dir / "redshift_pred_vs_actual.png",
        title="AION-base zero-shot redshift",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
