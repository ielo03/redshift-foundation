from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained redshift-token model on an external OOD bundle.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from data_pipeline import make_token_mask
    from neural_baseline import SpectraTransformerWithRedshiftToken
    from redshift_diagnostics import (
        decode_redshift_target,
        masked_normalized_reconstruction_metrics,
        plot_flux_reconstruction,
        plot_predicted_vs_actual_redshift,
        redshift_metrics,
        spectrum_reconstruction_metrics,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    batch = payload.get("batch", payload)
    X = batch["flux"].float()
    y = batch["z"].float().reshape(-1).numpy()

    # External spectra are never used for training. This standardizes the converted
    # OOD bundle for model compatibility using only unlabeled flux statistics.
    x_mean = X.mean(dim=0, keepdim=True)
    x_std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    X_norm = (X - x_mean) / x_std
    X_input = X_norm.unsqueeze(1)

    mask = make_token_mask(tuple(X_norm.shape), mask_prob=args.mask_prob, seed=args.mask_seed)
    model = SpectraTransformerWithRedshiftToken(
        input_length=X_norm.shape[1],
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        input_channels=1,
    ).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pred_chunks = []
    recon_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_input), args.batch_size):
            xb = X_input[start : start + args.batch_size].to(device)
            mb = mask[start : start + args.batch_size].to(device).float()
            recon, pred_encoded = model(xb * (1.0 - mb.unsqueeze(1)))
            recon_chunks.append(recon.cpu())
            pred_chunks.append(pred_encoded.cpu())

    recon = torch.cat(recon_chunks, dim=0).numpy()
    pred = decode_redshift_target(torch.cat(pred_chunks, dim=0).numpy())
    mask_np = mask.numpy().astype(bool)

    metrics = redshift_metrics(y, pred)
    metrics["reconstruction"] = masked_normalized_reconstruction_metrics(X_norm.numpy(), recon, mask_np)
    metrics["reconstruction_raw"] = spectrum_reconstruction_metrics(X_norm.numpy(), recon)
    metrics["n_ood"] = int(len(y))
    metrics["normalization"] = "per-wavelength standardization computed on OOD flux only"
    metrics["source_summary"] = payload.get("summary", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "redshift_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (args.output_dir / "redshift_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"actual": y.tolist(), "predicted": pred.tolist(), "metrics": metrics}, f, indent=2)
    plot_predicted_vs_actual_redshift(y, pred, args.output_dir / "redshift_pred_vs_actual.png", title="SDSS OOD redshift")
    plot_flux_reconstruction(
        X_norm.numpy()[0],
        recon[0],
        args.output_dir / "spectrum_reconstruction.png",
        title="SDSS OOD reconstruction example",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
