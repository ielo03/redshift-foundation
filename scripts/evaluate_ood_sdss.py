#!/usr/bin/env python3
"""Evaluate a crop/transform checkpoint on a prepared SDSS OOD bundle."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_experiments import DynamicSpectraTransformerWithRedshiftToken, decode_redshift, robust_normalize, robust_normalize_with_log_ivar
from redshift_metrics import redshift_metrics_numpy
from target_labels import TARGET_LABEL_NAMES


EMISSION_LINES = {
    "Lyα": 1215.67, "C IV": 1549.48, "C III]": 1908.73,
    "Mg II": 2798.75, "[O II]": 3727.09, "Hβ": 4861.33,
    "[O III]": 5006.84, "Hα": 6562.80,
}


def add_emission_line_overlays(ax, wavelength: np.ndarray, z_true: float, z_pred: float) -> None:
    """Mark the observed-frame locations implied by true and predicted redshift."""
    lower, upper = float(wavelength.min()), float(wavelength.max())
    for rest_wavelength in EMISSION_LINES.values():
        for redshift, color, linestyle in ((z_true, "tab:green", "--"), (z_pred, "tab:orange", ":")):
            observed = rest_wavelength * (1.0 + redshift)
            if lower <= observed <= upper:
                ax.axvline(observed, color=color, linestyle=linestyle, alpha=.55, lw=.8)
    ax.plot([], [], color="tab:green", linestyle="--", label=f"lines at true z={z_true:.4f}")
    ax.plot([], [], color="tab:orange", linestyle=":", label=f"lines at predicted z={z_pred:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--model", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--max-spectra", type=int, default=1000); p.add_argument("--examples", type=int, default=3)
    return p.parse_args()


def main() -> None:
    a = parse_args(); payload = torch.load(a.input, map_location="cpu", weights_only=True); checkpoint = torch.load(a.model, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    if cfg.get("architecture") not in (None, "DynamicSpectraTransformerWithRedshiftToken") and cfg.get("experiment") != "variable":
        raise RuntimeError("This evaluator supports variable/wavelength-aware checkpoints only.")
    n = min(a.max_spectra, len(payload["z"])); valid = payload["valid"][:n].bool(); targetid = payload.get("targetid")
    stored_wave = payload["wavelength"].float()
    wave = stored_wave.unsqueeze(0).expand(n, -1) if stored_wave.ndim == 1 else stored_wave[:n]
    if wave.shape != valid.shape:
        raise RuntimeError(f"Wavelength and validity shapes differ: {tuple(wave.shape)} versus {tuple(valid.shape)}")
    use_ivar_channel = bool(cfg.get("use_ivar_channel", False)); use_validity_channel = bool(cfg.get("use_validity_channel", False))
    if use_ivar_channel and "ivar" not in payload:
        raise RuntimeError("Checkpoint requires an IVAR channel, but the SDSS OOD bundle has no 'ivar' array")
    clean_rows=[]; confidence_rows=[]
    for index, (row, ok) in enumerate(zip(payload["flux"][:n], valid, strict=True)):
        if use_ivar_channel:
            normalized, confidence = robust_normalize_with_log_ivar(row.numpy(), payload["ivar"][index].numpy(), ok.numpy())
            clean_rows.append(torch.from_numpy(normalized)); confidence_rows.append(torch.from_numpy(confidence))
        else:
            clean_rows.append(torch.from_numpy(robust_normalize(row.numpy(), ok.numpy())))
    clean = torch.stack(clean_rows)
    channels = [clean]
    if use_ivar_channel:
        channels.append(torch.stack(confidence_rows))
    if use_validity_channel:
        channels.append(valid.float())
    input_channels = 1 + int(use_ivar_channel) + int(use_validity_channel)
    checkpoint_input_channels = int(cfg.get("input_channels", input_channels))
    if checkpoint_input_channels != input_channels:
        raise RuntimeError(f"Checkpoint expects {checkpoint_input_channels} input channels, evaluator built {input_channels}")
    model_input = torch.stack(channels, dim=1) if len(channels) > 1 else clean
    padding = ~valid
    num_classes = 2 if any(key.startswith("classification_head.") for key in checkpoint["model"]) else 0
    num_target_labels = len(TARGET_LABEL_NAMES) if any(key.startswith("target_selection_head.") for key in checkpoint["model"]) else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicSpectraTransformerWithRedshiftToken(patch_size=int(cfg["patch_size"]), d_model=int(cfg["d_model"]), nhead=int(cfg["nhead"]), num_layers=int(cfg["num_layers"]), input_channels=input_channels, num_classes=num_classes, num_target_labels=num_target_labels); model.load_state_dict(checkpoint["model"]); model.to(device); model.eval()
    preds=[]; recons=[]; class_logits_chunks=[]; target_logits_chunks=[]
    with torch.no_grad():
        for start in range(0, n, a.batch_size):
            stop=min(n,start+a.batch_size); recon,pred,class_logits,target_logits=model(model_input[start:stop].to(device),wave[start:stop].to(device),padding[start:stop].to(device)); recons.append(recon.cpu()); preds.append(pred.cpu())
            if class_logits is not None: class_logits_chunks.append(class_logits.cpu())
            if target_logits is not None: target_logits_chunks.append(target_logits.cpu())
    recon=torch.cat(recons); pred=decode_redshift(torch.cat(preds)).numpy(); actual=payload["z"][:n].numpy(); error=pred-actual
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5,5)); plt.scatter(actual,pred,s=5,alpha=.5); lim=[float(min(actual.min(),pred.min())),float(max(actual.max(),pred.max()))]; plt.plot(lim,lim,"k--"); plt.xlabel("Actual z"); plt.ylabel("Predicted z"); plt.title("SDSS OOD redshift"); plt.tight_layout(); plt.savefig(a.output_dir/"redshift_pred_vs_actual.png",dpi=160); plt.close()
    example_metadata = []
    for i in range(min(a.examples,n)):
        chosen = valid[i].numpy()
        x=wave[i].numpy(); fig, ax = plt.subplots(figsize=(11,3)); ax.plot(x[chosen],clean[i].numpy()[chosen],lw=.7,label="full-spectrum target"); ax.plot(x[chosen],recon[i].numpy()[chosen],lw=.7,label="reconstruction from full spectrum"); add_emission_line_overlays(ax, x[chosen], float(actual[i]), float(pred[i])); ax.legend(ncol=2, fontsize=8); ax.set_xlabel("Wavelength (Angstrom)"); fig.tight_layout(); fig.savefig(a.output_dir/f"reconstruction_{i:02d}.png",dpi=160); plt.close(fig)
        example_metadata.append({"example_index": i, "targetid": int(targetid[i]) if targetid is not None else i, "z_true": float(actual[i]), "z_pred": float(pred[i])})
    (a.output_dir / "reconstruction_examples.json").write_text(json.dumps(example_metadata, indent=2))
    grid = str(payload.get("grid", "desi"))
    input_protocol = "clean full spectra with normalized log-IVAR" if use_ivar_channel else "clean full spectra for every output head"
    metrics={"n":n,**redshift_metrics_numpy(pred, actual),"full_spectrum_reconstruction_mse":float(((recon[valid]-clean[valid])**2).mean()),"input_protocol":input_protocol,"wavelength_grid":grid,"checkpoint":str(a.model),"redshift_metrics_definition":{"dz_norm":"(z_pred-z_true)/(1+z_true)","sigma_nmad_scale":1.4826,"catastrophic_thresholds":[0.0033,0.05]},"redshift_sanity":{"invalid_excluded":0,"z_true_min":float(actual.min()),"z_true_max":float(actual.max()),"z_true_median":float(np.median(actual)),"z_pred_min":float(pred.min()),"z_pred_max":float(pred.max()),"z_pred_median":float(np.median(pred)),"input_wavelength_min":float(wave[valid].min()),"input_wavelength_max":float(wave[valid].max())}}
    if class_logits_chunks:
        class_pred = torch.cat(class_logits_chunks).argmax(dim=1)
        metrics["classification_predictions"] = {"classes": ["GALAXY", "QSO"], "predicted_counts": [int((class_pred == label).sum()) for label in range(2)]}
    if target_logits_chunks:
        target_pred = (torch.cat(target_logits_chunks).sigmoid() >= 0.5).numpy()
        metrics["target_selection_predictions"] = {"classes": list(TARGET_LABEL_NAMES), "predicted_counts": [int(target_pred[:, label].sum()) for label in range(len(TARGET_LABEL_NAMES))]}
    np.savez_compressed(a.output_dir / "redshift_predictions.npz", targetid=targetid[:n].numpy() if targetid is not None else np.arange(n), z_true=actual, z_pred=pred, dz=error, dz_norm=error / (1.0 + actual), abs_dz_norm=np.abs(error / (1.0 + actual)))
    (a.output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))


if __name__ == "__main__": main()
