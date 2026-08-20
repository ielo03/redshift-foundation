#!/usr/bin/env python3
"""Evaluate a wavelength-aware checkpoint on the reserved DESI benchmark split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_experiments import (  # noqa: E402
    DynamicSpectraTransformerWithRedshiftToken,
    decode_redshift,
    robust_normalize,
    robust_normalize_with_log_ivar,
    stitch_bands,
)
from redshift_metrics import redshift_metrics_numpy  # noqa: E402
from target_labels import TARGET_LABEL_NAMES, target_selection_labels  # noqa: E402


# Rest-frame vacuum wavelengths in Angstrom.  Each line is plotted twice: at
# z_true and at z_pred, so a redshift error is visible directly on the spectrum.
EMISSION_LINES = {
    "Lyα": 1215.67,
    "C IV": 1549.48,
    "C III]": 1908.73,
    "Mg II": 2798.75,
    "[O II]": 3727.09,
    "Hβ": 4861.33,
    "[O III]": 5006.84,
    "Hα": 6562.80,
}


def add_emission_line_overlays(ax, wavelength: np.ndarray, z_true: float, z_pred: float) -> None:
    """Mark expected and model-predicted observed-frame line positions."""
    lower, upper = float(wavelength.min()), float(wavelength.max())
    for name, rest_wavelength in EMISSION_LINES.items():
        for redshift, color, linestyle in (
            (z_true, "tab:green", "--"),
            (z_pred, "tab:orange", ":"),
        ):
            observed = rest_wavelength * (1.0 + redshift)
            if lower <= observed <= upper:
                ax.axvline(observed, color=color, linestyle=linestyle, alpha=.55, lw=.8)
    ax.plot([], [], color="tab:green", linestyle="--", label=f"lines at true z={z_true:.4f}")
    ax.plot([], [], color="tab:orange", linestyle=":", label=f"lines at predicted z={z_pred:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/preprocessed/split_manifest_v1.jsonl"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-spectra", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--examples", type=int, default=3)
    return p.parse_args()


def load_benchmark(
    manifest: Path,
    max_spectra: int,
    *,
    use_ivar_channel: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fluxes: list[np.ndarray] = []; log_ivars: list[np.ndarray] = []; valids: list[np.ndarray] = []; labels: list[float] = []; spectypes: list[int] = []; targetids: list[int] = []; target_labels: list[np.ndarray] = []; wavelength = None
    with manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") != "benchmark":
                continue
            with fits.open(record["coadd"], memmap=True) as coadd:
                waves = [coadd[f"{band}_WAVELENGTH"].data for band in ("B", "R", "Z")]
                fibermap = coadd["FIBERMAP"].data
                record_target_labels = target_selection_labels(fibermap[np.asarray(record["rows"], dtype=np.int64)], str(record["survey"]))
                for row, z, spectype, targetid, labels_for_row in zip(record["rows"], record["z"], record["spectype"], record["targetid"], record_target_labels, strict=True):
                    stitched = stitch_bands(
                        waves,
                        [coadd[f"{band}_FLUX"].data[int(row)] for band in ("B", "R", "Z")],
                        [coadd[f"{band}_IVAR"].data[int(row)] for band in ("B", "R", "Z")],
                        [coadd[f"{band}_MASK"].data[int(row)] != 0 for band in ("B", "R", "Z")],
                    )
                    valid = (~stitched["mask"]) & (stitched["ivar"] > 0)
                    if use_ivar_channel:
                        normalized_flux, log_ivar = robust_normalize_with_log_ivar(stitched["flux"], stitched["ivar"], valid)
                        log_ivars.append(log_ivar)
                    else:
                        normalized_flux = robust_normalize(stitched["flux"], valid)
                    fluxes.append(normalized_flux); valids.append(valid); labels.append(float(z)); spectypes.append(0 if spectype == "GALAXY" else 1); targetids.append(int(targetid)); target_labels.append(labels_for_row)
                    if wavelength is None:
                        wavelength = stitched["wavelength"].astype(np.float32)
                    if len(fluxes) >= max_spectra:
                        confidence = torch.from_numpy(np.stack(log_ivars)) if use_ivar_channel else None
                        return torch.from_numpy(np.stack(fluxes)), confidence, torch.from_numpy(np.stack(valids)), torch.from_numpy(wavelength), torch.tensor(labels), torch.tensor(spectypes), torch.tensor(targetids), torch.from_numpy(np.stack(target_labels))
    if not fluxes:
        raise RuntimeError("No reserved benchmark spectra found in manifest")
    confidence = torch.from_numpy(np.stack(log_ivars)) if use_ivar_channel else None
    return torch.from_numpy(np.stack(fluxes)), confidence, torch.from_numpy(np.stack(valids)), torch.from_numpy(wavelength), torch.tensor(labels), torch.tensor(spectypes), torch.tensor(targetids), torch.from_numpy(np.stack(target_labels))


def main() -> None:
    a = parse_args(); checkpoint = torch.load(a.model, map_location="cpu", weights_only=False); cfg = checkpoint["args"]
    use_ivar_channel = bool(cfg.get("use_ivar_channel", False)); use_validity_channel = bool(cfg.get("use_validity_channel", False))
    flux, log_ivar, valid, wave_1d, actual_t, spectype_t, targetid_t, target_label_t = load_benchmark(a.manifest, a.max_spectra, use_ivar_channel=use_ivar_channel); n = len(actual_t); wave = wave_1d.unsqueeze(0).expand(n, -1); padding = torch.zeros_like(valid)
    input_channels = 1 + int(use_ivar_channel) + int(use_validity_channel)
    checkpoint_input_channels = int(cfg.get("input_channels", input_channels))
    if checkpoint_input_channels != input_channels:
        raise RuntimeError(f"Checkpoint expects {checkpoint_input_channels} input channels, evaluator built {input_channels}")
    channels = [flux]
    if use_ivar_channel:
        if log_ivar is None:
            raise RuntimeError("Checkpoint requires an IVAR channel, but DESI benchmark confidence was not loaded")
        channels.append(log_ivar)
    if use_validity_channel:
        channels.append(valid.float())
    model_input = torch.stack(channels, dim=1) if len(channels) > 1 else flux
    num_classes = 2 if any(key.startswith("classification_head.") for key in checkpoint["model"]) else 0
    num_target_labels = len(TARGET_LABEL_NAMES) if any(key.startswith("target_selection_head.") for key in checkpoint["model"]) else 0
    model = DynamicSpectraTransformerWithRedshiftToken(patch_size=int(cfg["patch_size"]), d_model=int(cfg["d_model"]), nhead=int(cfg["nhead"]), num_layers=int(cfg["num_layers"]), input_channels=input_channels, num_classes=num_classes, num_target_labels=num_target_labels); model.load_state_dict(checkpoint["model"]); model.eval()
    pred_chunks=[]; recon_chunks=[]; class_chunks=[]; target_chunks=[]
    with torch.no_grad():
        for start in range(0, n, a.batch_size):
            stop=min(n,start+a.batch_size); recon,pred,class_logits,target_logits=model(model_input[start:stop],wave[start:stop],padding[start:stop]); recon_chunks.append(recon); pred_chunks.append(pred)
            if class_logits is not None: class_chunks.append(class_logits)
            if target_logits is not None: target_chunks.append(target_logits)
    recon=torch.cat(recon_chunks); predicted=decode_redshift(torch.cat(pred_chunks)).numpy(); actual=actual_t.numpy(); error=predicted-actual
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5,5)); plt.scatter(actual,predicted,s=3,alpha=.35); lim=[float(min(actual.min(),predicted.min())),float(max(actual.max(),predicted.max()))]; plt.plot(lim,lim,"k--"); plt.xlabel("Actual z"); plt.ylabel("Predicted z"); plt.title("Reserved DESI benchmark redshift"); plt.tight_layout(); plt.savefig(a.output_dir/"redshift_pred_vs_actual.png",dpi=160); plt.close()
    example_metadata = []
    for i in range(min(a.examples,n)):
        x=wave_1d.numpy(); fig, ax = plt.subplots(figsize=(11,3)); ax.plot(x,flux[i],lw=.7,label="full-spectrum target"); ax.plot(x,recon[i],lw=.7,label="reconstruction from full spectrum"); add_emission_line_overlays(ax, x, float(actual[i]), float(predicted[i])); ax.legend(ncol=2, fontsize=8); ax.set_xlabel("Wavelength (Angstrom)"); fig.tight_layout(); fig.savefig(a.output_dir/f"reconstruction_{i:02d}.png",dpi=160); plt.close(fig)
        example_metadata.append({"example_index": i, "targetid": int(targetid_t[i]), "z_true": float(actual[i]), "z_pred": float(predicted[i])})
    (a.output_dir / "reconstruction_examples.json").write_text(json.dumps(example_metadata, indent=2))
    input_protocol = "clean full spectra with normalized log-IVAR" if use_ivar_channel else "clean full spectra for every output head"
    metrics={"n":n,**redshift_metrics_numpy(predicted, actual),"full_spectrum_reconstruction_mse":float(((recon[valid]-flux[valid])**2).mean()),"input_protocol":input_protocol,"checkpoint":str(a.model),"split":"reserved benchmark"}
    metrics["redshift_metrics_definition"] = {"dz_norm": "(z_pred-z_true)/(1+z_true)", "sigma_nmad_scale": 1.4826, "catastrophic_thresholds": [0.0033, 0.05]}
    metrics["redshift_sanity"] = {"invalid_excluded": 0, "z_true_min": float(actual.min()), "z_true_max": float(actual.max()), "z_true_median": float(np.median(actual)), "z_pred_min": float(predicted.min()), "z_pred_max": float(predicted.max()), "z_pred_median": float(np.median(predicted))}
    per_target = {}
    labels_np = target_label_t.numpy().astype(bool)
    for index, name in enumerate(TARGET_LABEL_NAMES):
        selected = labels_np[:, index]
        if selected.any():
            per_target[name] = {"n": int(selected.sum()), **redshift_metrics_numpy(predicted[selected], actual[selected])}
    metrics["redshift_by_target_selection"] = per_target
    if class_chunks:
        class_pred=torch.cat(class_chunks).argmax(dim=1); class_acc=float((class_pred == spectype_t).float().mean()); confusion=[[int(((spectype_t == truth) & (class_pred == prediction)).sum()) for prediction in range(2)] for truth in range(2)]
        metrics["classification"]={"classes":["GALAXY","QSO"],"accuracy":class_acc,"confusion_matrix_rows_actual_columns_predicted":confusion}
    if target_chunks:
        target_pred = (torch.cat(target_chunks).sigmoid() >= 0.5).numpy()
        target_true = target_label_t.numpy().astype(bool)
        per_label = {}
        for index, name in enumerate(TARGET_LABEL_NAMES):
            truth, prediction = target_true[:, index], target_pred[:, index]
            tp = int((truth & prediction).sum()); fp = int((~truth & prediction).sum()); fn = int((truth & ~prediction).sum())
            per_label[name] = {"support": int(truth.sum()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
        metrics["target_selection_classification"] = {"classes": list(TARGET_LABEL_NAMES), "per_label": per_label}
    np.savez_compressed(a.output_dir / "redshift_predictions.npz", targetid=targetid_t.numpy(), z_true=actual, z_pred=predicted, dz=error, dz_norm=error / (1.0 + actual), abs_dz_norm=np.abs(error / (1.0 + actual)), object_type_true=spectype_t.numpy(), object_type_classes=np.asarray(["GALAXY", "QSO"]), target_labels=target_label_t.numpy(), target_label_names=np.asarray(TARGET_LABEL_NAMES))
    (a.output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))


if __name__ == "__main__": main()
