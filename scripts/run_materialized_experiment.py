#!/usr/bin/env python3
"""Launch one isolated materialized-training experiment arm from JSON config."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data, model, opt = config["data"], config["model"], config["optimization"]
    recon, aug, early = config["reconstruction"], config["augmentation"], config["early_stopping"]
    classification = config.get("classification", {"weight": 0.25})
    target_selection = config.get("target_selection", {"weight": 0.25})
    inputs = config.get("inputs", {})
    runtime = config.get("runtime", {})
    command = [
        sys.executable, "-u", config["trainer"], "--input", data["input"],
        "--checkpoint-dir", str(args.checkpoint_dir), "--model-dir", str(args.model_dir),
        "--epochs", str(opt["epochs"]), "--max-train-spectra-per-epoch", str(data["max_train_spectra_per_epoch"]),
        "--max-val-spectra", str(data["max_val_spectra"]), "--batch-size", str(opt["batch_size"]),
        "--lr", str(opt["lr"]), "--d-model", str(model["d_model"]), "--num-layers", str(model["num_layers"]),
        "--nhead", str(model["nhead"]), "--patch-size", str(model["patch_size"]), "--alpha", str(opt["alpha"]),
        "--reconstruction-weight", str(opt["reconstruction_weight"]), "--seed", str(opt["seed"]),
        "--classification-weight", str(classification["weight"]),
        "--target-label-weight", str(target_selection["weight"]),
        "--mask-prob", str(recon["mask_prob"]), "--mask-mode", recon["mask_mode"],
        "--variable-min-fraction", str(aug["variable_min_fraction"]), "--transform-gain", str(aug["transform_gain"]),
        "--transform-tilt", str(aug["transform_tilt"]), "--transform-offset", str(aug["transform_offset"]),
        "--transform-noise", str(aug["transform_noise"]), "--early-stopping-patience", str(early["patience"]),
        "--early-stopping-min-delta", str(early["min_delta"]),
        "--selection-metric", early.get("selection_metric", "z_sigma_nmad"),
        "--random-transforms" if aug["random_transforms"] else "--no-random-transforms",
        "--augmentation-policy", aug.get("policy", "combined"),
        "--exclusive-clean-prob", str(aug.get("clean_probability", 0.25)),
        "--exclusive-crop-prob", str(aug.get("crop_probability", 0.375)),
        "--exclusive-transform-prob", str(aug.get("transform_probability", 0.375)),
    ]
    if "prefetch_batches" in runtime:
        command += ["--prefetch-batches", str(runtime["prefetch_batches"])]
    if inputs.get("use_ivar_channel", False):
        command += ["--use-ivar-channel"]
    if inputs.get("use_validity_channel", False):
        command += ["--use-validity-channel"]
    if args.resume:
        command += ["--resume", str(args.resume)]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))
    print("[experiment]", config["experiment_id"], config["arm"], flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
