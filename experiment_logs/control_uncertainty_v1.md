# Control + Uncertainty v1

## Status

Queued as Slurm job `57318921` with a two-hour allocation. It has an `afterok` dependency on SDSS-IVAR preparation job `57318911`, so training starts only if the uncertainty benchmark bundle is built successfully.

## Purpose

Test whether per-pixel telescope uncertainty improves the plain full-spectrum Control. This is an input-only ablation against matched Control v2.

## Changed Factor

The model receives two channels per wavelength pixel:

1. Normalized flux.
2. `log1p` of inverse variance transformed into the normalized-flux units.

The validity mask remains operational metadata and is not a learned input channel. Architecture, data, seed, objectives, loss weights, checkpoint metric, and full-spectrum/no-augmentation policy match Control v2. Reconstruction loss remains unweighted so the uncertainty input is the only modeled change.

## Staged Protocol

1. Train through epoch 64 with practical early stopping disabled.
2. Apply the same conservative stage-1 gate as the augmentation study.
3. If competitive, continue the same model and optimizer through epoch 123.
4. Select by validation σ_NMAD and run fixed 10k DESI and SDSS OOD benchmarks using uncertainty at inference.

## Configuration and Artifacts

Configuration: `experiments/control_uncertainty_v1/`. Artifacts: `data/checkpoints/experiments/control_uncertainty_v1/main/` and `models/experiments/control_uncertainty_v1/main/`.
