# 50/50 Exclusive + Uncertainty v1

## Status

Queued as Slurm job `57318922` with a two-hour allocation. It has an `afterok` dependency on SDSS-IVAR preparation job `57318911`, so training starts only if the uncertainty benchmark bundle is built successfully.

## Purpose

Test whether per-pixel telescope uncertainty adds value on top of the leading cross-source 50/50 exclusive crop-or-transform recipe.

## Changed Factor

Relative to `crop_or_transform_50_50_v1`, the model adds one input channel: `log1p` of inverse variance transformed into normalized-flux units. Flux remains the first input channel. Validity remains an operational mask rather than a learned channel, and reconstruction loss remains unchanged.

Crop-only batches gather the matching uncertainty interval. Transform-only batches propagate uncertainty consistently: multiplicative gain/tilt scales measurement variance, injected noise adds variance in quadrature, and additive offset adds no variance.

## Staged Protocol

1. Train through epoch 64 with practical early stopping disabled.
2. Apply the same conservative stage-1 gate as the augmentation study.
3. If competitive, continue the same model and optimizer through epoch 123.
4. Select by validation σ_NMAD and run fixed 10k DESI and SDSS OOD benchmarks using uncertainty at inference.

## Configuration and Artifacts

Configuration: `experiments/crop_or_transform_50_50_uncertainty_v1/`. Artifacts: `data/checkpoints/experiments/crop_or_transform_50_50_uncertainty_v1/main/` and `models/experiments/crop_or_transform_50_50_uncertainty_v1/main/`.
