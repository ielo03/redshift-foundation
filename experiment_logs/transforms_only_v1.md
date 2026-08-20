# Transforms-Only v1

## Status

Completed as Slurm job `57269286`: 1h 29m 31s total, including a 13m 31s benchmark phase. The stage-1 gate chose `continue`; epoch 116 was selected.

## Purpose

Isolate the effect of input-only gain, calibration-tilt, offset, and Gaussian-noise transforms from the random crops used by `crop_transform_v2`. Training uses full-length spectra; architecture, data, seed, losses, and validation protocol match Control v2.

## Staged Protocol

1. Train without practical early stopping through epoch 64.
2. Compare best validation MAE and σ_NMAD through epoch 64 with Control v2.
3. Stop only if both metrics are at least 1.5× Control and neither improves by at least 5% across the last two 12-epoch windows. Invalid/non-finite results also stop.
4. Otherwise resume the same model and optimizer through epoch 123, select by validation σ_NMAD, and run the fixed 10k DESI and SDSS OOD benchmarks.

The gate is intentionally conservative: the known combined Crop/Transform run was about 1.26× Control on both stage-1 primary metrics and was still improving, so it would continue.

## Selected Validation Result

| Metric | Value |
| --- | ---: |
| MAE | 0.01907 |
| RMSE | 0.07393 |
| σ_NMAD | 0.01147 |
| Reconstruction loss | 0.46240 |
| R² | 0.79599 |

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| MAE | 0.02761 | 0.14282 |
| RMSE | 0.08627 | 0.38102 |
| σ_NMAD | 0.01678 | 0.03883 |
| Outlier fraction | 0.0097 | 0.1032 |
| Reconstruction MSE | 0.16236 | 0.19043 |
| R² | 0.76992 | 0.72243 |

Transforms-only is the leading redshift candidate: it has the best DESI redshift metrics and the best SDSS OOD MAE and outlier fraction. Crop-only remains better for SDSS RMSE/core scatter and reconstruction.

## Configuration

`experiments/transforms_only_v1/config_stage1.json`, `experiments/transforms_only_v1/config_stage2.json`, and `experiments/transforms_only_v1/train_staged.slurm`.

## Artifacts

`data/checkpoints/experiments/transforms_only_v1/main/` and `models/experiments/transforms_only_v1/main/`.
