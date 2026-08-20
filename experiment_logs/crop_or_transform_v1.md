# Crop-or-Transform v1

## Status

Completed as replacement Slurm job `57272017` in 1h 27m 51s, including fixed benchmarks. The stage-1 gate chose `continue`; epoch 112 was selected. Initial job `57271615` failed before training because the new exclusive-policy probability check was missing its `math` import; no checkpoint was produced by that attempt.

## Purpose

Test whether the combined Crop/Transform model underperformed the isolated augmentations because every augmented example was simultaneously cropped and corrupted.

## Augmentation Policy

Each training batch deterministically samples exactly one mode from the epoch, seed, and batch index:

| Mode | Probability | Input / target |
| --- | ---: | --- |
| Clean full spectrum | 25.0% | Clean full input; clean reconstruction target |
| Crop-only | 37.5% | Random 65–100% contiguous crop; matching clean crop target |
| Transforms-only | 37.5% | Full-length gain/tilt/offset/noise input; clean full target |

Crops and transforms are never applied simultaneously. Validation remains clean and full-length. The trainer records actual per-epoch augmentation batch counts in `history.jsonl`.

## Staged Protocol

1. Train through epoch 64 with practical early stopping disabled.
2. Stop only if best MAE and σ_NMAD are both at least 1.5× Control and neither metric is improving by 5% over the recent trend windows.
3. Otherwise continue the same model and optimizer through epoch 123.
4. Select by validation σ_NMAD and run the fixed 10k DESI and SDSS OOD benchmarks.

## Selected Validation Result

| Metric | Value |
| --- | ---: |
| MAE | 0.02236 |
| RMSE | 0.07478 |
| σ_NMAD | 0.01445 |
| Reconstruction loss | 0.46338 |
| R² | 0.79128 |

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| MAE | 0.02766 | 0.14403 |
| RMSE | 0.08912 | 0.39542 |
| σ_NMAD | 0.01794 | 0.03878 |
| Outlier fraction | 0.0097 | 0.1068 |
| Reconstruction MSE | 0.15253 | 0.12118 |
| R² | 0.75445 | 0.70106 |

This clean-inclusive mixture nearly matches transforms-only DESI MAE and improves reconstruction over the 50/50 no-clean mixture, but the 50/50 mixture is substantially better on SDSS OOD MAE, RMSE, outliers, and R².

## Configuration

`experiments/crop_or_transform_v1/config_stage1.json`, `experiments/crop_or_transform_v1/config_stage2.json`, and `experiments/crop_or_transform_v1/train_staged.slurm`.

## Artifacts

`data/checkpoints/experiments/crop_or_transform_v1/main/` and `models/experiments/crop_or_transform_v1/main/`.
