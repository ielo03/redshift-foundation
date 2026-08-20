# Crop-or-Transform 50/50 v1

## Status

Completed as Slurm job `57271985` in 1h 24m 58s, including fixed benchmarks. The stage-1 gate chose `continue`; epoch 112 was selected.

## Purpose

Test a true balanced exclusive mixture with no clean batches. Every training batch receives exactly one augmentation mode: crop-only or transforms-only. This separates the effect of equal augmentation weighting from the 25% clean exposure in `crop_or_transform_v1`.

## Augmentation Policy

| Mode | Probability | Approximate exposure through epoch 123 | Input / target |
| --- | ---: | ---: | --- |
| Clean full spectrum | 0% | 0 equivalent epochs | Not sampled during training |
| Crop-only | 50% | 61.5 equivalent epochs | Random 65–100% contiguous crop; matching clean crop target |
| Transforms-only | 50% | 61.5 equivalent epochs | Full-length gain/tilt/offset/noise input; clean full target |

Crop and transforms are never applied simultaneously. Validation remains clean and full-length, and actual augmentation batch counts are recorded in `history.jsonl`.

## Staged Protocol

1. Train through epoch 64 with practical early stopping disabled.
2. Stop only if best MAE and σ_NMAD are both at least 1.5× Control and neither metric is improving by 5% over the recent trend windows.
3. Otherwise continue the same model and optimizer through epoch 123.
4. Select by validation σ_NMAD and run the fixed 10k DESI and SDSS OOD benchmarks.

## Selected Validation Result

| Metric | Value |
| --- | ---: |
| MAE | 0.02685 |
| RMSE | 0.07663 |
| σ_NMAD | 0.01565 |
| Reconstruction loss | 0.46352 |
| R² | 0.78083 |

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| MAE | 0.03186 | **0.13498** |
| RMSE | 0.08726 | **0.36846** |
| σ_NMAD | 0.02227 | 0.04180 |
| Outlier fraction | **0.0092** | **0.0875** |
| Reconstruction MSE | 0.15174 | 0.13524 |
| R² | 0.76458 | **0.74043** |

The bold results are the best among the matched augmentation study. This is the strongest SDSS OOD redshift model overall, while transforms-only remains better on most DESI redshift metrics. Removing clean batches improves cross-source redshift performance relative to the clean-inclusive balanced mixture, at the cost of worse SDSS reconstruction.

## Configuration

`experiments/crop_or_transform_50_50_v1/config_stage1.json`, `experiments/crop_or_transform_50_50_v1/config_stage2.json`, and `experiments/crop_or_transform_50_50_v1/train_staged.slurm`.

## Artifacts

`data/checkpoints/experiments/crop_or_transform_50_50_v1/main/` and `models/experiments/crop_or_transform_50_50_v1/main/`.
