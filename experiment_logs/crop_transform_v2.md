# Crop/Transform v2

## Purpose

Test a combined robustness intervention: random contiguous crops spanning 65–100% of each spectrum plus input-only gain, calibration-tilt, offset, and Gaussian-noise transforms. Reconstruction targets remain clean, untransformed flux.

## Configuration Difference from Control v2

| Setting | Control v2 | Crop/Transform v2 |
| --- | ---: | ---: |
| Width / layers / heads | 256 / 4 / 8 | 256 / 4 / 8 |
| Crop fraction | 100% | 65–100% |
| Random transforms | No | Yes |
| Other architecture, loss, data, seed | Same | Same |

## Training Outcome

Training ran for 123 epochs (1h 02m 56s). Epoch 111 was selected by validation σ_NMAD.

| Selected validation metric | Value |
| --- | ---: |
| Redshift MAE | 0.02771 |
| Redshift RMSE | 0.07749 |
| σ_NMAD | 0.01773 |
| Reconstruction loss | 0.46568 |

At Control's epoch-64 budget, Crop/Transform was worse on clean validation redshift metrics. It continued improving later, so short runs can incorrectly reject augmentation-style interventions.

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| Redshift MAE | 0.04212 | 0.17000 |
| Redshift RMSE | 0.09590 | 0.39646 |
| σ_NMAD | 0.02756 | 0.04139 |
| Outlier fraction (`|Δz|/(1+z) > 0.15`) | 0.0122 | 0.1200 |
| Reconstruction MSE | 0.14652 | 0.10988 |
| Redshift R² | 0.71564 | 0.69948 |
| Galaxy/QSO accuracy | 0.9960 | — |

## Interpretation

The combined intervention is promising but does not isolate whether crops or transforms provide the gain. The matched Control continuation provides the more informative generalization comparison.

## Artifacts

`data/checkpoints/experiments/crop_transform_v2/main/` and `models/experiments/crop_transform_v2/main/`.
