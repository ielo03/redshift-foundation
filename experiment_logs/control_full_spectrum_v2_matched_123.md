# Matched Control v2 Continuation to Epoch 123

## Purpose

Resolve the unequal-training-budget concern in the Control-v2 versus Crop/Transform-v2 comparison. Control resumed from epoch 64 with its model and optimizer state intact, used the same full-size architecture and seed, disabled practical early stopping (patience 500), and ran through epoch 123.

Training took 35m 02s; fixed DESI and SDSS OOD benchmarks took 10m 30s. Both this continuation and Crop/Transform selected epoch 111.

## Validation at Selected Epoch 111

| Metric | Matched Control | Crop/Transform |
| --- | ---: | ---: |
| Redshift MAE | 0.02196 | 0.02771 |
| Redshift RMSE | 0.07966 | 0.07749 |
| Normalized MAE | 0.01640 | 0.02086 |
| σ_NMAD | 0.01409 | 0.01773 |
| Outlier fraction | 0.0073 | 0.0074 |
| Reconstruction loss | 0.46155 | 0.46568 |
| Redshift R² | 0.76311 | 0.77585 |

## Fixed 10k Held-Out Benchmarks

| Metric | Matched Control DESI | Crop/Transform DESI | Matched Control SDSS OOD | Crop/Transform SDSS OOD |
| --- | ---: | ---: | ---: | ---: |
| Redshift MAE | 0.08561 | 0.04212 | 0.20646 | 0.17000 |
| Redshift RMSE | 0.15008 | 0.09590 | 0.44307 | 0.39646 |
| Normalized MAE | 0.06575 | 0.03213 | 0.10929 | 0.08849 |
| σ_NMAD | 0.05117 | 0.02756 | 0.07930 | 0.04139 |
| Outlier fraction | 0.1456 | 0.0122 | 0.2137 | 0.1200 |
| Reconstruction MSE | 0.26626 | 0.14652 | 0.25059 | 0.10988 |
| Redshift R² | 0.30360 | 0.71564 | 0.62466 | 0.69948 |

## Interpretation

Crop/Transform wins strongly on DESI: MAE is 50.8% lower, RMSE 36.1% lower, σ_NMAD 46.1% lower, outliers 91.6% lower, and reconstruction MSE 45.0% lower. On SDSS OOD, the redshift MAE/RMSE gains are moderate (17.7% and 10.5% lower) while core-spread, outlier, and reconstruction improvements remain large.

The surprising result is that matched Control is better on the repeatedly evaluated clean validation set but much worse on both untouched benchmarks. Crop/Transform therefore appears to improve generalization and act as regularization, not merely benefit from additional epochs. This is one seed only: repeat with a second matched seed and split the combined method into crop-only and transforms-only ablations before adopting it for a large run.

## Artifacts

`data/checkpoints/experiments/control_full_spectrum_v2_matched_123_v1/main/` and `models/experiments/control_full_spectrum_v2_matched_123_v1/main/`.
