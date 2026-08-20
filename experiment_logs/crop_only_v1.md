# Crop-Only v1

## Status

Completed as Slurm job `57269287`: 1h 25m 56s total, including a 15m 48s benchmark phase. The stage-1 gate chose `continue`; epoch 112 was selected.

## Purpose

Isolate the effect of random 65–100% contiguous training crops from the calibration/noise transforms used by `crop_transform_v2`. Architecture, data, seed, losses, and validation protocol match Control v2.

## Staged Protocol

1. Train without practical early stopping through epoch 64.
2. Compare best validation MAE and σ_NMAD through epoch 64 with Control v2.
3. Stop only if both metrics are at least 1.5× Control and neither improves by at least 5% across the last two 12-epoch windows. Invalid/non-finite results also stop.
4. Otherwise resume the same model and optimizer through epoch 123, select by validation σ_NMAD, and run the fixed 10k DESI and SDSS OOD benchmarks.

The gate is intentionally conservative: the known combined Crop/Transform run was about 1.26× Control on both stage-1 primary metrics and was still improving, so it would continue.

## Selected Validation Result

| Metric | Value |
| --- | ---: |
| MAE | 0.03288 |
| RMSE | 0.08022 |
| σ_NMAD | 0.01785 |
| Reconstruction loss | 0.46383 |
| R² | 0.75981 |

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| MAE | 0.04575 | 0.15957 |
| RMSE | 0.10191 | 0.37277 |
| σ_NMAD | 0.02543 | 0.03470 |
| Outlier fraction | 0.0166 | 0.1231 |
| Reconstruction MSE | 0.15423 | 0.11279 |
| R² | 0.67889 | 0.73432 |

Crop-only produces the best SDSS OOD RMSE, σ_NMAD, and R² among the isolated/combined augmentation arms and nearly matches the combined model's reconstruction. It is weaker than transforms-only on MAE and catastrophic-outlier rate.

## Configuration

`experiments/crop_only_v1/config_stage1.json`, `experiments/crop_only_v1/config_stage2.json`, and `experiments/crop_only_v1/train_staged.slurm`.

## Artifacts

`data/checkpoints/experiments/crop_only_v1/main/` and `models/experiments/crop_only_v1/main/`.
