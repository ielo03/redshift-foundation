# Transform-Heavy Mixture v1

## Status

Completed as replacement Slurm job `57272019` in 1h 21m 56s, including fixed benchmarks. The stage-1 gate chose `continue`; epoch 112 was selected. Initial job `57271788` failed before training because the new exclusive-policy probability check was missing its `math` import; no checkpoint was produced by that attempt.

## Purpose

Test whether an exclusive mixture weighted strongly toward transforms-only batches preserves the leading redshift behavior while receiving limited crop and clean exposure.

## Mixture

| Mode | Probability | Approximate exposure by epoch 123 |
| --- | ---: | ---: |
| Clean full spectrum | 20% | 25 equivalent epochs |
| Crop-only | 15% | 18 equivalent epochs |
| Transforms-only | 65% | 80 equivalent epochs |

Modes are sampled deterministically per batch and are never combined. The run uses the conservative epoch-64 gate, continues to epoch 123 if it passes, and then runs fixed DESI and SDSS OOD benchmarks.

## Selected Validation Result

| Metric | Value |
| --- | ---: |
| MAE | 0.02345 |
| RMSE | 0.07717 |
| σ_NMAD | 0.01454 |
| Reconstruction loss | 0.46294 |
| R² | 0.77770 |

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| MAE | 0.02890 | 0.14533 |
| RMSE | 0.08950 | 0.39553 |
| σ_NMAD | 0.01919 | **0.03440** |
| Outlier fraction | 0.0119 | 0.1067 |
| Reconstruction MSE | 0.15538 | 0.17960 |
| R² | 0.75238 | 0.70089 |

Transform-heavy training has the best SDSS core scatter (σ_NMAD), but transforms-only remains better on DESI and slightly better on SDSS MAE. The 50/50 mixture is markedly better on SDSS RMSE, outliers, and R².

## Configuration and Artifacts

Configuration: `experiments/transform_heavy_mixture_v1/`. Artifacts: `data/checkpoints/experiments/transform_heavy_mixture_v1/main/` and `models/experiments/transform_heavy_mixture_v1/main/`.
