# Crop-Heavy Mixture v1

## Status

Completed as replacement Slurm job `57272018` in 1h 20m 49s, including fixed benchmarks. The stage-1 gate chose `continue`; epoch 111 was selected. Initial job `57271787` failed before training because the new exclusive-policy probability check was missing its `math` import; no checkpoint was produced by that attempt.

## Purpose

Test whether an exclusive mixture weighted strongly toward crop-only batches retains crop robustness and reconstruction while receiving limited transform and clean exposure.

## Mixture

| Mode | Probability | Approximate exposure by epoch 123 |
| --- | ---: | ---: |
| Clean full spectrum | 20% | 25 equivalent epochs |
| Crop-only | 65% | 80 equivalent epochs |
| Transforms-only | 15% | 18 equivalent epochs |

Modes are sampled deterministically per batch and are never combined. The run uses the conservative epoch-64 gate, continues to epoch 123 if it passes, and then runs fixed DESI and SDSS OOD benchmarks.

## Selected Validation Result

| Metric | Value |
| --- | ---: |
| MAE | 0.02397 |
| RMSE | 0.07760 |
| σ_NMAD | 0.01494 |
| Reconstruction loss | 0.46388 |
| R² | 0.77521 |

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| MAE | 0.03339 | 0.18118 |
| RMSE | 0.09109 | 0.42300 |
| σ_NMAD | 0.02578 | 0.03971 |
| Outlier fraction | 0.0097 | 0.1209 |
| Reconstruction MSE | 0.14909 | **0.10832** |
| R² | 0.74346 | 0.65790 |

Crop-heavy training produces the best SDSS reconstruction MSE in the matched study, but its SDSS redshift metrics are weaker than the balanced mixtures, crop-only, and transforms-only. It is a reconstruction specialist rather than the best foundation-model candidate.

## Configuration and Artifacts

Configuration: `experiments/crop_heavy_mixture_v1/`. Artifacts: `data/checkpoints/experiments/crop_heavy_mixture_v1/main/` and `models/experiments/crop_heavy_mixture_v1/main/`.
