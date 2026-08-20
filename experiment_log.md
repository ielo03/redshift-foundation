# Experiment Log

This is the project-level index for the DESI DR1 redshift and spectrum-reconstruction foundation-model project. `plan.md` explains why experiments are proposed; individual completed runs are recorded in [experiment_logs/README.md](experiment_logs/README.md).

## Dataset And Split

| Item | Status | Result / location |
| --- | --- | --- |
| DR1 clean-row index | Complete | `data/preprocessed/clean_rows.jsonl`: 15,834,159 galaxy/QSO spectra after the documented zcatalog and coadd quality cuts. |
| Fixed HEALPix split | Locked | `data/preprocessed/split_manifest_v1.jsonl`: 13,403,355 train, 1,615,938 validation, 814,866 benchmark-only spectra. |
| Benchmark protection | Enforced | Training loaders require the fixed split manifest and skip benchmark records before opening FITS files. |
| Legacy pre-split materialization | Archived; never train | `data/archived_pre_split_do_not_use/materialized_legacy_mixed_splits/` (107 GB) mixes benchmark rows. |

## Data-Preparation Results

| Run | Status | Result | Artifacts |
| --- | --- | --- | --- |
| Full clean-manifest preprocess, Slurm `55689699` | Complete | 15,834,159 clean rows across 28,148 HEALPix records; 5:35 runtime. | `data/preprocessed/` |
| Legacy materialization, Slurm `56047752` | Timed out; archived | 1,637,895 spectra in almost 24 hours. It predates the fixed split and must not be used for training. | Archived path above |
| Fresh materialization benchmark, Slurm `56472604` | Complete | 1,024 spectra in 89.76 s; materialized load + normalization: 0.531 s (1,929 spectra/s). | `data/materialized_benchmark_1024_cold/` |
| Split-clean 1M materialization, Slurm `56473809` | Complete | 1,000,000 train/validation spectra in 16:39:06; 112 `.pt` shards, 1,275 HEALPix records. | `data/materialized_training_v1_1m/` |
| SDSS OOD bundle with IVAR, Slurm `57318911` | Queued | Preserve and propagate SDSS inverse variance for uncertainty-aware inference; existing flux-only bundle remains unchanged. | `data/ood/sdss_eboss_ood_with_ivar.pt` |

The 1M job requested CPU `shared` QOS with 16 GiB memory; Slurm allocated 10 logical CPUs, while the serial materializer used about one core (97% CPU utilization). Its approximate shared allocation was 0.65 CPU node-hours; Iris is the source of truth for charged usage.

## Experiment Index

| Run | Status | Summary | Log |
| --- | --- | --- | --- |
| `control_full_spectrum_v1` | Complete | Pre-enrichment Control; useful historical context but not directly comparable to the post-enrichment study. | [README](experiments/control_full_spectrum_v1/README.md) |
| `control_full_spectrum_v2` | Complete | Full-spectrum 3.49M-parameter Control; early-stopped after 64 epochs. | [log](experiment_logs/control_full_spectrum_v2.md) |
| `crop_transform_v2` | Complete | 65–100% crops plus input-only calibration/noise transforms; selected epoch 111. | [log](experiment_logs/crop_transform_v2.md) |
| `control_full_spectrum_v2_matched_123_v1` | Complete | Control v2 resumed to epoch 123 for a matched-budget comparison. | [log](experiment_logs/control_full_spectrum_v2_matched_123.md) |
| `control_full_spectrum_v3` | Complete | 0.69M-parameter screening-size Control; two-hour saturation test. | [log](experiment_logs/control_full_spectrum_v3.md) |
| `crop_only_v1` | Complete, Slurm `57269287` | Strong SDSS OOD RMSE/core scatter and reconstruction, but surpassed overall by the 50/50 mixture. | [log](experiment_logs/crop_only_v1.md) |
| `transforms_only_v1` | Complete, Slurm `57269286` | Best DESI redshift results; still the strongest in-domain redshift candidate. | [log](experiment_logs/transforms_only_v1.md) |
| `crop_or_transform_v1` | Complete, Slurm `57272017` | Exclusive 25% clean / 37.5% crop / 37.5% transforms; selected epoch 112. | [log](experiment_logs/crop_or_transform_v1.md) |
| `crop_or_transform_50_50_v1` | Complete, Slurm `57271985` | Exclusive 50% crop / 50% transforms; best overall SDSS OOD redshift result. | [log](experiment_logs/crop_or_transform_50_50_v1.md) |
| `crop_heavy_mixture_v1` | Complete, Slurm `57272018` | Exclusive 20% clean / 65% crop / 15% transforms; best SDSS reconstruction MSE. | [log](experiment_logs/crop_heavy_mixture_v1.md) |
| `transform_heavy_mixture_v1` | Complete, Slurm `57272019` | Exclusive 20% clean / 15% crop / 65% transforms; best SDSS σ_NMAD. | [log](experiment_logs/transform_heavy_mixture_v1.md) |
| `control_uncertainty_v1` | Queued, Slurm `57318921` | Control with normalized log-IVAR as a second model input; depends on SDSS-IVAR job `57318911`. | [log](experiment_logs/control_uncertainty_v1.md) |
| `crop_or_transform_50_50_uncertainty_v1` | Queued, Slurm `57318922` | 50/50 exclusive mixture with normalized log-IVAR as a second model input; depends on SDSS-IVAR job `57318911`. | [log](experiment_logs/crop_or_transform_50_50_uncertainty_v1.md) |

## Current Decision

Transforms-only remains the strongest DESI redshift model. The exclusive 50/50 crop-or-transform mixture is the strongest cross-source candidate: it has the best SDSS OOD MAE, RMSE, catastrophic-outlier fraction, and R². Crop-heavy training is best for SDSS reconstruction, while transform-heavy has the best SDSS core scatter (σ_NMAD). This confirms that crop and transform invariances can be combined more effectively by sampling them separately than by always applying both simultaneously.

### Completed Augmentation Comparison

Control v2 below is its matched-budget continuation. All rows use the same full-size architecture, data, seed, checkpoint metric, maximum epoch 123, and fixed 10k benchmarks. Lower is better for every metric except R²; bold marks the best result in each benchmark table. `C/T` means crop and transforms are applied together; exclusive mixtures sample only one mode per batch.

#### DESI Reserved Benchmark

| Experiment | Training mixture | Epoch | MAE | RMSE | σ_NMAD | Outliers | Recon. MSE | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control v2 matched | 100% clean | 111 | 0.08561 | 0.15008 | 0.05117 | 0.1456 | 0.26626 | 0.30360 |
| Crop + transform | 100% combined C/T | 111 | 0.04212 | 0.09590 | 0.02756 | 0.0122 | **0.14652** | 0.71564 |
| Crop only | 100% crop | 112 | 0.04575 | 0.10191 | 0.02543 | 0.0166 | 0.15423 | 0.67889 |
| Transforms only | 100% transforms | 116 | **0.02761** | **0.08627** | **0.01678** | 0.0097 | 0.16236 | **0.76992** |
| Balanced + clean | 25% clean / 37.5% crop / 37.5% transform | 112 | 0.02766 | 0.08912 | 0.01794 | 0.0097 | 0.15253 | 0.75445 |
| 50/50 exclusive | 50% crop / 50% transform | 112 | 0.03186 | 0.08726 | 0.02227 | **0.0092** | 0.15174 | 0.76458 |
| Crop-heavy | 20% clean / 65% crop / 15% transform | 111 | 0.03339 | 0.09109 | 0.02578 | 0.0097 | 0.14909 | 0.74346 |
| Transform-heavy | 20% clean / 15% crop / 65% transform | 112 | 0.02890 | 0.08950 | 0.01919 | 0.0119 | 0.15538 | 0.75238 |

#### SDSS OOD Benchmark

| Experiment | Training mixture | Epoch | MAE | RMSE | σ_NMAD | Outliers | Recon. MSE | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control v2 matched | 100% clean | 111 | 0.20646 | 0.44307 | 0.07930 | 0.2137 | 0.25059 | 0.62466 |
| Crop + transform | 100% combined C/T | 111 | 0.17000 | 0.39646 | 0.04139 | 0.1200 | 0.10988 | 0.69948 |
| Crop only | 100% crop | 112 | 0.15957 | 0.37277 | 0.03470 | 0.1231 | 0.11279 | 0.73432 |
| Transforms only | 100% transforms | 116 | 0.14282 | 0.38102 | 0.03883 | 0.1032 | 0.19043 | 0.72243 |
| Balanced + clean | 25% clean / 37.5% crop / 37.5% transform | 112 | 0.14403 | 0.39542 | 0.03878 | 0.1068 | 0.12118 | 0.70106 |
| 50/50 exclusive | 50% crop / 50% transform | 112 | **0.13498** | **0.36846** | 0.04180 | **0.0875** | 0.13524 | **0.74043** |
| Crop-heavy | 20% clean / 65% crop / 15% transform | 111 | 0.18118 | 0.42300 | 0.03971 | 0.1209 | **0.10832** | 0.65790 |
| Transform-heavy | 20% clean / 15% crop / 65% transform | 112 | 0.14533 | 0.39553 | **0.03440** | 0.1067 | 0.17960 | 0.70089 |

Every augmentation arm beats matched Control v2 on every fixed-benchmark metric. The differences among the leading augmented models are smaller and come from one seed, so they should not yet be treated as statistically established rankings.

### Earlier and Non-Matched Benchmark Context

These runs are not candidates in the matched augmentation comparison: Control v1 predates target-label enrichment, Control v2 stopped at epoch 64, and Control v3 is a smaller 0.69M-parameter screening model.

| Experiment | Selected epoch | DESI MAE | DESI σ_NMAD | DESI outliers | SDSS MAE | SDSS σ_NMAD | SDSS outliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control v1 (pre-enrichment) | 56 | 0.06881 | 0.03979 | 0.0785 | 0.26357 | 0.08228 | 0.3197 |
| Control v2 (early stop) | 52 | 0.07223 | 0.05746 | 0.0567 | 0.22407 | 0.05163 | 0.2185 |
| Control v3 (small model) | 182 | 0.08655 | 0.05268 | 0.1493 | 0.23214 | 0.07319 | 0.3112 |

## Next Decisions

1. Repeat 50/50 exclusive and transforms-only with a second matched seed; choose between them based on reproducible cross-source versus in-domain performance.
2. Add another source benchmark and a separate development OOD subset before tuning mixture probabilities further.
3. Reserve the fixed 10k DESI and SDSS benchmarks for confirmation rather than repeatedly selecting methods against them.
