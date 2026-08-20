# Control Full Spectrum v3: Screening-Size Saturation Test

## Purpose

Test whether a smaller Control can saturate quickly enough to become a useful rapid-iteration proxy.

## Configuration

| Setting | V2 Control | V3 Control |
| --- | ---: | ---: |
| Width / layers / heads | 256 / 4 / 8 | 128 / 3 / 4 |
| Parameters | 3.49M | 0.69M |
| Patch size | 61 | 61 |
| Input / augmentation | Full clean / none | Full clean / none |

V3 was trained for 1h 58m 30s (188 epochs), followed by an 11m 11s benchmark phase.

## Saturation Result

Reconstruction effectively saturated: over the last 48 epochs, validation reconstruction loss moved only about 0.1% (0.46370 to 0.46309). Redshift did not clearly saturate before the time cap: the final 12 epochs improved validation MAE by 5.4% and RMSE by 4.7% from their first-half to second-half averages. Best MAE was 0.02841 at epoch 169; best σ_NMAD was 0.01913 at epoch 182.

The smaller model also did not improve observed per-epoch throughput: V3 averaged about 38 seconds/epoch, versus about 33 seconds/epoch for V2 Control. This indicates that data loading and validation, rather than parameter count, dominate current runtime.

## Fixed 10k Benchmark Result

| Metric | DESI |
| --- | ---: |
| Redshift MAE | 0.08655 |
| Redshift RMSE | 0.15042 |
| σ_NMAD | 0.05268 |
| Outlier fraction | 0.1493 |
| Reconstruction MSE | 0.26223 |
| Redshift R² | 0.30047 |

V3 is weaker than V2 Control on the DESI benchmark. It should be treated as an exploratory proxy only, not as a replacement final architecture.

## Artifacts

`data/checkpoints/experiments/control_full_spectrum_v3/main/` and `models/experiments/control_full_spectrum_v3/main/`.
