# Control Full Spectrum v2

## Purpose

Establish the full-size, no-augmentation Control for the variable-length redshift-token transformer.

## Configuration

| Setting | Value |
| --- | --- |
| Architecture | `DynamicSpectraTransformerWithRedshiftToken` |
| Width / layers / heads | 256 / 4 / 8 |
| Parameters | 3.49M |
| Patch size | 61 pixels |
| Train / validation spectra per epoch | 200k / 20k |
| Input | Full clean spectra; no random transforms |
| Checkpoint rule | Lowest validation `z_sigma_nmad`, minimum decrease 0.0001 |

## Training Outcome

Training ran for 64 epochs (35m 35s). The selected checkpoint is epoch 52 because later improvements did not exceed the configured minimum decrease.

| Selected validation metric | Value |
| --- | ---: |
| Redshift MAE | 0.04125 |
| σ_NMAD | 0.02257 |
| Reconstruction loss | 0.46435 |

At epoch 64, raw validation metrics still improved slightly (MAE 0.03250; σ_NMAD 0.02250), so the stop reflects the thresholded checkpoint rule rather than clear redshift saturation. Reconstruction had largely flattened by the middle of training.

## Fixed 10k Benchmarks

| Metric | DESI | SDSS OOD |
| --- | ---: | ---: |
| Redshift MAE | 0.07223 | 0.22407 |
| Redshift RMSE | 0.13099 | 0.42542 |
| σ_NMAD | 0.05746 | 0.05163 |
| Outlier fraction (`|Δz|/(1+z) > 0.15`) | 0.0567 | 0.2185 |
| Reconstruction MSE | 0.25775 | 0.30092 |
| Redshift R² | 0.46951 | 0.65398 |
| Galaxy/QSO accuracy | 0.9942 | — |

## Artifacts

`data/checkpoints/experiments/control_full_spectrum_v2/main/` and `models/experiments/control_full_spectrum_v2/main/`.
