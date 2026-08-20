# Individual Experiment Logs

Each completed run has a separate log containing its purpose, exact configuration, selected checkpoint, validation trajectory, benchmark outcomes, and interpretation.

| Run | Log | Key outcome |
| --- | --- | --- |
| `control_full_spectrum_v2` | [Control v2](control_full_spectrum_v2.md) | Baseline early-stopped at epoch 64; selected checkpoint at epoch 52. |
| `crop_transform_v2` | [Crop/Transform v2](crop_transform_v2.md) | Strong held-out DESI and reconstruction result; combined intervention. |
| `control_full_spectrum_v2_matched_123_v1` | [Matched Control continuation](control_full_spectrum_v2_matched_123.md) | Resolves unequal-epoch concern; benchmark generalization remains worse than Crop/Transform. |
| `control_full_spectrum_v3` | [Control v3 screening-size test](control_full_spectrum_v3.md) | Smaller model did not improve per-epoch speed or reach clear redshift saturation. |
| `crop_only_v1` | [Crop-only staged ablation](crop_only_v1.md) | Complete; best SDSS OOD RMSE, σ_NMAD, and R² among augmentation arms. |
| `transforms_only_v1` | [Transforms-only staged ablation](transforms_only_v1.md) | Complete; leading redshift-MAE and outlier results. |
| `crop_or_transform_v1` | [Exclusive augmentation mixture](crop_or_transform_v1.md) | Complete; clean-inclusive balance nearly matches the best DESI MAE. |
| `crop_or_transform_50_50_v1` | [50/50 exclusive augmentation mixture](crop_or_transform_50_50_v1.md) | Complete; best overall SDSS OOD redshift result. |
| `crop_heavy_mixture_v1` | [Crop-heavy exclusive mixture](crop_heavy_mixture_v1.md) | Complete; best SDSS reconstruction MSE. |
| `transform_heavy_mixture_v1` | [Transform-heavy exclusive mixture](transform_heavy_mixture_v1.md) | Complete; best SDSS σ_NMAD. |
| `control_uncertainty_v1` | [Control + uncertainty](control_uncertainty_v1.md) | Queued as Slurm `57318921`; isolates normalized log-IVAR input on the plain Control. |
| `crop_or_transform_50_50_uncertainty_v1` | [50/50 + uncertainty](crop_or_transform_50_50_uncertainty_v1.md) | Queued as Slurm `57318922`; isolates normalized log-IVAR input on the leading OOD recipe. |

The project-level data record and current decision are in [../experiment_log.md](../experiment_log.md).
