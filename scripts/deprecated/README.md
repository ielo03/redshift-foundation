# Deprecated Script Entry Points

Nothing in this directory is part of the active experiment workflow. These
files are retained only to reproduce or inspect earlier runs; do not submit
them for new work.

| File | Why it is deprecated | Replacement |
| --- | --- | --- |
| `train_crop_transform_ablation.slurm` | Combined short FITS-streaming ablation. | Separate experiment-local launchers in `experiments/control_full_spectrum_v1/` and `experiments/crop_transform_v1/`. |
| `train_materialized_crop_transform_ablation.slurm` | Combined two-arm production array with shared output paths. | Submit each separate experiment's `train.slurm`. |
| `train_materialized_subset.slurm` | Generic, unversioned materialized-training launcher. | An experiment-local `train.slurm` plus immutable `config.json`. |

`scripts/train_experiments.py` is also legacy FITS-streaming training code. It
remains at its original path temporarily because the benchmark readers reuse
its model and spectrum-preparation helpers. It is not an approved launcher for
new experiments.
