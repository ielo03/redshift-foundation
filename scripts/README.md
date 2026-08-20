# Workflow Scripts

The scripts stay in one directory deliberately: the active shell and Slurm commands use
these paths, and moving them would break resumable jobs. Their roles are grouped here.

For a new experiment, start from [`../experiments/README.md`](../experiments/README.md).
Keep these scripts shared only when their behavior is intentionally identical
across experiments; otherwise use an experiment-local implementation.

## Data Preparation

| Script | Role | Source of truth / output |
| --- | --- | --- |
| `preprocess.py` | Creates the clean DR1 row manifest. | `data/preprocessed/clean_rows.jsonl` |
| `build_split_manifest.py` | Assigns every clean row to fixed train, validation, or benchmark roles. Run once. | `data/preprocessed/split_manifest_v1.jsonl` |
| `materialize_shards.py` | Copies only train/validation spectra into contiguous PyTorch shards. | `data/materialized_training_v1/` |
| `preprocess_full.slurm` | Resumable full clean-manifest job. | `data/preprocessed/` |
| `materialize_full.slurm` | Resumable split-clean materialization job. | `data/materialized_training_v1/` |
| `materialize_1m_shared.slurm` | Resumable one-million-spectrum materialization on the fractional CPU `shared` QOS. | `data/materialized_training_v1_1m/` |

## Active Training

| Script | Status | Data path | Difference |
| --- | --- | --- | --- |
| `run_materialized_experiment.py` | Config launcher used by every active experiment. | Split-clean `.pt` shards | Resolves an immutable experiment config into the shared materialized trainer. |
| `train_materialized_streaming.py` | Shared trainer; not submitted directly for a study. | Split-clean `.pt` shards | GPU-native crop, transform, and contiguous-mask path. |
| `../experiments/control_full_spectrum_v1/train.slurm` | Active reusable control. | `materialized_training_v1_1m/` | Full spectra, no transforms. |
| `../experiments/crop_transform_v1/train.slurm` | Active treatment. | `materialized_training_v1_1m/` | 65–100% crops plus transforms. |

Legacy launchers live in [`deprecated/`](deprecated/README.md). `train_experiments.py` remains only as a helper dependency for the current benchmark readers and must not be used to start a new experiment.

`data/archived_pre_split_do_not_use/materialized_legacy_mixed_splits/` is legacy pre-split output. Do not train from it: it mixes benchmark rows and the current streaming trainer rejects it.

## Data Layout

```text
data/                                      # PSCRATCH symlink
  preprocessed/
    clean_rows.jsonl                       # cleaning result; never train from directly
    split_manifest_v1.jsonl                # fixed assignment; training source of truth
    split_manifest_v1_summary.json         # counts and split policy
  materialized_training_v1/                # future train/validation-only shards
  checkpoints/                             # resumable scratch checkpoints
  training_runs/                           # streaming trainer scratch outputs

models/                                    # durable best checkpoint + metadata only
logs/                                      # Slurm stdout/stderr
```
