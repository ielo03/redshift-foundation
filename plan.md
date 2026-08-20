# Redshift And Spectrum Reconstruction Foundation Model Plan

## Goal

Build a spectra-only foundation model on DESI DR1 that predicts redshift accurately from galaxy/QSO spectra and reconstructs masked or missing spectral regions. The long-term goal is for both redshift prediction and reconstruction to remain useful when the input spectrum is incomplete, corrupted, or intentionally tampered with.

Core constraints:

- DESI spectra only
- redshift `z` and masked spectrum reconstruction as the targets
- no stars in training
- no photometry, images, magnitudes, or other modalities
- evaluation must include both redshift accuracy and masked spectrum reconstruction

## Current Baseline To Rebuild

The first reconstruction baseline should follow the final-project model closely enough that results are comparable, but not copy it blindly. Treat final-project as the reference recipe: preserve the pieces that make the experiment fair, and improve anything that is clearly brittle, inefficient, or awkward for the new DR1 setup.

The final-project model used `SpectraTransformerWithRedshiftToken`:

- raw flux patches, no learned AION-style tokenizer
- learned redshift token prepended to spectrum patch tokens
- transformer encoder
- reconstruction head on spectral patch tokens
- redshift regression head on the redshift token
- redshift encoded as `z / (1 + z)`
- joint loss: masked reconstruction + weighted redshift loss

This is the first thing to reproduce with the new DR1 data filters. The goal is a faithful baseline, not a line-for-line port: if final-project does something avoidably stupid for data loading, masking, normalization, padding, logging, or evaluation, we should do the cleaner version and write down the difference.

## Data Plan

Use DESI DR1:

```text
/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/healpix
```

Use `scripts/preprocess.py` to build:

```text
data/preprocessed/clean_rows.jsonl
data/preprocessed/summary.json
data/preprocessed/allowed_targets_galaxy_qso_primary.npz
data/preprocessed/split_manifest_v1.jsonl
```

Required cuts:

```text
ZWARN == 0
ZCAT_PRIMARY == True
SPECTYPE in {GALAXY, QSO}
OBJTYPE == TGT
COADD_FIBERSTATUS == 0
nonzero B/R/Z flux
```

First choice: train from the fixed split manifest and original FITS files.

Fallback if training is I/O-bound: materialize train/validation spectra under `data/materialized_training_v1/`.

Materialization plan:

- use `scripts/materialize_shards.py`, not `scripts/preprocess.py --materialize`, for serious materialization
- write larger PyTorch `.pt` shards instead of one compressed `.npz` per HEALPix
- keep raw stitched `flux`, `ivar`, `valid`, `z`, `targetid`, `row`, `healpix`, `spectype`, and shared `wavelength`
- support subset runs with `--max-spectra` and `--max-healpix`
- resume through `materialized_processed_files.jsonl`

Subset materialization command:

```bash
$HOME/.conda/envs/venv/bin/python scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir data/materialized_smoke \
  --max-spectra 10000 \
  --shard-size 2048
```

Full materialization command, only if scratch space is sufficient:

```bash
$HOME/.conda/envs/venv/bin/python scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir data/materialized_training_v1 \
  --shard-size 4096
```

Preprocessing efficiency:

- reuse `allowed_targets_galaxy_qso_primary.npz`; building it is the expensive zcatalog pass
- use `--num-workers 1` for the full run unless new timing says otherwise
- the first `--num-workers 4` test was slower than cached single-worker timing, likely because the scan is CFS/FITS I/O-bound

Cached parallel timing command, for reference:

```bash
/usr/bin/time -v $HOME/.conda/envs/venv/bin/python scripts/preprocess.py \
  --max-healpix 100 \
  --out-dir data/preprocessed_timing_100_workers4 \
  --allowed-cache data/preprocessed_timing_100/allowed_targets_galaxy_qso_primary.npz \
  --num-workers 4
```

Full manifest command after timing:

```bash
$HOME/.conda/envs/venv/bin/python scripts/preprocess.py \
  --out-dir data/preprocessed \
  --allowed-cache data/preprocessed_timing_100/allowed_targets_galaxy_qso_primary.npz \
  --num-workers 1
```

Detached full manifest job:

```bash
sbatch -A <actual_account> scripts/preprocess_full.slurm
```

The Slurm script requests one task with 4 CPUs, not a full CPU node. The cached 100-HEALPix test took about 24 seconds and `--num-workers 4` was slower than `--num-workers 1`, so manifest preprocessing should avoid reserving hundreds of idle CPUs.

The first full Slurm attempt hit the 4-hour time limit without useful stdout because output was buffered and the single-worker path was accidentally pre-discovering the full tree before processing. Fixes:

- single-worker preprocessing streams HEALPix files again instead of pre-building the full file list
- Python runs with `-u` in `scripts/preprocess_full.slurm`
- status messages flush immediately
- `data/preprocessed/progress.json` updates every 25 processed HEALPix files
- preprocessing is resumable by default via `data/preprocessed/processed_files.jsonl`; resubmitting the same Slurm job skips completed HEALPix files and continues

## Experiment 1: Rebuild Final Project On Clean DR1

Purpose: establish a fair redshift + reconstruction baseline using the same model idea, but with better DR1 filtering.

Baseline rule:

- follow final-project for the model shape, redshift token, patching idea, reconstruction objective, and redshift transform
- allow targeted improvements where the old code is clearly worse for DR1 training or evaluation
- document every intentional difference from final-project so the comparison stays honest

Tasks:

- build a dataset loader for `split_manifest_v1.jsonl`
- read only clean rows from `coadd-*.fits`
- stitch B/R/Z bands the same way as `redshifty`
- normalize flux consistently
- train `SpectraTransformerWithRedshiftToken`
- optimize redshift prediction and masked flux reconstruction jointly
- evaluate both tasks on held-out HEALPix files, not random rows

Initial config:

```text
architecture: SpectraTransformerWithRedshiftToken
patch_size: 61
d_model: 512
layers: 8
heads: 8
mask_prob: 0.15
redshift_loss: SmoothL1
alpha: 1.25
sampling: tail-focused
```

30-minute test command:

```bash
timeout 30m $HOME/.conda/envs/venv/bin/python scripts/train_experiments.py \
  --experiment fixed \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir data/checkpoints/test_runs/exp1_fixed \
  --model-dir models/test_runs/exp1_fixed \
  --max-examples 4096 \
  --batch-size 32 \
  --epochs 100000 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --patch-size 61
```

Resume command:

```bash
timeout 30m $HOME/.conda/envs/venv/bin/python scripts/train_experiments.py \
  --experiment fixed \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir data/checkpoints/test_runs/exp1_fixed \
  --model-dir models/test_runs/exp1_fixed \
  --max-examples 4096 \
  --batch-size 32 \
  --epochs 100000 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --patch-size 61 \
  --resume data/checkpoints/test_runs/exp1_fixed/latest.pt
```

Scratch-loss resume command:

```bash
timeout 30m $HOME/.conda/envs/venv/bin/python scripts/train_experiments.py \
  --experiment fixed \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir data/checkpoints/test_runs/exp1_fixed \
  --model-dir models/test_runs/exp1_fixed \
  --max-examples 4096 \
  --batch-size 32 \
  --epochs 100000 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --patch-size 61 \
  --resume models/test_runs/exp1_fixed/best.pt
```

Success criteria:

- train/val/test split is HEALPix-clean
- beats or matches the final-project redshift MAE on comparable data scale
- reports masked reconstruction MAE/RMSE on comparable data scale
- no stars leak into training or validation
- reconstruction metrics are reported only on real, non-padded pixels

## Experiment 2: Input-Length Agnostic Model

Purpose: support redshift prediction and reconstruction for spectra with different wavelength coverage or missing regions without forcing everything into one fixed length.

Problems in the final-project model:

- `input_length` is baked into the model
- `padded_length` and `num_patches` are fixed at initialization
- learned `pos_embed` has fixed length
- reconstruction is always cropped to the constructor `input_length`

Proposed changes:

- compute padding and patch count dynamically in `forward`
- replace fixed learned positional embeddings with sinusoidal, rotary, or wavelength-aware embeddings
- return reconstruction cropped to each item length
- use padding masks in attention and reconstruction loss
- apply transforms to model input only and reconstruct the original clean crop
- embed absolute log-wavelength patch centers so crops retain physical wavelength location
- batch by padding to the longest spectrum in the batch

Preferred first implementation:

```text
dynamic patching
sinusoidal patch positions
batch padding mask
loss mask = real pixels & masked pixels & not padded pixels
```

30-minute test command:

```bash
timeout 30m $HOME/.conda/envs/venv/bin/python scripts/train_experiments.py \
  --experiment variable \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir data/checkpoints/test_runs/exp2_variable \
  --model-dir models/test_runs/exp2_variable \
  --max-examples 4096 \
  --batch-size 32 \
  --epochs 100000 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --patch-size 61 \
  --variable-min-fraction 0.65
```

Resume command:

```bash
timeout 30m $HOME/.conda/envs/venv/bin/python scripts/train_experiments.py \
  --experiment variable \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir data/checkpoints/test_runs/exp2_variable \
  --model-dir models/test_runs/exp2_variable \
  --max-examples 4096 \
  --batch-size 32 \
  --epochs 100000 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --patch-size 61 \
  --variable-min-fraction 0.65 \
  --resume data/checkpoints/test_runs/exp2_variable/latest.pt
```

Scratch-loss resume command:

```bash
timeout 30m $HOME/.conda/envs/venv/bin/python scripts/train_experiments.py \
  --experiment variable \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir data/checkpoints/test_runs/exp2_variable \
  --model-dir models/test_runs/exp2_variable \
  --max-examples 4096 \
  --batch-size 32 \
  --epochs 100000 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --patch-size 61 \
  --variable-min-fraction 0.65 \
  --resume models/test_runs/exp2_variable/best.pt
```

The 30-minute duration belongs in the shell command, not the trainer. The high `--epochs` value lets `timeout 30m` define the wall-clock test length. `latest.pt`, `metrics.jsonl`, `history.jsonl`, and `history_summary.json` are saved under `data/checkpoints/...` on PSCRATCH. `best.pt` and its small `best_metadata.json` are saved under `models/...` only when the selected validation metric improves. `best.pt` is also a full resumable checkpoint, so training can continue from `models/.../best.pt` if scratch checkpoints are lost. Resume checks refuse to continue if core metaparameters do not match the checkpoint, while allowing storage paths to change.

Longer-term version:

```text
wavelength-aware patch embeddings
patch-level wavelength center / range features
instrument coverage mask
```

## Experiment 3: Robustness To Missing Or Corrupted Data

Purpose: make redshift prediction and spectrum reconstruction reliable when spectra are incomplete, damaged, or intentionally tampered with.

Tampering modes to test:

- random pixel dropout
- contiguous wavelength band removal
- camera removal: B-only, R-only, Z-only, B+R, R+Z
- emission-line masking
- absorption-line masking
- 4000A break masking
- additive Gaussian noise
- flux scaling errors
- local spike injection
- wavelength-window truncation
- inverse-variance corruption

Training options:

- train with random masking only
- train with domain-informed masking
- train with curriculum corruption
- train with camera-dropout augmentation
- train with adversarial or hard-example corruption

Evaluation should report:

- redshift MAE/RMSE/bias by corruption type
- failure rate for catastrophic redshift errors
- performance by true redshift bin
- performance by `SPECTYPE` = GALAXY vs QSO
- reconstruction MAE/RMSE on tampered regions
- reconstruction MAE/RMSE on ordinary random masks

## Experiment 4: Data Representation Choices

Compare these input forms:

1. normalized flux only
2. normalized flux + log inverse variance
3. normalized flux + validity mask
4. flux + ivar + validity mask
5. wavelength-aware patch features

Decision rule:

- keep extra channels only if they improve redshift metrics, reconstruction metrics, or robustness without making training unstable
- prefer simple flux-only if performance is similar

## Experiment 5: Scaling

Scale along three axes:

- data size: smoke, 1k HEALPix, 10k HEALPix, full clean DR1
- model size: 26M, 87M, larger if data supports it
- training time: short sanity run, convergence run, long production run

Track:

- steps/sec
- GPU utilization
- dataloader wait time
- validation MAE/RMSE/bias
- reconstruction MAE/RMSE
- memory use

If FITS streaming bottlenecks training, materialize shards or save tokenized spectra.

## Splits

Use HEALPix-level splits:

```text
train: majority of HEALPix files
val: held-out HEALPix files
test: separate held-out HEALPix files
```

Do not split by row. Duplicate or nearby observations can leak otherwise.

Keep a separate OOD set for later:

- SDSS spectra resampled to DESI-like format
- possibly other non-DESI instruments

## Metrics

Primary:

- redshift MAE
- redshift RMSE
- redshift bias
- masked flux MAE
- masked flux RMSE

Secondary:

- normalized redshift error: `abs(z_pred - z_true) / (1 + z_true)`
- catastrophic outlier fraction
- metrics by redshift bin
- metrics by GALAXY/QSO
- metrics by corruption type
- reconstruction metrics by wavelength region

## Near-Term Checklist

- [x] Finish full `data/preprocessed/clean_rows.jsonl`
- [x] Create and lock the HEALPix-level split manifest
- [ ] Build a DR1 clean-row dataset loader
- [ ] Port final-project `SpectraTransformerWithRedshiftToken`
- [ ] Train smoke model on a tiny clean DR1 subset
- [ ] Train fixed-length baseline on clean DR1
- [ ] Verify both redshift and masked reconstruction losses decrease
- [x] Add HEALPix-level split utility
- [ ] Add variable-length batching and padding masks
- [ ] Replace fixed positional embeddings
- [ ] Add corruption/tampering evaluation suite
- [ ] Decide whether to materialize spectra or keep streaming from FITS

## Open Questions

- Should we require `OBJTYPE == TGT`, or is `SPECTYPE in {GALAXY, QSO}` plus `ZCAT_PRIMARY` enough?
- Should QSOs and galaxies share one model head, or do we need type-aware analysis only?
- Does inverse variance help redshift prediction or only reconstruction?
- How much of the spectrum can be missing before predictions become unreliable?
- Should tampering robustness be trained as augmentation or only evaluated as stress tests?
