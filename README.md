# Redshift And Spectrum Reconstruction Foundation Model

This project trains a spectra-only foundation model for **DESI redshift prediction and masked spectrum reconstruction**. The current data target is **DESI DR1**, specifically the `iron` HEALPix coadds:

```text
/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/healpix
```

The model scope is intentionally narrow:

- input: DESI spectra
- targets: redshift `z` and reconstructed flux in masked spectral regions
- no stars in the training set
- no images, photometry, magnitudes, or other modalities

## Setup

Use the existing venv Python environment:

```bash
$HOME/.conda/envs/venv/bin/python --version
```

If you want a local link to the DR1 spectra tree:

```bash
ln -s /global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/healpix data/desi_dr1_healpix
```

In this workspace, `data` is a symlink to scratch storage:

```text
data -> /pscratch/sd/c/colbyd/data
```

Generated manifests, caches, and optional materialized spectra should live under `data/`, not in the git-tracked source tree.

Current and planned experiment status, measured preprocessing/materialization results, and result locations are recorded in [`experiment_log.md`](experiment_log.md).

## New Agent Start Here

Read these files in order before changing a training workflow:

1. This README for data rules and the current pipeline.
2. [`experiments/README.md`](experiments/README.md) for experiment isolation.
3. [`scripts/README.md`](scripts/README.md) for runnable script roles.
4. The specific `experiments/<id>/README.md` and JSON arm config for the run being discussed.

Non-negotiable data rules:

- Train only from the fixed manifest's `train` rows and validate only on its `validation` rows.
- Never use manifest `benchmark` rows for training, validation, normalization fitting, model selection, or augmentation design.
- Keep SDSS strictly OOD: it is benchmark-only and never informs training choices.
- Do not modify an existing experiment's config or shared code in a way that changes a submitted/running experiment. Create a new experiment ID for a new scientific question.
- Treat the full-spectrum/no-transform control as its own reusable experiment; subsequent methods are separate experiments that compare to its fixed results.

Before submitting an experiment, verify the materialized input, prepared SDSS bundle, checkpoint/model output paths, Slurm account, walltime, and benchmark cap. Monitor logs and `history_summary.json`; select the run-best checkpoint by fixed validation `z_sigma_nmad`, then report both DESI benchmark and SDSS OOD results.

## Project Layout

The runnable workflow map is in [`scripts/README.md`](scripts/README.md). The important boundary is:

```text
data/preprocessed/clean_rows.jsonl        -> cleaning output only
data/preprocessed/split_manifest_v1.jsonl -> only allowed training/validation input
data/materialized_training_v1/            -> future train/validation-only fast loader input
data/archived_pre_split_do_not_use/
  materialized_legacy_mixed_splits/        -> legacy cache; contains benchmark rows, never train
models/                                    -> durable best checkpoint + metadata only
experiments/                               -> isolated experiment definitions and arm configs
```

## Preprocess DR1

The preprocessing script is:

```text
scripts/preprocess.py
```

It follows the same basic cleaning that `redshifty` applied at training time, then adds the project-specific cuts we discussed.

`redshifty`-style cuts:

```text
ZWARN == 0
COADD_FIBERSTATUS == 0
nonzero B/R/Z flux
```

Additional project cuts:

```text
ZCAT_PRIMARY == True
SPECTYPE in {GALAXY, QSO}
OBJTYPE == TGT
```

Run the full preprocessing pass:

```bash
$HOME/.conda/envs/venv/bin/python scripts/preprocess.py \
  --num-workers 1
```

Default outputs:

```text
data/preprocessed/clean_rows.jsonl
data/preprocessed/summary.json
data/preprocessed/allowed_targets_galaxy_qso_primary.npz
```

The `allowed_targets_galaxy_qso_primary.npz` file caches the expensive `zall-pix-iron.fits` pass. Building it took about 1-2 minutes in a smoke test; reusing it took about 2 seconds.

## Fixed Data Split

`data/preprocessed/split_manifest_v1.jsonl` is the source of truth for which clean rows may be used by each stage. It is a HEALPix-level split, so every row in a coadd stays together and later additions cannot reshuffle an existing row.

```bash
$HOME/.conda/envs/venv/bin/python scripts/build_split_manifest.py
```

The generated DR1 v1 split contains 13,403,355 train spectra, 1,615,938 validation spectra, and 814,866 benchmark-only spectra. Training scripts accept only `train` and `validation`; they reject or skip `benchmark` before reading its FITS file. The benchmark split is reserved for benchmarking code and must never be used for training, validation, model selection, normalization fitting, or augmentation design.

Do not regenerate this artifact with different fractions or `--overwrite` once experiments have started. Its accompanying `split_manifest_v1_summary.json` records the split version, assignment method, and exact counts.

Smoke test:

```bash
$HOME/.conda/envs/venv/bin/python scripts/preprocess.py \
  --max-healpix 1 \
  --out-dir data/preprocessed_smoke
```

The smoke test produced:

```text
files seen: 1
files kept: 1
clean rows: 21
```

Use a prebuilt allowed-target cache:

```bash
$HOME/.conda/envs/venv/bin/python scripts/preprocess.py \
  --allowed-cache data/preprocessed/allowed_targets_galaxy_qso_primary.npz \
  --num-workers 1
```

Timing test for the parallel scan path, which was slower in the first 4-worker test:

```bash
/usr/bin/time -v $HOME/.conda/envs/venv/bin/python scripts/preprocess.py \
  --max-healpix 100 \
  --out-dir data/preprocessed_timing_100_workers4 \
  --allowed-cache data/preprocessed_timing_100/allowed_targets_galaxy_qso_primary.npz \
  --num-workers 4
```

Detached full preprocessing run:

```bash
sbatch -A <actual_account> scripts/preprocess_full.slurm
```

Monitor it with:

```bash
squeue -u $USER
tail -f logs/preprocess-full-<jobid>.out
```

The Slurm script requests one task with 4 CPUs, not a full CPU node. The cached scan is I/O-bound and `--num-workers 1` was faster than 4 workers, so requesting hundreds of CPUs wastes allocation. The walltime is padded for the full tree, CFS variability, and startup overhead.

If the job is interrupted, check:

```bash
cat logs/preprocess-full-<jobid>.err
sed -n '1,220p' data/preprocessed/progress.json
```

`progress.json` is updated every 25 processed HEALPix files and records files seen, files kept, clean rows, cumulative cut counts, and elapsed seconds. The Slurm script runs Python unbuffered so progress should also appear in the `.out` log while the job is running.

Preprocessing is resumable by default. During the run, `scripts/preprocess.py` appends to:

```text
data/preprocessed/clean_rows.jsonl
data/preprocessed/processed_files.jsonl
data/preprocessed/progress.json
```

If Slurm kills the job for time, resubmit the same command. The next run reads `processed_files.jsonl`, rebuilds `clean_rows.jsonl` from completed HEALPix records, skips those files, and continues. To intentionally start over, pass `--no-resume`.

Optional materialization:

```bash
$HOME/.conda/envs/venv/bin/python scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir data/materialized_training_v1 \
  --shard-size 4096
```

Subset materialization:

```bash
$HOME/.conda/envs/venv/bin/python scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir data/materialized_smoke \
  --max-spectra 10000 \
  --shard-size 2048
```

By default, preprocessing saves a **clean row index**, not transformed spectra. That is the preferred first step because it avoids making a huge duplicate copy of DR1. If training is bottlenecked by repeated FITS reads, B/R/Z stitching, or per-row cleaning, use `scripts/materialize_shards.py` to write larger contiguous PyTorch `.pt` shards under `data/materialized_training_v1/`. It defaults to train and validation only; benchmark rows are never materialized there.

Materialized shard contents:

```text
flux:       float32 [N, L]
ivar:       float32 [N, L]
valid:      uint8 [N, L]
z:          float32 [N]
targetid:   int64 [N]
row:        int32 [N]
healpix:    int32 [N]
spectype:   int8 [N]
wavelength: float32 [L]
records:    source HEALPix metadata
```

Materialization is resumable by default through:

```text
data/materialized_training_v1/materialized_processed_files.jsonl
data/materialized_training_v1/shards_manifest.jsonl
data/materialized_training_v1/materialize_progress.json
```

Resubmit the same command to continue. Use `--no-resume` to intentionally start over.

## How To Train

Training should consume:

```text
data/preprocessed/split_manifest_v1.jsonl
```

Each JSONL record points to one clean HEALPix coadd/redrock pair and contains the rows, `TARGETID`s, redshifts, spectypes, and fixed `split` assignment to use. A training dataset should:

1. Open each `coadd-*.fits` and `redrock-*.fits` path from the manifest.
2. Read only the listed clean rows.
3. Stitch the B/R/Z camera bands using inverse-variance weighted overlaps.
4. Feed spectra into the model with redshift `z` and masked flux reconstruction as joint targets.

Current status: preprocessing is implemented, and a first short-run trainer consumes the fixed split manifest. A more scalable production loader still needs to be built.

The short-run trainer is:

```text
scripts/train_experiments.py
```

Recommended training direction:

```text
split_manifest_v1.jsonl
  -> clean-row DR1 Dataset
  -> raw spectrum patches or spectrum tokenizer
  -> transformer
  -> redshift prediction + masked spectrum reconstruction
```

Training should optimize a joint objective:

```text
loss = redshift_loss_weight * redshift_loss
     + reconstruction_loss_weight * masked_flux_reconstruction_loss
     + classification_loss_weight * GALAXY/QSO_cross_entropy
```

The reconstruction loss scores intentionally masked, real, non-padded spectral pixels against the original normalized flux. By default the mask is one contiguous wavelength span covering `--mask-prob` of the valid spectrum (15% by default), rather than isolated pixels. This makes reconstruction training match missing wavelength-region inference; fragmented valid coverage uses the fewest valid spans needed. Pass `--mask-mode independent` only for a pixel-denoising ablation. Random transforms corrupt only the model input; they are never used as reconstruction targets.

All active models classify the redshift-token representation as `GALAXY` (label `0`) or `QSO` (label `1`) from DESI `SPECTYPE`. New materializations also retain four separate, multi-label DESI target-selection flags: `BGS`, `LRG`, `ELG`, and `QSO`. Those flags are not mutually exclusive and are selection metadata, not spectroscopic truth. Both heads have weight `0.25` by default. Stars are excluded from the current training materialization, so `SPECTYPE` remains deliberately two-class. Training applies the reconstruction mask to every head; validation evaluates redshift and classification on clean full spectra, and evaluates reconstruction with the same fixed contiguous mask at every epoch.

If training is too slow from FITS, rerun preprocessing with `--materialize` or save tokenizer outputs as shards. For full DR1, a clean-row index is the safer first move; materialized spectra/tokens are a performance optimization, not the source of truth.

### Active Experiment Submission

New training must use one isolated experiment directory per hypothesis. The
reusable control and crop/transform treatment are already separate active
experiments:

```bash
sbatch -A m5374_g experiments/control_full_spectrum_v1/train.slurm
sbatch -A m5374_g experiments/crop_transform_v1/train.slurm
```

Prepare the shared fixed 10,000-spectrum SDSS bundle once before either GPU
job (or use a Slurm dependency):

```bash
SDSS_JOB=$(sbatch -A m5374 scripts/prepare_sdss_ood_cpu.slurm | awk '{print $4}')
sbatch -A m5374_g --dependency=afterok:${SDSS_JOB} experiments/control_full_spectrum_v1/train.slurm
sbatch -A m5374_g --dependency=afterok:${SDSS_JOB} experiments/crop_transform_v1/train.slurm
```

Each active launcher uses its own immutable `config.json`, checkpoints,
durable model directory, and the shared 10k DESI plus 10k SDSS benchmark
protocol. See [`experiments/README.md`](experiments/README.md) to create the
next experiment.

### DEPRECATED: FITS-Streaming Test Runs

The following commands describe the pre-experiment-layout FITS trainer. They
are retained only as historical context; do not use them for new work.

Do not hard-code the run duration in Python. Let the shell or scheduler own the wall-clock limit.

Current training status:

| Run | Ready now | Difference |
| --- | --- | --- |
| `train_experiments.py --experiment fixed` | Yes | Fixed full-spectrum input; closest to the final-project baseline. |
| `train_experiments.py --experiment variable` | Yes | New contiguous wavelength-window crop every training epoch plus conservative flux gain, tilt, offset, and noise transforms; tests variable input lengths. |
| `train_materialized_streaming.py` | Not yet | Requires `data/materialized_training_v1/`, which has not been created. It is intended to reduce FITS I/O. |

The first two are short-run experiment harnesses, not full-production trainers: `--max-examples` takes the first eligible spectra encountered in manifest order. They are appropriate for validating model behavior, checkpointing, and GPU throughput. A full training run should use the split-clean materialized path after it is built.

Experiment 1, fixed-length final-project-style baseline:

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

Experiment 2, input-length-agnostic baseline with random wavelength-window crops:

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

The variable experiment applies random transforms to training crops by default: gain up to 10%, linear calibration tilt up to 10% across a crop, normalized-flux offset up to 0.05, and Gaussian noise with standard deviation up to 0.02. The transformed crop is the model input, while reconstruction is scored against the corresponding original, untransformed crop. Validation always uses full-length, untransformed spectra so control and augmented runs are measured on identical inputs. Use `--no-random-transforms` for a crop-only ablation; the four `--transform-*` arguments control the respective magnitudes.

Variable-length patch tokens receive both relative sinusoidal positions and an embedding of their absolute log-wavelength center. This prevents every crop from looking as though it starts at the same wavelength and preserves information needed to identify redshifted spectral lines.

### DEPRECATED: Combined Crop/Transform Ablation

The first bounded GPU comparison uses the same 3,422,782-parameter variable/wavelength-aware model for both jobs. The control trains on full spectra with transforms disabled; the treatment trains on 65%-100% crops with transforms enabled. Both load 65,536 examples, train for eight epochs with batch size 128 and seed 42, and use identical full-length untransformed validation inputs and contiguous span masks. This model is about 7.6 times smaller than the 26,007,614-parameter smaller final-project checkpoint.

```bash
sbatch -A m5374_g scripts/deprecated/train_crop_transform_ablation.slurm
```

This submits a two-task array on one shared 40 GB GPU per task with a three-hour limit. The tasks run one at a time to avoid duplicating heavy FITS reads against the shared DESI filesystem. Each task exits when training finishes and is charged for actual runtime. Resubmitting the same command resumes either task from its PSCRATCH `latest.pt` when present.

Results are written to:

```text
data/checkpoints/ablation_crop_transforms_v2/control_full_no_transforms/
data/checkpoints/ablation_crop_transforms_v2/crop65_transforms/
models/ablation_crop_transforms_v2/control_full_no_transforms/best.pt
models/ablation_crop_transforms_v2/crop65_transforms/best.pt
```

### SDSS OOD Evaluation

Perlmutter already has local eBOSS plate spectra under `/global/cfs/cdirs/sdss/eBOSS/testFiles/` and SDSS redshift-result files under `/global/cfs/cdirs/sdss/data/sdss/dr16/eboss/`. Build one fixed labeled, DESI-grid OOD bundle before submitting or resubmitting the ablation:

```bash
$HOME/.conda/envs/venv/bin/python scripts/prepare_sdss_ood.py \
  --redshift-root /global/cfs/cdirs/sdss/data/sdss/dr16/eboss/spectro/redux/v5_13_0 \
  --max-plates 2 \
  --max-spectra 1000
```

The array job creates `data/ood/sdss_eboss_ood.pt` before its first training task, then reuses that fixed bundle for the second task and evaluates each run-best checkpoint before its task exits. It writes `redshift_pred_vs_actual.png`, three contiguous-gap reconstruction figures, and `metrics.json` to `models/ablation_crop_transforms_v2/<variant>/sdss_ood/`. Set `OOD_BUNDLE=/path/to/bundle` or `SDSS_REDSHIFT_ROOT=/path/to/spzall/tree` when submitting to override the defaults.

For the long 1M run, prepare this shared 10,000-spectrum SDSS OOD bundle on a CPU node before submitting GPU training:

```bash
sbatch -A m5374 scripts/prepare_sdss_ood_cpu.slurm
```

The GPU job still verifies the bundle exists and can build it as a fallback, but the CPU preparation job avoids reserving GPU time for FITS reading, label matching, and resampling.

Evaluation caps are deliberately fixed at 10,000 labeled SDSS OOD spectra and 10,000 spectra from the manifest's reserved `benchmark` split. After each arm, the single `scripts/run_benchmarks.py` entrypoint feeds clean full spectra to every output head, then writes a redshift predicted-vs-actual scatter plot, three full-spectrum reconstruction plots, prediction-level `.npz` tables, and metrics for both `sdss_ood/` and `desi_benchmark/`. Redshift reporting uses `dz_norm=(z_pred-z_true)/(1+z_true)`: physical and normalized MAE, bias, `sigma_NMAD=1.4826*median(|dz_norm-median(dz_norm)|)`, catastrophic fractions at `|dz_norm|>0.0033` and `>0.05`, and R². The DESI benchmark additionally reports the same table within BGS/LRG/ELG/QSO target selections. Reconstruction MSE and classification are secondary diagnostics, not substitutes for redshift accuracy.

The standard SDSS OOD bundle is resampled to DESI's wavelength coverage, making it a shared-coverage domain-shift benchmark. `scripts/prepare_sdss_ood_native_cpu.slurm` instead preserves each eBOSS spectrum's native wavelength grid and pads only for batching; the dynamic model returns a reconstruction on precisely that same grid. Treat its beyond-DESI wavelength region as an extrapolation stress test and write its results to a separate output directory rather than comparing it directly with the shared-coverage SDSS score.

### DEPRECATED: Combined Long 1M-Spectrum Comparison

`scripts/deprecated/train_materialized_crop_transform_ablation.slurm` is the historical combined-array counterpart of the short FITS-streaming ablation. It is superseded by the two independent experiment launchers above; do not submit it for new work.

```bash
SDSS_JOB=$(sbatch -A m5374 scripts/prepare_sdss_ood_cpu.slurm | awk '{print $4}')
sbatch -A m5374_g --dependency=afterok:${SDSS_JOB} scripts/deprecated/train_materialized_crop_transform_ablation.slurm
```

Each arm has a 500-epoch cap and processes up to 200,000 training spectra per epoch. Both arms can run simultaneously, each on one GPU, with a 10-hour walltime; training is capped at 9 hours 30 minutes to reserve the final 30 minutes for both 10k benchmarks. `latest.pt` is saved after every epoch, and resubmitting the same command resumes each arm. It stops an arm early after 12 consecutive epochs without a validation `z_sigma_nmad` decrease of at least `0.0001`; both values are configurable through `EARLY_STOPPING_PATIENCE` and `EARLY_STOPPING_MIN_DELTA` at submission. The trainer writes validation prediction tables each epoch and reports compact redshift metrics plus epoch-level data time, GPU compute time, and spectra/sec.

The high `--epochs` value is intentional for these tests: `timeout 30m` decides when the run stops. The script saves resumable checkpoints to `data/checkpoints/...` after each completed epoch, and saves only the run-best checkpoint artifact to `models/...`.

Both test runs are resumable. Each checkpoint directory contains:

```text
config.json
metrics.jsonl
history.jsonl
history_summary.json
latest.pt
```

Each model directory contains:

```text
best.pt
best_metadata.json
```

`latest.pt`, `metrics.jsonl`, `history.jsonl`, and `history_summary.json` stay under `data/checkpoints/...` on PSCRATCH. `best.pt` stores the same full resumable state, but only for the best validation checkpoint so far, and is the only model checkpoint copied to `models/...` in home storage. On resume, the trainer checks the current command against the checkpoint metaparameters and refuses to continue if core settings do not match. Storage paths are allowed to change, so `models/.../best.pt` can be used if scratch checkpoints disappear.

Resume Experiment 1:

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

Resume Experiment 1 if scratch checkpoints are gone:

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

Resume Experiment 2:

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

Resume Experiment 2 if scratch checkpoints are gone:

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

## Inference

Inference should use the same spectrum preparation as training:

1. Read a DESI coadd row.
2. Apply the same quality assumptions used during preprocessing.
3. Stitch B/R/Z bands the same way.
4. Run the trained model to predict `z` and reconstruct masked or missing spectral regions.

Current status: a final inference CLI is not implemented in this root project yet. Once the training loader/model entrypoint exists, inference should take either:

```text
coadd path + redrock path + row index
```

or a preprocessed/materialized shard from:

```text
data/preprocessed/
```

Expected inference outputs:

```text
predicted redshift z
reconstructed flux for masked/missing spectral pixels
optional uncertainty or diagnostic metrics
```

## Exploration Notes

Notebook:

```text
notebooks/explore.ipynb
```

The notebook checks the DR1 HEALPix file layout, verifies the relevant columns, fixes the spectrum plotting axis issue, and shows why `ZCAT_PRIMARY` requires the zcatalog rather than the per-HEALPix redrock file.

Important finding from one sample HEALPix file:

```text
redshifty current cuts: 1883 / 2617 rows
stars surviving those cuts: 63
after GALAXY/QSO cut: 1820 / 2617 rows
```

So the no-stars cut matters.
