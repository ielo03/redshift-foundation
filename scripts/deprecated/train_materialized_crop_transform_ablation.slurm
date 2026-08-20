#!/bin/bash
# Long, fair 1M-spectrum crop/transform comparison.  The high epoch cap is
# intentional: Slurm walltime ends each allocation and resubmission resumes.
# Submit: sbatch -A m5374_g scripts/train_materialized_crop_transform_ablation.slurm

#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The Perlmutter shared GPU queue requires 32 CPU cores per requested GPU.
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH --array=0-1%2
#SBATCH -J redshift-1m-ablation
#SBATCH -o logs/materialized-ablation-%A_%a.out
#SBATCH -e logs/materialized-ablation-%A_%a.err

set -euo pipefail
cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs data/checkpoints/materialized_crop_transform_v1 models/materialized_crop_transform_v1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}-${SLURM_JOB_ID:-manual}}
mkdir -p "$MPLCONFIGDIR"

INPUT_DIR="${INPUT_DIR:-data/materialized_training_v1_1m}"
BATCH_SIZE="${BATCH_SIZE:-3072}"
EPOCHS="${EPOCHS:-500}"
MAX_TRAIN_SPECTRA_PER_EPOCH="${MAX_TRAIN_SPECTRA_PER_EPOCH:-200000}"
MAX_VAL_SPECTRA="${MAX_VAL_SPECTRA:-20000}"
# Reserve the final 30 minutes of the 10-hour allocation for both fixed 10k
# benchmarks. timeout exit status 124 is expected and is treated as a clean
# training stop; latest.pt was saved at the preceding completed epoch.
TRAINING_TIME_LIMIT="${TRAINING_TIME_LIMIT:-34200s}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
OOD_BUNDLE="${OOD_BUNDLE:-data/ood/sdss_eboss_ood.pt}"
SDSS_REDSHIFT_ROOT="${SDSS_REDSHIFT_ROOT:-/global/cfs/cdirs/sdss/data/sdss/dr16/eboss/spectro/redux/v5_13_0}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) VARIANT="control_full_no_transforms"; AUGMENT_ARGS=(--variable-min-fraction 1.0 --no-random-transforms) ;;
  1) VARIANT="crop65_transforms"; AUGMENT_ARGS=(--variable-min-fraction 0.65 --random-transforms) ;;
  *) echo "ERROR: invalid array task"; exit 2 ;;
esac

CHECKPOINT_DIR="data/checkpoints/materialized_crop_transform_v1/${VARIANT}"
MODEL_DIR="models/materialized_crop_transform_v1/${VARIANT}"
RESUME_ARGS=()
if [ -s "${CHECKPOINT_DIR}/latest.pt" ]; then RESUME_ARGS=(--resume "${CHECKPOINT_DIR}/latest.pt"); fi

if [ ! -d "$INPUT_DIR/shards" ]; then echo "ERROR: missing materialized shards: $INPUT_DIR"; exit 1; fi
if [ ! -s "$OOD_BUNDLE" ]; then
  echo "ERROR: missing SDSS OOD bundle: $OOD_BUNDLE"
  echo "Submit scripts/prepare_sdss_ood_cpu.slurm first, then submit this job with an afterok dependency."
  exit 1
fi

echo "[setup] variant=$VARIANT batch_size=$BATCH_SIZE epochs=$EPOCHS input=$INPUT_DIR training_time_limit=$TRAINING_TIME_LIMIT"
set +e
timeout --signal=TERM --kill-after=2m "$TRAINING_TIME_LIMIT" \
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/train_materialized_streaming.py \
  --input "$INPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR" \
  --epochs "$EPOCHS" --max-train-spectra-per-epoch "$MAX_TRAIN_SPECTRA_PER_EPOCH" --max-val-spectra "$MAX_VAL_SPECTRA" \
  --batch-size "$BATCH_SIZE" --lr 3e-4 --d-model 256 --num-layers 4 --nhead 8 --patch-size 61 \
  --mask-prob 0.15 --mask-mode contiguous --alpha 1.25 --reconstruction-weight 1.0 --seed 42 \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
  "${AUGMENT_ARGS[@]}" "${RESUME_ARGS[@]}"
TRAIN_STATUS=$?
set -e
if [ "$TRAIN_STATUS" -ne 0 ] && [ "$TRAIN_STATUS" -ne 124 ]; then
  echo "ERROR: training failed with status $TRAIN_STATUS"
  exit "$TRAIN_STATUS"
fi
if [ "$TRAIN_STATUS" -eq 124 ]; then
  echo "[train] stopped at the configured time limit; running benchmarks from the best checkpoint"
fi

srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_benchmarks.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl --sdss-bundle "$OOD_BUNDLE" \
  --model "$MODEL_DIR/best.pt" --output-dir "$MODEL_DIR" \
  --max-spectra 10000 --examples 3 --batch-size 256 --mask-prob 0.15 --seed 314159
