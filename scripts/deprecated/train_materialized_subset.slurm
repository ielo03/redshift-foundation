#!/bin/bash
# Submit with, for example:
#   sbatch -A <gpu_account> scripts/train_materialized_subset.slurm
#
# Useful overrides:
#   MAX_TRAIN_SPECTRA_PER_EPOCH=500000 EPOCHS=20 BATCH_SIZE=32 D_MODEL=512 NUM_LAYERS=8 \
#     sbatch -A <gpu_account> scripts/train_materialized_subset.slurm
#
# This streams materialize_shards.py output one shard at a time. It caps how
# many training spectra are consumed per epoch without loading the subset into
# RAM all at once.

#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J redshift-train-subset
#SBATCH -o logs/train-subset-%j.out
#SBATCH -e logs/train-subset-%j.err

set -euo pipefail

cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs data/training_runs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}-${SLURM_JOB_ID:-manual}}
mkdir -p "$MPLCONFIGDIR"

INPUT_DIR="${INPUT_DIR:-data/materialized_training_v1}"
MAX_TRAIN_SPECTRA_PER_EPOCH="${MAX_TRAIN_SPECTRA_PER_EPOCH:-100000}"
MAX_VAL_SPECTRA="${MAX_VAL_SPECTRA:-20000}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-5e-5}"
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NHEAD="${NHEAD:-8}"
PATCH_SIZE="${PATCH_SIZE:-61}"
MASK_PROB="${MASK_PROB:-0.15}"
ALPHA="${ALPHA:-1.25}"
RECONSTRUCTION_WEIGHT="${RECONSTRUCTION_WEIGHT:-1.0}"
TAIL_POWER="${TAIL_POWER:-1.0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-materialized_stream_${MAX_TRAIN_SPECTRA_PER_EPOCH}_${SLURM_JOB_ID:-manual}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/checkpoints/${RUN_NAME}}"
MODEL_DIR="${MODEL_DIR:-models/${RUN_NAME}}"

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
echo "[slurm] node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "[slurm] hostname=$(hostname)"
echo "[slurm] start=$(date -Is)"
echo "[slurm] workdir=$(pwd)"
echo "[slurm] python=$HOME/.conda/envs/venv/bin/python"
echo "[train] input=$INPUT_DIR"
echo "[train] checkpoints=$CHECKPOINT_DIR"
echo "[train] best_model=$MODEL_DIR/best.pt"
echo "[train] seed=$SEED"
echo "[train] max_train_spectra_per_epoch=$MAX_TRAIN_SPECTRA_PER_EPOCH max_val_spectra=$MAX_VAL_SPECTRA"
echo "[train] split_manifest=data/preprocessed/split_manifest_v1.jsonl (fixed train/validation; benchmark excluded)"
echo "[train] epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[train] d_model=$D_MODEL layers=$NUM_LAYERS nhead=$NHEAD patch_size=$PATCH_SIZE"

if [ ! -e "$INPUT_DIR" ]; then
  echo "ERROR: INPUT_DIR does not exist yet: $INPUT_DIR"
  echo "Wait for materialization to write data/materialized_training_v1, or set INPUT_DIR."
  exit 1
fi

/usr/bin/time -v "$HOME/.conda/envs/venv/bin/python" -u \
  scripts/train_materialized_streaming.py \
  --input "$INPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-dir "$MODEL_DIR" \
  --max-train-spectra-per-epoch "$MAX_TRAIN_SPECTRA_PER_EPOCH" \
  --max-val-spectra "$MAX_VAL_SPECTRA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --nhead "$NHEAD" \
  --patch-size "$PATCH_SIZE" \
  --mask-prob "$MASK_PROB" \
  --alpha "$ALPHA" \
  --reconstruction-weight "$RECONSTRUCTION_WEIGHT" \
  --tail-power "$TAIL_POWER" \
  --seed "$SEED"

echo "[slurm] end=$(date -Is)"
