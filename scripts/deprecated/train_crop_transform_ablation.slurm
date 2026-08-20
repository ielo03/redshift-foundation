#!/bin/bash
# Submit both fair-comparison jobs with:
#   sbatch -A m5374_g scripts/train_crop_transform_ablation.slurm
#
# Array task 0: full spectra, no random transforms.
# Array task 1: 65%-100% crops plus random transforms.

#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 03:00:00
#SBATCH --array=0-1%1
#SBATCH -J redshift-ablation
#SBATCH -o logs/ablation-%A_%a.out
#SBATCH -e logs/ablation-%A_%a.err

set -euo pipefail

cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs data/checkpoints/ablation_crop_transforms_v2 models/ablation_crop_transforms_v2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    VARIANT="control_full_no_transforms"
    AUGMENT_ARGS=(--variable-min-fraction 1.0 --no-random-transforms)
    ;;
  1)
    VARIANT="crop65_transforms"
    AUGMENT_ARGS=(--variable-min-fraction 0.65 --random-transforms)
    ;;
  *)
    echo "ERROR: unexpected array task ${SLURM_ARRAY_TASK_ID}"
    exit 2
    ;;
esac

CHECKPOINT_DIR="data/checkpoints/ablation_crop_transforms_v2/${VARIANT}"
MODEL_DIR="models/ablation_crop_transforms_v2/${VARIANT}"
OOD_BUNDLE="${OOD_BUNDLE:-data/ood/sdss_eboss_ood.pt}"
SDSS_REDSHIFT_ROOT="${SDSS_REDSHIFT_ROOT:-/global/cfs/cdirs/sdss/data/sdss/dr16/eboss/spectro/redux/v5_13_0}"
RESUME_ARGS=()
if [ -s "${CHECKPOINT_DIR}/latest.pt" ]; then
  RESUME_ARGS=(--resume "${CHECKPOINT_DIR}/latest.pt")
fi

echo "[slurm] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} variant=${VARIANT}"
echo "[slurm] node=$(hostname) start=$(date -Is)"
echo "[train] model=d256 layers=4 heads=8 parameters=3422782"
echo "[train] examples=65536 epochs=8 batch_size=128 best_metric=val_mae"
echo "[train] checkpoints=${CHECKPOINT_DIR} best_model=${MODEL_DIR}/best.pt"

# The array runs serially (--array=0-1%1), so task 0 creates this fixed OOD
# dataset once and task 1 reuses the exact same labeled SDSS examples.
if [ ! -s "$OOD_BUNDLE" ]; then
  echo "[ood] preparing fixed SDSS bundle: ${OOD_BUNDLE}"
  mkdir -p "$(dirname "$OOD_BUNDLE")"
  srun "$HOME/.conda/envs/venv/bin/python" -u scripts/prepare_sdss_ood.py \
    --plate-dir data/raw-sdss \
    --redshift-root "$SDSS_REDSHIFT_ROOT" \
    --reference-manifest data/preprocessed/split_manifest_v1.jsonl \
    --output "$OOD_BUNDLE" \
    --max-plates 16 \
    --max-spectra 10000
else
  echo "[ood] reusing fixed SDSS bundle: ${OOD_BUNDLE}"
fi

srun "$HOME/.conda/envs/venv/bin/python" -u scripts/train_experiments.py \
  --experiment variable \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-dir "$MODEL_DIR" \
  --max-examples 65536 \
  --batch-size 128 \
  --epochs 8 \
  --lr 3e-4 \
  --mask-prob 0.15 \
  --redshift-weight 1.25 \
  --reconstruction-weight 1.0 \
  --patch-size 61 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --seed 42 \
  --best-metric val_mae \
  --num-workers 0 \
  "${AUGMENT_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

OOD_DIR="${MODEL_DIR}/sdss_ood"
echo "[ood] evaluating ${MODEL_DIR}/best.pt on ${OOD_BUNDLE}"
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/evaluate_ood_sdss.py \
  --input "$OOD_BUNDLE" \
  --model "${MODEL_DIR}/best.pt" \
  --output-dir "$OOD_DIR" \
  --max-spectra 10000 \
  --examples 3 \
  --mask-prob 0.15 \
  --seed 314159

echo "[slurm] end=$(date -Is)"
