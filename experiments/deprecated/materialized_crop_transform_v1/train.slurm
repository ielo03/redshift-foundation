#!/bin/bash
# Isolated long-run entrypoint for materialized_crop_transform_v1.

#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH --array=0-1%2
#SBATCH -J mct-v1
#SBATCH -o logs/mct-v1-%A_%a.out
#SBATCH -e logs/mct-v1-%A_%a.err

set -euo pipefail
cd /global/homes/c/colbyd/redshift-foundation
EXPERIMENT_ID="materialized_crop_transform_v1"
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM="control_full_no_transforms" ;;
  1) ARM="crop65_transforms" ;;
  *) exit 2 ;;
esac
CONFIG="experiments/${EXPERIMENT_ID}/configs/${ARM}.json"
CHECKPOINT_DIR="data/checkpoints/experiments/${EXPERIMENT_ID}/${ARM}"
MODEL_DIR="models/experiments/${EXPERIMENT_ID}/${ARM}"
OOD_BUNDLE="${OOD_BUNDLE:-data/ood/sdss_eboss_ood.pt}"
RESUME_ARGS=()
if [ -s "${CHECKPOINT_DIR}/latest.pt" ]; then RESUME_ARGS=(--resume "${CHECKPOINT_DIR}/latest.pt"); fi
mkdir -p logs "$CHECKPOINT_DIR" "$MODEL_DIR"
if [ ! -s "$OOD_BUNDLE" ]; then
  echo "ERROR: missing prepared SDSS bundle: $OOD_BUNDLE"
  exit 1
fi
set +e
timeout --signal=TERM --kill-after=2m 34200s \
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_materialized_experiment.py \
  --config "$CONFIG" --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR" "${RESUME_ARGS[@]}"
STATUS=$?
set -e
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 124 ]; then exit "$STATUS"; fi
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_benchmarks.py \
  --model "$MODEL_DIR/best.pt" --output-dir "$MODEL_DIR" --sdss-bundle "$OOD_BUNDLE" \
  --max-spectra 10000 --examples 3 --batch-size 256 --mask-prob 0.15 --seed 314159
