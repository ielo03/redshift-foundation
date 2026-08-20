#!/bin/bash
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH -J crop-transform-v1
#SBATCH -o logs/crop-transform-v1-%j.out
#SBATCH -e logs/crop-transform-v1-%j.err

set -euo pipefail
cd /global/homes/c/colbyd/redshift-foundation
EXPERIMENT_ID="crop_transform_v1"
CONFIG="experiments/${EXPERIMENT_ID}/config.json"
CHECKPOINT_DIR="data/checkpoints/experiments/${EXPERIMENT_ID}/main"
MODEL_DIR="models/experiments/${EXPERIMENT_ID}/main"
OOD_BUNDLE="${OOD_BUNDLE:-data/ood/sdss_eboss_ood.pt}"
TRAINING_TIME_LIMIT="${TRAINING_TIME_LIMIT:-34200s}"
RESUME_ARGS=()
if [ -s "${CHECKPOINT_DIR}/latest.pt" ]; then RESUME_ARGS=(--resume "${CHECKPOINT_DIR}/latest.pt"); fi
if [ ! -s "$OOD_BUNDLE" ]; then echo "ERROR: missing SDSS OOD bundle"; exit 1; fi
mkdir -p logs "$CHECKPOINT_DIR" "$MODEL_DIR"
set +e
timeout --signal=TERM --kill-after=2m "$TRAINING_TIME_LIMIT" srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_materialized_experiment.py --config "$CONFIG" --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR" "${RESUME_ARGS[@]}"
STATUS=$?
set -e
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 124 ]; then exit "$STATUS"; fi
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_benchmarks.py --model "$MODEL_DIR/best.pt" --output-dir "$MODEL_DIR" --sdss-bundle "$OOD_BUNDLE" --max-spectra 10000 --examples 3 --batch-size 256
