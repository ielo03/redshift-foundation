#!/bin/bash
# Two-hour, smaller-control saturation run followed by a 30-minute benchmark.
#SBATCH -C gpu&hbm40g
#SBATCH -A m5374_g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 02:30:00
#SBATCH -J control-full-v3
#SBATCH -o logs/control-full-v3-%j.out
#SBATCH -e logs/control-full-v3-%j.err

set -euo pipefail
cd /global/u1/c/colbyd/redshift-foundation
EXPERIMENT_ID="control_full_spectrum_v3"
CONFIG="experiments/${EXPERIMENT_ID}/config.json"
CHECKPOINT_DIR="data/checkpoints/experiments/${EXPERIMENT_ID}/main"
MODEL_DIR="models/experiments/${EXPERIMENT_ID}/main"
# Reserve 90 seconds for the trainer to flush its final checkpoint/history.
TRAINING_TIME_LIMIT="${TRAINING_TIME_LIMIT:-7110s}"
RESUME_ARGS=()
if [ -s "${CHECKPOINT_DIR}/latest.pt" ]; then RESUME_ARGS=(--resume "${CHECKPOINT_DIR}/latest.pt"); fi
mkdir -p logs "$CHECKPOINT_DIR" "$MODEL_DIR"
set +e
timeout --signal=TERM --kill-after=60s "$TRAINING_TIME_LIMIT" srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_materialized_experiment.py --config "$CONFIG" --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR" "${RESUME_ARGS[@]}"
STATUS=$?
set -e
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 124 ]; then exit "$STATUS"; fi

# Keep evaluation bounded independently from the two-hour training budget.
BENCHMARK_TIME_LIMIT="${BENCHMARK_TIME_LIMIT:-1800s}"
set +e
timeout --signal=TERM --kill-after=60s "$BENCHMARK_TIME_LIMIT" srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_benchmarks.py --model "$MODEL_DIR/best.pt" --output-dir "$MODEL_DIR" --sdss-bundle data/ood/sdss_eboss_ood.pt --max-spectra 10000 --examples 3 --batch-size 256
STATUS=$?
set -e
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi
