#!/bin/bash
# Continue Control v2 from epoch 64 through epoch 123, then benchmark it.
#SBATCH -C gpu&hbm40g
#SBATCH -A m5374_g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J control-v2-match123
#SBATCH -o logs/control-v2-match123-%j.out
#SBATCH -e logs/control-v2-match123-%j.err

set -euo pipefail
cd /global/u1/c/colbyd/redshift-foundation
EXPERIMENT_ID="control_full_spectrum_v2_matched_123_v1"
CONFIG="experiments/${EXPERIMENT_ID}/config.json"
SOURCE_CHECKPOINT="data/checkpoints/experiments/control_full_spectrum_v2/main/latest.pt"
CHECKPOINT_DIR="data/checkpoints/experiments/${EXPERIMENT_ID}/main"
MODEL_DIR="models/experiments/${EXPERIMENT_ID}/main"
OOD_BUNDLE="data/ood/sdss_eboss_ood.pt"
TRAINING_TIME_LIMIT="${TRAINING_TIME_LIMIT:-5400s}"
BENCHMARK_TIME_LIMIT="${BENCHMARK_TIME_LIMIT:-1800s}"

if [ ! -s "$SOURCE_CHECKPOINT" ]; then echo "ERROR: missing source checkpoint: $SOURCE_CHECKPOINT"; exit 1; fi
if [ ! -s "$OOD_BUNDLE" ]; then echo "ERROR: missing SDSS OOD bundle: $OOD_BUNDLE"; exit 1; fi
mkdir -p logs "$CHECKPOINT_DIR" "$MODEL_DIR"

timeout --signal=TERM --kill-after=60s "$TRAINING_TIME_LIMIT" srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_materialized_experiment.py --config "$CONFIG" --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR" --resume "$SOURCE_CHECKPOINT"
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/run_benchmarks.py --model "$MODEL_DIR/best.pt" --output-dir "$MODEL_DIR" --sdss-bundle "$OOD_BUNDLE" --max-spectra 10000 --examples 3 --batch-size 256
