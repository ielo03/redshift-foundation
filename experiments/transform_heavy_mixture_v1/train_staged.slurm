#!/bin/bash
# Exclusive mixture: 20% clean, 15% crop, 65% transforms.
#SBATCH -C gpu&hbm40g
#SBATCH -A m5374_g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J transform-heavy-mix-v1
#SBATCH -o logs/transform-heavy-mix-v1-%j.out
#SBATCH -e logs/transform-heavy-mix-v1-%j.err

set -euo pipefail
cd /global/u1/c/colbyd/redshift-foundation
EXPERIMENT_ID="transform_heavy_mixture_v1"
CHECKPOINT_DIR="data/checkpoints/experiments/${EXPERIMENT_ID}/main"
MODEL_DIR="models/experiments/${EXPERIMENT_ID}/main"
PYTHON_BIN="/global/homes/c/colbyd/.conda/envs/venv/bin/python"
CONTROL_HISTORY="data/checkpoints/experiments/control_full_spectrum_v2/main/history.jsonl"
OOD_BUNDLE="data/ood/sdss_eboss_ood.pt"
mkdir -p logs "$CHECKPOINT_DIR" "$MODEL_DIR"

srun "$PYTHON_BIN" -u scripts/run_materialized_experiment.py --config experiments/${EXPERIMENT_ID}/config_stage1.json --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR"
set +e
"$PYTHON_BIN" -u scripts/assess_stage1.py --candidate-history "$CHECKPOINT_DIR/history.jsonl" --control-history "$CONTROL_HISTORY" --output "$CHECKPOINT_DIR/stage1_decision.json" --stage-epoch 64 --worse-factor 1.5 --trend-window 12 --minimum-improving-trend 0.05
ASSESS_STATUS=$?
set -e
if [ "$ASSESS_STATUS" -eq 2 ]; then echo "[stage1] clearly worse and not improving; stopping"; exit 0; fi
if [ "$ASSESS_STATUS" -ne 0 ]; then exit "$ASSESS_STATUS"; fi

srun "$PYTHON_BIN" -u scripts/run_materialized_experiment.py --config experiments/${EXPERIMENT_ID}/config_stage2.json --checkpoint-dir "$CHECKPOINT_DIR" --model-dir "$MODEL_DIR" --resume "$CHECKPOINT_DIR/latest.pt"
timeout --signal=TERM --kill-after=60s 1800s srun "$PYTHON_BIN" -u scripts/run_benchmarks.py --model "$MODEL_DIR/best.pt" --output-dir "$MODEL_DIR" --sdss-bundle "$OOD_BUNDLE" --max-spectra 10000 --examples 3 --batch-size 256
