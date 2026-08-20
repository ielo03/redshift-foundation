#!/bin/bash
# Bounded compute-node timing test for FITS materialization and .pt reload.
# Submit with: sbatch -A m5374 scripts/materialize_benchmark.slurm

#SBATCH -C cpu
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -J materialize-bench
#SBATCH -o logs/materialize-bench-%j.out
#SBATCH -e logs/materialize-bench-%j.err

set -euo pipefail

cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs

BENCHMARK_DIR="${BENCHMARK_DIR:-data/materialized_benchmark_1024}"
PYTHON="$HOME/.conda/envs/venv/bin/python"

echo "[slurm] job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "[benchmark] materializing 1024 train/validation spectra to ${BENCHMARK_DIR}"

/usr/bin/time -v "$PYTHON" -u scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir "$BENCHMARK_DIR" \
  --format pt \
  --shard-size 1024 \
  --max-spectra 1024 \
  --progress-interval 1

echo "[benchmark] timing .pt load plus training-equivalent robust normalization"
/usr/bin/time -v "$PYTHON" -u scripts/benchmark_materialized_load.py \
  --input "$BENCHMARK_DIR" \
  --output "$BENCHMARK_DIR/load_benchmark.json"

echo "[slurm] end=$(date -Is)"
