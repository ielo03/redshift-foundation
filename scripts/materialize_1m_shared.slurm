#!/bin/bash
# Materialize exactly one million train/validation spectra on a fractional CPU node.
# Submit with:
#   sbatch -A m5374 scripts/materialize_1m_shared.slurm
#
# The run is resumable. Re-submit the same command after a timeout or failure;
# completed HEALPix records are listed in the output directory's
# materialized_processed_files.jsonl. Benchmark rows are excluded by default.

#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 30:00:00
#SBATCH -J redshift-mat-1m
#SBATCH -o logs/materialize-1m-%j.out
#SBATCH -e logs/materialize-1m-%j.err

set -euo pipefail

cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT_DIR="data/materialized_training_v1_1m"

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
echo "[slurm] node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "[slurm] cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "[slurm] requested_memory=${SLURM_MEM_PER_NODE:-unknown}"
echo "[slurm] hostname=$(hostname)"
echo "[slurm] start=$(date -Is)"
echo "[slurm] manifest=data/preprocessed/split_manifest_v1.jsonl"
echo "[slurm] output=${OUT_DIR}"
echo "[slurm] max_spectra=1000000"

/usr/bin/time -v "$HOME/.conda/envs/venv/bin/python" -u scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir "$OUT_DIR" \
  --format pt \
  --shard-size 8192 \
  --max-spectra 1000000 \
  --progress-interval 10

echo "[slurm] end=$(date -Is)"
