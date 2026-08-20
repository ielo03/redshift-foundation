#!/bin/bash
# Submit with:
#   sbatch -A m5374 scripts/materialize_full.slurm
#
# This job is resumable. If it stops, submit the same command again and
# scripts/materialize_shards.py will skip completed HEALPix records recorded in
# data/materialized_training_v1/materialized_processed_files.jsonl.

#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -J redshift-materialize
#SBATCH -o logs/materialize-full-%j.out
#SBATCH -e logs/materialize-full-%j.err

set -euo pipefail

cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
echo "[slurm] node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "[slurm] cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "[slurm] ntasks=${SLURM_NTASKS:-unknown}"
echo "[slurm] hostname=$(hostname)"
echo "[slurm] start=$(date -Is)"
echo "[slurm] workdir=$(pwd)"
echo "[slurm] python=$HOME/.conda/envs/venv/bin/python"
echo "[slurm] manifest=data/preprocessed/split_manifest_v1.jsonl"
echo "[slurm] output=data/materialized_training_v1"
echo "[slurm] shard_size=8192"
echo "[slurm] format=pt"
echo "[slurm] running materialization"

/usr/bin/time -v "$HOME/.conda/envs/venv/bin/python" -u scripts/materialize_shards.py \
  --manifest data/preprocessed/split_manifest_v1.jsonl \
  --out-dir data/materialized_training_v1 \
  --format pt \
  --shard-size 8192 \
  --progress-interval 10

echo "[slurm] end=$(date -Is)"
