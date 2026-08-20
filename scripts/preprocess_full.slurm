#!/bin/bash
# Submit with:
#   sbatch -A <actual_account> scripts/preprocess_full.slurm
#

#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -J redshift-preprocess
#SBATCH -o logs/preprocess-full-%j.out
#SBATCH -e logs/preprocess-full-%j.err

set -euo pipefail

cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
echo "[slurm] node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "[slurm] cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "[slurm] ntasks=${SLURM_NTASKS:-unknown}"
echo "[slurm] hostname=$(hostname)"
echo "[slurm] start=$(date -Is)"
echo "[slurm] workdir=$(pwd)"
echo "[slurm] python=$HOME/.conda/envs/venv/bin/python"
echo "[slurm] output=data/preprocessed"
echo "[slurm] allowed_cache=data/preprocessed_timing_100/allowed_targets_galaxy_qso_primary.npz"
echo "[slurm] running preprocess"

/usr/bin/time -v "$HOME/.conda/envs/venv/bin/python" -u scripts/preprocess.py \
  --out-dir data/preprocessed \
  --allowed-cache data/preprocessed_timing_100/allowed_targets_galaxy_qso_primary.npz \
  --num-workers 1

echo "[slurm] end=$(date -Is)"
