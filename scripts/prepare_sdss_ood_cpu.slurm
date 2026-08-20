#!/bin/bash
# Build the shared labeled/resampled SDSS OOD bundle without reserving a GPU.
# Submit before the GPU training array:
#   sbatch -A m5374 scripts/prepare_sdss_ood_cpu.slurm

#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
# This converter is single-process; request one shared CPU core rather than a
# whole CPU-node slice. No explicit memory reservation is made.
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J prepare-sdss-ood
#SBATCH -o logs/prepare-sdss-ood-%j.out
#SBATCH -e logs/prepare-sdss-ood-%j.err

set -euo pipefail
cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs data/ood

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

OOD_BUNDLE="${OOD_BUNDLE:-data/ood/sdss_eboss_ood.pt}"
SDSS_REDSHIFT_ROOT="${SDSS_REDSHIFT_ROOT:-/global/cfs/cdirs/sdss/data/sdss/dr16/eboss/spectro/redux/v5_13_0}"
SDSS_MAX_PLATES="${SDSS_MAX_PLATES:-16}"
SDSS_MAX_SPECTRA="${SDSS_MAX_SPECTRA:-10000}"

if [ -s "$OOD_BUNDLE" ]; then
  echo "[sdss] bundle already exists; leaving it unchanged: $OOD_BUNDLE"
  exit 0
fi

echo "[sdss] start=$(date -Is) output=$OOD_BUNDLE max_plates=$SDSS_MAX_PLATES max_spectra=$SDSS_MAX_SPECTRA"
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/prepare_sdss_ood.py \
  --plate-dir data/raw-sdss \
  --redshift-root "$SDSS_REDSHIFT_ROOT" \
  --reference-manifest data/preprocessed/split_manifest_v1.jsonl \
  --output "$OOD_BUNDLE" \
  --max-plates "$SDSS_MAX_PLATES" \
  --max-spectra "$SDSS_MAX_SPECTRA"
echo "[sdss] end=$(date -Is)"
