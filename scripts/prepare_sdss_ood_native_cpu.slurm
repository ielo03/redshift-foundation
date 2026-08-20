#!/bin/bash
# Build a native-grid SDSS/eBOSS OOD bundle. This is a separate extrapolation
# protocol; it does not replace the DESI-grid, shared-coverage OOD bundle.
# Submit: sbatch -A m5374 scripts/prepare_sdss_ood_native_cpu.slurm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J prepare-sdss-native
#SBATCH -o logs/prepare-sdss-native-%j.out
#SBATCH -e logs/prepare-sdss-native-%j.err

set -euo pipefail
cd /global/homes/c/colbyd/redshift-foundation
mkdir -p logs data/ood
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

OOD_BUNDLE="${OOD_BUNDLE:-data/ood/sdss_eboss_ood_native.pt}"
SDSS_REDSHIFT_ROOT="${SDSS_REDSHIFT_ROOT:-/global/cfs/cdirs/sdss/data/sdss/dr16/eboss/spectro/redux/v5_13_0}"
SDSS_MAX_PLATES="${SDSS_MAX_PLATES:-16}"
SDSS_MAX_SPECTRA="${SDSS_MAX_SPECTRA:-10000}"
if [ -s "$OOD_BUNDLE" ]; then
  echo "[sdss-native] bundle already exists; leaving it unchanged: $OOD_BUNDLE"
  exit 0
fi
srun "$HOME/.conda/envs/venv/bin/python" -u scripts/prepare_sdss_ood.py \
  --plate-dir data/raw-sdss --redshift-root "$SDSS_REDSHIFT_ROOT" \
  --output "$OOD_BUNDLE" --max-plates "$SDSS_MAX_PLATES" \
  --max-spectra "$SDSS_MAX_SPECTRA" --grid native
