#!/bin/bash
# One-time metadata-only enrichment; it does not touch flux values.
# Submit: sbatch -A m5374 scripts/enrich_materialized_target_labels_cpu.slurm
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -J enrich-target-labels
#SBATCH -o logs/enrich-target-labels-%j.out
#SBATCH -e logs/enrich-target-labels-%j.err

set -euo pipefail
cd /global/homes/c/colbyd/redshift-foundation
"$HOME/.conda/envs/venv/bin/python" -u scripts/enrich_materialized_target_labels.py \
  --input data/materialized_training_v1_1m
