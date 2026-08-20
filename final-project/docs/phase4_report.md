# Phase 4 Plan: Redshift-Focused Variant

## Goal

Keep the repaired raw-spectrum baseline fixed as the reference model, then test one redshift-focused change at a time.

## Baseline Rule

The baseline is now locked in:

- [`scripts/train_redshift_baseline.py`](../scripts/train_redshift_baseline.py:1)
- [`outputs/redshift_baseline/`](../outputs/redshift_baseline)

Any new experiment must write to a separate output directory so we never overwrite the baseline results.

## Next Experiment

The next change should target redshift directly:

- stronger loss weighting on the continuous redshift head
- fixed validation/test masks
- compare against the baseline with the same split and evaluation code

## Success Criteria

The next experiment is worth keeping only if it improves redshift MAE/RMSE without breaking the reconstruction pipeline or the reproducibility of the baseline.
