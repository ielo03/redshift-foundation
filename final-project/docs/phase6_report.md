# Phase 6 Findings: Prediction Collapse Diagnosis

## Observation

The tokenized ablation still underperforms the raw-spectrum baseline on redshift, even after the refactor. These observations should be refreshed again if we decide to keep iterating on the tokenized path.

Because the actual redshift values are spread across a wide range, a weak or collapsed prediction stream produces poor MAE/RMSE even when token reconstruction looks good.

## What This Means

The objective is currently too reconstruction-heavy on the tokenized path. It learns the discrete spectrum tokens well enough, but the redshift path is not being forced to represent redshift accurately enough.

So the next step is **not** another small tweak to the same token-prediction setup.

## Next Step

The next experiment should separate redshift from generic token prediction more aggressively, for example by:

- adding a dedicated redshift regression head
- using the redshift token only as conditioning, not as the main target
- predicting `z` as a continuous output instead of decoding it only through the codec token path

## Keep the Baseline Separate

The baseline remains frozen in:

- [`scripts/train_redshift_main_baseline.py`](../scripts/train_redshift_main_baseline.py:1)
- [`outputs/redshift_baseline/`](../outputs/redshift_baseline)

The underlying implementation lives in [`scripts/train_redshift_baseline.py`](../scripts/train_redshift_baseline.py:1), but the official baseline configuration is now locked behind the dedicated wrapper above.

The tokenized ablation remains separate in:

- [`scripts/train_aion_codec_baseline.py`](../scripts/train_aion_codec_baseline.py:1)
- [`outputs/aion_codec_baseline/`](../outputs/aion_codec_baseline)

Any new model should write to a new output directory so we can compare it cleanly.
