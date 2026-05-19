# Phase 5 Findings: Redshift-Focused Variant

## Goal

Test one redshift-focused change while keeping the raw-spectrum baseline separate and unchanged.

## What I Ran

I wrapped the tokenized ablation in [`train_aion_codec_redshift_variant.py`](../scripts/train_aion_codec_redshift_variant.py:1) and kept its outputs separate in:

- [`outputs/aion_codec_redshift_variant/`](../outputs/aion_codec_redshift_variant)

The variant uses the same cached token bundle as the baseline, but gives the redshift head a larger loss weight and a different deterministic mask seed.

## Result

The run stopped early and saved outputs successfully, but the numbers below are legacy and should be regenerated after the refactor.

Metrics:

- `MAE = 0.5786536921264089`
- `RMSE = 0.8014667308603987`
- `bias = -0.5786156456118076`
- `masked_token_accuracy = 0.9788658016122774`

## Comparison to the Baseline

The raw-spectrum baseline in [`outputs/redshift_baseline/redshift_metrics.json`](../outputs/redshift_baseline/redshift_metrics.json) had:

- `MAE = 0.2344225294635197`
- `RMSE = 0.36506266787417263`
- `bias = 0.006562201790194053`

So the tokenized redshift-focused variant is not the main baseline. It is only useful if it demonstrates an ablation effect that can be explained clearly.

## Interpretation

This confirms that the tokenized route should stay an ablation unless it clearly beats the raw-spectrum baseline. The next step should focus on the raw-spectrum baseline and a clean redshift-aware improvement.

## Next Direction

The most promising next change is to make redshift a more dedicated pathway in the raw-spectrum transformer, then revisit the tokenized ablation only if time permits.
