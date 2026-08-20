# Phase 3 Findings: AION Tokenized Ablation

## Goal

Run the AION spectrum tokenizer end to end on the cached DESI spectra bundle and record a working tokenized ablation for comparison against the raw-spectrum baseline.

## What I Added

The spectra-token path now lives in:

- [`scripts/process_aion_codec_tokens.py`](../scripts/process_aion_codec_tokens.py:1)
- [`scripts/train_aion_codec_baseline.py`](../scripts/train_aion_codec_baseline.py:1)
- [`src/neural_baseline.py`](../src/neural_baseline.py:124)

### Tokenization

[`process_aion_codec_tokens.py`](../scripts/process_aion_codec_tokens.py:50) now:

- loads the raw DESI HDF5 inputs
- encodes spectra with AION's DESI spectrum codec
- uses `cuda` → `mps` → `cpu` device selection
- moves each batch to the selected device before encoding
- caches the resulting token bundle in a `.pt` file

The cached bundle includes:

- `tok_spectrum_desi` tokens
- redshift values
- raw flux and wavelength arrays for reconstruction evaluation

### Training

[`train_aion_codec_baseline.py`](../scripts/train_aion_codec_baseline.py:62) trains a spectra-token transformer with:

- smaller defaults for MPS memory safety
- minibatched validation and test evaluation
- early stopping
- best-checkpoint saving
- a continuous redshift head
- flux-level reconstruction scoring

## Final Results

The successful run on the cached bundle produced, but these numbers should be treated as legacy until the repaired spectrum-token pipeline is rerun:

- `MAE = 0.5924519253092057`
- `RMSE = 0.8249922134569417`
- `bias = -0.5924159750375758`
- `masked_token_accuracy = 0.9967358461436668`

Saved outputs:

- [`redshift_predictions.json`](../outputs/aion_codec_baseline/redshift_predictions.json)
- [`redshift_metrics.json`](../outputs/aion_codec_baseline/redshift_metrics.json)
- [`redshift_pred_vs_actual.png`](../outputs/aion_codec_baseline/redshift_pred_vs_actual.png)
- [`best_model.pt`](../outputs/aion_codec_baseline/best_model.pt)

## Comparison to the Main Baseline

The raw-spectrum baseline in [`outputs/redshift_baseline/redshift_metrics.json`](../outputs/redshift_baseline/redshift_metrics.json) achieved:

- `MAE = 0.2344225294635197`
- `RMSE = 0.36506266787417263`
- `bias = 0.006562201790194053`

That means the tokenized ablation is not the main result. It is useful as a comparison point, but the raw-spectrum baseline is the better headline baseline for the assignment.

## Interpretation

This is still a useful result because it shows the AION spectrum tokenizer is now wired into a from-scratch training loop locally. The next improvement should target redshift handling specifically, not just reconstruction quality.

The most likely next experiments are:

- a stronger redshift path / redshift token strategy
- always masking the redshift token
- domain-informed masking around physically meaningful spectral regions

## Next Step

Use the raw-spectrum baseline as the reference point and keep the tokenized ablation separate so the effect of tokenization can be measured cleanly.
