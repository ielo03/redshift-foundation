# Phase 2 Findings: Data Pipeline

## Goal

Build the smallest useful data-loading and preprocessing path for the project:

- load DESI-style spectra
- carry redshift `z`
- prepare batches that can feed a spectra-only baseline

## What I Added

The new pipeline lives in [`src/data_pipeline.py`](../src/data_pipeline.py:1).

### Core data structure

[`SpectrumRecord`](../src/data_pipeline.py:13) stores the fields we need for each example:

- `flux`
- `ivar`
- `mask`
- `wavelength`
- `z`

### Loading helpers

The module can load spectra from:

- JSONL via [`load_jsonl`](../src/data_pipeline.py:65)
- CSV via [`load_csv`](../src/data_pipeline.py:75)
- NPZ via [`load_npz`](../src/data_pipeline.py:84)

### Batching and preprocessing

The batching path is handled by [`collate_records`](../src/data_pipeline.py:138), which:

- pads or trims spectra to a common length
- preserves the mask
- normalizes flux using unmasked values only
- returns a dictionary with stacked tensors

### Synthetic smoke-test data

[`synthetic_records`](../src/data_pipeline.py:191) creates a tiny artificial spectra-only dataset so we can test the pipeline before real DESI/MMU data is available.

## Why This Is Enough for Now

We do not yet have a documented local DESI/MMU file format in the workspace, so the right phase-2 move is to build a loader that can handle several common storage formats and a synthetic fallback.

This gives us a stable interface for the next step:

1. inspect the actual dataset files
2. map them into `SpectrumRecord`
3. connect the batch output to the raw-spectrum baseline first, then the tokenized ablation

## Current Limitation

This pipeline is **format-flexible**, not yet dataset-specific. It is ready to be pointed at the real DESI/MMU slice once we know whether the source files are JSONL, CSV, NPZ, parquet, or something else.

## Next Step

The next task is to wire these records into the raw-spectrum baseline and run one end-to-end smoke test on synthetic data and then on the real dataset once it is located. After that, we can compare against the tokenized ablation if it still looks useful.
