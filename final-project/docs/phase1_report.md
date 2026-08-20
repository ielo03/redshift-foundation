# Phase 1 Findings: AION Reference Audit

## Executive Summary

The AION repo already contains the pieces we need for a **tokenized spectra ablation**:

- a DESI spectrum modality with the correct token count
- a spectrum codec/tokenizer path
- a generic encoder-decoder model wrapper that can predict a target modality

However, the repo does **not** appear to include a clear end-to-end training entrypoint for our project scope. In practice, we can reuse AION’s spectra codec as a reference or ablation, but we will likely need to build our own small training script and data loader for the DESI/MMU subset.

## What the Repo Shows

### 1. DESI spectra are already a first-class modality

The spectra modality is defined in [`aion/modalities.py`](../AION/aion/modalities.py:89). The key details are:

- [`DESISpectrum`](../AION/aion/modalities.py:111) exists explicitly
- it uses [`token_key = "tok_spectrum_desi"`](../AION/aion/modalities.py:114)
- it expects [`num_tokens = 273`](../AION/aion/modalities.py:115)
- it stores spectra as `flux`, `ivar`, `mask`, and `wavelength` tensors

This matches the assignment’s description of DESI spectra as the core input.

### 2. The spectra tokenizer / codec is already implemented

The main spectra codec is in [`aion/codecs/spectrum.py`](../AION/aion/codecs/spectrum.py:15). Important observations:

- it normalizes flux and inverse variance
- it projects the spectrum onto a latent grid
- it quantizes the latent representation
- it prepends a normalization token to the learned code sequence

So the AION spectra path is not a simple hand-written binning scheme; it is a learned autoencoder-style codec with discrete tokenization.

### 3. The model wrapper can handle masked targets

The model API in [`aion/model.py`](../AION/aion/model.py:13) shows that AION is built around:

- `embed_inputs(...)` for encoder inputs
- `embed_targets(...)` for decoder targets
- `forward(...)` / `_forward(...)` for producing logits over a requested modality

That is useful because our project needs both:

- masked spectral reconstruction
- redshift prediction

### 4. Redshift is a scalar modality

The redshift class is defined in [`aion/modalities.py`](../AION/aion/modalities.py:273):

- [`class Z(Scalar)`](../AION/aion/modalities.py:274)
- Legacy AION redshift token path in [`AION/aion/modalities.py`](../AION/aion/modalities.py:278), which we are no longer using as the project target
- [`num_tokens = 1`](../AION/aion/modalities.py:279)

That means redshift is already represented in the AION design as a single-token scalar target, which aligns with the assignment’s discussion.

## What Appears Reusable

For our project, the most reusable parts are:

1. **The DESI spectrum modality definition**
2. **The spectrum codec/tokenizer implementation**
3. **The encoder-decoder model interface**
4. **The redshift scalar modality definition**

If we want to reproduce a spectra-only baseline, these give us a strong starting point without needing to invent the architecture from scratch.

## What Seems Missing for Our Project

### 1. A clean project-specific training entrypoint

The repo contains scripts like [`scripts/export_gaia_codecs.py`](../AION/scripts/export_gaia_codecs.py:399) and [`scripts/export_hsc_codecs.py`](../AION/scripts/export_hsc_codecs.py:325), but those are maintenance/export tools rather than the baseline training pipeline we need.

From the current audit, there is no obvious top-level training script dedicated to:

- loading DESI/MMU spectra
- training only on spectra + redshift
- comparing baseline and improved redshift handling

### 2. A local DESI/MMU data loader

The repo shows how a spectrum should look internally, but not how we should load the actual course dataset slice.

So we still need to build:

- a dataset reader
- a batch collation path
- a tiny smoke-test subset

### 3. The project-specific redshift fix

The assignment’s main contribution is to improve redshift handling beyond the baseline. That part is not solved by the stock AION repo and needs new code or a new training head.

## Minimum Baseline to Reproduce Locally

The smallest defensible baseline is:

- **Input:** DESI spectra only
- **Auxiliary target:** redshift `z`
- **Main baseline:** a raw-spectrum transformer trained from scratch
- **Ablation:** AION’s [`DESISpectrum`](../AION/aion/modalities.py:111) and spectra codec path from [`aion/codecs/spectrum.py`](../AION/aion/codecs/spectrum.py:15)
- **Objective:** masked reconstruction plus redshift prediction

In other words, the assignment baseline is **spectra modeling with everything except spectra and redshift removed**, with tokenization chosen as an implementation detail rather than the headline result.

## Practical Next Step After Phase 1

The next implementation step should be:

1. build a tiny DESI/MMU loader
2. confirm one batch matches the expected spectrum fields
3. run a forward pass through the spectra codec / model path
4. check output shapes and finite losses

That will tell us whether we can directly reuse the AION codec/model or whether we need a thinner custom baseline.
