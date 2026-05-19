# Implementation Plan

## Phase 1: Environment and Reference Audit

- Verify local packages and GPU availability
- Inspect the public AION repo
- Identify the minimal tokenizer, dataset, and training code paths we can reuse
- Write down what belongs in the raw-spectrum baseline versus the tokenized ablation

## Phase 2: Data Pipeline

- Download or locate the DESI/MMU subset
- Inspect file formats and sample counts
- Build a loader for spectra and redshift only
- Start with a tiny subset for smoke tests

## Phase 3: Baseline Reproduction

- Train a raw-spectrum transformer from scratch
- Keep the redshift target normalized and continuous
- Validate masked reconstruction and redshift prediction on fixed splits
- Train a small local version first

## Phase 4: Redshift Fixes

- Add a joint regression head
- Add always-mask-redshift training
- Compare:
  - baseline
  - joint head only
  - always-mask only
  - combined

## Phase 5: Domain-Informed Masking

- Define wavelength regions for informative masking
- Compare against random masking
- Measure impact on both reconstruction and redshift

## Phase 6: Stretch Work

- Tokenization-count experiments
- AION-tokenized ablation
- Compact model from scratch
- Wavelength mismatch handling for OOD robustness

## Decision Rule

If time gets tight, prioritize:

1. raw-spectrum baseline reproduction
2. joint redshift head
3. always-mask redshift token
4. one domain-informed masking experiment
