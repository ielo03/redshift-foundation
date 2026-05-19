# Proposal Outline

## Working Title

Domain-Informed Spectroscopic Foundation Models: From Brute-Force Scaling to Expert-Guided Learning

## Motivation

- AION-1 spreads model capacity across `39` modalities, which may limit deep competence on spectra.
- The assignment asks for the reverse trade-off: take one modality seriously.
- The main technical critique is that AION-1 underweights redshift and does not force `z` into the representation space during pretraining.

## Dataset

- Dataset name:
  DESI DR1 / EDR / SV3 one-percent subset through Multimodal Universe
- Source:
  MMU packaging of DESI spectra
- Number of examples:
  Start tiny for pipeline validation, then scale toward the assignment subset
- Input shape or feature types:
  Spectrum on a `7081`-pixel wavelength grid covering about `3600-9800 Å`
- Labels or targets:
  Redshift `z`

## Method

- Baseline:
  Reproduce a spectra-only baseline using raw spectra plus redshift, with an AION-tokenized ablation if helpful
- Planned improved model:
  Transformer with domain-informed masking and improved redshift handling
- Candidate improvements:
  - joint redshift regression head
  - always-masked redshift token
  - masking focused on high-information spectral regions
  - token-count experiments if the tokenized ablation is useful
- Tools/framework:
  Python, PyTorch, AION reference code as needed, MMU data tools

## Evaluation

- Primary metric:
  Redshift prediction on held-out spectra
- Secondary metrics:
  Reconstruction quality on masked spectral regions
- Validation plan:
  Local held-out validation during development, then instructor benchmark including non-DESI OOD spectra

## Risks and Backup Plan

- Main technical risk:
  The tokenized ablation may not outperform the raw-spectrum baseline
- Data risk:
  Large data footprint and preprocessing complexity
- Backup plan:
  Use a smaller curated subset and focus on one strong intervention: joint redshift training plus always-masked `z`
