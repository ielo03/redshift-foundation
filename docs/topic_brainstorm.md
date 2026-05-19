# Topic and Experiment Paths

The assignment already fixes the broad topic. What remains is choosing the strongest improvement path over the spectra-only AION baseline.

## Fixed Problem

- Build a unimodal foundation model for DESI spectra.
- Use spectra plus redshift only.
- Produce both masked-spectrum reconstruction and redshift prediction.

## Candidate Path 1: Redshift-Fixed AION

- Problem:
  Reproduce the AION-style spectra pipeline, then redesign the redshift mechanism.
- Why it is interesting:
  This directly addresses the central critique in the assignment.
- Dataset:
  DESI MMU one-percent subset.
- Model idea:
  Transformer with spectra tokens plus redshift token, trained with joint redshift loss and always-masked redshift token.
- Risks:
  Reproducing tokenization and training details may take time.
- Success metric:
  Better redshift prediction than the reproduced baseline without hurting reconstruction too much.

## Candidate Path 2: Domain-Informed Masking

- Problem:
  Replace uniform random masking with astrophysics-informed masking.
- Why it is interesting:
  It tests whether expert-guided learning improves efficiency and physical understanding.
- Dataset:
  Same DESI spectra-only setup as the baseline.
- Model idea:
  Preferential masking of emission lines, absorption lines, and the `4000 Å` break.
- Risks:
  Requires a careful definition of informative wavelength regions.
- Success metric:
  Better reconstruction and/or redshift performance at equal or smaller model size.

## Candidate Path 3: Compact Challenger

- Problem:
  Train a much smaller model from scratch that is better aligned to the domain.
- Why it is interesting:
  It tests whether domain alignment can beat brute-force scale.
- Dataset:
  Curated subset of the DESI spectra-only corpus.
- Model idea:
  Small transformer with dual heads for reconstruction and redshift regression.
- Risks:
  Harder optimization and less safety from pretrained weights.
- Success metric:
  Comparable or better benchmark behavior than the baseline with far fewer parameters.

## Recommended Primary Plan

- Primary path:
  `Candidate Path 1 + Candidate Path 2`
- Why:
  This is the most assignment-aligned and the most defensible if time is limited.

## Stretch Goal

- Stretch path:
  `Candidate Path 3`

## Final Choice

- Chosen topic:
  A spectra-only AION baseline with improved redshift handling and domain-informed masking.
- One-sentence research question:
  Can a unimodal transformer trained on DESI spectra learn better redshift-aware representations than AION-style random masking and post hoc redshift prediction?
- Input data:
  DESI spectra on the fixed wavelength grid plus redshift labels.
- Output target:
  Masked-spectrum reconstruction and direct redshift prediction.
- Baseline model:
  Reproduced spectra-only AION-style pipeline.
- Stretch goal:
  A compact from-scratch transformer challenger.
