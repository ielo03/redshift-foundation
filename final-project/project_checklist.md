# Final Project Checklist

## Hard Requirements

- [ ] Final deadline recorded: `11 PM, Tuesday, May 19, 2026`
- [ ] Confirm deliverables from the handout
- [ ] Keep the project strictly spectra-only plus redshift
- [ ] Do not include imaging, magnitudes, Subaru, or other extra modalities

## Baseline Reproduction

- [ ] Find the public AION code path for spectra tokenization
- [ ] Find the public AION code path for redshift handling
- [ ] Document the baseline architecture in our own words
- [ ] Reproduce a spectra-only baseline using DESI spectra and redshift only
- [ ] Verify the baseline can do masked reconstruction
- [ ] Verify the baseline can produce redshift predictions

## Data Pipeline

- [ ] Acquire the DESI DR1 / EDR / SV3 one-percent slice through MMU
- [ ] Start with a tiny subset to validate I/O and training
- [ ] Scale toward the assignment-sized working subset
- [ ] Record wavelength grid assumptions: `7081` pixels, roughly `3600-9800 Å`
- [ ] Document preprocessing and normalization choices
- [ ] Build train/validation splits without leaking benchmark information

## Redshift Improvements

- [ ] Implement `Approach A`: joint training with a redshift prediction head
- [ ] Implement `Approach B`: always-mask the redshift token
- [ ] Evaluate combined `A + B`
- [ ] Record whether redshift enters the representation more effectively than baseline

## Domain-Informed Masking

- [ ] Define candidate high-information masking regions
- [ ] Include narrow lines such as emission or absorption features
- [ ] Include broad features such as the `4000 Å` break
- [ ] Compare domain-informed masking against random masking

## Generalization and Data Engineering

- [ ] Test normalization sensitivity
- [ ] Plan for wavelength mismatch across instruments
- [ ] Decide how to handle missing or extra wavelength coverage
- [ ] Record any OOD robustness experiments on non-DESI-style inputs

## Tokenization and Architecture

- [ ] Explain the original AION tokenization clearly
- [ ] Decide whether to reuse about `273` tokens or modify token count
- [ ] Try a higher-token variant if compute allows
- [ ] Keep the model transformer-based, not a pure CNN regressor
- [ ] Preserve both tasks: reconstruction and redshift prediction

## Experiment Paths

- [ ] Path A: strategic fine-tuning on a curated subset
- [ ] Path B: compact challenger model from scratch
- [ ] Decide which path is primary and which is stretch

## Results and Writeup

- [ ] Save checkpoints in `outputs/models/`
- [ ] Save report-ready plots in `outputs/figures/`
- [ ] Save metric summaries in `outputs/tables/`
- [ ] Track every meaningful run in `docs/experiment_log.md`
- [ ] Keep `docs/report_outline.md` updated with actual findings

## Final Submission

- [ ] Redshift benchmark results ready
- [ ] Masked-spectrum reconstruction results ready
- [ ] OOD notes on non-DESI spectra ready
- [ ] Final model and code organized cleanly
