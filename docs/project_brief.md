# Project Brief

> Source note: this file is a working summary derived from the professor’s PDF assignment plus our planning notes. The PDF in [`PHYS303_Final-Project_2026.pdf`](../PHYS303_Final-Project_2026.pdf) is the authoritative source.

## Core Assignment

Build a unimodal foundation model for DESI spectra that uses:

- the spectrum
- redshift `z` as the only auxiliary signal

The model must do both:

- redshift prediction
- reconstruction of masked spectral regions

## Central Critique of AION-1

The assignment focuses on a specific weakness in AION-1:

- the redshift token is treated like any other token under random masking
- the redshift token is only occasionally masked
- a separate prediction head is trained after freezing the encoder

This means:

- redshift is underweighted relative to its physical importance
- redshift does not shape the encoder representation during training

These bullet points are a paraphrase of the professor’s critique, not a verbatim quote.

## Required Baseline

Before improving anything, reproduce the assignment baseline in a reduced form:

- keep spectral data
- keep redshift
- remove all other modalities
- prefer a raw-spectrum transformer as the main baseline
- optionally use AION-style tokenization as an ablation

This baseline gives us the comparison point for every later claim.

This is an implementation plan inferred from the assignment’s requirement to use spectra only and then improve redshift handling.

## Best Improvement Directions

### 1. Redshift-Aware Training

- joint training with a lightweight redshift regression head
- always-mask the redshift token
- combine both if possible

These are recommended directions, synthesized from the assignment’s two suggested redshift mechanisms.

### 2. Domain-Informed Masking

Replace purely random masking with masking that emphasizes physically meaningful regions:

- absorption lines
- emission lines
- `4000 Å` break

This is a proposed experiment direction, not a strict requirement from the PDF.

### 3. Data Engineering for OOD Generalization

- test normalization sensitivity
- handle wavelength mismatch across instruments
- build a strategy for missing or extra wavelength ranges

These points are inferred from the assignment’s emphasis on held-out and out-of-distribution evaluation.

### 4. Tokenization and Scale

- understand the original `273`-token AION design
- use it only if it helps the baseline or an ablation
- optionally try a smaller from-scratch model aligned to the task

The exact token-count and model-scale experiments are planning ideas, not required deliverables.

## Most Defensible Project Story

If we need a clear narrative for the report and benchmark:

1. Reproduce a spectra-only baseline.
2. Compare raw-spectrum and AION-tokenized variants.
3. Introduce one or two principled redshift fixes.
4. Test whether domain-informed masking helps.
5. Measure redshift, reconstruction, and OOD behavior.

This section is purely for project framing and report structure; it is not stated directly in the assignment PDF.
