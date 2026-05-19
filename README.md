# PHYS 303 Final Project

This project is now scoped to the course assignment:

`A Unimodal Foundation Model for DESI Spectra`

The target is not a generic ML project. The target is a spectra-only transformer baseline, followed by domain-informed improvements to redshift handling, masking, tokenization, and generalization.

## Assignment Snapshot

- Due: `11 PM, Tuesday, May 19, 2026`
- Inputs allowed: `DESI spectra + redshift only`
- Inputs forbidden: imaging, Subaru/HSC, Legacy Survey photometry, magnitudes, and other non-spectral modalities
- Core outputs:
  - redshift prediction
  - reconstruction of masked spectral regions
- Evaluation:
  - held-out benchmark from the instructor
  - includes out-of-distribution non-DESI spectra

## Project Strategy

We should treat this as a two-stage project.

### Stage 1: Reproduce a Spectra-Only Baseline

The first milestone is to reproduce the assignment baseline as closely as practical while:

- keeping only spectral data and redshift
- removing all other modalities
- establishing a baseline for reconstruction and redshift prediction

We can use AION's tokenizer as a reference or ablation, but the main baseline is a spectra-only transformer trained from scratch on raw spectra with a continuous redshift head.

This gives us a fair comparison point before trying improvements.

### Stage 2: Domain-Informed Improvements

After the baseline works, the strongest project directions are:

- redshift-aware joint training
- always-masked redshift token
- high-information masking around physically meaningful features
- tokenization experiments beyond the original `273` tokens
- normalization and wavelength-mismatch experiments for OOD robustness
- compact challenger models trained from scratch

## Folder Guide

- `PHYS303_Final-Project_2026.pdf`: assignment handout
- `project_checklist.md`: concrete project milestones
- `docs/project_brief.md`: condensed technical brief from the assignment and instructor guidance
- `docs/implementation_plan.md`: staged execution plan
- `docs/topic_brainstorm.md`: narrowed to assignment-compatible experiment paths
- `docs/proposal_outline.md`: project writeup scaffold
- `docs/report_outline.md`: final report scaffold
- `docs/experiment_log.md`: experiment tracking
- `scripts/check_environment.py`: local package check
- `scripts/process_raw_hdf5.py`: build the raw-spectrum tensor cache
- `scripts/process_aion_codec_tokens.py`: build the AION spectrum-token cache
- `scripts/train_redshift_main_baseline.py`: official frozen raw-spectrum baseline entrypoint
- `scripts/train_redshift_baseline.py`: shared raw-spectrum training implementation used by the baseline and variants
- `scripts/train_aion_codec_baseline.py`: train the AION-token reconstruction + redshift baseline
- `scripts/train_aion_codec_redshift_variant.py`: run the redshift-weighted AION-token variant
- `scripts/train_redshift_domain_variant.py`: run the domain-masking raw-spectrum variant
- `scripts/train_redshift_tail_variant.py`: run the tail-focused raw-spectrum variant
- `scripts/train_redshift_coarse_variant.py`: run the coarse-to-fine raw-spectrum variant
- `src/project_paths.py`: project path helpers
- `src/baseline.py`: legacy starter placeholder kept for reference
- `data/raw/`: original downloaded spectra or metadata files
- `data/processed/`: cleaned/tokenized/intermediate data
- `outputs/figures/`: plots
- `outputs/models/`: checkpoints
- `outputs/tables/`: metrics and summaries
- `references/`: papers, links, and copied notes
- `notebooks/`: exploration notebooks

## Immediate Next Steps

1. Read [docs/project_brief.md](/Users/colbydobson/cs/b-dl/hw/final_project/docs/project_brief.md).
2. Work through [project_checklist.md](/Users/colbydobson/cs/b-dl/hw/final_project/project_checklist.md).
3. Inspect the public AION repo and identify the minimal spectra-only path we can reuse.
4. Acquire the DESI/MMU subset and document its format in `data/`.
5. Decide whether our first implementation path is:
   - raw-spectrum transformer baseline
   - AION-tokenized ablation
   - compact model first for pipeline validation
6. Run:

```bash
./tf-env/bin/python hw/final_project/scripts/check_environment.py
./tf-env/bin/python hw/final_project/scripts/process_raw_hdf5.py
./tf-env/bin/python hw/final_project/scripts/process_aion_codec_tokens.py
./tf-env/bin/python hw/final_project/scripts/train_redshift_main_baseline.py
./tf-env/bin/python hw/final_project/scripts/train_aion_codec_baseline.py
```

## Recommended Technical Direction

If we want the best chance of finishing strong in the course timeframe, the most pragmatic sequence is:

1. Get a small end-to-end spectra-only baseline running.
2. Compare raw-spectrum and AION-tokenized ablations.
3. Add one principled redshift fix:
   - joint redshift regression head
   - always-mask redshift token
   - or both
4. Add one domain-informed masking strategy.
5. If time allows, test compact-model or tokenization variants.

## Deliverables to Keep in View

- a trained model that ingests DESI spectra
- redshift predictions
- masked-spectrum reconstruction
- benchmark results on instructor-held data
- evidence of out-of-distribution behavior on non-DESI spectra

## Notes

The project documents in this folder now reflect the actual assignment and the extra instructor guidance you pasted. The next productive step is implementation planning against the AION codebase and dataset format, not more generic brainstorming.

The baseline is now explicitly frozen behind `scripts/train_redshift_main_baseline.py`. Any new idea should go into a separate variant entrypoint and output directory so the baseline remains comparable.
