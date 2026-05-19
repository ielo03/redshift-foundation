# Report Outline

## 1. Introduction

- What is a unimodal spectroscopic foundation model?
- Why spectra-only may be better than diluted multimodal breadth for this assignment
- The critique of AION-1 redshift handling
- Research question and hypothesis

## 2. Dataset

- DESI/MMU subset description
- Wavelength coverage and fixed-grid assumptions
- Preprocessing pipeline
- Normalization choices
- Train/validation split

## 3. Baseline Reproduction

- Raw-spectrum baseline architecture
- AION-tokenized ablation, if included
- How redshift is treated in the baseline
- Baseline training setup

## 4. Proposed Improvements

- Joint redshift head
- Always-masked redshift token
- Domain-informed masking
- Tokenization changes
- Compact-model variant if attempted

## 5. Results

- Redshift prediction metrics
- Reconstruction metrics
- Comparison to the raw-spectrum baseline
- AION-tokenized ablation results, if included
- OOD behavior on non-DESI spectra
- Efficiency or parameter-count discussion if relevant

## 6. Discussion

- Did redshift-aware training improve the representation?
- Did domain-informed masking outperform random masking?
- Where the model still failed
- Limits of data, compute, and evaluation

## 7. Conclusion

- Main takeaway
- Best-performing design choice
- What a stronger future version would try next
