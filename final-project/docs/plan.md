# Final Project Plan

## Goal

Build a **unimodal DESI-spectra foundation model** that uses only:

- DESI spectra
- redshift `z` as the auxiliary signal

The model must support:

- masked-spectrum reconstruction
- redshift prediction

The project should begin with a spectra-only reproduction of the AION-style pipeline, then improve redshift handling and masking.

## Working Data Scale Assumption

For good results, we should aim for **at least 100,000 spectra** if the dataset slice and preprocessing budget allow it. That is a practical minimum target for meaningful training, and we should build up to it gradually even if we start with a much smaller subset for smoke tests.

## Empirical Training Note

An additional working assumption for the final training run is that a **300M-parameter transformer trained for about 12 hours on an A100 GPU with ~100,000 spectra** produced good results. We should treat that as the target scale to build toward if resources allow.

## What This Project Is Based On

Primary source:

- [`PHYS303_Final-Project_2026.pdf`](../PHYS303_Final-Project_2026.pdf)

Reference implementation:

- the cloned AION repo in [`AION/`](../AION)

Supporting planning docs:

- [`project_brief.md`](project_brief.md)
- [`implementation_plan.md`](implementation_plan.md)

## Core Requirements From the Assignment

- Use DESI spectra only
- Use redshift `z` as the only auxiliary signal
- Do not use imaging, catalog, magnitude, or Subaru/HSC data
- Train a transformer with masked-token prediction
- Be able to explain the tokenization approach
- Target a realistic model scale, but start small enough to debug locally
- Evaluate both reconstruction and redshift prediction
- Check robustness on out-of-distribution spectra

## Main Baseline to Reproduce

The baseline is the AION-style spectra pipeline, reduced to:

1. spectra input
2. redshift scalar
3. masked-token reconstruction objective
4. model output for redshift prediction

The baseline should be close enough to AION to serve as a fair comparison point.

## Planned Improvements

### 1. Redshift-Aware Training

Compare these variants:

- baseline AION-style redshift handling
- joint training with a lightweight redshift head
- always-mask the redshift token
- joint head + always-mask combined

### How We Will Change the Model, and Why

This project should not jump straight from the current baseline to a large final architecture. Instead, we should change one design choice at a time so we can tell what each change actually does.

The guiding idea is:

- keep the same data split and evaluation code
- keep the same reconstruction + redshift metrics
- change only one model component at a time
- compare every new model against the same baseline

That gives us a controlled experiment instead of a moving target.

#### Step 1: Keep the current model as the reference baseline

Current baseline:

- spectra-only transformer
- patch embedding over the flux array
- reconstruction head
- redshift head
- random masking during training

Why this stays the baseline:

- it is already working end to end
- it gives us a repeatable score to beat
- it is small enough to train quickly
- it matches the assignment’s core ingredients without extra complexity

What we learn from it:

- whether the data pipeline is sane
- whether redshift prediction is possible at all from the spectra alone
- what the current error floor looks like on our split

#### Step 2: Improve the redshift path before making the model much bigger

The model may reconstruct spectra reasonably well while still being weak at redshift. That is why redshift deserves its own design work.

Possible changes:

- strengthen the redshift head with a better pooled representation
- add a dedicated redshift token or latent path
- separate the reconstruction and redshift representations more clearly
- try always masking redshift during training if the AION-style setup expects that

Why do this now:

- it is cheaper than scaling the entire model
- it directly targets the metric we care about for astrophysics interpretation
- it lets us see whether redshift failure is an architectural issue or a data-scale issue

What to watch for:

- redshift MAE and RMSE
- bias in predicted redshift
- whether reconstruction gets worse when redshift improves, or vice versa

#### Step 3: Change the masking strategy

The current setup uses random masking. That is a good generic baseline, but it may not force the model to learn the most informative spectral features.

Potential next changes:

- mask emission-line regions more often
- mask the 4000 Å break more often
- compare against fully random masking

Why this matters:

- DESI spectra are not just arbitrary 1D signals; the physically important wavelength regions matter
- if the model learns to reconstruct meaningful lines, it may also learn better redshift features
- this is a principled way to test whether domain knowledge helps

What to watch for:

- reconstruction on line-rich regions
- redshift prediction quality
- whether masking policy improves generalization or just training loss

#### Step 4: Only then scale model size

If the model design looks promising on the current data slice, we can increase capacity.

Possible scale-up directions:

- more transformer layers
- wider hidden size
- more attention heads
- more patch tokens / finer tokenization

Why scale later:

- if the design is wrong, a bigger model just wastes time
- scaling after the design is stable tells us whether extra capacity actually helps
- this keeps the experiments interpretable

What to watch for:

- train/val gap
- stability of training
- whether improvements continue or saturate

#### Step 5: Scale data only after the model is worth testing harder

Once the architecture is in a good place, we should increase the data volume.

Why not start here:

- data scale is expensive
- it is hard to debug a bad architecture on a huge run
- a small slice is enough to compare design ideas quickly

Why do it eventually:

- the project needs a realistic demonstration of the method
- the assignment target and the AION-style reference both imply that the model should work at a meaningful scale
- larger data gives us a better sense of whether the model will hold up beyond the debug set

#### Step 6: Keep the evaluation protocol fixed

Every change should be judged with the same:

- train/val/test split
- normalization logic
- redshift metrics
- reconstruction metrics
- predicted-vs-actual plot

Why:

- otherwise we cannot tell whether one model is truly better
- a fixed evaluation setup makes the final writeup defensible

#### Practical model-change order

Recommended sequence:

1. baseline spectra-only transformer
2. better redshift head / redshift token experiment
3. always-mask or joint redshift-handling experiment
4. domain-informed masking experiment
5. only then scale width/depth
6. only after that scale the dataset further if needed

That order keeps the project scientifically controlled and avoids mixing too many unknowns at once.

### 2. Domain-Informed Masking

Replace purely random masking with masking that emphasizes physically meaningful regions:

- emission lines
- absorption lines
- the `4000 Å` break

### 3. Data Engineering and OOD Robustness

Test how the model behaves under:

- different normalization schemes
- wavelength-range mismatch
- missing or extra wavelength coverage

### 4. Tokenization and Scale Experiments

Investigate:

- the original AION learned-token setup
- whether a different token count helps
- whether a smaller from-scratch model is competitive

## Implementation Phases

### Phase 1: Reference Audit

Done / in progress work:

- inspect AION spectra codec
- inspect AION model input/output flow
- identify what can be reused for a spectra-only baseline

### Phase 2: Data Pipeline

Need to build:

- a loader for DESI/MMU spectra
- a parser for flux, inverse variance, mask, wavelength, and redshift
- a tiny smoke-test subset for quick iteration

### Phase 3: Baseline Reproduction

Need to implement:

- spectra-only tokenizer / codec path
- baseline masked reconstruction
- redshift prediction path
- a first small training run

### Phase 4: Redshift Fixes

Need to compare:

- baseline
- joint head only
- always-mask only
- combined approach

### Phase 5: Masking and OOD Experiments

Need to test:

- domain-informed masking versus random masking
- normalization sensitivity
- out-of-distribution spectra

### Phase 6: Final Report / Writeup

Need to document:

- what the baseline was
- what changed
- which variant worked best
- reconstruction and redshift metrics
- OOD results

## File Map

Useful files in this workspace:

- [`README.md`](../README.md)
- [`project_checklist.md`](../project_checklist.md)
- [`project_brief.md`](project_brief.md)
- [`implementation_plan.md`](implementation_plan.md)
- [`experiment_log.md`](experiment_log.md)
- [`proposal_outline.md`](proposal_outline.md)
- [`report_outline.md`](report_outline.md)

Reference code in AION:

- [`aion/codecs/spectrum.py`](../AION/aion/codecs/spectrum.py)
- [`aion/model.py`](../AION/aion/model.py)
- [`aion/modalities.py`](../AION/aion/modalities.py)
- [`scripts/`](../AION/scripts)

## Immediate Next Steps

1. Find the exact spectra dataset format we will load.
2. Build a tiny loader that returns one batch of spectra + redshift.
3. Verify the AION spectrum codec or a local approximation on that batch.
4. Run a smoke-test forward pass.
5. Only then start training comparisons.

## Success Criteria

The project is in good shape if we can show:

- a working spectra-only baseline
- a clear redshift improvement over the baseline
- at least one masking improvement or robustness result
- a short, defensible explanation of the tokenization and training setup
