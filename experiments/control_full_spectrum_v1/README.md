# Control Full Spectrum v1

This is the reusable baseline for subsequent experiments. It trains the
wavelength-aware transformer on full, clean materialized DESI spectra with
contiguous reconstruction spans, but no crop or flux transforms.

The shared materialized trainer recognizes this full/no-transform configuration
and uses its direct materialized-tensor fast path: it does not run the
per-spectrum crop, repadding, or transform loop during control training or
fixed full-length validation.

Use this experiment as the fixed reference unless a later project explicitly
changes the baseline definition. New experiments compare their best checkpoint
against this experiment's fixed-validation, DESI-benchmark, and SDSS-OOD
results; they never write into this experiment's output directories.
