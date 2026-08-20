# Crop Transform v1

Hypothesis: training on 65%-100% contiguous wavelength crops with conservative
flux gain, tilt, offset, and noise improves robustness while retaining strong
full-spectrum DESI performance.

This is a standalone experiment. Its reference is
`experiments/control_full_spectrum_v1/`; keep all unrelated settings equal to
that baseline and compare each experiment's run-best fixed-validation checkpoint
on the same 10k DESI and 10k SDSS benchmark sets.
