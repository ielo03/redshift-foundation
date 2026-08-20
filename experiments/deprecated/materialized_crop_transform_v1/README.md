# DEPRECATED — Materialized Crop/Transform v1 (Superseded Scaffold)

Hypothesis: training with variable wavelength coverage plus conservative flux
calibration/noise corruption improves robustness without harming full-spectrum
DESI redshift performance.

Both arms use the same fixed 1M materialized training source, wavelength-aware
transformer, optimizer, seed, contiguous reconstruction mask, validation set,
and 10k DESI plus 10k SDSS benchmark protocol. Only `augmentation` differs.

- `configs/control_full_no_transforms.json`: full clean spectrum control.
- `configs/crop65_transforms.json`: 65%-100% crop plus gain/tilt/offset/noise.

This early two-arm scaffold is retained only as historical context. Do not
submit it for new work. Use the separate `experiments/control_full_spectrum_v1`
and `experiments/crop_transform_v1` experiments instead.
