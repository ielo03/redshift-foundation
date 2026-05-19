# Baseline Lock

## Official Baseline

The official raw-spectrum baseline is the command:

```bash
./tf-env/bin/python3 hw/final_project/scripts/train_redshift_main_baseline.py
```

This wrapper freezes the baseline configuration so later experiments do not silently change the reference result.

## Locked Configuration

- output directory: `hw/final_project/outputs/redshift_baseline`
- redshift loss: `mse`
- sampling strategy: `uniform`
- tail power: `1.0`
- mask strategy: `random`
- mask probability: `0.15`
- mask seed: `42`
- alpha: `1.0`

## Variant Rule

Any experiment that changes architecture, masking, sampling, redshift loss, or weighting must run through a separate script and a separate output directory.

Current variant entrypoints:

- `scripts/train_redshift_domain_variant.py`
- `scripts/train_redshift_tail_variant.py`
- `scripts/train_redshift_coarse_variant.py`
- `scripts/train_aion_codec_baseline.py`
- `scripts/train_aion_codec_redshift_variant.py`
