"""Consistent physical-redshift metrics for training and benchmark reports.

All redshift comparisons in this repository use ``dz_norm = (z_pred-z_true) /
(1+z_true)``.  The definitions intentionally match the common DESI reporting
convention so results can be compared without having to infer metric details.
"""
from __future__ import annotations

import numpy as np
import torch


SIGMA_NMAD_SCALE = 1.4826


def redshift_metrics_numpy(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return physical-z and normalized residual metrics for two 1-D arrays."""

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("predicted and actual must be same-length non-empty arrays")
    if not np.isfinite(actual).all():
        raise ValueError("z_true contains non-finite values; no rows were silently excluded")
    if not np.isfinite(predicted).all():
        raise ValueError("z_pred contains non-finite values; no rows were silently excluded")
    dz = predicted - actual
    dz_norm = dz / (1.0 + actual)
    bias = float(np.median(dz_norm))
    centered = dz_norm - bias
    ss_res = float(np.square(dz).sum())
    ss_tot = float(np.square(actual - actual.mean()).sum())
    return {
        # Explicit names are the canonical interface.  ``mae`` and ``rmse``
        # remain aliases for existing scripts and old result readers.
        "z_mae": float(np.abs(dz).mean()),
        "z_rmse": float(np.sqrt(np.square(dz).mean())),
        "z_mae_norm": float(np.abs(dz_norm).mean()),
        "z_bias": bias,
        "z_sigma_nmad": float(SIGMA_NMAD_SCALE * np.median(np.abs(centered))),
        "z_eta_0033": float((np.abs(dz_norm) > 0.0033).mean()),
        "z_eta_005": float((np.abs(dz_norm) > 0.05).mean()),
        "z_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "mae": float(np.abs(dz).mean()),
        "rmse": float(np.sqrt(np.square(dz).mean())),
        "normalized_bias": bias,
        "sigma_nmad": float(SIGMA_NMAD_SCALE * np.median(np.abs(centered))),
        "outlier_fraction_abs_dz_over_1_plus_z_gt_0p15": float((np.abs(dz_norm) > 0.15).mean()),
    }


def redshift_metrics_torch(predicted: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Torch convenience wrapper that computes metric definitions on CPU."""

    return redshift_metrics_numpy(
        predicted.detach().float().cpu().numpy(), actual.detach().float().cpu().numpy()
    )
