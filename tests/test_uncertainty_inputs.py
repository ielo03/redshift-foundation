from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_sdss_ood import interpolate_flux_and_ivar  # noqa: E402
from train_experiments import robust_normalize, robust_normalize_with_log_ivar  # noqa: E402
from train_materialized_streaming import transform_log_ivar  # noqa: E402


def test_robust_log_ivar_matches_flux_normalization_and_masks_invalid() -> None:
    flux = np.asarray([1.0, 3.0, 5.0, 99.0], dtype=np.float32)
    ivar = np.asarray([4.0, 1.0, 0.25, 100.0], dtype=np.float32)
    valid = np.asarray([True, True, True, False])

    normalized, log_ivar = robust_normalize_with_log_ivar(flux, ivar, valid)

    np.testing.assert_allclose(normalized, robust_normalize(flux, valid))
    scale = np.percentile(flux[valid], 95) - np.percentile(flux[valid], 5)
    np.testing.assert_allclose(log_ivar[valid], np.log1p(ivar[valid] * scale**2), rtol=1e-6)
    assert log_ivar[~valid].tolist() == [0.0]


def test_linear_resampling_propagates_variance() -> None:
    wave = np.asarray([0.0, 2.0], dtype=np.float64)
    flux = np.asarray([0.0, 2.0], dtype=np.float64)
    ivar = np.asarray([4.0, 4.0], dtype=np.float64)
    output_wave = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    output_flux, output_ivar, valid = interpolate_flux_and_ivar(output_wave, wave, flux, ivar)

    np.testing.assert_allclose(output_flux[:3], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(output_ivar[:3], [4.0, 8.0, 4.0])
    assert valid.tolist() == [True, True, True, False]
    assert output_ivar[3] == 0.0


def test_transform_confidence_scales_and_adds_noise_in_quadrature() -> None:
    ivar = torch.tensor([[4.0, 4.0, 4.0]])
    log_ivar = torch.log1p(ivar)
    multiplier = torch.tensor([[2.0, 1.0, 1.0]])
    noise = torch.tensor([[0.0, 0.5, 0.0]])
    valid = torch.tensor([[True, True, False]])

    transformed = torch.expm1(transform_log_ivar(log_ivar, multiplier, noise, valid))

    torch.testing.assert_close(transformed, torch.tensor([[1.0, 2.0, 0.0]]))
