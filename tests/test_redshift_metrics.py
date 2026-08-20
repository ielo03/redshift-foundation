"""Small deterministic checks for physical-redshift reporting definitions."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from redshift_metrics import redshift_metrics_numpy  # noqa: E402


class RedshiftMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        z = np.array([0.0, 0.3, 1.1, 2.7])
        metric = redshift_metrics_numpy(z, z)
        for name in ("z_mae", "z_mae_norm", "z_bias", "z_sigma_nmad", "z_eta_0033", "z_eta_005"):
            self.assertEqual(metric[name], 0.0)
        self.assertEqual(metric["z_r2"], 1.0)

    def test_uses_one_plus_true_redshift(self) -> None:
        # dz=[0.1, 0.1], but normalized residuals are [0.1, 0.025].
        actual = np.array([0.0, 3.0])
        predicted = np.array([0.1, 3.1])
        metric = redshift_metrics_numpy(predicted, actual)
        self.assertAlmostEqual(metric["z_mae"], 0.1)
        self.assertAlmostEqual(metric["z_mae_norm"], 0.0625)
        self.assertAlmostEqual(metric["z_bias"], 0.0625)
        self.assertEqual(metric["z_eta_0033"], 1.0)
        self.assertEqual(metric["z_eta_005"], 0.5)


if __name__ == "__main__":
    unittest.main()
