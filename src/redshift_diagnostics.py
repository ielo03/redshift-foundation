from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def encode_redshift_target(redshift: Iterable[float]) -> np.ndarray:
    redshift_arr = np.asarray(list(redshift), dtype=np.float64)
    return redshift_arr / (1.0 + redshift_arr)


def decode_redshift_target(encoded_redshift: Iterable[float]) -> np.ndarray:
    encoded_arr = np.asarray(list(encoded_redshift), dtype=np.float64)
    denom = np.clip(1.0 - encoded_arr, 1e-6, None)
    return encoded_arr / denom


def redshift_sample_weights(redshift: Iterable[float]) -> np.ndarray:
    redshift_arr = np.asarray(list(redshift), dtype=np.float64)
    clipped = np.clip(redshift_arr, 0.0, 4.0)
    return 1.0 + clipped / 4.0


def redshift_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_arr = np.asarray(list(actual), dtype=np.float64)
    predicted_arr = np.asarray(list(predicted), dtype=np.float64)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual and predicted must have the same shape, got {actual_arr.shape} and {predicted_arr.shape}"
        )

    errors = predicted_arr - actual_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    return {"mae": mae, "rmse": rmse, "bias": bias}


def spectrum_reconstruction_metrics(
    actual_flux: Iterable[float],
    predicted_flux: Iterable[float],
) -> dict[str, float]:
    actual_arr = np.asarray(list(actual_flux), dtype=np.float64)
    predicted_arr = np.asarray(list(predicted_flux), dtype=np.float64)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual and predicted flux must have the same shape, got {actual_arr.shape} and {predicted_arr.shape}"
        )

    errors = predicted_arr - actual_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {"flux_mae": mae, "flux_rmse": rmse}


def masked_normalized_reconstruction_metrics(
    actual_flux: Iterable[float],
    predicted_flux: Iterable[float],
    mask: Iterable[bool],
) -> dict[str, float]:
    actual_arr = np.asarray(actual_flux, dtype=np.float64)
    predicted_arr = np.asarray(predicted_flux, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual and predicted flux must have the same shape, got {actual_arr.shape} and {predicted_arr.shape}"
        )
    if actual_arr.shape != mask_arr.shape:
        raise ValueError(f"mask must match flux shape, got {mask_arr.shape} and {actual_arr.shape}")

    if actual_arr.ndim == 1:
        actual_arr = actual_arr[None, :]
        predicted_arr = predicted_arr[None, :]
        mask_arr = mask_arr[None, :]

    observed_arr = ~mask_arr
    observed_count = observed_arr.sum(axis=-1, keepdims=True)

    fallback_mean = actual_arr.mean(axis=-1, keepdims=True)
    fallback_std = actual_arr.std(axis=-1, keepdims=True)
    fallback_std = np.clip(fallback_std, 1e-6, None)

    observed_sum = (actual_arr * observed_arr).sum(axis=-1, keepdims=True)
    observed_mean = np.divide(
        observed_sum,
        np.clip(observed_count, 1, None),
        out=fallback_mean.copy(),
        where=observed_count > 0,
    )
    centered = (actual_arr - observed_mean) * observed_arr
    observed_var = np.divide(
        (centered**2).sum(axis=-1, keepdims=True),
        np.clip(observed_count, 1, None),
        out=(fallback_std**2).copy(),
        where=observed_count > 0,
    )
    observed_std = np.sqrt(np.clip(observed_var, 1e-6, None))

    actual_norm = (actual_arr - observed_mean) / observed_std
    predicted_norm = (predicted_arr - observed_mean) / observed_std

    masked_actual = actual_norm[mask_arr]
    masked_predicted = predicted_norm[mask_arr]
    if masked_actual.size == 0:
        return {"masked_flux_mae": float("nan"), "masked_flux_rmse": float("nan"), "masked_fraction": 0.0}

    errors = masked_predicted - masked_actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    masked_fraction = float(mask_arr.sum() / mask_arr.size)
    return {
        "masked_flux_mae": mae,
        "masked_flux_rmse": rmse,
        "masked_fraction": masked_fraction,
    }


def plot_predicted_vs_actual_redshift(
    actual: Iterable[float],
    predicted: Iterable[float],
    output_path: str | Path,
    title: str = "Predicted vs Actual Redshift",
) -> Path:
    actual_arr = np.asarray(list(actual), dtype=np.float64)
    predicted_arr = np.asarray(list(predicted), dtype=np.float64)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual and predicted must have the same shape, got {actual_arr.shape} and {predicted_arr.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lim_min = float(min(actual_arr.min(), predicted_arr.min()))
    lim_max = float(max(actual_arr.max(), predicted_arr.max()))
    padding = 0.05 * max(lim_max - lim_min, 1e-6)
    lower = lim_min - padding
    upper = lim_max + padding

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_arr, predicted_arr, s=18, alpha=0.7, edgecolors="none")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.2, label="ideal")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Actual redshift z")
    ax.set_ylabel("Predicted redshift ẑ")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_flux_reconstruction(
    actual_flux: Iterable[float],
    predicted_flux: Iterable[float],
    output_path: str | Path,
    title: str = "Flux Reconstruction",
) -> Path:
    actual_arr = np.asarray(list(actual_flux), dtype=np.float64)
    predicted_arr = np.asarray(list(predicted_flux), dtype=np.float64)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual and predicted flux must have the same shape, got {actual_arr.shape} and {predicted_arr.shape}"
        )

    if actual_arr.ndim == 1:
        actual_arr = actual_arr[None, :]
        predicted_arr = predicted_arr[None, :]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(actual_arr.shape[-1])
    ax.plot(x, actual_arr[0], linewidth=1.2, label="actual")
    ax.plot(x, predicted_arr[0], linewidth=1.0, alpha=0.8, label="predicted")
    ax.set_xlabel("Wavelength index")
    ax.set_ylabel("Flux")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
