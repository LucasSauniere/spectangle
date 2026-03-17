"""
spectangle.utils.metrics
=========================
Quantitative quality metrics for evaluating spectral cube reconstructions.

Metrics
-------
``psnr``        — Peak Signal-to-Noise Ratio (dB)
``ssim_cube``   — Mean SSIM over all wavelength slices
``sam``         — Spectral Angle Mapper (degrees) — pure spectral fidelity
``rmse``        — Root Mean Square Error
``cube_metrics``— Convenience function returning all metrics in one dict
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity as skimage_ssim


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float | None = None) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    pred, target : ndarray
        Predicted and reference cubes (any shape, same dtype).
    data_range : float, optional
        Maximum possible value.  Defaults to ``target.max() - target.min()``.
    """
    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            return float("inf")
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def ssim_cube(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean SSIM computed per wavelength slice then averaged over λ.

    Parameters
    ----------
    pred, target : ndarray, shape (n_lambda, ny, nx)
    """
    scores = []
    data_range = float(target.max() - target.min())
    if data_range == 0:
        return 1.0 if np.allclose(pred, target) else 0.0
    for i in range(pred.shape[0]):
        s = skimage_ssim(
            pred[i], target[i],
            data_range=data_range,
        )
        scores.append(s)
    return float(np.mean(scores))


def spectral_angle_mapper(pred: np.ndarray, target: np.ndarray) -> float:
    """Spectral Angle Mapper (SAM) — mean angle in degrees.

    Measures the similarity between predicted and target spectra at each
    spatial pixel independently of amplitude.  0° = perfect, 90° = worst.

    Parameters
    ----------
    pred, target : ndarray, shape (n_lambda, ny, nx)
    """
    # Reshape to (n_lambda, n_pixels)
    n_lam, ny, nx = pred.shape
    p = pred.reshape(n_lam, -1).astype(np.float64)
    t = target.reshape(n_lam, -1).astype(np.float64)

    # Dot product and norms
    dot = (p * t).sum(axis=0)
    norm_p = np.linalg.norm(p, axis=0)
    norm_t = np.linalg.norm(t, axis=0)

    cos_theta = dot / (norm_p * norm_t + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_theta))

    # Ignore pixels with zero flux in the target
    valid = norm_t > 0
    return float(angles_deg[valid].mean()) if valid.any() else 0.0


def cube_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all standard reconstruction quality metrics.

    Parameters
    ----------
    pred, target : ndarray, shape (n_lambda, ny, nx)
        Predicted and ground-truth spectral cubes.

    Returns
    -------
    dict with keys: ``psnr``, ``rmse``, ``ssim``, ``sam``
        ``sam`` is the Spectral Angle Mapper in degrees (lower = better).
    """
    return {
        "psnr": psnr(pred, target),
        "rmse": rmse(pred, target),
        "ssim": ssim_cube(pred, target),
        "sam":  spectral_angle_mapper(pred, target),
    }
