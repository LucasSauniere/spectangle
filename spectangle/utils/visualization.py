"""
spectangle.utils.visualisation
================================
Plotting helpers for spectrograms, spectral cubes, and RGB previews.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


K_SEQUENCE_NAMES = ["RGS000+0", "RGS180+4", "RGS000-4", "RGS180+0"]


def plot_spectrograms(
    spectrograms: np.ndarray,
    titles: Optional[Sequence[str]] = None,
    cmap: str = "inferno",
    figsize: Tuple[float, float] = (14, 4),
    log_scale: bool = True,
    suptitle: str = "K-sequence spectrograms",
) -> plt.Figure:
    """Plot the four K-sequence dispersed images side by side.

    Parameters
    ----------
    spectrograms : ndarray, shape (4, ny, nx)
    titles : list of str, optional
    cmap : str
    figsize : tuple
    log_scale : bool
        Apply ``log1p`` before plotting for better dynamic range.
    suptitle : str

    Returns
    -------
    matplotlib Figure
    """
    titles = titles or K_SEQUENCE_NAMES
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, img, title in zip(axes, spectrograms, titles):
        data = np.log1p(img) if log_scale else img
        im = ax.imshow(data, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, y=1.01, fontsize=12)
    plt.tight_layout()
    return fig


def plot_cube_slice(
    cube: np.ndarray,
    wavelengths_AA: Optional[np.ndarray] = None,
    slice_indices: Optional[Sequence[int]] = None,
    cmap: str = "magma",
    figsize: Tuple[float, float] = (14, 3),
    suptitle: str = "Spectral cube — wavelength slices",
) -> plt.Figure:
    """Plot a few wavelength slices of a 3-D spectral cube.

    Parameters
    ----------
    cube : ndarray, shape (n_lambda, ny, nx)
    wavelengths_AA : ndarray, optional
    slice_indices : list of int, optional
        Indices along the spectral axis to show.  Defaults to 5 evenly
        spaced indices.
    """
    n_lam = cube.shape[0]
    if slice_indices is None:
        slice_indices = np.linspace(0, n_lam - 1, 5, dtype=int).tolist()

    n_plots = len(slice_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, slice_indices):
        ax.imshow(cube[idx], cmap=cmap, origin="lower")
        if wavelengths_AA is not None:
            ax.set_title(f"λ = {wavelengths_AA[idx]:.0f} Å", fontsize=9)
        else:
            ax.set_title(f"slice {idx}", fontsize=9)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    return fig


def plot_rgb(
    cube: np.ndarray,
    wavelengths_AA: Optional[np.ndarray] = None,
    r_range: Tuple[float, float] = (15_000, 18_500),
    g_range: Tuple[float, float] = (12_000, 15_000),
    b_range: Tuple[float, float] = (9_250, 12_000),
    figsize: Tuple[float, float] = (5, 5),
    title: str = "Broadband RGB preview",
) -> plt.Figure:
    """Create a false-colour RGB image by integrating the cube over sub-bands.

    Parameters
    ----------
    cube : ndarray, shape (n_lambda, ny, nx)
    wavelengths_AA : ndarray, shape (n_lambda,), optional
        If None, assumes linear spacing from 9250 to 18500 Å.
    r_range, g_range, b_range : tuple of float
        Wavelength ranges (Å) for the R, G, B channels respectively.
    """
    n_lam, ny, nx = cube.shape
    if wavelengths_AA is None:
        wavelengths_AA = np.linspace(9250, 18500, n_lam)

    def band_integrate(lo, hi):
        mask = (wavelengths_AA >= lo) & (wavelengths_AA <= hi)
        if not mask.any():
            return np.zeros((ny, nx))
        return cube[mask].sum(axis=0)

    r = band_integrate(*r_range)
    g = band_integrate(*g_range)
    b = band_integrate(*b_range)

    def normalise(arr):
        p1, p99 = np.percentile(arr, [1, 99])
        arr = np.clip((arr - p1) / (p99 - p1 + 1e-8), 0, 1)
        return arr

    rgb = np.stack([normalise(r), normalise(g), normalise(b)], axis=-1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, origin="lower")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_spectrum(
    cube: np.ndarray,
    x: int,
    y: int,
    wavelengths_AA: Optional[np.ndarray] = None,
    label: str = "Extracted spectrum",
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Plot the extracted 1-D spectrum at pixel (x, y).

    Parameters
    ----------
    cube : ndarray, shape (n_lambda, ny, nx)
    x, y : int
        Pixel coordinates.
    wavelengths_AA : ndarray, optional
    """
    spectrum = cube[:, y, x]
    x_axis = wavelengths_AA if wavelengths_AA is not None else np.arange(len(spectrum))
    xlabel = "Wavelength (Å)" if wavelengths_AA is not None else "Spectral pixel"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_axis, spectrum, lw=1.2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Flux (a.u.)")
    ax.set_title(f"Spectrum at pixel ({x}, {y})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    wavelengths_AA: Optional[np.ndarray] = None,
    slice_index: int | None = None,
    cmap: str = "magma",
    figsize: Tuple[float, float] = (14, 4),
) -> plt.Figure:
    """Side-by-side comparison of predicted and ground-truth cube slices.

    Shows (from left to right): prediction, ground truth, and absolute
    residual for a single wavelength slice.

    Parameters
    ----------
    pred : ndarray, shape (n_lambda, ny, nx)
        Predicted spectral cube.
    target : ndarray, shape (n_lambda, ny, nx)
        Ground-truth spectral cube.
    wavelengths_AA : ndarray, optional
        Wavelength axis.  Used only for the figure title.
    slice_index : int, optional
        Spectral slice to display.  Defaults to the midpoint.
    cmap : str
        Colormap.
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    n_lam = pred.shape[0]
    if slice_index is None:
        slice_index = n_lam // 2

    pred_sl = pred[slice_index]
    tgt_sl  = target[slice_index]
    res_sl  = np.abs(pred_sl - tgt_sl)

    lam_str = (
        f"λ = {wavelengths_AA[slice_index]:.0f} Å"
        if wavelengths_AA is not None
        else f"slice {slice_index}"
    )

    vmin = min(pred_sl.min(), tgt_sl.min())
    vmax = max(pred_sl.max(), tgt_sl.max())

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, title in zip(
        axes,
        [pred_sl, tgt_sl, res_sl],
        ["Prediction", "Ground Truth", "Residual |pred − truth|"],
    ):
        vn = 0 if title.startswith("Res") else vmin
        vx = res_sl.max() if title.startswith("Res") else vmax
        im = ax.imshow(img, cmap=cmap, origin="lower", vmin=vn, vmax=vx)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Cube reconstruction comparison — {lam_str}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_spectrum_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    x: int,
    y: int,
    wavelengths_AA: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Overlay predicted and ground-truth spectra at a single spatial pixel.

    Parameters
    ----------
    pred, target : ndarray, shape (n_lambda, ny, nx)
    x, y : int
        Pixel coordinates.
    wavelengths_AA : ndarray, optional
    """
    pred_spec   = pred[:, y, x]
    target_spec = target[:, y, x]
    x_axis = wavelengths_AA if wavelengths_AA is not None else np.arange(len(pred_spec))
    xlabel = "Wavelength (Å)" if wavelengths_AA is not None else "Spectral pixel"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_axis, target_spec, lw=1.5, label="Ground truth", color="steelblue")
    ax.plot(x_axis, pred_spec,   lw=1.2, label="Prediction",   color="tomato",
            linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Flux (a.u.)")
    ax.set_title(f"Spectral comparison at pixel ({x}, {y})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
