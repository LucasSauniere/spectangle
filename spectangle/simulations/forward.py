"""
spectangle.simulations.forward
===============================
Core forward model: spectral cube → dispersed 2-D spectrograms.

Physics summary
---------------
Given a 3-D spectral cube F(x, y, λ) representing the astrophysical scene,
the dispersed image for grism orientation θ is:

    I_θ(u, v) = ∫ F(x, y, λ) · PSF(u − x − D_θ(λ)cos θ,
                                       v − y − D_θ(λ)sin θ) dλ dx dy

In the discrete (pixel-space) approximation we:
  1. Project each wavelength slice of the cube onto a 2-D image, shifted by
     the dispersion offset D(λ) along (cos θ, sin θ).
  2. Convolve with the PSF kernel.
  3. Sum over wavelengths.

The forward model also supports:
  * Multiple diffraction orders (each with its own dispersion factor and
    efficiency curve).
  * An optional 0th-order "peanut" morphology (2× PSF FWHM, broadened).

Spectrogram shape
-----------------
Spectral dispersion physically elongates the footprint of any source along
the dispersion direction.  Even for a compact (point) source, its spectrum
spans D(λ_max) − D(λ_min) pixels; for an extended source the dispersed image
is the convolution of the spatial extent with the spectral trace.

To avoid silently truncating spectra that fall outside the input scene
boundaries, the output spectrogram array is padded by ``pad_pixels`` on all
sides (default: half the spectral length, rounded up to the next even number).
The spectrogram shape is therefore

    (ny + 2*pad_y, nx + 2*pad_x)

where pad_x = ceil(|max D_x(λ)|) and pad_y = ceil(|max D_y(λ)|)
over the full wavelength grid and all active orders.

The ``spectrogram_shape`` property exposes this padded shape and can be used
by downstream callers (dataset, ML models) to set the correct tensor size.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import shift as ndimage_shift

from spectangle.physics.dispersion import DispersionModel, KSequence
from spectangle.physics.psf import PSFModel


# ---------------------------------------------------------------------------
# Diffraction-order efficiency curves (approximate)
# ---------------------------------------------------------------------------

def _order_efficiency(order: int, wavelengths_AA: np.ndarray) -> np.ndarray:
    """Return the blaze efficiency for a given diffraction order.

    These are rough analytic approximations; replace with measured NISP
    calibration curves when available.

    Parameters
    ----------
    order : int
        Diffraction order (0, 1, 2, ...).
    wavelengths_AA : ndarray
        Wavelength array in ångströms.

    Returns
    -------
    ndarray
        Efficiency in [0, 1], same shape as ``wavelengths_AA``.
    """
    if order == 0:
        # 0th order: flat, ~5 % throughput
        return np.full_like(wavelengths_AA, 0.05, dtype=float)
    elif order == 1:
        # 1st order: blaze peak at band centre, Gaussian envelope
        lam_centre = 13_500.0  # Å
        sigma = 3_000.0         # Å
        eff = 0.6 * np.exp(-0.5 * ((wavelengths_AA - lam_centre) / sigma) ** 2)
        return eff
    elif order == 2:
        # 2nd order: weaker, shifted blaze
        lam_centre = 10_000.0
        sigma = 2_000.0
        eff = 0.1 * np.exp(-0.5 * ((wavelengths_AA - lam_centre) / sigma) ** 2)
        return eff
    else:
        return np.zeros_like(wavelengths_AA, dtype=float)


# ---------------------------------------------------------------------------
# ForwardModel
# ---------------------------------------------------------------------------

class ForwardModel:
    """Simulate dispersed 2-D spectrograms from a 3-D spectral cube.

    Parameters
    ----------
    ksequence : KSequence
        Dispersion models for the four K-sequence grism orientations.
    psf_model : PSFModel
        Spatial PSF applied to each wavelength slice.
    image_shape : tuple of int
        (ny, nx) — shape of the **scene** (input cube spatial dimensions).
    orders : list of int
        Diffraction orders to simulate.  Use ``[1]`` for the simple
        (1st-order only) simulator and ``[0, 1, 2]`` for the complex one.

    Attributes
    ----------
    spectrogram_shape : tuple of int
        (ny_spec, nx_spec) — the output spectrogram shape, which is larger
        than ``image_shape`` by the dispersion padding in each axis.
    pad_y, pad_x : int
        Pixel padding added on each side (top/bottom and left/right).
    """

    def __init__(
        self,
        ksequence: KSequence,
        psf_model: PSFModel,
        image_shape: Tuple[int, int],
        orders: List[int] = None,
    ) -> None:
        self.ksequence = ksequence
        self.psf_model = psf_model
        self.image_shape = image_shape  # (ny, nx) of the scene cube
        self.orders = orders if orders is not None else [1]

        # ------------------------------------------------------------------
        # Compute the dispersion padding required so that spectra from any
        # source position are never clipped at the image boundary.
        #
        # For each K-step and each active order, the maximum pixel shift of a
        # spectral trace is:
        #     max |D(λ)| * order   along x  →  pad_x
        #     max |D(λ)| * order   along y  →  pad_y
        #
        # We take the ceiling and ensure a minimum of 1 pixel padding so the
        # spectrogram is always strictly larger than the scene, making the
        # rectangular footprint of dispersion visible.
        # ------------------------------------------------------------------
        max_pad_x = 0
        max_pad_y = 0
        # Sample the wavelength grid from the first model in the sequence
        _wav = ksequence.wavelength_grid(256)
        for disp in ksequence:
            for order in self.orders:
                offsets = disp.wavelength_to_offset(_wav)
                px = int(math.ceil(np.abs(offsets.delta_x * order).max()))
                py = int(math.ceil(np.abs(offsets.delta_y * order).max()))
                max_pad_x = max(max_pad_x, px)
                max_pad_y = max(max_pad_y, py)

        # Ensure at least 1 pixel of padding so the shape is always rectangular
        self.pad_x: int = max(max_pad_x, 1)
        self.pad_y: int = max(max_pad_y, 1)

        ny, nx = image_shape
        self.spectrogram_shape: Tuple[int, int] = (
            ny + 2 * self.pad_y,
            nx + 2 * self.pad_x,
        )

    # ------------------------------------------------------------------

    def project_cube(
        self,
        cube: np.ndarray,
        wavelengths_AA: np.ndarray,
        dispersion_model: DispersionModel,
        order: int = 1,
    ) -> np.ndarray:
        """Project one spectral cube onto a 2-D detector image.

        The output array is padded so that spectra extending beyond the scene
        boundary are preserved rather than truncated.

        Parameters
        ----------
        cube : ndarray, shape (n_lambda, ny, nx)
            Spectral cube where ``cube[i]`` is the flux image at
            ``wavelengths_AA[i]``.
        wavelengths_AA : ndarray, shape (n_lambda,)
            Wavelength grid.
        dispersion_model : DispersionModel
            Dispersion law + direction for this grism orientation.
        order : int
            Diffraction order; controls dispersion offset scaling and
            blaze efficiency.

        Returns
        -------
        ndarray, shape (ny + 2*pad_y, nx + 2*pad_x)
            Simulated dispersed image on the padded canvas.
        """
        ny_out, nx_out = self.spectrogram_shape
        image = np.zeros((ny_out, nx_out), dtype=np.float64)

        efficiency = _order_efficiency(order, wavelengths_AA)

        for i, (lam, eff) in enumerate(zip(wavelengths_AA, efficiency)):
            if eff == 0:
                continue

            # Pixel shift for this wavelength and order
            # Higher orders scale the offset: D_n(λ) = n · D_1(λ)
            offsets = dispersion_model.wavelength_to_offset(lam)
            dx = offsets.delta_x * order
            dy = offsets.delta_y * order

            # Embed the scene slice into the centre of the padded canvas
            slice_2d = cube[i] * eff
            padded = np.pad(
                slice_2d,
                ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x)),
                mode="constant",
                constant_values=0.0,
            )

            # Shift along the dispersion direction; scipy uses (row, col) = (y, x)
            # mode='constant' with cval=0 drops flux shifted outside the canvas
            shifted = ndimage_shift(
                padded, shift=(dy, dx), order=1, mode="constant", cval=0.0
            )

            # 0th-order: broaden PSF to simulate peanut morphology
            if order == 0:
                broad_psf = PSFModel(
                    fwhm_pixels=self.psf_model.fwhm_pixels * 2.0,
                    kernel_size=self.psf_model.kernel_size,
                )
                shifted = broad_psf.convolve(shifted)
            else:
                shifted = self.psf_model.convolve(shifted)

            image += shifted

        return image

    def __call__(
        self,
        cube: np.ndarray,
        wavelengths_AA: np.ndarray,
    ) -> np.ndarray:
        """Run the full K-sequence forward model.

        Parameters
        ----------
        cube : ndarray, shape (n_lambda, ny, nx)
            Input 3-D spectral cube.
        wavelengths_AA : ndarray, shape (n_lambda,)
            Wavelength grid in ångströms.

        Returns
        -------
        spectrograms : ndarray, shape (4, ny + 2*pad_y, nx + 2*pad_x)
            Stack of four dispersed images (one per K-sequence step).
            The extra padding along each axis is ``self.pad_y`` (top & bottom)
            and ``self.pad_x`` (left & right).
        """
        ny_out, nx_out = self.spectrogram_shape
        spectrograms = np.zeros((4, ny_out, nx_out), dtype=np.float64)

        for k, disp_model in enumerate(self.ksequence):
            for order in self.orders:
                spectrograms[k] += self.project_cube(
                    cube, wavelengths_AA, disp_model, order=order
                )

        return spectrograms

    def forward_with_direct(
        self,
        cube: np.ndarray,
        wavelengths_AA: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward model returning both the 4 spectrograms and the direct image.

        The direct (undispersed) image is the wavelength-integrated cube
        convolved with the PSF — it acts as a spatial prior in the 5-channel
        model variant.  It is **also padded** to match the spectrogram shape
        so all 5 input channels have the same spatial dimensions.

        Returns
        -------
        spectrograms : ndarray, shape (4, ny + 2*pad_y, nx + 2*pad_x)
        direct_image : ndarray, shape (ny + 2*pad_y, nx + 2*pad_x)
        """
        spectrograms = self(cube, wavelengths_AA)
        # Integrate cube over wavelength and convolve with PSF
        broadband = cube.sum(axis=0)  # (ny, nx)
        broadband_convolved = self.psf_model.convolve(broadband)
        # Pad to match spectrogram shape
        direct_image = np.pad(
            broadband_convolved,
            ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x)),
            mode="constant",
            constant_values=0.0,
        )
        return spectrograms, direct_image
