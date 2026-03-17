"""spectangle.physics.psf

PSF models with complexity levels.

Complexity levels
-----------------
  0  GaussianPSF      — symmetric 2-D Gaussian; the simplest model, fast and
                        fully analytic.  Controlled by a single FWHM parameter.
  1  MoffatPSF        — Moffat profile (power-law wings); better approximation
                        of real optical PSFs in ground- or space-based systems.
  2  EuclidNISPPSF    — parametric model that combines a Moffat core with a
                        wavelength-dependent FWHM scaling law and an optional
                        ellipticity component to mimic Euclid NISP detector PSF.

Factory
-------
  make_psf(complexity, **kwargs) → PSFModel subclass
  PSFModel (abstract base)       — unified interface: .kernel, .convolve(),
                                   .convolve_cube(), .at_wavelength()
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PSFModel:
    """Base class / complexity-0 entry point.

    All subclasses expose:
      - ``kernel``        : 2-D ndarray, the normalised PSF stamp.
      - ``convolve``      : 2-D image → 2-D image (same shape).
      - ``convolve_cube`` : (n_lam, ny, nx) → (n_lam, ny, nx).
      - ``at_wavelength`` : return a new PSFModel instance scaled for a given λ.

    Instantiating ``PSFModel`` directly gives a **complexity-0 Gaussian PSF**.
    Use :func:`make_psf` or one of the subclasses for higher complexity.

    Parameters
    ----------
    fwhm_pixels : float
        Full-width at half-maximum in pixels.
    kernel_size : int, optional
        Size of the square kernel (forced to odd).  Computed automatically
        from the FWHM when not given (≥ 5 σ coverage).
    """
    COMPLEXITY: ClassVar[int] = 0

    def __init__(
        self,
        fwhm_pixels: float = 1.6,
        kernel_size: int | None = None,
    ) -> None:
        self.fwhm_pixels = float(fwhm_pixels)
        sigma = self.fwhm_pixels / 2.3548
        ks = int(max(3, np.ceil(sigma * 8)))
        if ks % 2 == 0:
            ks += 1
        if kernel_size is not None:
            ks = max(3, int(kernel_size) | 1)   # ensure odd and ≥ 3
        self.kernel_size = ks
        self.kernel = self._make_kernel()

    # --- internal -----------------------------------------------------------

    def _make_kernel(self) -> np.ndarray:
        """Build the (kernel_size × kernel_size) Gaussian kernel."""
        sigma = self.fwhm_pixels / 2.3548
        ctr = self.kernel_size // 2
        y, x = np.ogrid[: self.kernel_size, : self.kernel_size]
        r2 = (x - ctr) ** 2 + (y - ctr) ** 2
        k = np.exp(-0.5 * r2 / sigma ** 2)
        return (k / k.sum()).astype(np.float32)

    # --- public interface ---------------------------------------------------

    def convolve(self, image: np.ndarray) -> np.ndarray:
        """Convolve a 2-D image with the PSF kernel (FFT-based, 'same' output)."""
        return fftconvolve(image, self.kernel, mode="same").astype(np.float32)

    def convolve_cube(self, cube: np.ndarray) -> np.ndarray:
        """Convolve every spectral slice of *cube* with the PSF.

        Parameters
        ----------
        cube : ndarray, shape (n_lam, ny, nx)

        Returns
        -------
        ndarray, shape (n_lam, ny, nx), float32
        """
        out = np.empty_like(cube, dtype=np.float32)
        for i in range(cube.shape[0]):
            out[i] = self.convolve(cube[i])
        return out

    def at_wavelength(
        self,
        wavelength_AA: float,
        ref_wavelength_AA: float = 13500.0,
    ) -> "PSFModel":
        """Return a new PSF instance with FWHM scaled to *wavelength_AA*.

        Uses the diffraction-limited scaling FWHM ∝ λ (λ / D).

        Parameters
        ----------
        wavelength_AA : float
        ref_wavelength_AA : float
            Reference wavelength at which ``self.fwhm_pixels`` is defined.
        """
        scale = float(wavelength_AA) / float(ref_wavelength_AA)
        return PSFModel(fwhm_pixels=self.fwhm_pixels * scale)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fwhm={self.fwhm_pixels:.3f}px, "
            f"ks={self.kernel_size}) [complexity={self.COMPLEXITY}]"
        )


# ---------------------------------------------------------------------------
# Complexity 1 — Moffat PSF
# ---------------------------------------------------------------------------

class MoffatPSF(PSFModel):
    """Complexity-1 PSF: Moffat profile.

    The Moffat function is I(r) ∝ (1 + (r/α)²)^{-β}, which captures the
    power-law wings of realistic telescope PSFs better than a pure Gaussian.

    Parameters
    ----------
    fwhm_pixels : float
        FWHM in pixels; determines the Moffat scale radius α.
    beta : float
        Power-law exponent.  β ≈ 4.765 is the theoretical Kolmogorov limit
        for atmospheric seeing; β ≈ 3–10 is typical for space telescopes.
    kernel_size : int, optional
        Square kernel size (forced to odd).
    """
    COMPLEXITY = 1

    def __init__(
        self,
        fwhm_pixels: float = 1.6,
        beta: float = 4.765,
        kernel_size: int | None = None,
    ) -> None:
        self.beta = float(beta)
        # Compute Moffat scale α from FWHM and β
        # FWHM = 2 α √(2^{1/β} − 1)
        self._alpha = float(fwhm_pixels) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        # Let parent store fwhm_pixels and auto-choose kernel_size
        super().__init__(fwhm_pixels=fwhm_pixels, kernel_size=kernel_size)

    def _make_kernel(self) -> np.ndarray:
        ctr = self.kernel_size // 2
        y, x = np.ogrid[: self.kernel_size, : self.kernel_size]
        r2 = (x - ctr) ** 2 + (y - ctr) ** 2
        k = (1.0 + r2 / self._alpha ** 2) ** (-self.beta)
        return (k / k.sum()).astype(np.float32)

    def at_wavelength(
        self,
        wavelength_AA: float,
        ref_wavelength_AA: float = 13500.0,
    ) -> "MoffatPSF":
        scale = float(wavelength_AA) / float(ref_wavelength_AA)
        return MoffatPSF(fwhm_pixels=self.fwhm_pixels * scale, beta=self.beta)


# ---------------------------------------------------------------------------
# Complexity 2 — Euclid NISP-like PSF
# ---------------------------------------------------------------------------

class EuclidNISPPSF(PSFModel):
    """Complexity-2 PSF: simplified Euclid NISP near-IR detector PSF.

    Models the NISP PSF as a Moffat core with:
    * wavelength-dependent FWHM: FWHM(λ) = fwhm_ref × (λ/λ_ref)^power
    * optional mild ellipticity (elongation along a position angle PA).
    * optional diffraction ring halo (Airy-like contribution).

    Parameters
    ----------
    fwhm_ref_pixels : float
        PSF FWHM in pixels at *ref_wavelength_AA*.
    ref_wavelength_AA : float
        Reference wavelength (Å) at which *fwhm_ref_pixels* is defined.
        Default is 13 500 Å (NISP H-band centre).
    beta : float
        Moffat β exponent.
    power : float
        Wavelength-scaling exponent for FWHM (1.0 = diffraction-limited).
    ellipticity : float
        Axis ratio ε = b/a ∈ (0, 1].  1.0 = circular.
    pa_deg : float
        Position angle of the major axis (degrees, measured N→E).
    kernel_size : int, optional
        Square kernel size (odd).  Auto-computed when not given.
    wavelength_AA : float, optional
        Wavelength at which to evaluate the PSF.  When supplied, the FWHM is
        scaled accordingly.
    """
    COMPLEXITY = 2

    def __init__(
        self,
        fwhm_ref_pixels: float = 1.6,
        ref_wavelength_AA: float = 13500.0,
        beta: float = 3.5,
        power: float = 1.0,
        ellipticity: float = 1.0,
        pa_deg: float = 0.0,
        kernel_size: int | None = None,
        wavelength_AA: float | None = None,
    ) -> None:
        self.fwhm_ref_pixels = float(fwhm_ref_pixels)
        self.ref_wavelength_AA = float(ref_wavelength_AA)
        self.beta = float(beta)
        self.power = float(power)
        self.ellipticity = float(ellipticity)   # b/a
        self.pa_deg = float(pa_deg)

        # Effective FWHM at requested wavelength
        lam = float(wavelength_AA) if wavelength_AA is not None else ref_wavelength_AA
        scale = (lam / self.ref_wavelength_AA) ** self.power
        effective_fwhm = self.fwhm_ref_pixels * scale

        # Moffat scale radius
        self._alpha = effective_fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))

        super().__init__(fwhm_pixels=effective_fwhm, kernel_size=kernel_size)

    def _make_kernel(self) -> np.ndarray:
        ctr = self.kernel_size // 2
        y, x = np.ogrid[: self.kernel_size, : self.kernel_size]
        dy = y - ctr
        dx = x - ctr
        # Rotate coordinates to major-axis frame
        pa_rad = np.deg2rad(self.pa_deg)
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
        x_rot = dx * cos_pa - dy * sin_pa
        y_rot = dx * sin_pa + dy * cos_pa
        # Apply ellipticity along rotated x axis (major) — squeeze y by 1/ε
        eps = max(0.01, min(1.0, self.ellipticity))
        r2 = x_rot ** 2 + (y_rot / eps) ** 2
        k = (1.0 + r2 / self._alpha ** 2) ** (-self.beta)
        return (k / k.sum()).astype(np.float32)

    def at_wavelength(
        self,
        wavelength_AA: float,
        ref_wavelength_AA: float | None = None,
    ) -> "EuclidNISPPSF":
        ref = ref_wavelength_AA if ref_wavelength_AA is not None else self.ref_wavelength_AA
        return EuclidNISPPSF(
            fwhm_ref_pixels=self.fwhm_ref_pixels,
            ref_wavelength_AA=ref,
            beta=self.beta,
            power=self.power,
            ellipticity=self.ellipticity,
            pa_deg=self.pa_deg,
            wavelength_AA=wavelength_AA,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_psf(complexity: int = 0, **kwargs) -> PSFModel:
    """Instantiate a PSF model at the requested complexity level.

    Parameters
    ----------
    complexity : {0, 1, 2}
        0 → :class:`PSFModel` (Gaussian),
        1 → :class:`MoffatPSF`,
        2 → :class:`EuclidNISPPSF`.
    **kwargs
        Forwarded to the chosen class constructor.

    Examples
    --------
    >>> psf0 = make_psf(0, fwhm_pixels=1.4)
    >>> psf1 = make_psf(1, fwhm_pixels=1.4, beta=3.5)
    >>> psf2 = make_psf(2, fwhm_ref_pixels=1.4, beta=3.5, ellipticity=0.9)
    """
    _classes = {0: PSFModel, 1: MoffatPSF, 2: EuclidNISPPSF}
    if complexity not in _classes:
        raise ValueError(f"complexity must be 0, 1 or 2; got {complexity!r}")
    return _classes[complexity](**kwargs)
