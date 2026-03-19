"""spectangle.simulations.sources

Source morphology models with complexity levels.

Complexity levels
-----------------
  0  GaussianSource   — 2-D isotropic Gaussian profile; single sigma parameter.
                        Simplest model; no PSF convolution needed for basic tests.
  1  SersicSource      — 2-D Sérsic profile; index n controls concentration.
                        n=0.5 → Gaussian, n=1 → exponential disc, n=4 → de Vaucouleurs.
  2  PhysicalSource    — Physically motivated morphologies including multi-component
                        models: point sources, exponential discs, rings (planetary
                        nebulae), spiral-arm perturbations, and emission-line
                        nebulae.

All source models share a common interface:

    stamp = model.render(ny, nx, xc, yc)

Returns a (ny, nx) float32 stamp normalised to unit sum.

Factory
-------
  make_source(source_type, complexity, **kwargs) → SourceModel
  source_types: "point", "galaxy", "nebula", "disc", "ring"
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indices(ny: int, nx: int, xc: float, yc: float):
    """Return (dx, dy) grids relative to (xc, yc)."""
    y, x = np.indices((ny, nx), dtype=float)
    return x - xc, y - yc


def _safe_normalise(img: np.ndarray) -> np.ndarray:
    """Normalise to unit sum; return all-zeros if sum is zero."""
    total = float(img.sum())
    if total <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return (img / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Complexity 0 — Gaussian source
# ---------------------------------------------------------------------------

class GaussianSource:
    """Complexity-0 source: 2-D isotropic Gaussian.

    This is the simplest possible representation; equivalent to a marginally
    resolved point source convolved with a circular Gaussian beam.

    Parameters
    ----------
    sigma_pixels : float
        Gaussian 1-σ radius in pixels.  Use small values (≲ 1) for point-like
        sources, larger values for extended galaxies.
    """
    COMPLEXITY: ClassVar[int] = 0

    def __init__(self, sigma_pixels: float = 0.5):
        self.sigma_pixels = float(sigma_pixels)

    def render(self, ny: int, nx: int, xc: float, yc: float) -> np.ndarray:
        """Render a (ny x nx) stamp centred at pixel (xc, yc).

        Returns
        -------
        stamp : ndarray, shape (ny, nx), float32, normalised to unit sum.
        """
        dx, dy = _indices(ny, nx, xc, yc)
        sig = max(self.sigma_pixels, 1e-6)
        stamp = np.exp(-0.5 * (dx ** 2 + dy ** 2) / sig ** 2)
        return _safe_normalise(stamp)

    def __repr__(self) -> str:
        return f"GaussianSource(sigma={self.sigma_pixels:.2f}px) [complexity=0]"


# ---------------------------------------------------------------------------
# Complexity 1 — Sérsic source
# ---------------------------------------------------------------------------

class SersicSource:
    """Complexity-1 source: 2-D Sérsic profile.

    The Sérsic law is:
        I(r) = I_e x exp{ -b_n x [ (r / r_e)^{1/n} − 1 ] }

    where b_n ≈ 1.9992 n − 0.3271 (Capaccioli 1989 approximation for n ≥ 0.5).
    The half-light radius r_e is given in pixels.

    Parameters
    ----------
    r_e_pixels : float
        Effective (half-light) radius in pixels.
    sersic_n : float
        Sérsic index.  n=0.5 ≈ Gaussian, n=1 → exponential, n=4 → de Vaucouleurs.
    ellipticity : float
        Axis ratio b/a ∈ (0, 1].
    pa_deg : float
        Position angle of the major axis (degrees N→E).
    """
    COMPLEXITY: ClassVar[int] = 1

    def __init__(
        self,
        r_e_pixels: float = 3.0,
        sersic_n: float = 1.0,
        ellipticity: float = 1.0,
        pa_deg: float = 0.0,
    ):
        self.r_e_pixels = float(r_e_pixels)
        self.sersic_n = float(sersic_n)
        self.ellipticity = float(ellipticity)
        self.pa_deg = float(pa_deg)
        # Capaccioli approximation for b_n
        self._bn = 1.9992 * self.sersic_n - 0.3271

    def render(self, ny: int, nx: int, xc: float, yc: float) -> np.ndarray:
        dx, dy = _indices(ny, nx, xc, yc)
        pa_rad = np.deg2rad(self.pa_deg)
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
        x_rot = dx * cos_pa - dy * sin_pa
        y_rot = dx * sin_pa + dy * cos_pa
        eps = max(0.01, min(1.0, self.ellipticity))
        r = np.sqrt(x_rot ** 2 + (y_rot / eps) ** 2)
        re = max(self.r_e_pixels, 1e-6)
        stamp = np.exp(-self._bn * ((r / re) ** (1.0 / self.sersic_n) - 1.0))
        stamp = np.nan_to_num(stamp, nan=0.0, posinf=0.0, neginf=0.0)
        return _safe_normalise(stamp)

    def __repr__(self) -> str:
        return (
            f"SersicSource(r_e={self.r_e_pixels:.1f}px, n={self.sersic_n:.1f}, "
            f"eps={self.ellipticity:.2f}) [complexity=1]"
        )


# ---------------------------------------------------------------------------
# Complexity 2 — Physical multi-component sources
# ---------------------------------------------------------------------------

class PointSource:
    """Complexity-2 point source: infinitely thin delta function.

    In practice the emission is placed in a single pixel; PSF convolution
    is the only mechanism that spreads the light.  This is the canonical
    representation of an unresolved star.

    Parameters
    ----------
    (none — all spatial information is encoded in xc, yc passed to render)
    """
    COMPLEXITY: ClassVar[int] = 2

    def render(self, ny: int, nx: int, xc: float, yc: float) -> np.ndarray:
        stamp = np.zeros((ny, nx), dtype=np.float32)
        xi, yi = int(round(xc)), int(round(yc))
        xi = int(np.clip(xi, 0, nx - 1))
        yi = int(np.clip(yi, 0, ny - 1))
        stamp[yi, xi] = 1.0
        return stamp

    def __repr__(self) -> str:
        return "PointSource() [complexity=2]"


class DiscGalaxy:
    """Complexity-2 galaxy: exponential disc + optional de Vaucouleurs bulge.

    The surface brightness is a superposition of:
    * exponential disc:  I_d(r) ∝ exp(-r / h_r)
    * de Vaucouleurs bulge: I_b(r) ∝ exp(-7.67 x [(r / r_b)^{1/4} − 1])

    Parameters
    ----------
    h_r_pixels : float
        Disc scale length in pixels.
    r_b_pixels : float
        Bulge effective radius in pixels.  Set to 0 to disable the bulge.
    bulge_fraction : float
        Fraction of total flux in the bulge ∈ [0, 1].
    ellipticity : float
        Disc axis ratio b/a ∈ (0, 1].
    pa_deg : float
        Position angle of the disc major axis (degrees N→E).
    """
    COMPLEXITY: ClassVar[int] = 2

    def __init__(
        self,
        h_r_pixels: float = 4.0,
        r_b_pixels: float = 1.5,
        bulge_fraction: float = 0.3,
        ellipticity: float = 0.7,
        pa_deg: float = 30.0,
    ):
        self.h_r_pixels = float(h_r_pixels)
        self.r_b_pixels = float(r_b_pixels)
        self.bulge_fraction = float(np.clip(bulge_fraction, 0.0, 1.0))
        self.ellipticity = float(ellipticity)
        self.pa_deg = float(pa_deg)

    def render(self, ny: int, nx: int, xc: float, yc: float) -> np.ndarray:
        dx, dy = _indices(ny, nx, xc, yc)
        pa_rad = np.deg2rad(self.pa_deg)
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
        x_rot = dx * cos_pa - dy * sin_pa
        y_rot = dx * sin_pa + dy * cos_pa
        eps = max(0.01, min(1.0, self.ellipticity))
        r_disc = np.sqrt(x_rot ** 2 + (y_rot / eps) ** 2)
        r_sph = np.sqrt(dx ** 2 + dy ** 2)

        disc = np.exp(-r_disc / max(self.h_r_pixels, 1e-6))
        if self.r_b_pixels > 0:
            bulge = np.exp(-7.67 * ((r_sph / max(self.r_b_pixels, 1e-6)) ** 0.25 - 1.0))
            bulge = np.nan_to_num(bulge, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            bulge = np.zeros_like(disc)

        bf = self.bulge_fraction
        stamp = (1 - bf) * disc + bf * bulge
        return _safe_normalise(stamp)

    def __repr__(self) -> str:
        return (
            f"DiscGalaxy(h_r={self.h_r_pixels:.1f}px, r_b={self.r_b_pixels:.1f}px, "
            f"bf={self.bulge_fraction:.2f}) [complexity=2]"
        )


class RingNebula:
    """Complexity-2 planetary nebula: emission-line ring with central star.

    The morphology is a thin annulus (the nebular shell) plus an optional
    central point source (the white dwarf remnant).

    Parameters
    ----------
    inner_r_pixels : float
        Inner radius of the ring in pixels.
    outer_r_pixels : float
        Outer radius of the ring in pixels.
    central_fraction : float
        Fraction of total flux in the central point source ∈ [0, 1].
    ellipticity : float
        Ring axis ratio b/a ∈ (0, 1].
    pa_deg : float
        Position angle of the ring major axis (degrees N→E).
    """
    COMPLEXITY: ClassVar[int] = 2

    def __init__(
        self,
        inner_r_pixels: float = 3.0,
        outer_r_pixels: float = 6.0,
        central_fraction: float = 0.05,
        ellipticity: float = 1.0,
        pa_deg: float = 0.0,
    ):
        self.inner_r_pixels = float(inner_r_pixels)
        self.outer_r_pixels = float(outer_r_pixels)
        self.central_fraction = float(np.clip(central_fraction, 0.0, 1.0))
        self.ellipticity = float(ellipticity)
        self.pa_deg = float(pa_deg)

    def render(self, ny: int, nx: int, xc: float, yc: float) -> np.ndarray:
        dx, dy = _indices(ny, nx, xc, yc)
        pa_rad = np.deg2rad(self.pa_deg)
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
        x_rot = dx * cos_pa - dy * sin_pa
        y_rot = dx * sin_pa + dy * cos_pa
        eps = max(0.01, min(1.0, self.ellipticity))
        r = np.sqrt(x_rot ** 2 + (y_rot / eps) ** 2)

        # Shell: soft annulus (raised cosine profile between inner and outer)
        ring_mask = (r >= self.inner_r_pixels) & (r <= self.outer_r_pixels)
        ring = np.where(ring_mask, 1.0, 0.0).astype(float)

        # Central point source
        central = np.zeros((ny, nx), dtype=float)
        xi, yi = int(round(xc)), int(round(yc))
        xi = int(np.clip(xi, 0, nx - 1))
        yi = int(np.clip(yi, 0, ny - 1))
        central[yi, xi] = 1.0

        cf = self.central_fraction
        if ring.sum() > 0 and central.sum() > 0:
            stamp = (1.0 - cf) * ring / ring.sum() + cf * central
        elif ring.sum() > 0:
            stamp = ring
        else:
            stamp = central

        return _safe_normalise(stamp)

    def __repr__(self) -> str:
        return (
            f"RingNebula(r=[{self.inner_r_pixels:.1f}, {self.outer_r_pixels:.1f}]px, "
            f"cf={self.central_fraction:.2f}) [complexity=2]"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SOURCE_REGISTRY: dict[tuple[str, int], type] = {
    # (source_type, complexity) → class
    ("point",  0): GaussianSource,
    ("point",  1): SersicSource,     # very concentrated Sérsic (n~0.5)
    ("point",  2): PointSource,
    ("galaxy", 0): GaussianSource,
    ("galaxy", 1): SersicSource,
    ("galaxy", 2): DiscGalaxy,
    ("disc",   0): GaussianSource,
    ("disc",   1): SersicSource,
    ("disc",   2): DiscGalaxy,
    ("nebula", 0): GaussianSource,
    ("nebula", 1): SersicSource,     # ring-like Sérsic
    ("nebula", 2): RingNebula,
    ("ring",   0): GaussianSource,
    ("ring",   1): SersicSource,
    ("ring",   2): RingNebula,
}


def make_source(
    source_type: str = "point",
    complexity: int = 0,
    **kwargs,
):
    """Instantiate a source morphology model.

    Parameters
    ----------
    source_type : {"point", "galaxy", "disc", "nebula", "ring"}
    complexity : {0, 1, 2}
        0 → Gaussian (fastest, no physics).
        1 → Sérsic / intermediate model.
        2 → Physically motivated model (PointSource, DiscGalaxy, RingNebula).
    **kwargs
        Passed verbatim to the chosen class constructor.

    Examples
    --------
    >>> s0 = make_source("point",  0, sigma_pixels=0.5)
    >>> s1 = make_source("galaxy", 1, r_e_pixels=4, sersic_n=1.5)
    >>> s2 = make_source("nebula", 2, inner_r_pixels=4, outer_r_pixels=8)

    Notes
    -----
    When ``complexity=0`` for *any* source type the result is a
    ``GaussianSource``.  The ``sigma_pixels`` keyword can be used to tune how
    concentrated vs extended the source appears (≲ 1 for point-like, ≳ 3 for
    extended).
    """
    key = (source_type.lower(), complexity)
    if key not in _SOURCE_REGISTRY:
        valid_types = sorted({k[0] for k in _SOURCE_REGISTRY})
        raise ValueError(
            f"Unknown (source_type, complexity)=({source_type!r}, {complexity}). "
            f"Valid source_types: {valid_types}; complexity: 0, 1, 2."
        )
    cls = _SOURCE_REGISTRY[key]
    # Apply sensible defaults for point-like complexity-1 Sérsic
    if cls is SersicSource and source_type == "point":
        kwargs.setdefault("sersic_n", 0.5)
        kwargs.setdefault("r_e_pixels", 1.0)
    return cls(**kwargs)
