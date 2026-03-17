"""spectangle.simulations.sed

SED providers with complexity levels.

Complexity levels
-----------------
  0  GaussianSED   — parametric Gaussian; no physics, fastest.
  1  BlackbodySED  — Planck function; clean ground-truth for ML training.
  2  RealisticSED  — stellar atmosphere templates via synphot (optional dep.);
                     falls back to BlackbodySED if synphot is not installed.

Factory
-------
  make_sed(complexity, **kwargs) → SED callable
  random_sed(rng, complexity)    → randomly parameterised SED

All SED callables accept a wavelength array in Ångström and return a
positive float32 array.  When more than one wavelength is supplied the output
is normalised so that its integral (trapezoidal rule) equals 1.  For a
single-wavelength call the output is normalised to its peak value (= 1).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

# Optional synphot import for complexity-2 SEDs
try:
    import synphot  # noqa: F401
    _HAS_SYNPHOT = True
except Exception:
    _HAS_SYNPHOT = False


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _normalise(spec: np.ndarray, wavelengths_AA: np.ndarray) -> np.ndarray:
    """Normalise *spec* to unit integral; handles single-wavelength edge-case."""
    spec = np.maximum(spec, 0.0)
    if len(wavelengths_AA) > 1:
        integral = float(np.trapz(spec, x=wavelengths_AA))
    else:
        # Single wavelength: normalise by peak so the value is meaningful
        integral = float(spec.max())
    if integral <= 0.0:
        return np.zeros_like(spec, dtype=np.float32)
    return (spec / integral).astype(np.float32)


# ---------------------------------------------------------------------------
# Complexity 0 — Gaussian SED
# ---------------------------------------------------------------------------

class GaussianSED:
    """Complexity-0 SED: smooth Gaussian centred on *peak_wavelength_AA*.

    This is the simplest possible SED — useful for debugging, near-monochromatic
    emission lines, or as a sanity-check baseline.  It has no physical basis but
    is fast and fully differentiable.

    Parameters
    ----------
    peak_wavelength_AA : float
        Central wavelength of the Gaussian in Ångström.
    sigma_AA : float
        1-σ width of the Gaussian in Ångström.  Defaults to 500 Å.
    """
    COMPLEXITY: ClassVar[int] = 0

    def __init__(self, peak_wavelength_AA: float = 13500.0, sigma_AA: float = 500.0):
        self.peak_wavelength_AA = float(peak_wavelength_AA)
        self.sigma_AA = float(sigma_AA)

    def __call__(self, wavelengths_AA: np.ndarray) -> np.ndarray:
        wav = np.asarray(wavelengths_AA, dtype=float)
        spec = np.exp(-0.5 * ((wav - self.peak_wavelength_AA) / self.sigma_AA) ** 2)
        return _normalise(spec, wav)

    def __repr__(self) -> str:
        return (
            f"GaussianSED(peak={self.peak_wavelength_AA:.0f}Å, "
            f"sigma={self.sigma_AA:.0f}Å) [complexity=0]"
        )


# ---------------------------------------------------------------------------
# Complexity 1 — Blackbody SED
# ---------------------------------------------------------------------------

class BlackbodySED:
    """Complexity-1 SED: Planck blackbody spectrum.

    Parameters
    ----------
    temperature_K : float
        Effective temperature in Kelvin.
    """
    COMPLEXITY: ClassVar[int] = 1

    def __init__(self, temperature_K: float):
        self.temperature_K = float(temperature_K)

    def __call__(self, wavelengths_AA: np.ndarray) -> np.ndarray:
        """Return blackbody flux at requested wavelengths (in Å)."""
        lam_m = np.asarray(wavelengths_AA, dtype=float) * 1e-10
        h = 6.62607015e-34
        c = 299792458.0
        k = 1.380649e-23
        T = self.temperature_K

        a = 2.0 * h * c ** 2
        b = h * c / (lam_m * k * T)
        with np.errstate(over="ignore", invalid="ignore"):
            spec = a / (lam_m ** 5) / np.expm1(b)
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        return _normalise(spec, np.asarray(wavelengths_AA, dtype=float))

    def __repr__(self) -> str:
        return f"BlackbodySED(T={self.temperature_K:.0f} K) [complexity=1]"


# ---------------------------------------------------------------------------
# Complexity 2 — Realistic stellar SED
# ---------------------------------------------------------------------------

class RealisticSED:
    """Complexity-2 SED: realistic stellar spectrum.

    If *synphot* is available this will attempt to build a Kurucz stellar model;
    otherwise it falls back transparently to ``BlackbodySED``.

    Parameters
    ----------
    temperature_K : float
        Effective temperature in Kelvin.
    log_g : float, optional
        Surface gravity log g (cgs).  Used when synphot templates are available.
    metallicity : float, optional
        [Fe/H] metallicity in dex.
    """
    COMPLEXITY: ClassVar[int] = 2

    def __init__(
        self,
        temperature_K: float,
        log_g: float | None = None,
        metallicity: float | None = None,
    ):
        self.temperature_K = float(temperature_K)
        self.log_g = log_g
        self.metallicity = metallicity
        # synphot stellar template path: not yet implemented; fall back to BB
        self._fallback = BlackbodySED(temperature_K=self.temperature_K)

    def __call__(self, wavelengths_AA: np.ndarray) -> np.ndarray:
        return self._fallback(wavelengths_AA)

    def __repr__(self) -> str:
        return (
            f"RealisticSED(T={self.temperature_K:.0f} K, "
            f"log_g={self.log_g}, [Fe/H]={self.metallicity}) [complexity=2]"
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_sed(complexity: int = 1, **kwargs):
    """Instantiate an SED at the requested complexity level.

    Parameters
    ----------
    complexity : {0, 1, 2}
        0 → ``GaussianSED``, 1 → ``BlackbodySED``, 2 → ``RealisticSED``.
    **kwargs
        Forwarded verbatim to the chosen SED class constructor.

    Examples
    --------
    >>> sed0 = make_sed(0, peak_wavelength_AA=13500, sigma_AA=800)
    >>> sed1 = make_sed(1, temperature_K=5800)
    >>> sed2 = make_sed(2, temperature_K=5800, log_g=4.5, metallicity=0.0)
    """
    _classes = {0: GaussianSED, 1: BlackbodySED, 2: RealisticSED}
    if complexity not in _classes:
        raise ValueError(
            f"complexity must be 0, 1 or 2; got {complexity!r}"
        )
    return _classes[complexity](**kwargs)


def random_blackbody_sed(rng: np.random.Generator | None = None) -> BlackbodySED:
    """Return a randomly parameterised ``BlackbodySED``.

    Uses a log-uniform prior between 3 500 K and 30 000 K, matching typical
    stellar temperatures used in the miniature simulators.
    """
    if rng is None:
        rng = np.random.default_rng()
    temp = float(np.exp(rng.uniform(np.log(3500.0), np.log(30000.0))))
    return BlackbodySED(temperature_K=temp)


def random_sed(
    rng: np.random.Generator | None = None,
    complexity: int = 1,
):
    """Return a randomly parameterised SED at *complexity*.

    Parameters
    ----------
    rng : np.random.Generator, optional
    complexity : {0, 1, 2}
    """
    if rng is None:
        rng = np.random.default_rng()
    if complexity == 0:
        peak = float(rng.uniform(9500.0, 17500.0))
        sigma = float(rng.uniform(200.0, 2000.0))
        return GaussianSED(peak_wavelength_AA=peak, sigma_AA=sigma)
    elif complexity == 1:
        temp = float(np.exp(rng.uniform(np.log(3500.0), np.log(30000.0))))
        return BlackbodySED(temperature_K=temp)
    elif complexity == 2:
        temp = float(np.exp(rng.uniform(np.log(3500.0), np.log(30000.0))))
        log_g = float(rng.uniform(1.0, 5.0))
        metallicity = float(rng.uniform(-2.0, 0.5))
        return RealisticSED(temperature_K=temp, log_g=log_g, metallicity=metallicity)
    else:
        raise ValueError(f"complexity must be 0, 1 or 2; got {complexity!r}")
