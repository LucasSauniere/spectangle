"""
spectangle.simulations.complex
================================
Realistic multi-order noise simulator for Euclid NISP.

Design choices
--------------
* **Realistic SEDs** — ``RealisticSED`` (synphot / BaSeL); blackbody fallback.
* **Multi-order** — 0th (peanut), 1st, and 2nd diffraction orders simulated.
* **NISP noise model**:
    - Poisson photon noise (source + sky background)
    - Read noise (Gaussian, per-pixel, per-read)
    - Dark current (Poisson)
  All noise is added *after* the forward projection, matching the physical
  readout process of the H2RG detector.

Spectrogram shape vs scene shape
---------------------------------
Multi-order dispersion physically elongates the detector footprint.  The
output spectrograms therefore have a **different (larger) shape** than the
input cube — see ``spectangle.simulations.forward`` for details.

Output HDF5 format
------------------
Same structure as ``SimpleSimulator`` but with an additional
``/noise_maps`` group and extended ``/metadata`` attributes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from spectangle.physics.dispersion import KSequence
from spectangle.physics.grism import NISP_DETECTOR, SKY_BACKGROUND
from spectangle.physics.psf import PSFModel
from spectangle.simulations.forward import ForwardModel
from spectangle.simulations.io import save_simulation
from spectangle.simulations.sed import BlackbodySED, RealisticSED, random_blackbody_sed


# ---------------------------------------------------------------------------
# NISP noise model
# ---------------------------------------------------------------------------

def add_nisp_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    exposure_time_s: float = NISP_DETECTOR["exposure_time_s"],
    n_reads: int = NISP_DETECTOR["n_reads"],
    read_noise_e: float = NISP_DETECTOR["read_noise_e"],
    dark_current_e_s: float = NISP_DETECTOR["dark_current_e_s"],
    sky_e_s_pix: float = SKY_BACKGROUND["sky_e_s_pix"],
    gain: float = NISP_DETECTOR["gain_e_adu"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Add Euclid NISP H2RG noise to a noiseless count-rate image.

    The image is assumed to be in units of **electrons/s/pixel**.

    Noise components
    ----------------
    1. **Sky background** — additive Poisson noise from zodiacal light.
    2. **Poisson photon noise** — from the source signal.
    3. **Dark current** — Poisson noise from thermal leakage.
    4. **Read noise** — Gaussian noise scaled by ``1/sqrt(n_reads)``
       (Fowler / MACC sampling reduces read noise by ``sqrt(n_reads)``).

    Parameters
    ----------
    image : ndarray
        Noiseless image in electrons/s/pixel.
    rng : numpy.random.Generator
    exposure_time_s, n_reads, read_noise_e, dark_current_e_s, sky_e_s_pix, gain
        NISP detector parameters.

    Returns
    -------
    noisy_image : ndarray
        Noisy image in ADU.
    noise_map : ndarray
        Per-pixel noise standard deviation in electrons.
    """
    # Total signal in electrons
    signal_e = image * exposure_time_s  # source electrons
    sky_e = sky_e_s_pix * exposure_time_s  # per pixel
    dark_e = dark_current_e_s * exposure_time_s  # per pixel

    # Poisson noise on source + sky + dark
    total_e = np.clip(signal_e + sky_e + dark_e, 0, None)
    noisy_e = rng.poisson(total_e).astype(np.float64)

    # Gaussian read noise (reduced by Fowler sampling)
    rn_eff = read_noise_e / np.sqrt(n_reads)
    read_e = rng.normal(0.0, rn_eff, size=image.shape)
    noisy_e += read_e

    # Noise map (theoretical sigma per pixel, in electrons)
    noise_map = np.sqrt(total_e + rn_eff**2)

    # Convert to ADU
    noisy_adu = noisy_e / gain
    return noisy_adu.astype(np.float32), noise_map.astype(np.float32)


# ---------------------------------------------------------------------------
# ComplexSimulator
# ---------------------------------------------------------------------------

class ComplexSimulator:
    """Realistic Euclid NISP multi-order noisy slitless spectrogram simulator.

    Parameters
    ----------
    n_sources : int
        Number of point sources in the field.
    image_shape : tuple of int
        (ny, nx) — output image dimensions.  For a full NISP detector
        use (2048, 2048); for miniature tests use (128, 128).
    n_spectral_pixels : int
        Spectral dimension of the output cube (128 for miniature, ~690 for full).
    psf_fwhm_pixels : float
        Spatial PSF FWHM.
    orders : list of int
        Diffraction orders to simulate.  Default: [0, 1, 2].
    add_noise : bool
        Whether to add NISP noise.  Set to ``False`` for debugging.
    use_realistic_seds : bool
        Use ``RealisticSED`` (requires synphot).  Falls back to blackbody.
    seed : int, optional
        RNG seed.
    include_direct : bool
        Also produce and save the direct (undispersed) image.
    exposure_time_s : float
        Integration time per exposure in seconds.
    """

    def __init__(
        self,
        n_sources: int = 10,
        image_shape: Tuple[int, int] = (128, 128),
        n_spectral_pixels: int = 128,
        psf_fwhm_pixels: float = 1.6,
        orders: List[int] = None,
        add_noise: bool = True,
        use_realistic_seds: bool = True,
        seed: int | None = None,
        include_direct: bool = True,
        exposure_time_s: float = NISP_DETECTOR["exposure_time_s"],
    ) -> None:
        self.n_sources = n_sources
        self.image_shape = image_shape
        self.n_spectral_pixels = n_spectral_pixels
        self.psf_fwhm_pixels = psf_fwhm_pixels
        self.orders = orders if orders is not None else [0, 1, 2]
        self.add_noise = add_noise
        self.use_realistic_seds = use_realistic_seds
        self.seed = seed
        self.include_direct = include_direct
        self.exposure_time_s = exposure_time_s

        self.kseq = KSequence.miniature(n_spectral_pixels)
        self.psf = PSFModel(fwhm_pixels=psf_fwhm_pixels)
        self.fwd = ForwardModel(
            ksequence=self.kseq,
            psf_model=self.psf,
            image_shape=image_shape,
            orders=self.orders,
        )
        self.wavelengths_AA = self.kseq[0].wavelength_grid(n_spectral_pixels)

        # Expose the padded spectrogram shape for downstream use
        self.spectrogram_shape: Tuple[int, int] = self.fwd.spectrogram_shape
        self.pad_x: int = self.fwd.pad_x
        self.pad_y: int = self.fwd.pad_y

    # ------------------------------------------------------------------

    def _make_sed(self, rng: np.random.Generator, temperature_K: float):
        """Return a RealisticSED or BlackbodySED depending on config."""
        if self.use_realistic_seds:
            log_g = rng.uniform(3.5, 5.0)
            feh = rng.choice([-0.5, 0.0, 0.5])
            return RealisticSED(temperature_K, log_g=log_g, metallicity=feh)
        return BlackbodySED(temperature_K)

    def _place_sources(self, rng: np.random.Generator):
        ny, nx = self.image_shape
        margin = max(ny, nx) // 8
        xs = rng.uniform(margin, nx - margin, size=self.n_sources)
        ys = rng.uniform(margin, ny - margin, size=self.n_sources)
        temps = np.exp(rng.uniform(np.log(3500), np.log(30_000), size=self.n_sources))
        seds = [self._make_sed(rng, t) for t in temps]
        return xs, ys, seds, temps

    def _build_cube(self, xs, ys, seds, amplitudes):
        ny, nx = self.image_shape
        cube = np.zeros((self.n_spectral_pixels, ny, nx), dtype=np.float32)
        for x, y, sed, amp in zip(xs, ys, seds, amplitudes):
            flux = sed(self.wavelengths_AA).astype(np.float32) * amp
            xi = int(np.clip(round(x), 0, nx - 1))
            yi = int(np.clip(round(y), 0, ny - 1))
            cube[:, yi, xi] += flux
        cube = self.psf.convolve_cube(cube)
        return cube

    def generate_one(self, rng: np.random.Generator) -> Dict:
        """Generate one noisy multi-order training sample."""
        xs, ys, seds, temps = self._place_sources(rng)
        amplitudes = rng.uniform(0.5, 5.0, size=self.n_sources).astype(np.float32)

        cube = self._build_cube(xs, ys, seds, amplitudes)

        if self.include_direct:
            spectrograms_clean, direct_image = self.fwd.forward_with_direct(
                cube, self.wavelengths_AA
            )
        else:
            spectrograms_clean = self.fwd(cube, self.wavelengths_AA)
            direct_image = None

        spectrograms = spectrograms_clean.copy()
        noise_maps = np.zeros_like(spectrograms_clean)

        if self.add_noise:
            for k in range(4):
                noisy, nm = add_nisp_noise(
                    spectrograms_clean[k],
                    rng,
                    exposure_time_s=self.exposure_time_s,
                )
                spectrograms[k] = noisy
                noise_maps[k] = nm

        return {
            "cube": cube.astype(np.float32),
            "wavelengths": self.wavelengths_AA.astype(np.float32),
            "spectrograms": spectrograms.astype(np.float32),
            "spectrograms_clean": spectrograms_clean.astype(np.float32),
            "noise_maps": noise_maps.astype(np.float32),
            "direct_image": (
                direct_image.astype(np.float32) if direct_image is not None else None
            ),
            "source_xs": xs.astype(np.float32),
            "source_ys": ys.astype(np.float32),
            "source_temps": temps.astype(np.float32),
        }

    def run(
        self,
        n_samples: int,
        output_path: str | Path,
        show_progress: bool = True,
    ) -> Path:
        """Generate and save ``n_samples`` complex simulation examples."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(self.seed)

        metadata = {
            "simulator": "ComplexSimulator",
            "n_sources": self.n_sources,
            "image_shape_ny": self.image_shape[0],
            "image_shape_nx": self.image_shape[1],
            "spectrogram_shape_ny": self.spectrogram_shape[0],
            "spectrogram_shape_nx": self.spectrogram_shape[1],
            "pad_y": self.pad_y,
            "pad_x": self.pad_x,
            "n_spectral_pixels": self.n_spectral_pixels,
            "psf_fwhm_pixels": self.psf_fwhm_pixels,
            "orders": str(self.orders),
            "noise": "nisp" if self.add_noise else "none",
            "sed_type": "realistic" if self.use_realistic_seds else "blackbody",
            "exposure_time_s": self.exposure_time_s,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        samples = []
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating", unit="sample")

        for _ in iterator:
            samples.append(self.generate_one(rng))

        save_simulation(samples, output_path, metadata)
        return output_path
