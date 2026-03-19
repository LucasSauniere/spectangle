"""
spectangle.simulations.simple
==============================
Noiseless, 1st-order-only simulator for proof-of-concept training.

Key design choices
------------------
* **Blackbody SEDs** — analytically exact, no external data needed.
* **1st diffraction order only** — avoids confusion with 0th/2nd order ghosts.
* **Miniature geometry** — spectrum length ≤ 128 pixels for fast ML training.
* **No noise** — clean ground-truth pairs for the initial learning task.
* **Separate files per spectrogram** — prevents single large files when
  generating large datasets.

Spectrogram shape vs scene shape
---------------------------------
Spectral dispersion physically elongates the footprint of the scene along the
dispersion direction.  The output spectrograms therefore have a **different
(larger) shape** than the input cube:

    scene cube        : (n_lambda, ny, nx)
    output spectrograms : (4, ny + 2·pad_y, nx + 2·pad_x)

where ``pad_y`` and ``pad_x`` are computed automatically by ``ForwardModel``
from the maximum spectral displacement.  The ``spectrogram_shape`` attribute
of ``SimpleSimulator`` exposes this padded shape.

Output format (HDF5)
--------------------
Each call to ``SimpleSimulator.run()`` produces an HDF5 file with groups:

    /cube            — (n_lambda, ny, nx)              float32 ground-truth cube
    /wavelengths     — (n_lambda,)                     float32 wavelength grid [Å]
    /spectrograms    — (4, ny+2·pad_y, nx+2·pad_x)    float32 dispersed images
    /direct_image    — (ny+2·pad_y, nx+2·pad_x)       float32 broadband direct image
    /sources         — table of source positions and temperatures
    /metadata        — simulation config as HDF5 attributes
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from spectangle.physics.dispersion import KSequence
from spectangle.physics.psf import PSFModel
from spectangle.simulations.forward import ForwardModel
from spectangle.simulations.io import save_simulation
from spectangle.simulations.sed import BlackbodySED, random_blackbody_sed


# ---------------------------------------------------------------------------
# Module-level worker (must live at module scope to be picklable by
# multiprocessing on all platforms, including macOS spawn-based processes).
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> Dict:
    """Generate a single sample in a subprocess.

    Parameters
    ----------
    args : (sim_params_dict, seed_int)
        ``sim_params_dict`` contains the keyword arguments for
        ``SimpleSimulator.__init__`` (everything except ``seed``).
        ``seed_int`` is the per-sample RNG seed for reproducibility.
    """
    sim_params, seed = args
    sim = SimpleSimulator(**sim_params)
    rng = np.random.default_rng(seed)
    return sim.generate_one(rng)


# ---------------------------------------------------------------------------

class SimpleSimulator:
    """Generate noiseless 1st-order slitless spectroscopy training samples.

    Parameters
    ----------
    n_sources : int
        Number of point sources randomly placed in the field.
    image_shape : tuple of int
        (ny, nx) — spatial dimensions of the detector image.
    n_spectral_pixels : int
        Number of wavelength channels in the output cube.  Setting this to
        128 compresses the full NISP bandpass into 128 pixels (miniature mode).
    psf_fwhm_pixels : float
        PSF FWHM in pixels.
    seed : int, optional
        RNG seed for reproducibility.
    include_direct : bool
        If ``True``, also save the broadband direct (undispersed) image.
    """

    def __init__(
        self,
        n_sources: int = 5,
        image_shape: Tuple[int, int] = (128, 128),
        n_spectral_pixels: int = 128,
        psf_fwhm_pixels: float = 1.6,
        seed: int | None = None,
        include_direct: bool = True,
    ) -> None:
        self.n_sources = n_sources
        self.image_shape = image_shape
        self.n_spectral_pixels = n_spectral_pixels
        self.psf_fwhm_pixels = psf_fwhm_pixels
        self.seed = seed
        self.include_direct = include_direct

        # Build physics objects
        self.kseq = KSequence.miniature(n_spectral_pixels)
        self.psf = PSFModel(fwhm_pixels=psf_fwhm_pixels)
        self.fwd = ForwardModel(
            ksequence=self.kseq,
            psf_model=self.psf,
            image_shape=image_shape,
            orders=[1],  # 1st order only
        )
        self.wavelengths_AA = self.kseq.wavelength_grid(n_spectral_pixels)

        # Expose the padded spectrogram shape for downstream use
        self.spectrogram_shape: Tuple[int, int] = self.fwd.spectrogram_shape
        self.pad_x: int = self.fwd.pad_x
        self.pad_y: int = self.fwd.pad_y

    # ------------------------------------------------------------------

    def _place_sources(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, List[BlackbodySED]]:
        """Randomly place point sources in the field.

        Returns
        -------
        xs, ys : ndarray, shape (n_sources,)
            Pixel coordinates (floating point).
        seds : list of BlackbodySED
        """
        ny, nx = self.image_shape
        # Keep sources away from edges by half the image size
        margin = max(ny, nx) // 8
        xs = rng.uniform(margin, nx - margin, size=self.n_sources)
        ys = rng.uniform(margin, ny - margin, size=self.n_sources)
        seds = [random_blackbody_sed(rng) for _ in range(self.n_sources)]
        return xs, ys, seds

    def _build_cube(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        seds: List[BlackbodySED],
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        """Build a 3-D spectral cube from point sources.

        Parameters
        ----------
        xs, ys : arrays of source pixel positions
        seds : list of SED callables
        amplitudes : flux normalisation per source

        Returns
        -------
        cube : ndarray, shape (n_lambda, ny, nx)
        """
        ny, nx = self.image_shape
        n_lam = self.n_spectral_pixels
        cube = np.zeros((n_lam, ny, nx), dtype=np.float32)

        for x, y, sed, amp in zip(xs, ys, seds, amplitudes):
            flux = sed(self.wavelengths_AA).astype(np.float32) * amp

            # Place source at nearest-integer pixel (point source)
            xi, yi = int(round(x)), int(round(y))
            xi = np.clip(xi, 0, nx - 1)
            yi = np.clip(yi, 0, ny - 1)
            cube[:, yi, xi] += flux

        # Convolve each spectral slice with the PSF
        cube = self.psf.convolve_cube(cube)
        return cube

    def generate_one(
        self, rng: np.random.Generator
    ) -> Dict:
        """Generate a single training sample.

        Returns a dict with keys:
          ``cube``, ``wavelengths``, ``spectrograms``, ``direct_image``,
          ``source_xs``, ``source_ys``, ``source_temps``.
        """
        xs, ys, seds = self._place_sources(rng)
        amplitudes = rng.uniform(0.5, 2.0, size=self.n_sources).astype(np.float32)

        cube = self._build_cube(xs, ys, seds, amplitudes)

        if self.include_direct:
            spectrograms, direct_image = self.fwd.forward_with_direct(
                cube, self.wavelengths_AA
            )
        else:
            spectrograms = self.fwd(cube, self.wavelengths_AA)
            direct_image = None

        return {
            "cube": cube.astype(np.float32),
            "wavelengths": self.wavelengths_AA.astype(np.float32),
            "spectrograms": spectrograms.astype(np.float32),
            "direct_image": direct_image.astype(np.float32) if direct_image is not None else None,
            "source_xs": xs.astype(np.float32),
            "source_ys": ys.astype(np.float32),
            "source_temps": np.array(
                [s.temperature_K for s in seds], dtype=np.float32
            ),
        }

    def run(
        self,
        n_samples: int,
        output_path: str | Path,
        show_progress: bool = True,
        n_workers: int = 1,
    ) -> Path:
        """Generate *n_samples* training examples and save them to one HDF5 file.

        Generation can be parallelised across CPU cores with the ``n_workers``
        argument.  Each worker process re-creates the simulator from the same
        parameters and generates its sample with a unique, deterministic seed
        derived from ``self.seed`` so the dataset is fully reproducible.

        Parameters
        ----------
        n_samples : int
            Number of (cube, spectrograms) pairs to generate.
        output_path : str or Path
            Destination ``.h5`` file.
        show_progress : bool
            Display a ``tqdm`` progress bar.
        n_workers : int
            Number of parallel worker processes.  ``1`` (default) runs
            sequentially in the calling process.  ``-1`` uses
            ``os.cpu_count()`` workers (all available cores).

        Returns
        -------
        Path
            The written HDF5 file path.
        """
        import os

        if n_workers == -1:
            n_workers = os.cpu_count() or 1

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Derive one deterministic integer seed per sample from the master seed.
        base_rng = np.random.default_rng(self.seed)
        seeds = base_rng.integers(0, 2**31, size=n_samples).tolist()

        # Pack simulator constructor kwargs (without 'seed' — each worker
        # receives its own per-sample seed via the args tuple).
        sim_params = dict(
            n_sources=self.n_sources,
            image_shape=self.image_shape,
            n_spectral_pixels=self.n_spectral_pixels,
            psf_fwhm_pixels=self.psf_fwhm_pixels,
            include_direct=self.include_direct,
            seed=None,
        )
        args = [(sim_params, s) for s in seeds]

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = executor.map(_worker, args)
                if show_progress:
                    futures = tqdm(futures, total=n_samples, desc="Simulating", unit="sample")
                samples = list(futures)
        else:
            iterator = args
            if show_progress:
                iterator = tqdm(iterator, total=n_samples, desc="Simulating", unit="sample")
            samples = [_worker(a) for a in iterator]

        metadata = {
            "simulator": "SimpleSimulator",
            "n_sources": self.n_sources,
            "image_shape_ny": self.image_shape[0],
            "image_shape_nx": self.image_shape[1],
            "spectrogram_shape_ny": self.spectrogram_shape[0],
            "spectrogram_shape_nx": self.spectrogram_shape[1],
            "pad_y": self.pad_y,
            "pad_x": self.pad_x,
            "n_spectral_pixels": self.n_spectral_pixels,
            "psf_fwhm_pixels": self.psf_fwhm_pixels,
            "orders": "1",
            "noise": "none",
            "sed_type": "blackbody",
            "n_workers": n_workers,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        save_simulation(samples, output_path, metadata)
        return output_path
