"""
spectangle.simulations.io
==========================
HDF5 I/O utilities for saving and loading simulation datasets.

HDF5 file layout
----------------
/metadata          — group of scalar string/float attributes (sim config)
                     Includes: image_shape_ny/nx, spectrogram_shape_ny/nx,
                               pad_y, pad_x, n_spectral_pixels, simulator, …
/wavelengths       — (n_lambda,)                     float32 — shared wavelength grid
/samples/
    /0000/
        cube               — (n_lambda, ny, nx)              float32  ground-truth cube
        spectrograms       — (4, ny+2·pad_y, nx+2·pad_x)    float32  [noisy if complex]
        spectrograms_clean — (4, ny+2·pad_y, nx+2·pad_x)    float32  [complex only]
        noise_maps         — (4, ny+2·pad_y, nx+2·pad_x)    float32  [complex only]
        direct_image       — (ny+2·pad_y, nx+2·pad_x)       float32  [if include_direct]
        source_xs          — (n_src,)                        float32
        source_ys          — (n_src,)                        float32
        source_temps       — (n_src,)                        float32
    /0001/
        ...

Shape note
----------
``spectrograms`` and ``direct_image`` are **larger** than the scene cube
because spectral dispersion physically extends the detector footprint.
The padding (``pad_y``, ``pad_x``) is stored in ``/metadata`` and is the
same for every sample in the file.  Downstream models must be aware of
this shape difference (input ≠ output spatial size).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


def save_simulation(
    samples: List[Dict[str, Any]],
    output_path: str | Path,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Write simulation samples to an HDF5 file.

    Parameters
    ----------
    samples : list of dict
        Each dict is the output of ``SimpleSimulator.generate_one()`` or
        ``ComplexSimulator.generate_one()``.
    output_path : str or Path
        Destination ``.h5`` file (created / overwritten).
    metadata : dict, optional
        Simulation configuration written as HDF5 attributes on ``/metadata``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # -- metadata -------------------------------------------------------
        meta_grp = f.create_group("metadata")
        if metadata:
            for key, val in metadata.items():
                meta_grp.attrs[key] = str(val) if not isinstance(val, (int, float)) else val

        # -- shared wavelength grid -----------------------------------------
        if samples and "wavelengths" in samples[0]:
            f.create_dataset(
                "wavelengths",
                data=samples[0]["wavelengths"],
                compression="gzip",
            )

        # -- individual samples ---------------------------------------------
        sample_grp = f.create_group("samples")
        for idx, sample in enumerate(samples):
            grp = sample_grp.create_group(f"{idx:04d}")
            for key, val in sample.items():
                if key == "wavelengths":
                    continue  # stored once at root level
                if val is None:
                    continue
                grp.create_dataset(key, data=val, compression="gzip", compression_opts=4)


def load_simulation(
    path: str | Path,
    indices: List[int] | None = None,
) -> Dict[str, Any]:
    """Load simulation samples from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.h5`` file.
    indices : list of int, optional
        Sample indices to load.  If ``None``, all samples are loaded.

    Returns
    -------
    dict with keys:
        ``"metadata"`` — dict of simulation config,
        ``"wavelengths"`` — ndarray,
        ``"samples"`` — list of per-sample dicts.
    """
    path = Path(path)
    result: Dict[str, Any] = {"metadata": {}, "wavelengths": None, "samples": []}

    with h5py.File(path, "r") as f:
        # Metadata
        if "metadata" in f:
            for key, val in f["metadata"].attrs.items():
                result["metadata"][key] = val

        # Wavelength grid
        if "wavelengths" in f:
            result["wavelengths"] = f["wavelengths"][:]

        # Samples
        if "samples" in f:
            sample_keys = sorted(f["samples"].keys())
            if indices is not None:
                sample_keys = [sample_keys[i] for i in indices]

            for key in sample_keys:
                grp = f["samples"][key]
                sample: Dict[str, np.ndarray] = {}
                for dset_name in grp:
                    sample[dset_name] = grp[dset_name][:]
                result["samples"].append(sample)

    return result


def load_spectrograms(
    path: str | Path,
    indices: List[int] | None = None,
) -> np.ndarray:
    """Fast loader returning only the stacked spectrograms array.

    Parameters
    ----------
    path : str or Path
    indices : list of int, optional

    Returns
    -------
    ndarray, shape (n_samples, 4, ny+2·pad_y, nx+2·pad_x)
        Padded dispersed images.  The padding values are stored in
        ``/metadata`` as ``pad_y`` and ``pad_x``.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        sample_keys = sorted(f["samples"].keys())
        if indices is not None:
            sample_keys = [sample_keys[i] for i in indices]
        arrays = [f["samples"][k]["spectrograms"][:] for k in sample_keys]
    return np.stack(arrays, axis=0)


def load_cubes(
    path: str | Path,
    indices: List[int] | None = None,
) -> np.ndarray:
    """Fast loader returning only the stacked ground-truth cube array.

    Returns
    -------
    ndarray, shape (n_samples, n_lambda, ny, nx)
        Unpadded scene cubes at scene resolution.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        sample_keys = sorted(f["samples"].keys())
        if indices is not None:
            sample_keys = [sample_keys[i] for i in indices]
        arrays = [f["samples"][k]["cube"][:] for k in sample_keys]
    return np.stack(arrays, axis=0)
