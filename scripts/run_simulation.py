#!/usr/bin/env python
"""
scripts/run_simulation.py
--------------------------
Command-line simulation runner for spectangle.

Generates a training dataset using either the SimpleSimulator (noiseless,
blackbody SEDs, 1st order only) or the ComplexSimulator (noisy, multi-order,
realistic SEDs) and saves it as a compressed HDF5 file.

Usage
-----
    # Miniature simple dataset (fast)
    python scripts/run_simulation.py --config configs/simulations/simple_mini.yaml

    # Euclid-like complex dataset
    python scripts/run_simulation.py --config configs/simulations/complex_euclid.yaml

    # Override number of samples at runtime
    python scripts/run_simulation.py --config configs/simulations/simple_mini.yaml \\
        --n_samples 500 --output data/raw/simple_mini_500s.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make sure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectangle.simulations.simple import SimpleSimulator
from spectangle.simulations.complex import ComplexSimulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="spectangle simulation runner")
    p.add_argument(
        "--config",
        required=True,
        help="Path to a simulation YAML config file.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Override the number of samples from the config.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override the output HDF5 path from the config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg  = cfg["simulation"]
    data_cfg = cfg["data"]

    # Resolve parameters (CLI overrides config)
    n_samples   = args.n_samples  or sim_cfg.get("n_samples",   100)
    output_path = args.output     or data_cfg.get("output_path", "data/raw/dataset.h5")
    seed        = args.seed       or sim_cfg.get("seed", 42)

    sim_type = sim_cfg.get("type", "simple").lower()

    common_kwargs = dict(
        n_sources        = sim_cfg.get("n_sources",         5),
        image_shape      = tuple(sim_cfg.get("image_shape", [128, 128])),
        n_spectral_pixels= sim_cfg.get("n_spectral_pixels", 128),
        psf_fwhm_pixels  = sim_cfg.get("psf_fwhm_pixels",  1.6),
        seed             = seed,
        include_direct   = sim_cfg.get("include_direct",    True),
    )

    if sim_type == "simple":
        sim = SimpleSimulator(**common_kwargs)
        print(f"[spectangle] SimpleSimulator | scene={sim.image_shape} | "
              f"spec={sim.spectrogram_shape} | n_lambda={sim.n_spectral_pixels}")
    elif sim_type == "complex":
        sim = ComplexSimulator(
            **common_kwargs,
            orders             = sim_cfg.get("orders",             [0, 1, 2]),
            add_noise          = sim_cfg.get("add_noise",          True),
            use_realistic_seds = sim_cfg.get("use_realistic_seds", False),
            exposure_time_s    = sim_cfg.get("exposure_time_s",    565.0),
        )
        print(f"[spectangle] ComplexSimulator | scene={sim.image_shape} | "
              f"spec={sim.spectrogram_shape} | n_lambda={sim.n_spectral_pixels} | "
              f"noise={sim.add_noise} | orders={sim.orders}")
    else:
        raise ValueError(f"Unknown simulator type: {sim_type!r}. "
                         "Choose 'simple' or 'complex'.")

    print(f"[spectangle] Generating {n_samples} samples → {output_path}")
    out = sim.run(n_samples=n_samples, output_path=output_path, show_progress=True)
    print(f"[spectangle] Done. Saved to {out}")


if __name__ == "__main__":
    main()
