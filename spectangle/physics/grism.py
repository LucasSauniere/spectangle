"""spectangle.physics.grism

Simple grism/detector parameter definitions used by simulators and noise model.
"""

from __future__ import annotations

from typing import Dict

# Simple detector parameters approximating Euclid NISP H2RG behavior (miniature)
NISP_DETECTOR = {
    "exposure_time_s": 565.0,
    "n_reads": 4,
    "read_noise_e": 15.0,
    "dark_current_e_s": 0.01,
    "gain_e_adu": 1.0,
}

# Zodiacal sky background in electrons/s/pixel (approx for miniature tests)
SKY_BACKGROUND = {"sky_e_s_pix": 0.2}

# Placeholder for grism configuration dictionary; PINN imports GRISM_PARAMS in
# a few places — provide a minimal structure.
GRISM_PARAMS: Dict[str, float] = {
    "dispersion_coefficient": 1.0,  # unitless placeholder
}
