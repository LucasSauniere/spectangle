"""spectangle.physics.dispersion

Simple dispersion models and the K-sequence used by Euclid NISP miniature
simulators. This implements a small API expected by the forward models and
PINN differentiable forward model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


@dataclass
class Offset:
    delta_x: float
    delta_y: float

    # Allow numpy to unpack Offset as a 2-element sequence so that
    #   np.array([disp.wavelength_to_offset(l) for l in wav])
    # produces a (n_lambda, 2) array rather than a 1-D object array.
    def __iter__(self):
        yield self.delta_x
        yield self.delta_y

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.delta_x
        if index == 1:
            return self.delta_y
        raise IndexError(f"Offset index {index} out of range (0 or 1)")


class DispersionModel:
    """Linear dispersion model mapping wavelength → pixel offsets.

    For the miniature tests we use a simple linear relation D(λ) = s · (λ − λ0)
    where s is pixels per Ångström and λ0 is a reference wavelength.
    """

    def __init__(self, dispersion_vector: Tuple[float, float], lam_ref: float = 13500.0, scale: float = 1.0):
        self._dx_unit, self._dy_unit = dispersion_vector
        self.lam_ref = lam_ref
        self.scale = scale

    def wavelength_to_offset(self, wavelength: float) -> Offset:
        # simple linear mapping relative to lam_ref
        delta = (wavelength - self.lam_ref) * self.scale
        return Offset(delta_x=self._dx_unit * delta, delta_y=self._dy_unit * delta)


class KSequence(List[DispersionModel]):
    """Container for the four K-sequence dispersion models.

    Provide a miniature factory to produce compact dispersion laws scaled to
    the reduced spectral sampling used in the toy datasets.
    """

    @staticmethod
    def miniature(n_lambda: int = 128) -> "KSequence":
        # Define four dispersion directions that mimic the Euclid K-sequence
        # unit vectors (approx): RGS000 (along +x), RGS180 (along -x) and
        # small tilts in y for the ±4° GWA tilt arms. We scale the dispersion
        # such that the total spectral length maps into roughly 60 pixels for
        # the miniature setup.
        lam_min, lam_max = 9250.0, 18500.0
        total_span = lam_max - lam_min
        # target pixel span for miniature (half on each side from center)
        target_span_px = 64.0
        # pixels per Å
        s = target_span_px / total_span

        # unit vectors for the four K-steps
        vecs = [ (1.0, 0.0), (-1.0, 0.0), (1.0, np.tan(np.deg2rad(-4.0))), (-1.0, np.tan(np.deg2rad(4.0))) ]

        kseq = KSequence()
        for vx, vy in vecs:
            kseq.append(DispersionModel(dispersion_vector=(vx, vy), lam_ref=(lam_min+lam_max)/2.0, scale=s))
        return kseq

    def wavelength_grid(self, n_lambda: int) -> np.ndarray:
        lam_min, lam_max = 9250.0, 18500.0
        return np.linspace(lam_min, lam_max, n_lambda)
