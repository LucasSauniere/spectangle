"""spectangle.simulations

Simulation utilities and entry points for dataset generation.
"""

from spectangle.simulations.simple import SimpleSimulator
from spectangle.simulations.complex import ComplexSimulator
from spectangle.simulations.io import save_simulation, load_simulation

__all__ = ["SimpleSimulator", "ComplexSimulator", "save_simulation", "load_simulation"]
