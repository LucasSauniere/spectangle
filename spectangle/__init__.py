"""spectangle — Spectroscopic disentangling package

Top-level package initialisation: expose version and subpackages.
"""

__version__ = "0.1.0"

# Expose commonly used subpackages for convenience
from spectangle import physics  # noqa: F401
from spectangle import simulations  # noqa: F401
from spectangle import models  # noqa: F401
from spectangle import utils  # noqa: F401

__all__ = ["__version__", "physics", "simulations", "models", "utils"]
