"""spectangle — Spectroscopic disentangling package

Top-level package initialisation: expose version and subpackages.
"""

__version__ = "0.1.0"

# Centralised data-path configuration — import before subpackages so they
# can use spectangle.paths without circular dependencies.
from spectangle import paths  # noqa: F401
from spectangle.paths import DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR  # noqa: F401

# Expose commonly used subpackages for convenience
from spectangle import physics  # noqa: F401
from spectangle import simulations  # noqa: F401
from spectangle import models  # noqa: F401
from spectangle import utils  # noqa: F401

__all__ = [
    "__version__",
    "paths",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "physics",
    "simulations",
    "models",
    "utils",
]
