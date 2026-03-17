"""
spectangle.utils
=================
Shared utilities: visualisation, metrics, I/O helpers, and training tools.
"""

from spectangle.utils.visualization import (
    plot_cube_slice,
    plot_spectrograms,
    plot_rgb,
    plot_spectrum,
    plot_comparison,
    plot_spectrum_comparison,
)
from spectangle.utils.metrics import cube_metrics
from spectangle.utils.training import Trainer, get_device

__all__ = [
    "plot_cube_slice",
    "plot_spectrograms",
    "plot_rgb",
    "plot_spectrum",
    "plot_comparison",
    "plot_spectrum_comparison",
    "cube_metrics",
    "Trainer",
    "get_device",
]
