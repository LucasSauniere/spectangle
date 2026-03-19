"""
spectangle.paths
================
Centralised data-directory configuration for the spectangle package.

By default the data directory is ``<project_root>/data/``, where
``<project_root>`` is the parent of the installed ``spectangle/`` package
directory.

Override
--------
Set the environment variable ``SPECTANGLE_DATA_DIR`` to any absolute path
to redirect all data I/O to a custom location::

    export SPECTANGLE_DATA_DIR=/Volumes/MyDrive/spectangle_data

The sub-directory layout mirrors the default::

    $SPECTANGLE_DATA_DIR/
        raw/          ← HDF5 files produced by the simulators
        interim/      ← pre-processed / cached artefacts
        processed/    ← model-ready tensors

Usage in notebooks / scripts
-----------------------------
>>> from spectangle.paths import DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR
>>> h5_path = RAW_DIR / 'simple_mini_100s.h5'
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve data root
# ---------------------------------------------------------------------------

_env = os.environ.get("SPECTANGLE_DATA_DIR")

if _env:
    DATA_DIR: Path = Path(_env).expanduser().resolve()
else:
    # Default: <project_root>/data  (project_root = parent of spectangle/)
    _pkg_dir = Path(__file__).parent          # …/spectangle/
    DATA_DIR = (_pkg_dir.parent / "data").resolve()  # …/data/

# ---------------------------------------------------------------------------
# Convenience sub-directories  (created lazily when first used)
# ---------------------------------------------------------------------------

RAW_DIR: Path       = DATA_DIR / "raw"
INTERIM_DIR: Path   = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"


def ensure_dirs() -> None:
    """Create all data sub-directories if they do not already exist."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "ensure_dirs",
]
