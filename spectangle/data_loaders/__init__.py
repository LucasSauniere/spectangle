"""
spectangle.data_loaders
========================
PyTorch Dataset and DataModule classes for loading HDF5 simulation files.

Exports
-------
``SpectangleDataset``   — torch.utils.data.Dataset wrapping an HDF5 file.
``SpectangleDataModule``— Convenience class that builds train/val/test splits
                          and exposes DataLoaders.
"""

from spectangle.data_loaders.dataset import SpectangleDataset, SpectangleDataModule

__all__ = ["SpectangleDataset", "SpectangleDataModule"]
