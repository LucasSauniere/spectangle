"""
spectangle.models
==================
Deep learning architectures for 3-D spectral cube reconstruction.

Architectures
-------------
unet         : 2D→3D U-Net with spectral expansion decoder
vit          : Vision Transformer adapted for 3-D output
pinn         : Physics-Informed Neural Network with dispersion loss
losses       : Custom loss functions (MSE, SSIM, spectral, physics)
"""

from spectangle.models.unet import UNet2Dto3D
from spectangle.models.vit import SpectralViT
from spectangle.models.pinn import PINN
from spectangle.models.losses import (
    ReconstructionLoss,
    SpectralConsistencyLoss,
    PhysicsInformedLoss,
    CombinedLoss,
)

__all__ = [
    "UNet2Dto3D",
    "SpectralViT",
    "PINN",
    "ReconstructionLoss",
    "SpectralConsistencyLoss",
    "PhysicsInformedLoss",
    "CombinedLoss",
]
