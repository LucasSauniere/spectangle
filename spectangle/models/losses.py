"""
spectangle.models.losses
=========================
Custom loss functions for 3-D spectral cube reconstruction.

Loss components
---------------
``ReconstructionLoss``      — MSE + optional SSIM on the predicted cube.
``SpectralConsistencyLoss`` — Penalises spectral smoothness violations
                              (encourages physically plausible spectra).
``PhysicsInformedLoss``     — MSE between re-projected cube and observed
                              spectrograms (measurement-space residual).
``CombinedLoss``            — Weighted sum of all components; the main loss
                              used by the Trainer.

Design notes
------------
* All losses operate on PyTorch tensors of shape (B, n_lambda, H, W).
* SSIM is computed per wavelength slice and averaged over λ.
* The physics loss requires a differentiable forward model (DifferentiableForwardModel
  from spectangle.models.pinn) to re-project the predicted cube.
* Weights are controlled via a dict (matching the YAML config ``loss:`` block).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM helper (pure PyTorch, no external dependency)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1-D Gaussian kernel."""
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _ssim_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM for a batch of 2-D images (B, 1, H, W).

    Returns a scalar (mean over batch and spatial dims).
    """
    device, dtype = pred.device, pred.dtype
    k1d = _gaussian_kernel_1d(window_size, sigma, device, dtype)
    kernel = k1d.unsqueeze(0) * k1d.unsqueeze(1)  # (ks, ks)
    kernel = kernel.unsqueeze(0).unsqueeze(0)       # (1, 1, ks, ks)
    pad = window_size // 2

    mu_x = F.conv2d(pred,   kernel, padding=pad)
    mu_y = F.conv2d(target, kernel, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred   * pred,   kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred   * target, kernel, padding=pad) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


def ssim_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SSIM loss over all wavelength slices of a 3-D cube.

    Parameters
    ----------
    pred, target : Tensor, shape (B, n_lambda, H, W)

    Returns
    -------
    Scalar tensor — 1 − mean_SSIM  (so that minimising = maximising SSIM).
    """
    B, n_lam, H, W = pred.shape
    # Reshape to (B*n_lam, 1, H, W) for batched 2-D SSIM
    p = pred.reshape(B * n_lam, 1, H, W)
    t = target.reshape(B * n_lam, 1, H, W)
    return 1.0 - _ssim_2d(p, t)


# ---------------------------------------------------------------------------
# Individual loss modules
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """Pixel-wise reconstruction loss: weighted MSE + SSIM.

    Parameters
    ----------
    mse_weight : float
        Weight for the MSE term.
    ssim_weight : float
        Weight for the SSIM term (1 − SSIM).
    """

    def __init__(self, mse_weight: float = 1.0, ssim_weight: float = 0.5) -> None:
        super().__init__()
        self.mse_weight  = mse_weight
        self.ssim_weight = ssim_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse  = F.mse_loss(pred, target)
        ssim = ssim_loss_3d(pred, target) if self.ssim_weight > 0 else torch.tensor(0.0)
        total = self.mse_weight * mse + self.ssim_weight * ssim
        return total, {"mse": mse.item(), "ssim": ssim.item()}


class SpectralConsistencyLoss(nn.Module):
    """Penalise high-frequency spectral noise in the predicted cube.

    Computes the mean squared second-order finite difference along the
    spectral (λ) axis.  This encourages smooth, physically plausible spectra
    without enforcing a specific SED shape.

    Parameters
    ----------
    weight : float
        Loss weight.
    """

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor, shape (B, n_lambda, H, W)

        Returns
        -------
        Scalar tensor.
        """
        if self.weight == 0:
            return torch.tensor(0.0, device=pred.device)
        # Second-order finite difference along λ axis (dim=1)
        d2 = pred[:, 2:, :, :] - 2 * pred[:, 1:-1, :, :] + pred[:, :-2, :, :]
        return self.weight * d2.pow(2).mean()


class PhysicsInformedLoss(nn.Module):
    """Physics residual loss: ||A(ŷ) − x_obs||².

    Requires a differentiable forward model ``A`` that maps a predicted cube
    to the observation space (4 K-sequence spectrograms).

    Parameters
    ----------
    forward_model : nn.Module
        Differentiable forward model (DifferentiableForwardModel).
    weight : float
        Loss weight.
    """

    def __init__(self, forward_model: nn.Module, weight: float = 1.0) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.weight = weight

    def forward(
        self, pred_cube: torch.Tensor, observed: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_cube : Tensor, shape (B, n_lambda, H, W)
            Predicted spectral cube.
        observed : Tensor, shape (B, 4, H_spec, W_spec)
            Observed spectrograms (first 4 channels of the input).

        Returns
        -------
        Scalar tensor.
        """
        if self.weight == 0:
            return torch.tensor(0.0, device=pred_cube.device)
        reprojected = self.forward_model(pred_cube)
        return self.weight * F.mse_loss(reprojected, observed)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """Weighted combination of reconstruction, spectral and physics losses.

    This is the main loss used by the ``Trainer``.  It accepts a ``weights``
    dict (matching the ``loss:`` block in the YAML configs) and an optional
    differentiable forward model for the physics term.

    Parameters
    ----------
    weights : dict
        Keys: ``mse``, ``ssim``, ``spectral``, ``physics``.
        Missing keys default to 0.
    forward_model : nn.Module, optional
        Required when ``weights["physics"] > 0``.

    Usage
    -----
    For standard (non-PINN) training::

        loss_fn = CombinedLoss(weights={"mse": 1.0, "ssim": 0.5, "spectral": 0.1})
        total, breakdown = loss_fn(pred, target)

    For PINN training (physics loss enabled)::

        loss_fn = CombinedLoss(
            weights={"mse": 1.0, "ssim": 0.5, "spectral": 0.1, "physics": 1.0},
            forward_model=pinn.physics_model,
        )
        total, breakdown = loss_fn(pred, target, observed=x[:, :4])
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        forward_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        w = weights or {}
        self.w_mse      = float(w.get("mse",      1.0))
        self.w_ssim     = float(w.get("ssim",      0.5))
        self.w_spectral = float(w.get("spectral",  0.1))
        self.w_physics  = float(w.get("physics",   0.0))

        self.rec_loss      = ReconstructionLoss(mse_weight=self.w_mse, ssim_weight=self.w_ssim)
        self.spectral_loss = SpectralConsistencyLoss(weight=self.w_spectral)
        self.physics_loss: Optional[PhysicsInformedLoss] = None
        if self.w_physics > 0 and forward_model is not None:
            self.physics_loss = PhysicsInformedLoss(forward_model, weight=self.w_physics)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss.

        Parameters
        ----------
        pred : Tensor, shape (B, n_lambda, H, W)
            Predicted spectral cube.
        target : Tensor, shape (B, n_lambda, H, W)
            Ground-truth spectral cube.
        observed : Tensor, shape (B, 4, H_spec, W_spec), optional
            Observed spectrograms.  Required when physics loss is active.

        Returns
        -------
        total : scalar Tensor
        breakdown : dict with per-component loss values (for logging)
        """
        total = torch.tensor(0.0, device=pred.device)
        breakdown: Dict[str, float] = {}

        # Reconstruction (MSE + SSIM)
        rec, rec_bd = self.rec_loss(pred, target)
        total = total + rec
        breakdown.update(rec_bd)

        # Spectral smoothness
        spec = self.spectral_loss(pred)
        total = total + spec
        breakdown["spectral"] = spec.item()

        # Physics residual
        if self.physics_loss is not None and observed is not None:
            phys = self.physics_loss(pred, observed)
            total = total + phys
            breakdown["physics"] = phys.item()
        else:
            breakdown["physics"] = 0.0

        breakdown["total"] = total.item()
        return total, breakdown
