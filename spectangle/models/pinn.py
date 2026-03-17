"""
spectangle.models.pinn
========================
Physics-Informed Neural Network (PINN) for spectral cube reconstruction.

Physics-informed strategy
--------------------------
The PINN augments any backbone (U-Net or ViT) with a differentiable forward
model that re-projects the predicted cube into the observation space.  The
total loss is:

    L_total = λ_rec  · L_reconstruction(ŷ, y)
            + λ_phys · L_physics(A(ŷ), x)

where:
    ŷ        = predicted spectral cube
    y        = ground-truth cube
    x        = observed spectrograms
    A(ŷ)     = differentiable forward projection of ŷ
    L_rec    = MSE or SSIM reconstruction loss
    L_phys   = MSE between re-projected ŷ and observed x
               ("residual in measurement space")

The physics loss L_phys enforces that the predicted cube is *consistent with
the observed data* even in regions where the ground truth is not available
(e.g. at inference time).  This is the key advantage of the PINN approach.

Differentiable forward model
------------------------------
The differentiable forward model ``DifferentiableForwardModel`` implements the
same dispersion + PSF projection as ``spectangle.simulations.ForwardModel`` but
using only PyTorch operations (``torch.nn.functional.conv2d``,
``torch.nn.functional.grid_sample``-based shifts) so that gradients flow back
to the network weights.

The output spectrograms have the same padded shape as in the numpy forward
model: ``(B, 4, ny + 2·pad_y, nx + 2·pad_x)``.  The padding is computed from
the maximum spectral displacement over all wavelengths.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectangle.models.unet import UNet2Dto3D
from spectangle.physics.dispersion import KSequence
from spectangle.physics.grism import GRISM_PARAMS
from spectangle.physics.psf import PSFModel


# ---------------------------------------------------------------------------
# Differentiable forward model (PyTorch)
# ---------------------------------------------------------------------------

class DifferentiableForwardModel(nn.Module):
    """Differentiable re-projection of a cube onto four K-sequence spectrograms.

    The output shape is padded to match the numpy ``ForwardModel``:
        (B, 4, ny + 2·pad_y, nx + 2·pad_x)

    Dispersion shifts are applied via ``torch.nn.functional.grid_sample``
    (bilinear, zeros padding) so that flux shifted outside the padded canvas
    is correctly set to zero — not wrapped as with ``torch.roll``.

    Parameters
    ----------
    ksequence : KSequence
        K-sequence dispersion models.
    wavelengths_AA : Tensor, shape (n_lambda,)
        Wavelength grid in ångströms.
    psf_kernel : Tensor, shape (1, 1, ks, ks)
        PSF convolution kernel (already normalised).
    image_shape : tuple of int
        (ny, nx) scene (cube) spatial size.
    pad_y, pad_x : int
        Padding on each side (computed from ForwardModel or passed explicitly).
    """

    def __init__(
        self,
        ksequence: KSequence,
        wavelengths_AA: torch.Tensor,
        psf_kernel: torch.Tensor,
        image_shape: Tuple[int, int],
        pad_y: int = 0,
        pad_x: int = 0,
    ) -> None:
        super().__init__()
        self.ksequence = ksequence
        self.register_buffer("wavelengths_AA", wavelengths_AA)
        self.register_buffer("psf_kernel", psf_kernel)
        self.image_shape = image_shape   # (ny, nx) — scene size
        self.pad_y = pad_y
        self.pad_x = pad_x
        # Padded output shape
        ny, nx = image_shape
        self.spectrogram_shape: Tuple[int, int] = (ny + 2 * pad_y, nx + 2 * pad_x)

    # ------------------------------------------------------------------

    def _apply_psf(self, image: torch.Tensor) -> torch.Tensor:
        """Convolve a single (B, 1, H, W) image slice with the PSF."""
        ks = self.psf_kernel.shape[-1]
        pad = ks // 2
        return F.conv2d(image, self.psf_kernel, padding=pad)

    @staticmethod
    def _shift_grid_sample(
        x: torch.Tensor, dx: float, dy: float
    ) -> torch.Tensor:
        """Sub-pixel shift via bilinear grid_sample (zero-padding, no wrap).

        Parameters
        ----------
        x : Tensor, shape (B, H, W)
        dx, dy : float — shift in pixels (+x = right, +y = down)

        Returns
        -------
        Tensor, shape (B, H, W)
        """
        B, H, W = x.shape
        # Normalise shift to [-1, 1] grid_sample coordinates
        # grid_sample convention: theta maps (x_norm, y_norm) → pixel
        #   x_norm = 2 * col / (W-1) - 1,  y_norm = 2 * row / (H-1) - 1
        # To shift the image by (+dx, +dy) pixels, move the sampling grid
        # in the opposite direction.
        delta_x_norm = -2.0 * dx / max(W - 1, 1)
        delta_y_norm = -2.0 * dy / max(H - 1, 1)

        # Base identity grid
        base = F.affine_grid(
            torch.eye(2, 3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1),
            (B, 1, H, W),
            align_corners=True,
        )
        # Apply translation
        base[..., 0] += delta_x_norm
        base[..., 1] += delta_y_norm

        return F.grid_sample(
            x.unsqueeze(1).float(),
            base,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

    def _project_one(
        self,
        cube: torch.Tensor,
        dispersion_model,
        order: int = 1,
    ) -> torch.Tensor:
        """Project cube onto one padded detector image.

        Parameters
        ----------
        cube : Tensor, shape (B, n_lambda, H, W)  — scene cube
        dispersion_model : DispersionModel
        order : int

        Returns
        -------
        image : Tensor, shape (B, H_out, W_out)
            where H_out = H + 2·pad_y, W_out = W + 2·pad_x.
        """
        B, n_lam, H, W = cube.shape
        H_out, W_out = self.spectrogram_shape
        image = torch.zeros(B, H_out, W_out, device=cube.device, dtype=cube.dtype)

        for i, lam in enumerate(self.wavelengths_AA):
            lam_val = lam.item()
            offsets = dispersion_model.wavelength_to_offset(lam_val)
            dx = offsets.delta_x * order
            dy = offsets.delta_y * order

            # Embed the scene slice into the padded canvas
            slice_2d = cube[:, i, :, :]  # (B, H, W)
            padded = F.pad(slice_2d, (self.pad_x, self.pad_x, self.pad_y, self.pad_y))
            # (B, H_out, W_out)

            # Sub-pixel shift via grid_sample (no wrapping)
            shifted = self._shift_grid_sample(padded, dx, dy)

            # PSF convolution
            shifted = self._apply_psf(shifted.unsqueeze(1)).squeeze(1)
            image = image + shifted

        return image

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """Re-project the predicted cube into K-sequence spectrogram space.

        Parameters
        ----------
        cube : Tensor, shape (B, n_lambda, H, W)

        Returns
        -------
        spectrograms : Tensor, shape (B, 4, H + 2·pad_y, W + 2·pad_x)
        """
        B = cube.shape[0]
        H_out, W_out = self.spectrogram_shape
        out = torch.zeros(B, 4, H_out, W_out, device=cube.device, dtype=cube.dtype)
        for k, disp in enumerate(self.ksequence):
            out[:, k] = self._project_one(cube, disp, order=1)
        return out


# ---------------------------------------------------------------------------
# PINN wrapper
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Physics-Informed Neural Network for slitless spectrogram inversion.

    Wraps a backbone network (default: UNet2Dto3D) and adds a differentiable
    physics constraint evaluated at training time.

    Parameters
    ----------
    backbone : nn.Module, optional
        The reconstruction network.  Defaults to ``UNet2Dto3D``.
    ksequence : KSequence
        K-sequence dispersion models.
    wavelengths_AA : array-like
        Wavelength grid in ångströms.
    psf_fwhm_pixels : float
        PSF FWHM for the differentiable forward model.
    image_shape : tuple of int
        (ny, nx).
    in_channels : int
        4 or 5.
    n_lambda : int
        Number of spectral output channels.
    lambda_physics : float
        Weight of the physics loss (default 0.1).  Set to 0 to disable.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        ksequence: Optional[KSequence] = None,
        wavelengths_AA=None,
        psf_fwhm_pixels: float = 1.6,
        image_shape: Tuple[int, int] = (128, 128),
        in_channels: int = 4,
        n_lambda: int = 128,
        lambda_physics: float = 0.1,
        pad_y: int | None = None,
        pad_x: int | None = None,
    ) -> None:
        super().__init__()
        self.lambda_physics = lambda_physics
        self.image_shape = image_shape  # (ny, nx) — unpadded scene size

        # Build KSequence / wavelength grid first so we can derive padding
        if ksequence is None:
            ksequence = KSequence.miniature(n_lambda)
        if wavelengths_AA is None:
            wavelengths_AA = ksequence[0].wavelength_grid(n_lambda)

        wav_tensor = torch.tensor(wavelengths_AA, dtype=torch.float32)

        # Build PSF kernel tensor (1, 1, ks, ks)
        psf = PSFModel(fwhm_pixels=psf_fwhm_pixels)
        k = torch.tensor(psf.kernel, dtype=torch.float32)
        psf_tensor = k.unsqueeze(0).unsqueeze(0)  # (1, 1, ks, ks)

        # Compute padding from the numpy ForwardModel so they match exactly
        from spectangle.simulations.forward import ForwardModel as _FwdModel
        _fwd_np = _FwdModel(
            ksequence=ksequence,
            psf_model=psf,
            image_shape=image_shape,
            orders=[1],
        )
        _pad_y = pad_y if pad_y is not None else _fwd_np.pad_y
        _pad_x = pad_x if pad_x is not None else _fwd_np.pad_x

        # Build backbone with scene_shape so it center-crops its output to
        # (ny, nx) — the backbone receives padded spectrograms but must output
        # the unpadded cube to match ground-truth targets.
        if backbone is None:
            self.backbone = UNet2Dto3D(
                in_channels=in_channels,
                n_lambda=n_lambda,
                scene_shape=image_shape,
            )
        else:
            self.backbone = backbone
            # Best-effort: propagate scene_shape if the backbone supports it
            if hasattr(backbone, "scene_shape") and backbone.scene_shape is None:
                backbone.scene_shape = image_shape

        # Build differentiable forward model for the physics loss
        self.physics_model = DifferentiableForwardModel(
            ksequence=ksequence,
            wavelengths_AA=wav_tensor,
            psf_kernel=psf_tensor,
            image_shape=image_shape,
            pad_y=_pad_y,
            pad_x=_pad_x,
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard inference forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, C_in, H, W)

        Returns
        -------
        cube : Tensor, shape (B, n_lambda, H, W)
        """
        return self.backbone(x)

    def forward_with_physics_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass that also computes the physics residual.

        Parameters
        ----------
        x : Tensor, shape (B, C_in, H_spec, W_spec)
            Observed spectrograms (4 channels) + optional direct image.
            H_spec / W_spec are the padded spectrogram dimensions.
        y : Tensor, shape (B, n_lambda, H, W), optional
            Ground-truth cube (unpadded scene dimensions).

        Returns
        -------
        cube_pred : Tensor, shape (B, n_lambda, H, W)
        physics_loss : scalar Tensor — ||A(ŷ) − x_4ch||²
        rec_loss : scalar Tensor — ||ŷ − y||²  (0 if y is None)
        """
        cube_pred = self.backbone(x)

        # Physics loss: re-project and compare to the 4 observed spectrograms.
        # Both x[:, :4] and reprojected are in padded spectrogram space.
        x_4ch = x[:, :4]
        reprojected = self.physics_model(cube_pred)
        physics_loss = F.mse_loss(reprojected, x_4ch)

        rec_loss = F.mse_loss(cube_pred, y) if y is not None else torch.tensor(0.0)

        return cube_pred, physics_loss, rec_loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
