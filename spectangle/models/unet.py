"""
spectangle.models.unet
========================
2D-to-3D U-Net for spectral cube reconstruction.

Architecture overview
---------------------
The network maps a multi-channel 2-D input (4 or 5 spectrograms, shape
``(B, C_in, H_spec, W_spec)``) to a 3-D spectral cube (shape
``(B, n_lambda, ny, nx)``).

Input vs. output shape
-----------------------
Because spectral dispersion pads the spectrograms beyond the original scene
boundaries (see ``spectangle.simulations.forward``), the input spatial size
``(H_spec, W_spec) = (ny + 2·pad_y, nx + 2·pad_x)`` is **larger** than the
output cube size ``(ny, nx)``.  When ``scene_shape=(ny, nx)`` is provided,
the network automatically **center-crops** its output to ``(ny, nx)`` so that
``(x_batch, y_batch)`` pairs remain correctly aligned.

If ``scene_shape`` is ``None`` the crop is skipped and the output has the
same spatial size as the input (backward-compatible for square inputs).

Design choices
--------------
1. **2-D encoder** — standard U-Net contracting path using 2-D convolutions.
   Feature maps capture both spatial and cross-channel (cross-dispersion)
   correlations.

2. **Spectral expansion bottleneck** — a learned linear projection expands
   the bottleneck feature map from spatial 2-D to a pseudo-3-D representation
   by treating the channel dimension as the spectral axis.

3. **3-D decoder** — the decoder uses 3-D convolutions to jointly refine
   spatial and spectral structure.  Skip connections from the 2-D encoder are
   broadcast along the spectral axis before being concatenated.

4. **Output head** — a 3-D 1×1×1 convolution maps to the desired number of
   spectral channels, followed by a ReLU to ensure non-negative flux.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv2d(nn.Module):
    """Two consecutive (Conv2d → BN → ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None) -> None:
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down2d(nn.Module):
    """MaxPool2d downsampling followed by DoubleConv2d."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv2d(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DoubleConv3d(nn.Module):
    """Two consecutive (Conv3d → BN → ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up3d(nn.Module):
    """Trilinear upsampling + DoubleConv3d with skip-connection concatenation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # in_ch = channels from below + channels from skip
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims don't match perfectly (edge case)
        diff_h = skip.shape[-2] - x.shape[-2]
        diff_w = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet2Dto3D
# ---------------------------------------------------------------------------

class UNet2Dto3D(nn.Module):
    """2-D→3-D U-Net for slitless spectrogram → spectral cube reconstruction.

    Parameters
    ----------
    in_channels : int
        Number of input channels (4 for 4-spectrogram or 5 with direct image).
    n_lambda : int
        Number of output spectral (wavelength) channels.
    base_features : int
        Feature-map width at the first encoder level.  Doubles at each
        downsampling step.  Default 32 balances speed and capacity.
    depth : int
        Number of encoder downsampling levels (default 4).
    scene_shape : tuple of int, optional
        ``(ny, nx)`` — the **unpadded** scene (cube) spatial size.  When
        provided, the network center-crops its output from the (potentially
        padded) spectrogram spatial size to ``(ny, nx)``.  Set this to match
        the ground-truth cube shape so that inputs and targets align.
        If ``None``, no cropping is performed (output H = input H).
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_lambda: int = 128,
        base_features: int = 32,
        depth: int = 4,
        scene_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_lambda = n_lambda
        self.depth = depth
        self.scene_shape = scene_shape  # (ny, nx) of the ground-truth cube

        # ----------------------------------------------------------------
        # 2-D Encoder
        # ----------------------------------------------------------------
        feats: List[int] = [base_features * (2**i) for i in range(depth + 1)]
        # feats = [32, 64, 128, 256, 512] for depth=4, base=32

        self.enc_in = DoubleConv2d(in_channels, feats[0])
        self.enc_downs = nn.ModuleList([Down2d(feats[i], feats[i + 1]) for i in range(depth)])

        # ----------------------------------------------------------------
        # Spectral expansion bottleneck
        # Converts (B, C_bottleneck, H', W') → (B, n_lambda, C_3d, H', W')
        # by projecting the channel dim to n_lambda via a 1×1 conv then
        # reshaping.
        # ----------------------------------------------------------------
        self.bottleneck_ch = feats[-1]  # e.g. 512
        self.n_3d_ch = feats[-2]        # e.g. 256

        # Project: (B, bottleneck_ch, H', W') → (B, n_lambda * n_3d_ch, H', W')
        self.spectral_expand = nn.Conv2d(
            self.bottleneck_ch, n_lambda * self.n_3d_ch, kernel_size=1, bias=False
        )

        # ----------------------------------------------------------------
        # 3-D Decoder
        # Skip connections from 2-D encoder are broadcast along λ.
        # Each Up3d takes (B, C_below + C_skip_3d, λ, H, W) → (B, C_out, λ, H, W)
        # ----------------------------------------------------------------
        # Skip channel counts after broadcasting (same as 2-D encoder features)
        self.dec_ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            # from below: feats[i+1] if i < depth-1 else n_3d_ch
            below_ch = self.n_3d_ch if i == depth - 1 else feats[i + 1]
            skip_ch = feats[i]
            out_ch = feats[i]
            self.dec_ups.append(Up3d(below_ch + skip_ch, out_ch))

        # Output: 1×1×1 conv → n_lambda channels (already the λ axis)
        # Final shape: (B, 1, n_lambda, H, W) → squeeze → (B, n_lambda, H, W)
        self.output_head = nn.Sequential(
            nn.Conv3d(feats[0], 1, kernel_size=1),
            nn.ReLU(),  # non-negative flux
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _center_crop(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Center-crop a (..., H, W) tensor to (target_h, target_w)."""
        h, w = tensor.shape[-2], tensor.shape[-1]
        top  = (h - target_h) // 2
        left = (w - target_w) // 2
        return tensor[..., top: top + target_h, left: left + target_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C_in, H_spec, W_spec)
            Padded spectrograms.  H_spec ≥ ny, W_spec ≥ nx.

        Returns
        -------
        cube : Tensor, shape (B, n_lambda, ny, nx)
            If ``scene_shape`` is set, the output is center-cropped to
            ``(ny, nx)``.  Otherwise the output has shape
            ``(B, n_lambda, H_spec, W_spec)``.
        """
        # ---- 2-D Encoder ----
        skips = []
        x2d = self.enc_in(x)           # (B, feats[0], H, W)
        skips.append(x2d)
        for down in self.enc_downs:
            x2d = down(x2d)
            skips.append(x2d)

        # x2d is now (B, bottleneck_ch, H/2^depth, W/2^depth)
        bottleneck = skips.pop()  # deepest level

        # ---- Spectral expansion ----
        B, _, Hb, Wb = bottleneck.shape
        expanded = self.spectral_expand(bottleneck)  # (B, n_lambda*n_3d_ch, Hb, Wb)
        # Reshape to (B, n_3d_ch, n_lambda, Hb, Wb)
        x3d = expanded.view(B, self.n_3d_ch, self.n_lambda, Hb, Wb)

        # ---- 3-D Decoder ----
        for up, skip2d in zip(self.dec_ups, reversed(skips)):
            # Broadcast 2-D skip along spectral dimension
            _, C_skip, H_skip, W_skip = skip2d.shape
            skip3d = skip2d.unsqueeze(2).expand(
                B, C_skip, self.n_lambda, H_skip, W_skip
            )
            x3d = up(x3d, skip3d)

        # ---- Output head ----
        # x3d: (B, feats[0], n_lambda, H, W)
        out = self.output_head(x3d)  # (B, 1, n_lambda, H_spec, W_spec)
        out = out.squeeze(1)          # (B, n_lambda, H_spec, W_spec)

        # ---- Center-crop to scene shape ----
        if self.scene_shape is not None:
            ny, nx = self.scene_shape
            out = self._center_crop(out, ny, nx)

        return out

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
