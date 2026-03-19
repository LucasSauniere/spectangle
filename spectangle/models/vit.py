"""
spectangle.models.vit
======================
Vision Transformer (ViT) adapted for 3-D spectral cube reconstruction.

Architecture overview
---------------------
1. **Patch embedding** — the multi-channel 2-D input is split into non-
   overlapping spatial patches; each patch is linearly embedded into a token.

2. **Transformer encoder** — standard multi-head self-attention blocks process
   the sequence of spatial tokens, allowing long-range cross-dispersion
   correlations to be captured.

3. **Spectral expansion head** — the output token sequence is reshaped and
   decoded via a learned MLP (or lightweight Conv3d) to produce a full
   (n_lambda, H, W) spectral cube.

Key design decisions
--------------------
* Positional encodings are 2-D sinusoidal, matching the spatial grid.
* The decoder projects each token to ``patch_size² x n_lambda`` values, then
  reshapes (pixel-shuffle-like) to the full spatial resolution.
* Uses ``einops`` for readable tensor manipulations.

Input vs. output shape
-----------------------
Spectrograms are padded by the forward model, so the input has shape
``(B, C_in, H_spec, W_spec)`` where H_spec ≥ ny, W_spec ≥ nx.
Use ``image_size_h`` / ``image_size_w`` to specify the **padded** input
dimensions.  When ``scene_shape=(ny, nx)`` is given, the model
center-crops its output to ``(B, n_lambda, ny, nx)``.
For square, unpadded inputs all three values can simply be identical.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# 2-D sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPosEmbedding2D(nn.Module):
    """Learnable-free 2-D sinusoidal positional encoding.

    Produces encodings of shape ``(1, n_patches_h * n_patches_w, embed_dim)``.
    """

    def __init__(self, embed_dim: int, n_patches_h: int, n_patches_w: int) -> None:
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sin/cos."

        half = embed_dim // 2
        # Frequency bands
        freq = torch.arange(0, half // 2, dtype=torch.float32)
        freq = torch.pow(10_000.0, -2.0 * freq / half)

        row = torch.arange(n_patches_h, dtype=torch.float32).unsqueeze(1)
        col = torch.arange(n_patches_w, dtype=torch.float32).unsqueeze(1)

        row_enc = torch.zeros(n_patches_h, half)
        row_enc[:, 0::2] = torch.sin(row * freq[: half // 2].unsqueeze(0))
        row_enc[:, 1::2] = torch.cos(row * freq[: half // 2].unsqueeze(0))

        col_enc = torch.zeros(n_patches_w, half)
        col_enc[:, 0::2] = torch.sin(col * freq[: half // 2].unsqueeze(0))
        col_enc[:, 1::2] = torch.cos(col * freq[: half // 2].unsqueeze(0))

        # Combine: (n_h, n_w, embed_dim)
        pe = torch.cat(
            [
                row_enc.unsqueeze(1).expand(n_patches_h, n_patches_w, half),
                col_enc.unsqueeze(0).expand(n_patches_h, n_patches_w, half),
            ],
            dim=-1,
        ).reshape(1, n_patches_h * n_patches_w, embed_dim)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token sequence x, shape (B, N, D)."""
        return x + self.pe


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (attention + MLP)."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SpectralViT
# ---------------------------------------------------------------------------

class SpectralViT(nn.Module):
    """Vision Transformer for 2-D spectrogram → 3-D spectral cube.

    Parameters
    ----------
    in_channels : int
        Input channels (4 or 5).
    image_size : int or None
        **Deprecated convenience alias** for ``image_size_h`` when the input
        is square.  Ignored if ``image_size_h`` / ``image_size_w`` are set.
    image_size_h : int
        Height of the **padded** input spectrogram (H_spec).
        Must be divisible by ``patch_size``.
    image_size_w : int
        Width of the **padded** input spectrogram (W_spec).
        Must be divisible by ``patch_size``.
    patch_size : int
        Spatial patch side length.
    embed_dim : int
        Token embedding dimension (must be divisible by 4 and ``n_heads``).
    depth : int
        Number of Transformer encoder blocks.
    n_heads : int
        Number of attention heads.
    n_lambda : int
        Number of spectral output channels.
    mlp_ratio : float
        MLP hidden-dim / embed-dim ratio in Transformer blocks.
    dropout : float
        Dropout rate.
    scene_shape : tuple of int, optional
        ``(ny, nx)`` — unpadded output cube shape.  When provided the network
        center-crops the decoder output to ``(ny, nx)``.  If ``None`` the
        output has the same spatial size as the input.
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 128,
        image_size_h: Optional[int] = None,
        image_size_w: Optional[int] = None,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        n_lambda: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        scene_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        # Resolve actual input spatial size (prefer explicit h/w args)
        H_spec = image_size_h if image_size_h is not None else image_size
        W_spec = image_size_w if image_size_w is not None else image_size

        assert H_spec % patch_size == 0, (
            f"image_size_h ({H_spec}) must be divisible by patch_size ({patch_size})."
        )
        assert W_spec % patch_size == 0, (
            f"image_size_w ({W_spec}) must be divisible by patch_size ({patch_size})."
        )
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4."

        self.in_channels = in_channels
        self.H_spec = H_spec
        self.W_spec = W_spec
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_lambda = n_lambda
        self.scene_shape = scene_shape  # (ny, nx) or None

        n_patches_h = H_spec // patch_size
        n_patches_w = W_spec // patch_size
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.n_patches = n_patches_h * n_patches_w
        patch_dim = in_channels * patch_size * patch_size

        # ---- Patch embedding ----
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ---- Positional encoding ----
        self.pos_embed = SinusoidalPosEmbedding2D(embed_dim, n_patches_h, n_patches_w)

        # ---- Transformer encoder ----
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Spectral decoder head ----
        # Project each token → patch_size² * n_lambda values, then pixel-shuffle
        self.decoder_head = nn.Linear(embed_dim, patch_size * patch_size * n_lambda)
        self.final_relu = nn.ReLU()  # non-negative flux

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
            Padded spectrograms.

        Returns
        -------
        cube : Tensor, shape (B, n_lambda, ny, nx)
            Center-cropped to ``scene_shape`` when set, otherwise
            ``(B, n_lambda, H_spec, W_spec)``.
        """
        B, C, H, W = x.shape
        P = self.patch_size

        # ---- Patchify: (B, C, H, W) → (B, n_patches, patch_dim) ----
        tokens = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=P, p2=P)

        # ---- Embed + positional encoding ----
        tokens = self.patch_embed(tokens)    # (B, N, D)
        tokens = self.pos_embed(tokens)      # (B, N, D)

        # ---- Transformer ----
        tokens = self.transformer(tokens)    # (B, N, D)
        tokens = self.norm(tokens)           # (B, N, D)

        # ---- Decode to spatial-spectral ----
        decoded = self.decoder_head(tokens)  # (B, N, P²·n_lambda)

        # Pixel-shuffle: (B, n_h*n_w, P²·n_lambda) → (B, n_lambda, H_spec, W_spec)
        nh = self.n_patches_h
        nw = self.n_patches_w
        cube = rearrange(
            decoded,
            "b (h w) (p1 p2 l) -> b l (h p1) (w p2)",
            h=nh, w=nw, p1=P, p2=P, l=self.n_lambda,
        )
        cube = self.final_relu(cube)

        # ---- Center-crop to scene shape ----
        if self.scene_shape is not None:
            ny, nx = self.scene_shape
            cube = self._center_crop(cube, ny, nx)

        return cube

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
