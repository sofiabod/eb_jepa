"""Shared utilities for neural network initialization and common patterns."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def init_module_weights(m, std: float = 0.02):
    """
    Initialize weights for common layer types using truncated normal distribution.

    This is a unified weight initialization function used across the codebase.
    Apply it via module.apply(init_module_weights) or as a method wrapper.

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_vit_weights(m: nn.Module) -> None:
    """Initialize weights for ViT-style modules (Linear + LayerNorm).

    Used by AttentiveXYHead and ViTVisualDecoder for consistent
    initialization of transformer-based decoders.

    Args:
        m: PyTorch module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    final_ln: bool = False,
) -> nn.Sequential:
    """Build a simple MLP with ReLU activations.

    Used by ActionMLP and MLPEncoder to avoid duplicating the same
    layer-construction loop.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output feature dimension.
        final_ln: Whether to append LayerNorm on the output.

    Returns:
        nn.Sequential containing the MLP layers.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_ln:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


def spatial_layer_norm(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    """Apply LayerNorm to a spatial feature map ``[B, D, H, W]``.

    Permutes to ``[B*H*W, D]``, applies ``ln``, then restores shape.

    Args:
        x: Input tensor ``[B, D, H, W]``.
        ln: LayerNorm module with ``normalized_shape = D``.

    Returns:
        Tensor ``[B, D, H, W]`` after layer normalization.
    """
    B, D, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(B * H * W, D)  # [B*H*W, D]
    x = ln(x)
    x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
    return x


class TemporalBatchMixin:
    """
    Mixin class that handles automatic temporal batching for 4D/5D tensors.

    This mixin provides a unified forward() method that:
    - For 5D tensors [B, C, T, H, W]: flattens temporal dim, applies _forward(), restores shape
    - For 4D tensors [B, C, H, W]: directly applies _forward()

    Subclasses must implement _forward(self, x) for 4D tensors.
    """

    def _forward(self, x):
        """
        Process 4D tensor [B, C, H, W]. Must be implemented by subclasses.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C_out, H_out, W_out]
        """
        raise NotImplementedError("Subclasses must implement _forward()")

    def _forward_temporal(self, x):
        """
        Process 5D tensor [B, C, T, H, W] by folding T into B.

        Default: fold T into B, call _forward(4D), unfold.
        Override in subclasses for custom temporal processing (e.g. causal context).

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            Output tensor of shape [B, C_out, T, H_out, W_out]
        """
        b = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        out = self._forward(x)
        out = rearrange(out, "(b t) c h w -> b c t h w", b=b)
        return out

    def forward(self, x):
        """
        Forward pass supporting both 4D and 5D tensors.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, C, T, H, W]

        Returns:
            Output tensor with same batch and temporal dimensions as input
        """
        assert x.ndim in [
            4,
            5,
        ], "Supports only 4D [B, C, H, W] or 5D [B, C, T, H, W] tensors"
        if x.ndim == 5:
            return self._forward_temporal(x)
        else:
            return self._forward(x)


# ---------------------------------------------------------------------------
# Sincos positional embeddings (shared between AttentivePooler, decoders, etc.)
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """Generate 1D sincos positional embeddings.

    Args:
        embed_dim: Output dimension (must be even).
        positions: [N] array of position values.

    Returns:
        [N, embed_dim] array of positional embeddings.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Generate 2D sincos positional embeddings for a square grid.

    Args:
        embed_dim: Output dimension (must be divisible by 2).
        grid_size: Side length of the square grid (H = W = grid_size).

    Returns:
        [grid_size*grid_size, embed_dim] array of positional embeddings.
    """
    grid = np.arange(grid_size, dtype=np.float32)
    grid_h, grid_w = np.meshgrid(grid, grid, indexing="ij")
    half = embed_dim // 2
    emb_h = _get_1d_sincos_pos_embed(half, grid_h.flatten())
    emb_w = _get_1d_sincos_pos_embed(half, grid_w.flatten())
    return np.concatenate([emb_h, emb_w], axis=1)


# ---------------------------------------------------------------------------
# AttentivePooler: shared between position probes and IDM
# ---------------------------------------------------------------------------


class AttentivePooler(nn.Module):
    """Attentive pooling via a learnable query token and transformer layers.

    Prepends a learnable query token to a sequence of spatial tokens, processes
    through transformer encoder layers with optional sincos 2D positional
    embeddings, and reads out from the query token position.

    Used by both ``AttentiveXYHead`` (position decoding) and
    ``AttentiveInverseDynamicsModel`` (action prediction from state pairs).

    Args:
        input_dim: Dimension of input tokens (encoder output dim C).
        embed_dim: Internal transformer embedding dimension.
        output_dim: Dimension of the output vector.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads (default: embed_dim // 64).
        mlp_ratio: MLP expansion ratio in transformer layers.
        grid_size: Spatial grid size for sincos 2D positional embeddings.
            If None, no positional embeddings are used.
        num_query_tokens: Number of learnable query tokens prepended.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 384,
        output_dim: int = 1,
        depth: int = 3,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        grid_size: Optional[int] = None,
        num_query_tokens: int = 1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        if num_heads is None:
            num_heads = max(1, embed_dim // 64)

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        if grid_size is not None:
            pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
            self.register_buffer(
                "pos_embed",
                torch.from_numpy(pos_embed).float().unsqueeze(0),
            )
        else:
            self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_proj = nn.Linear(embed_dim, output_dim)

        self.apply(init_vit_weights)
        self._rescale_blocks()

    def _rescale_blocks(self) -> None:
        for layer_id, layer in enumerate(self.transformer.layers):
            factor = math.sqrt(2.0 * (layer_id + 1))
            layer.self_attn.out_proj.weight.data.div_(factor)
            layer.linear2.weight.data.div_(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a sequence of spatial tokens into a fixed-size output.

        Args:
            x: [B, N, C] spatial tokens (N = H*W).

        Returns:
            [B, output_dim] pooled output vector.
        """
        x = self.input_proj(x)  # [B, N, embed_dim]

        if self.pos_embed is not None:
            x = x + self.pos_embed

        queries = self.query_tokens.expand(x.size(0), -1, -1)  # [B, Q, D]
        x = torch.cat([queries, x], dim=1)  # [B, Q+N, D]

        x = self.transformer(x)  # [B, Q+N, D]

        query_out = x[:, : self.num_query_tokens]  # [B, Q, D]
        if self.num_query_tokens == 1:
            query_out = query_out.squeeze(1)  # [B, D]
        else:
            query_out = query_out.mean(dim=1)  # [B, D]

        return self.output_proj(query_out)  # [B, output_dim]
