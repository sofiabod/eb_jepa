from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from eb_jepa.models.nn import get_2d_sincos_pos_embed, init_vit_weights


class _PositionHeadMixin:
    """Mixin providing Z-score normalization for position probe heads.

    Registers ``pos_mean`` / ``pos_std`` buffers and exposes
    ``normalize_targets`` and ``denormalize`` helpers.
    """

    def _register_pos_stats(
        self,
        pos_mean: Optional[torch.Tensor] = None,
        pos_std: Optional[torch.Tensor] = None,
    ) -> None:
        if pos_mean is not None:
            self.register_buffer("pos_mean", pos_mean.view(1, -1, 1))
            self.register_buffer("pos_std", pos_std.view(1, -1, 1).clamp(min=1e-6))
        else:
            self.pos_mean = None
            self.pos_std = None

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Z-score normalize targets. Shape: [B, D, T]."""
        if self.pos_mean is not None:
            return (targets - self.pos_mean) / self.pos_std
        return targets

    def denormalize(self, predictions: torch.Tensor) -> torch.Tensor:
        """Invert Z-score normalization. Shape: [B, D, T]."""
        if self.pos_mean is not None:
            return predictions * self.pos_std + self.pos_mean
        return predictions


class MLPXYHead(_PositionHeadMixin, nn.Module):
    """A head to recover the xy location from features.

    Args:
        input_shape: Input feature dimension.
        output_dim: Number of output dimensions (default 2 for xy).
        hidden_dims: List of hidden layer sizes. Depth is implicit from
            ``len(hidden_dims)``. Default ``[512]`` preserves legacy behavior.
        normalizer: Optional normalizer applied to predictions.
        pos_mean: Per-dim position mean for Z-score normalization ``[output_dim]``.
        pos_std: Per-dim position std for Z-score normalization ``[output_dim]``.
    """

    def __init__(
        self,
        input_shape,
        output_dim=2,
        hidden_dims=None,
        normalizer=None,
        pos_mean=None,
        pos_std=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        if hidden_dims is None:
            hidden_dims = [512]

        layers = []
        in_dim = input_shape
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.normalizer = normalizer
        self._register_pos_stats(pos_mean, pos_std)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            pred: [B, output_dim, T]
        """
        bs, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(bs * t, c, h, w)  # [B*T, C, H, W]

        # Global average pool for spatial dims > 1, squeeze for 1x1
        if h > 1 or w > 1:
            x = x.mean(dim=(-2, -1))  # [B*T, C]
        else:
            x = x.squeeze(-1).squeeze(-1)  # [B*T, C]

        pred = self.mlp(x)

        pred = pred.view(bs, t, self.output_dim).permute(0, 2, 1)  # [B, output_dim, T]

        return pred


class AttentiveXYHead(_PositionHeadMixin, nn.Module):
    """ViT-based state readout head for position decoding.

    Follows the StateReadoutViT pattern from jepa-wms: a learnable state token
    is prepended to the spatial token sequence, processed through transformer
    blocks with sincos positional embeddings, then projected to output_dim.

    This preserves spatial information (unlike MLPXYHead's global average
    pooling), enabling accurate position decoding from spatially-structured
    encoder features.

    Args:
        input_shape: Encoder output dimension (C).
        output_dim: Number of output dimensions (e.g. 3 for xyz).
        decoder_embed_dim: Internal embedding dimension of the decoder.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads (default: decoder_embed_dim // 64).
        mlp_ratio: MLP expansion ratio in transformer layers.
        grid_size: Spatial grid size (H=W) for sincos positional embeddings.
            If None, uses learnable pos embed sized dynamically on first forward.
        normalizer: Optional normalizer applied to predictions.
        pos_mean: Per-dim position mean for Z-score normalization ``[output_dim]``.
        pos_std: Per-dim position std for Z-score normalization ``[output_dim]``.
    """

    def __init__(
        self,
        input_shape: int,
        output_dim: int = 2,
        decoder_embed_dim: int = 384,
        depth: int = 3,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        grid_size: Optional[int] = None,
        normalizer=None,
        pos_mean=None,
        pos_std=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.decoder_embed_dim = decoder_embed_dim
        if num_heads is None:
            num_heads = max(1, decoder_embed_dim // 64)

        self.decoder_embed = nn.Linear(input_shape, decoder_embed_dim)

        self.state_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.state_token, std=0.02)

        if grid_size is not None:
            pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size)
            self.register_buffer(
                "decoder_pos_embed",
                torch.from_numpy(pos_embed).float().unsqueeze(0),
            )
        else:
            self.decoder_pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.state_proj = nn.Linear(decoder_embed_dim, output_dim)
        self.normalizer = normalizer
        self._register_pos_stats(pos_mean, pos_std)

        self.apply(init_vit_weights)
        self._rescale_blocks()

    def _rescale_blocks(self) -> None:
        for layer_id, layer in enumerate(self.transformer.layers):
            factor = math.sqrt(2.0 * (layer_id + 1))
            layer.self_attn.out_proj.weight.data.div_(factor)
            layer.linear2.weight.data.div_(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with state-token readout.

        Args:
            x: [B, C, T, H, W]

        Returns:
            pred: [B, output_dim, T]
        """
        bs, c, t, h, w = x.shape

        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        x = x.reshape(bs * t, h * w, c)  # [B*T, H*W, C]

        x = self.decoder_embed(x)  # [B*T, H*W, decoder_embed_dim]

        if self.decoder_pos_embed is not None:
            x = x + self.decoder_pos_embed

        state_tokens = self.state_token.expand(bs * t, -1, -1)  # [B*T, 1, D]
        x = torch.cat([state_tokens, x], dim=1)  # [B*T, 1+H*W, D]

        x = self.transformer(x)  # [B*T, 1+H*W, D]

        state_out = x[:, 0]  # [B*T, D]
        pred = self.state_proj(state_out)  # [B*T, output_dim]
        pred = pred.view(bs, t, self.output_dim).permute(0, 2, 1)  # [B, output_dim, T]

        return pred
