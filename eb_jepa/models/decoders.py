"""Visual decoder classes for reconstructing images from learned representations."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from eb_jepa.models.nn import TemporalBatchMixin, init_module_weights, init_vit_weights

VIT_DECODER_CONFIGS = {
    "tiny": {"hidden_size": 192, "num_layers": 12, "nhead": 3, "dim_feedforward": 768},
    "small": {
        "hidden_size": 384,
        "num_layers": 12,
        "nhead": 6,
        "dim_feedforward": 1536,
    },
    "base": {
        "hidden_size": 768,
        "num_layers": 12,
        "nhead": 12,
        "dim_feedforward": 3072,
    },
}


class _VisualDecoderBase(nn.Module):
    """Base class providing shared normalize buffer registration and decode_to_uint8."""

    def _register_normalize_buffers(
        self,
        normalize_mean: Optional[List[float]],
        normalize_std: Optional[List[float]],
    ):
        if normalize_mean is not None:
            self.register_buffer(
                "normalize_mean",
                torch.tensor(normalize_mean, dtype=torch.float32).view(1, -1, 1, 1),
            )
        else:
            self.normalize_mean = None

        if normalize_std is not None:
            self.register_buffer(
                "normalize_std",
                torch.tensor(normalize_std, dtype=torch.float32).view(1, -1, 1, 1),
            )
        else:
            self.normalize_std = None

    def decode_to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """Decode encoder states to uint8 numpy images with inverse normalization.

        Runs the forward pass, applies inverse channel normalization if
        ``normalize_mean`` / ``normalize_std`` were provided, clamps to [0, 1],
        and returns uint8 [B, T, H, W, C].

        Args:
            x: Encoded states [B, D, T, H, W].

        Returns:
            np.ndarray of shape [B, T, H, W, C] with dtype uint8.
        """
        decoded = self.forward(x)  # [B, C, T, H, W]
        if self.normalize_std is not None:
            decoded = decoded * self.normalize_std.unsqueeze(2)
        if self.normalize_mean is not None:
            decoded = decoded + self.normalize_mean.unsqueeze(2)
        decoded = decoded.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        decoded = (decoded * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return decoded


class VisualDecoder(TemporalBatchMixin, _VisualDecoderBase):
    """Decodes [B, D, T, 1, 1] encoder representations into [B, C, T, H, W] images.

    Progressive upsampling from init_spatial via ConvTranspose2d doublings,
    followed by bilinear interpolation to the exact target size.

    Args:
        input_dim: Encoder output dimension D.
        output_channels: Number of output image channels (default: 3 for RGB).
        target_h: Target image height.
        target_w: Target image width.
        base_channels: Number of channels after the initial linear projection.
        init_spatial: Initial spatial size for reshaping (default 1 for 1x1 start).
            Set to 4 or 8 to reduce the number of upsample stages.
        min_channels: Minimum channel count in upsample stages (default 64).
        normalize_mean: Per-channel mean used by the data pipeline (for inverse
            normalization at decode time). None means no normalization was applied.
        normalize_std: Per-channel std used by the data pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        output_channels: int = 3,
        target_h: int = 224,
        target_w: int = 224,
        base_channels: int = 256,
        init_spatial: int = 1,
        min_channels: int = 64,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_h = target_h
        self.target_w = target_w
        self.base_channels = base_channels
        self.init_spatial = init_spatial

        self._register_normalize_buffers(normalize_mean, normalize_std)

        self.linear = nn.Linear(input_dim, base_channels * init_spatial * init_spatial)

        n_ups = math.ceil(math.log2(max(target_h, target_w) / init_spatial))

        up_blocks = []
        ch_in = base_channels
        for i in range(n_ups):
            ch_out = max(ch_in // 2, min_channels)
            up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ch_in, ch_out, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(min(32, ch_out), ch_out),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = ch_out
        self.up_blocks = nn.ModuleList(up_blocks)

        self.upsample = nn.Upsample(
            size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        self.head = nn.Conv2d(ch_in, output_channels, kernel_size=1)

        self.apply(init_module_weights)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode 4D tensor [B*T, D, 1, 1] -> [B*T, C_out, H, W].

        Args:
            x: Input tensor of shape [B*T, D, 1, 1].

        Returns:
            Decoded image tensor [B*T, C_out, target_h, target_w].
        """
        bt = x.shape[0]
        x = x.squeeze(-1).squeeze(-1)  # [B*T, D]
        x = self.linear(x)  # [B*T, base_channels * init_h * init_w]
        s = self.init_spatial
        x = x.view(bt, self.base_channels, s, s)  # [B*T, base_channels, init_h, init_w]

        for block in self.up_blocks:
            x = block(x)

        x = self.upsample(x)  # [B*T, ch, target_h, target_w]
        x = self.head(x)  # [B*T, C_out, target_h, target_w]
        return x


class ImageDecoder(TemporalBatchMixin, nn.Module):
    """Simple 2D convolutional decoder for reconstructing images from representations.

    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(
        self,
        in_dim,
        out_dim=1,
        hidden_dim=16,
        tk=1,  # unused in 2D; kept for API compatibility
        ts=1,  # unused in 2D; kept for API compatibility
        sk=4,  # spatial kernel for ConvTranspose2d
        ss=2,  # spatial stride (controls the upsample factor)
        pad_mode="same",
        scale_factor=1.0,
        shift_factor=0.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
        )

        self.apply(init_module_weights)

    def _forward(self, x):
        # x: (B,C,H,W)
        y = self.net(x)
        return y


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """Find the largest valid GroupNorm num_groups <= max_groups for channels."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _DecoderResBlock(nn.Module):
    """Pre-activation residual block for ResNetVisualDecoder.

    GroupNorm + SiLU + Conv2d(3x3) + GroupNorm + SiLU + Conv2d(3x3), with a
    1x1 skip projection when ``ch_in != ch_out``.
    """

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(ch_in), ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_gn_groups(ch_out), ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(ch_in, ch_out, kernel_size=1)
            if ch_in != ch_out
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.act(self.norm2(self.conv1(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class ResNetVisualDecoder(TemporalBatchMixin, _VisualDecoderBase):
    """Decodes [B, D, T, H', W'] spatially-structured features into images.

    Uses ResBlocks at each resolution with bilinear upsample + 3x3 conv
    instead of ConvTranspose2d, avoiding checkerboard artifacts. Each
    resolution level has a residual block for feature refinement.

    Args:
        input_dim: Encoder output dimension D.
        input_spatial: Encoder output spatial size (H'=W').
        output_channels: Number of output image channels (default: 3 for RGB).
        target_h: Target image height.
        target_w: Target image width.
        base_channels: Working channel count at the lowest resolution.
            Halved at each upsample stage (clamped by min_channels).
        min_channels: Minimum channel count in upsample stages (default 64).
        normalize_mean: Per-channel mean for inverse normalization.
        normalize_std: Per-channel std for inverse normalization.
    """

    def __init__(
        self,
        input_dim: int,
        input_spatial: int,
        output_channels: int = 3,
        target_h: int = 224,
        target_w: int = 224,
        base_channels: int = 256,
        min_channels: int = 64,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

        self._register_normalize_buffers(normalize_mean, normalize_std)

        self.input_conv = nn.Conv2d(input_dim, base_channels, kernel_size=3, padding=1)
        self.input_block = _DecoderResBlock(base_channels, base_channels)

        n_ups = math.ceil(math.log2(max(target_h, target_w) / input_spatial))

        up_blocks = nn.ModuleList()
        ch_in = base_channels
        for _ in range(n_ups):
            ch_out = max(ch_in // 2, min_channels)
            up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                    _DecoderResBlock(ch_out, ch_out),
                )
            )
            ch_in = ch_out
        self.up_blocks = up_blocks

        self.final_norm = nn.GroupNorm(_gn_groups(ch_in), ch_in)
        self.final_act = nn.SiLU(inplace=True)
        self.head = nn.Conv2d(ch_in, output_channels, kernel_size=1)

        self.apply(init_module_weights)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode 4D tensor [B*T, D, H', W'] -> [B*T, C_out, H, W].

        Args:
            x: Input tensor of shape [B*T, D, H', W'].

        Returns:
            Decoded image tensor [B*T, C_out, target_h, target_w].
        """
        x = self.input_conv(x)  # [B*T, base_channels, H', W']
        x = self.input_block(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.final_act(self.final_norm(x))

        # Exact resize for non-power-of-2 targets
        if x.shape[-2] != self.target_h or x.shape[-1] != self.target_w:
            x = nn.functional.interpolate(
                x,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            )

        x = self.head(x)  # [B*T, C_out, target_h, target_w]
        return x


class ViTVisualDecoder(TemporalBatchMixin, _VisualDecoderBase):
    """Decodes encoder representations into images using a ViT decoder.

    Supports two input modes:
    - CLS input ``[B, D, T, 1, 1]``: project, broadcast to N patches, add pos
      embed, then decode via transformer + unpatchify.
    - Spatial patch input ``[B, D, T, H', W']`` where ``H'*W' == n_patches``:
      reshape to token sequence, project, add pos embed, then decode.

    Args:
        input_dim: Encoder output dimension D.
        scale: ViT scale ('tiny', 'small', 'base') to select hidden_size,
            num_layers, nhead, dim_feedforward from VIT_DECODER_CONFIGS.
        patch_size: Patch size used for patchify/unpatchify.
        image_size: Target image size (square images assumed).
        output_channels: Number of output image channels (default: 3).
        depth_override: If set, use this many transformer layers instead of
            the default from VIT_DECODER_CONFIGS.
        use_refine: If True, apply a lightweight residual conv head after
            unpatchify to smooth patch-boundary artifacts (default: False).
        normalize_mean: Per-channel mean for inverse normalization.
        normalize_std: Per-channel std for inverse normalization.
    """

    def __init__(
        self,
        input_dim: int,
        scale: str = "tiny",
        patch_size: int = 16,
        image_size: int = 224,
        output_channels: int = 3,
        depth_override: Optional[int] = None,
        use_refine: bool = False,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__()
        cfg = VIT_DECODER_CONFIGS[scale]
        hidden_size = cfg["hidden_size"]
        num_layers = depth_override if depth_override is not None else cfg["num_layers"]
        nhead = cfg["nhead"]
        dim_feedforward = cfg["dim_feedforward"]

        self.patch_size = patch_size
        self.image_size = image_size
        self.output_channels = output_channels
        self.n_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size

        self._register_normalize_buffers(normalize_mean, normalize_std)

        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(hidden_size, patch_size * patch_size * output_channels)

        self.use_refine = use_refine
        if use_refine:
            refine_ch = max(output_channels * 4, 32)
            self.refine = nn.Sequential(
                nn.Conv2d(output_channels, refine_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(16, refine_ch), refine_ch),
                nn.GELU(),
                nn.Conv2d(refine_ch, refine_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(16, refine_ch), refine_ch),
                nn.GELU(),
                nn.Conv2d(refine_ch, output_channels, kernel_size=3, padding=1),
            )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(init_vit_weights)
        if use_refine:
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode 4D tensor to image pixels.

        Supports two input modes:
        - CLS input ``[B*T, D, 1, 1]``: project, broadcast to N patches,
          add pos embed.
        - Spatial patch input ``[B*T, D, H', W']`` where ``H'*W' == n_patches``:
          reshape to token sequence, project, add pos embed.

        Args:
            x: Input tensor of shape [B*T, D, 1, 1] or [B*T, D, H', W'].

        Returns:
            Decoded image tensor [B*T, C_out, image_size, image_size].
        """
        bt, d, h, w = x.shape
        if h == 1 and w == 1:
            x = x.squeeze(-1).squeeze(-1)  # [B*T, D]
            x = self.input_proj(x)  # [B*T, hidden_size]
            x = x.unsqueeze(1).expand(-1, self.n_patches, -1)  # [B*T, N, hidden_size]
        else:
            x = x.permute(0, 2, 3, 1).reshape(bt, h * w, d)  # [B*T, N, D]
            x = self.input_proj(x)  # [B*T, N, hidden_size]
        x = x + self.pos_embed  # [B*T, N, hidden_size]

        x = self.transformer(x)  # [B*T, N, hidden_size]

        x = self.head(x)  # [B*T, N, patch_size^2 * C]

        # Unpatchify: [B*T, N, P*P*C] -> [B*T, C, H, W]
        p = self.patch_size
        c = self.output_channels
        gh = gw = self.image_size // p
        x = x.view(bt, gh, gw, p, p, c)  # [B*T, gh, gw, p, p, C]
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B*T, C, gh, p, gw, p]
        x = x.reshape(bt, c, self.image_size, self.image_size)  # [B*T, C, H, W]
        if self.use_refine:
            x = x + self.refine(x)
        return x


class SpatialVisualDecoder(TemporalBatchMixin, _VisualDecoderBase):
    """Decodes [B, D, T, H', W'] spatially-structured encoder representations into images.

    Starts from H'>1 spatial dims, uses fewer upsample stages than VisualDecoder.
    Initial projection via Conv2d(input_dim, base_channels, 1), then progressive
    ConvTranspose2d upsampling to target size.

    Args:
        input_dim: Encoder output dimension D.
        input_spatial: Encoder output spatial size (H'=W').
        output_channels: Number of output image channels (default: 3 for RGB).
        target_h: Target image height.
        target_w: Target image width.
        base_channels: Number of channels after the initial Conv2d projection.
        min_channels: Minimum channel count in upsample stages (default 64).
        normalize_mean: Per-channel mean for inverse normalization.
        normalize_std: Per-channel std for inverse normalization.
    """

    def __init__(
        self,
        input_dim: int,
        input_spatial: int,
        output_channels: int = 3,
        target_h: int = 256,
        target_w: int = 256,
        base_channels: int = 256,
        min_channels: int = 64,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_spatial = input_spatial
        self.target_h = target_h
        self.target_w = target_w
        self.base_channels = base_channels

        self._register_normalize_buffers(normalize_mean, normalize_std)

        self.proj = nn.Conv2d(input_dim, base_channels, kernel_size=1)

        n_ups = math.ceil(math.log2(max(target_h, target_w) / input_spatial))

        up_blocks = []
        ch_in = base_channels
        for i in range(n_ups):
            ch_out = max(ch_in // 2, min_channels)
            up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ch_in, ch_out, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(min(32, ch_out), ch_out),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = ch_out
        self.up_blocks = nn.ModuleList(up_blocks)

        self.upsample = nn.Upsample(
            size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        self.head = nn.Conv2d(ch_in, output_channels, kernel_size=1)

        self.apply(init_module_weights)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode 4D tensor [B*T, D, H', W'] -> [B*T, C_out, H, W].

        Args:
            x: Input tensor of shape [B*T, D, H', W'].

        Returns:
            Decoded image tensor [B*T, C_out, target_h, target_w].
        """
        x = self.proj(x)  # [B*T, base_channels, H', W']

        for block in self.up_blocks:
            x = block(x)

        x = self.upsample(x)  # [B*T, ch, target_h, target_w]
        x = self.head(x)  # [B*T, C_out, target_h, target_w]
        return x
