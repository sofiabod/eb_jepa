from __future__ import annotations

import os
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.models.components import (
    Projector,
    _CTTransformer,
    build_frame_causal_mask,
)
from eb_jepa.models.nn import TemporalBatchMixin, spatial_layer_norm

# Suppress xFormers availability warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is not available")

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, 65, 65),
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape)  # (1, C, H, W)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)  # b c w h
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1]  # c * w * h

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, D, T, 1, 1]
        """

        # [B, C, T, H, W] --> [T, B, C, H, W]
        (
            _,
            _,
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            if self.layer_norm:
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)  # b c w h
            # flatten
            out = conv_out.view(conv_out.size(0), -1)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        return features


class _ViTBase(TemporalBatchMixin, nn.Module):
    """Base class for ViT encoders with optional frame-causal context.

    Consolidates shared init (VIT_CONFIGS, HF ViT creation, hidden_size,
    grid_size, output_dim) and adds causal temporal infrastructure when
    ``context_frames != 0``.

    Args:
        scale: ViT scale ('tiny', 'small', 'base').
        patch_size: Patch size in pixels.
        image_size: Expected input image size.
        output_dim: Output channel dim. If None, uses ViT hidden_size.
        input_channels: Number of input channels (default 3).
        context_frames: Number of past frames to attend to when encoding
            frame t. 0 (default) = independent per-frame encoding.
        max_seq_len: Maximum temporal sequence length for positional
            embeddings (only used when context_frames != 0).
        causal_depth: Number of causal transformer blocks (default 4).
        causal_heads: Number of attention heads in causal blocks (default 4).
        causal_dim_head: Dimension per attention head (default 32).
        causal_mlp_ratio: MLP expansion ratio in causal blocks (default 4.0).
    """

    VIT_CONFIGS = {
        "tiny": {
            "hidden_size": 192,
            "num_hidden_layers": 12,
            "num_attention_heads": 3,
            "intermediate_size": 768,
        },
        "small": {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
    }

    def __init__(
        self,
        scale: str = "tiny",
        patch_size: int = 16,
        image_size: int = 256,
        output_dim: Optional[int] = None,
        input_channels: int = 3,
        context_frames: int = 0,
        max_seq_len: int = 16,
        causal_depth: int = 4,
        causal_heads: int = 4,
        causal_dim_head: int = 32,
        causal_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        vit_kwargs = self.VIT_CONFIGS[scale]
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=input_channels,
            **vit_kwargs,
        )
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.hidden_size = vit_kwargs["hidden_size"]
        self.grid_size = image_size // patch_size
        self.context_frames = context_frames

        if output_dim is None:
            output_dim = self.hidden_size
        self.output_dim = output_dim

        if context_frames != 0:
            D = self.hidden_size
            G = self.grid_size
            self.patch_embed = nn.Conv2d(
                input_channels, D, kernel_size=patch_size, stride=patch_size
            )
            self.temporal_pos = nn.Parameter(0.02 * torch.randn(1, max_seq_len, D))
            self.spatial_pos = nn.Parameter(0.02 * torch.randn(1, G * G, D))
            self.causal_transformer = _CTTransformer(
                input_dim=D,
                hidden_dim=D,
                output_dim=D,
                depth=causal_depth,
                heads=causal_heads,
                dim_head=causal_dim_head,
                mlp_ratio=causal_mlp_ratio,
                action_cond_mode="none",
            )
            self._cached_mask = None
            self._cached_T = None

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Get or build the frame-causal attention mask."""
        if self._cached_T != T or self._cached_mask is None:
            G = self.grid_size
            cw = self.context_frames if self.context_frames > 0 else None
            self._cached_mask = build_frame_causal_mask(T, G, G, context_window=cw)
            self._cached_T = T
        return self._cached_mask.to(device)

    def _extract_output(
        self, tokens: torch.Tensor, B: int, T: int, G: int
    ) -> torch.Tensor:
        """Extract final output from causal transformer tokens.

        Args:
            tokens: [B, T*G*G, D] output from causal transformer.
            B: Batch size.
            T: Number of frames.
            G: Spatial grid size.

        Returns:
            Output tensor in subclass-specific shape.
        """
        raise NotImplementedError

    def _forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Process 5D tensor with optional frame-causal context.

        Args:
            x: [B, C, T, H, W]

        Returns:
            Output tensor of shape [B, D_out, T, ...] (subclass-dependent).
        """
        if self.context_frames == 0:
            return super()._forward_temporal(x)

        B, C, T, H, W = x.shape
        G = self.grid_size
        D = self.hidden_size

        # Patch-embed: fold T into B, Conv2d, reshape to [B, T, G*G, D]
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        patches = self.patch_embed(x_flat)  # [B*T, D, G, G]
        patches = patches.reshape(B, T, D, G * G).permute(0, 1, 3, 2)  # [B, T, G*G, D]

        # Add temporal + spatial positional embeddings
        patches = patches + self.temporal_pos[:, :T, None, :]  # [B, T, G*G, D]
        patches = patches + self.spatial_pos[:, None, :, :]  # [B, T, G*G, D]

        # Flatten to sequence, run causal transformer
        tokens = patches.reshape(B, T * G * G, D)  # [B, T*G*G, D]
        mask = self._get_mask(T, tokens.device)
        tokens = self.causal_transformer(tokens, attn_mask=mask)  # [B, T*G*G, D]

        return self._extract_output(tokens, B, T, G)


class ViTEncoder(_ViTBase):
    """ViT encoder with spatial patch token output.

    Wraps HuggingFace ViT. Per-frame: [B, D_out, H', W']. Optionally uses
    frame-causal context when ``context_frames != 0``.

    Args:
        scale: ViT scale ('tiny', 'small', 'base').
        patch_size: Patch size in pixels.
        image_size: Expected input image size.
        output_dim: Desired output channel dim. If None, uses ViT hidden_size.
        input_channels: Number of input channels (default 3).
        context_frames: Past frames for causal context (0 = independent).
        max_seq_len: Max temporal length for positional embeddings.
        causal_depth: Number of causal transformer blocks.
        causal_heads: Number of attention heads in causal blocks.
        causal_dim_head: Dimension per attention head.
        causal_mlp_ratio: MLP expansion ratio in causal blocks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.output_dim != self.hidden_size:
            self.proj = nn.Conv2d(self.hidden_size, self.output_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.final_ln = nn.LayerNorm(self.output_dim)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, D_out, H', W'] where H'=W'=image_size/patch_size
        """
        outputs = self.vit(pixel_values=x)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, N, D] (drop CLS)
        B, N, D = patch_tokens.shape
        G = self.grid_size
        x = patch_tokens.reshape(B, G, G, D).permute(0, 3, 1, 2)  # [B, D, G, G]
        x = self.proj(x)
        x = spatial_layer_norm(x, self.final_ln)
        return x

    def _extract_output(
        self, tokens: torch.Tensor, B: int, T: int, G: int
    ) -> torch.Tensor:
        """Reshape causal tokens to spatial grid, project, and normalize.

        Args:
            tokens: [B, T*G*G, D]
        Returns:
            [B, D_out, T, G, G]
        """
        D = self.hidden_size
        x = tokens.reshape(B * T, G, G, D).permute(0, 3, 1, 2)  # [B*T, D, G, G]
        x = self.proj(x)
        x = spatial_layer_norm(x, self.final_ln)  # [B*T, D_out, G, G]
        D_out = x.shape[1]
        x = x.reshape(B, T, D_out, G, G).permute(0, 2, 1, 3, 4)  # [B, D_out, T, G, G]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handles [B,C,H,W] and [B,C,T,H,W] via TemporalBatchMixin."""
        return super().forward(x)


class ViTCLSEncoder(_ViTBase):
    """ViT encoder that extracts the CLS token (or mean-pools in causal mode).

    Produces [B, D, 1, 1] per frame, suitable for sequence-level predictors.
    Optionally uses frame-causal context when ``context_frames != 0``.

    When ``projector_spec`` is provided, the final LayerNorm is replaced by a
    ``Projector`` (BatchNorm + activation + Linear), matching the le-wm
    architecture where the projector replaces the encoder's final LayerNorm to
    allow the anti-collapse regularizer to work effectively. The projector
    operates directly on the CLS token (no intermediate linear), so its first
    dimension must equal ``hidden_size`` (e.g. ``"192-2048-192"`` for ViT-Tiny).

    Args:
        scale: ViT scale ('tiny', 'small', 'base').
        patch_size: Patch size in pixels.
        image_size: Expected input image size.
        output_dim: Output embedding dimension. If None, uses ViT hidden_size.
            Ignored when ``projector_spec`` is set (output_dim is determined
            by the projector's last dimension).
        input_channels: Number of input channels (default 3).
        context_frames: Past frames for causal context (0 = independent).
        max_seq_len: Max temporal length for positional embeddings.
        causal_depth: Number of causal transformer blocks.
        causal_heads: Number of attention heads in causal blocks.
        causal_dim_head: Dimension per attention head.
        causal_mlp_ratio: MLP expansion ratio in causal blocks.
        projector_spec: Optional MLP spec string (e.g. ``"192-2048-192"``).
            When set, replaces ``proj`` + ``final_ln`` with a ``Projector``.
        projector_activation: Activation for hidden layers (default ``"gelu"``).
        projector_final_bias: Whether the final linear has bias (default True).
    """

    def __init__(self, **kwargs):
        projector_spec = kwargs.pop("projector_spec", None)
        projector_activation = kwargs.pop("projector_activation", "gelu")
        projector_final_bias = kwargs.pop("projector_final_bias", True)

        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)

        if projector_spec is not None:
            self.proj = None
            self.final_ln = None
            self.projector = Projector(
                projector_spec,
                activation=projector_activation,
                final_bias=projector_final_bias,
            )
            self.output_dim = self.projector.out_dim
        else:
            if self.output_dim != self.hidden_size:
                self.proj = nn.Linear(self.hidden_size, self.output_dim)
            else:
                self.proj = nn.Identity()
            self.final_ln = nn.LayerNorm(self.output_dim)
            self.projector = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, output_dim, 1, 1]
        """
        cls = self.vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[
            :, 0
        ]  # [B, hidden_size]
        if self.projector is not None:
            cls = self.projector(cls)  # [B, output_dim]
        else:
            cls = self.proj(cls)  # [B, output_dim]
            cls = self.final_ln(cls)  # [B, output_dim]
        return cls.unsqueeze(-1).unsqueeze(-1)  # [B, output_dim, 1, 1]

    def _extract_output(
        self, tokens: torch.Tensor, B: int, T: int, G: int
    ) -> torch.Tensor:
        """Mean-pool patches per frame, project, and normalize.

        Args:
            tokens: [B, T*G*G, D]
        Returns:
            [B, D_out, T, 1, 1]
        """
        D = self.hidden_size
        x = tokens.reshape(B, T, G * G, D)  # [B, T, G*G, D]
        x = x.mean(dim=2)  # [B, T, D]
        if self.projector is not None:
            BT = B * T
            x = x.reshape(BT, D)  # [B*T, D]
            x = self.projector(x)  # [B*T, D_out]
            D_out = x.shape[-1]
            x = x.reshape(B, T, D_out)  # [B, T, D_out]
        else:
            x = self.proj(x)  # [B, T, D_out]
            x = self.final_ln(x)  # [B, T, D_out]
        return x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D_out, T, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handles [B,C,H,W] and [B,C,T,H,W] via TemporalBatchMixin."""
        return super().forward(x)


class DinoEncoder(TemporalBatchMixin, nn.Module):
    """Frozen DINOv2/v3 encoder with optional output projection.

    Loads a pretrained DINOv2 or DINOv3 backbone, freezes all weights, and
    extracts patch or CLS token features. Reshapes the flat token sequence
    to a spatial grid [B, D, H', W'] matching eb-jepa conventions.

    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via
    TemporalBatchMixin.

    Args:
        name: Model name (e.g. 'dinov2_vitl14', 'dinov3_vitl16').
        image_size: Expected input image size (used to compute grid_size).
        output_dim: If set, project from emb_dim to output_dim via 1x1 Conv2d
            + LayerNorm. If None, output_dim = emb_dim.
        feature_key: Feature to extract from forward_features().
            'x_norm_patchtokens' for spatial patch tokens (default),
            'x_norm_clstoken' for the CLS token.
    """

    def __init__(
        self,
        name: str,
        image_size: int,
        output_dim: Optional[int] = None,
        feature_key: str = "x_norm_patchtokens",
    ):
        super().__init__()
        self.name = name
        self.feature_key = feature_key

        if self.name.startswith("dinov2"):
            self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        elif self.name.startswith("dinov3"):
            pretrained_ckpt_root = os.environ.get("EBJEPA_CKPTS")
            dinov3_path = os.path.join(
                os.environ.get("EBJEPA_HOME", os.path.expanduser("~")), "dinov3"
            )
            if "vitl16" in self.name:
                self.base_model = torch.hub.load(
                    dinov3_path,
                    name,
                    source="local",
                    backbone_weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m-7c1da9a5.pth",
                    weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m-7c1da9a5.pth",
                )
            else:
                self.base_model = torch.hub.load(
                    dinov3_path,
                    name,
                    source="local",
                    weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m.pth",
                )
        else:
            raise ValueError(f"Unknown DINO model family: {name}")

        self.emb_dim = self.base_model.num_features
        self.patch_size = self.base_model.patch_size
        self.grid_size = image_size // self.patch_size

        # Freeze pretrained weights
        self.base_model.requires_grad_(False)
        self.base_model.eval()

        # Optional projection
        if output_dim is not None:
            self.proj = nn.Conv2d(self.emb_dim, output_dim, kernel_size=1)
            self.final_ln = nn.LayerNorm(output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.final_ln = nn.LayerNorm(self.emb_dim)
            self.output_dim = self.emb_dim

    def train(self, mode: bool = True):
        """Override to keep base_model always in eval mode."""
        super().train(mode)
        self.base_model.eval()
        return self

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and reshape to spatial grid.

        Args:
            x: [B, C, H, W]
        Returns:
            [B, D_out, H', W'] where H'=W'=image_size/patch_size
        """
        with torch.no_grad():
            emb = self.base_model.forward_features(x)[self.feature_key]
            # emb: [B, N, emb_dim] for patchtokens, [B, emb_dim] for clstoken

        if self.feature_key == "x_norm_clstoken":
            emb = emb.unsqueeze(1)  # [B, 1, emb_dim]

        B, N, D = emb.shape
        G = self.grid_size
        x = emb.reshape(B, G, G, D).permute(0, 3, 1, 2)  # [B, D, G, G]
        x = self.proj(x)
        x = spatial_layer_norm(x, self.final_ln)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handles [B,C,H,W] and [B,C,T,H,W] via TemporalBatchMixin."""
        return super().forward(x)


class TorchVisionEncoder(TemporalBatchMixin, nn.Module):
    """Encoder wrapping torchvision backbones (ResNet, EfficientNet).

    Preserves spatial structure by truncating the backbone at an intermediate
    layer, with optional adaptive pooling and 1x1 projection.

    Args:
        backbone: Backbone name (e.g. 'resnet18', 'efficientnet_b0').
        truncate_after: Layer to truncate after (e.g. 'layer3', 'layer4').
        output_dim: Desired output channel dimension. If None, uses the native
            channel count of the truncated backbone (no projection).
        output_spatial: Target spatial size (e.g. 16 for 16x16), or None to keep native.
        input_channels: Number of input channels (default 3).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        truncate_after: str = "layer3",
        output_dim: Optional[int] = None,
        output_spatial: Optional[int] = None,
        input_channels: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        import torchvision.models as models

        weights = "DEFAULT" if pretrained else None
        if backbone == "resnet18":
            base = models.resnet18(weights=weights)
            layer_map = {
                "layer1": list(base.children())[:5],
                "layer2": list(base.children())[:6],
                "layer3": list(base.children())[:7],
                "layer4": list(base.children())[:8],
            }
            native_channels = {
                "layer1": 64,
                "layer2": 128,
                "layer3": 256,
                "layer4": 512,
            }
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            # EfficientNet features: 8 blocks, take first 6 for ~16x16 at 256x256
            feature_layers = list(base.features.children())
            layer_map = {
                "features5": feature_layers[:6],
                "features7": feature_layers[:8],
            }
            native_channels = {"features5": 112, "features7": 1280}
            if truncate_after not in layer_map:
                truncate_after = "features5"
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        trunk_layers = layer_map[truncate_after]

        # Adapt first conv for non-3-channel input
        if input_channels != 3:
            first_conv = trunk_layers[0]
            if isinstance(first_conv, nn.Conv2d):
                trunk_layers[0] = nn.Conv2d(
                    input_channels,
                    first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None,
                )

        self.trunk = nn.Sequential(*trunk_layers)

        backbone_ch = native_channels[truncate_after]

        if output_dim is None:
            output_dim = backbone_ch

        post_layers: List[nn.Module] = []

        if output_spatial is not None:
            post_layers.append(nn.AdaptiveAvgPool2d((output_spatial, output_spatial)))

        if output_dim != backbone_ch:
            post_layers.append(nn.Conv2d(backbone_ch, output_dim, kernel_size=1))

        self.post = nn.Sequential(*post_layers) if post_layers else nn.Identity()
        self.output_dim = output_dim

        # Final LayerNorm (applied per-spatial-position)
        self.final_ln = nn.LayerNorm(output_dim)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, D_out, H', W']
        """
        x = self.trunk(x)
        x = self.post(x)
        x = spatial_layer_norm(x, self.final_ln)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handles [B,C,H,W] and [B,C,T,H,W] via TemporalBatchMixin."""
        return super().forward(x)
