from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.models.nn import (
    AttentivePooler,
    TemporalBatchMixin,
    build_mlp,
    init_module_weights,
    spatial_layer_norm,
)


class CostModule(nn.Module):
    """Cost module with projector and loss for planning objectives."""

    def __init__(self, projector: nn.Module, loss: nn.Module):
        super().__init__()
        self.projector = projector
        self.loss = loss

    def forward(self, x: torch.Tensor):
        """Returns (loss, loss_dict) tuple."""
        return self.loss(x)


class conv3d2(nn.Sequential):
    """Simple 3D convnet with 2 layers."""

    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super(conv3d2, self).__init__(
            nn.Conv3d(
                in_d, h_d, kernel_size=(tk, sk, sk), stride=(1, 1, 1), padding=pad
            ),
            nn.ReLU(),
            nn.Conv3d(
                h_d, out_d, kernel_size=(tk, sk, sk), stride=(ts, ss, ss), padding=pad
            ),
        )
        self.apply(init_module_weights)
        self.input_dim = in_d
        self.hidden_dim = h_d
        self.output_dim = out_d
        # t_shift is the index (in the time dimension) of the first output
        # cannot see its coresponding input
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)
        else:
            raise NameError("invalid padding for con3d2. Must be 'valid' or 'same'")


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet5(TemporalBatchMixin, nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks).
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else torch.nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class SimplePredictor(nn.Module):
    """Wrapper that concatenates states and actions channel-wise before prediction."""

    def __init__(self, predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x, a], dim=1))


class StateOnlyPredictor(SimplePredictor):
    """Wrapper for a simple predictor which concatenates states and actions channel wise."""

    def forward(self, x, a):
        # action not used on purpose
        prev_state = x[:, :, :-1]  # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]  # [B, C, T-1, H, W]
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)


class ResUNet(TemporalBatchMixin, nn.Module):
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling.
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        # Stem
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)  # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)  # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)  # H/4, W/4
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)  # H/8, W/8

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # Guards against odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)  # 8h

        # Decoder stage 3
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # 4h + 4h = 8h
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # 2h + 2h = 4h
        d2 = self.dec2(d2)  # → 2h

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # h + h = 2h
        d1 = self.dec1(d1)  # → h

        out = self.head(d1)  # → out_d channels
        return out


class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'.

    Args:
        mlp_spec: Dash-separated layer dimensions, e.g. ``"192-2048-192"``.
        activation: Activation function for hidden layers (``"relu"`` or
            ``"gelu"``).  Default ``"relu"`` for backward compatibility.
        final_bias: Whether to include bias in the final linear layer.
            Default ``False`` for backward compatibility.
    """

    def __init__(
        self,
        mlp_spec: str,
        activation: str = "relu",
        final_bias: bool = False,
    ):
        super().__init__()
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU(True)
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(act_fn)
        layers.append(nn.Linear(f[-2], f[-1], bias=final_bias))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]

    def forward(self, x):
        return self.net(x)


class DetHead(nn.Module):
    """Detection head that pools features and predicts binary maps."""

    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        """Forward pass on predictor output of shape (B, C, T, H, W)."""
        # (Batch, Feature, Time, Height, Width)
        # [8, 8, T, 8, 8]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        # [8, T, 8, 8]
        x = self.head(x).squeeze(1)

        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):
        from sklearn.metrics import average_precision_score

        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)

            y = targets[:, T:]
            x = x[:, T:]

            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)

        return scores


class InverseDynamicsModel(nn.Module):
    """MLP-based inverse dynamics model for non-spatial (1x1) encoder outputs.

    Concatenates flattened state pairs and predicts the action via a small MLP.
    Only appropriate when the encoder output has no spatial structure (H=W=1),
    e.g. Impala or CLS-token encoders.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        """Predict action from consecutive state vectors.

        Args:
            state_t: [B, D] state at time t.
            state_t_plus_1: [B, D] state at time t+1.

        Returns:
            [B, A] predicted action.
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)  # [B, 2*D]
        return self.model(combined_states)


class AttentiveInverseDynamicsModel(nn.Module):
    """Attentive inverse dynamics model for spatially-structured encoder outputs.

    Instead of flattening C*H*W tokens into a massive vector (which creates
    a first linear layer with ~50M params for typical ViT encoders), this model
    uses two AttentivePooler instances to independently pool each state's
    spatial tokens into a compact vector, then predicts the action from the
    concatenated pooled representations via a small MLP.

    Args:
        input_dim: Per-token feature dimension (encoder output dim C).
        action_dim: Action dimension to predict.
        embed_dim: Internal embedding dimension for the attentive pooler.
        depth: Number of transformer layers in each pooler.
        num_heads: Number of attention heads (default: embed_dim // 64).
        mlp_ratio: MLP expansion ratio in transformer layers.
        grid_size: Spatial grid size for sincos 2D positional embeddings.
        hidden_dim: Hidden dimension of the action prediction MLP.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        embed_dim: int = 384,
        depth: int = 3,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        grid_size: Optional[int] = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            input_dim=input_dim,
            embed_dim=embed_dim,
            output_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            grid_size=grid_size,
        )
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.action_head.apply(init_module_weights)

    def forward(
        self, state_t: torch.Tensor, state_t_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """Predict action from consecutive spatial state tokens.

        Args:
            state_t: [B, N, C] spatial tokens at time t.
            state_t_plus_1: [B, N, C] spatial tokens at time t+1.

        Returns:
            [B, A] predicted action.
        """
        z_t = self.pooler(state_t)  # [B, embed_dim]
        z_tp1 = self.pooler(state_t_plus_1)  # [B, embed_dim]
        combined = torch.cat([z_t, z_tp1], dim=1)  # [B, 2*embed_dim]
        return self.action_head(combined)  # [B, action_dim]


class ActionMLP(nn.Module):
    """Simple MLP for action encoding (flat 2D input, no spatial dims).

    Processes flat [B, input_dim] tensors, typically used for aggregating
    windowed actions at higher hierarchy levels.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        final_ln: bool = False,
    ):
        """Initialize ActionMLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output feature dimension.
            final_ln: Whether to apply LayerNorm to final layer.
        """
        super().__init__()
        self.output_dim = output_dim
        self.net = build_mlp(input_dim, hidden_dims, output_dim, final_ln)
        self.apply(init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, input_dim] (flat action window)
        Returns:
            Output tensor [B, output_dim]
        """
        return self.net(x)


class MLPEncoder(nn.Module):
    """MLP-based encoder for vector embeddings (operates on last dim).

    Processes [B, D, T, H, W] tensors by treating spatial dims (H, W) as batch dims.
    Useful for encoding abstract representations at higher hierarchy levels.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        final_ln: bool = False,
    ):
        """Initialize MLPEncoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output feature dimension.
            final_ln: Whether to apply LayerNorm to final layer.
        """
        super().__init__()
        self.output_dim = output_dim
        self.net = build_mlp(input_dim, hidden_dims, output_dim, final_ln)
        self.apply(init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, D, T, H, W] or [B, D, T, 1, 1] (vector embeddings)
        Returns:
            Output tensor [B, D_out, T, 1, 1]
        """
        B, D, T, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, D)  # [B*T*H*W, D]
        x_out = self.net(x_flat)  # [B*T*H*W, D_out]
        x_out = x_out.reshape(B, T, H, W, self.output_dim).permute(
            0, 4, 1, 2, 3
        )  # [B, D_out, T, H, W]
        return x_out


class ConvEncoder(nn.Module):
    """Conv2d encoder for spatial downsampling at higher hierarchy levels.

    Processes [B, D, T, H, W] tensors using Conv2d layers with stride for
    spatial downsampling. Unlike MLPEncoder, this enables spatial interaction
    between features via convolution kernels.

    Args:
        input_dim: Input feature dimension (channels).
        output_dim: Output feature dimension.
        hidden_dims: List of hidden layer channel dimensions.
        kernel_size: Convolution kernel size (default 3).
        stride: Convolution stride for downsampling (default 2).
        final_ln: Whether to apply LayerNorm to output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        kernel_size: int = 3,
        stride: int = 2,
        final_ln: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.stride = stride

        if hidden_dims is None:
            hidden_dims = []

        layers: List[nn.Module] = []
        prev_dim = input_dim
        padding = kernel_size // 2
        for hidden_dim in hidden_dims:
            layers.append(
                nn.Conv2d(
                    prev_dim, hidden_dim, kernel_size, stride=stride, padding=padding
                )
            )
            layers.append(nn.GroupNorm(min(32, hidden_dim), hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
            stride = 1  # only first layer downsamples by default

        layers.append(
            nn.Conv2d(prev_dim, output_dim, kernel_size=1, stride=1, padding=0)
        )
        self.conv_layers = nn.Sequential(*layers)
        self.final_ln = nn.LayerNorm(output_dim) if final_ln else nn.Identity()
        self.apply(init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, D, T, H, W].
        Returns:
            Output tensor [B, D_out, T, H//s, W//s].
        """
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W)  # [B*T, D, H, W]
        x = self.conv_layers(x)  # [B*T, D_out, H', W']
        _, D_out, H_out, W_out = x.shape
        # LayerNorm over channel dim: reshape to [B*T*H'*W', D_out]
        x = x.permute(0, 2, 3, 1).reshape(-1, D_out)  # [B*T*H'*W', D_out]
        x = self.final_ln(x)
        x = x.reshape(B, T, H_out, W_out, D_out).permute(
            0, 4, 1, 2, 3
        )  # [B, D_out, T, H', W']
        return x


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for spatially-structured state propagation.

    Replaces the linear gates in a standard GRU with Conv2d operations,
    preserving spatial structure throughout recurrence.

    Args:
        hidden_dim: Number of channels in the hidden state.
        input_dim: Number of channels in the input (action broadcast).
        kernel_size: Convolution kernel size for gates (default 3).
    """

    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.gate_conv = nn.Conv2d(
            hidden_dim + input_dim, 2 * hidden_dim, kernel_size, padding=padding
        )
        self.candidate_conv = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, input_dim, H, W].
            h: Hidden state [B, hidden_dim, H, W].
        Returns:
            New hidden state [B, hidden_dim, H, W].
        """
        combined = torch.cat([h, x], dim=1)  # [B, hidden_dim+input_dim, H, W]
        gates = torch.sigmoid(self.gate_conv(combined))  # [B, 2*hidden_dim, H, W]
        r, z = gates.chunk(2, dim=1)  # reset, update: each [B, hidden_dim, H, W]

        candidate_input = torch.cat(
            [r * h, x], dim=1
        )  # [B, hidden_dim+input_dim, H, W]
        n = torch.tanh(self.candidate_conv(candidate_input))  # [B, hidden_dim, H, W]
        h_new = (1 - z) * n + z * h  # [B, hidden_dim, H, W]
        return h_new


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with depthwise large-kernel conv and inverted bottleneck.

    Uses depthwise convolution with kernel_size = spatial_size - 1 for
    near-global receptive field on the spatial grid, followed by pointwise
    expand/contract (inverted bottleneck).

    Args:
        dim: Number of input/output channels.
        spatial_size: Spatial grid size (H=W). Kernel is spatial_size - 1.
        expansion_ratio: Channel expansion factor for inverted bottleneck.
    """

    def __init__(self, dim: int, spatial_size: int, expansion_ratio: int = 4):
        super().__init__()
        kernel_size = spatial_size - 1 if spatial_size > 1 else 1
        padding = kernel_size // 2
        hidden_dim = dim * expansion_ratio
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw_expand = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.pw_contract = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, C, H, W].
        Returns:
            [B, C, H, W] with residual connection.
        """
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_contract(x)
        return x + residual


class MiniUNet(nn.Module):
    """Mini U-Net for hierarchical multi-scale spatial mixing.

    Uses GroupNorm throughout (not BatchNorm2d) for stability with small
    batch sizes. Encoder: stem -> level1 (skip) -> level2 (stride=2, skip)
    -> bottleneck (stride=2). Decoder: upsample + skip concat + conv.
    Residual connection from input to output.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_channels: Base channel width (default 64).
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        super().__init__()
        bc = base_channels

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, bc, 3, padding=1),
            nn.GroupNorm(1, bc),
            nn.GELU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(bc, bc, 3, padding=1),
            nn.GroupNorm(1, bc),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(bc, 2 * bc, 3, stride=2, padding=1),
            nn.GroupNorm(1, 2 * bc),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * bc, 4 * bc, 3, stride=2, padding=1),
            nn.GroupNorm(1, 4 * bc),
            nn.GELU(),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(4 * bc, 2 * bc, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(4 * bc, 2 * bc, 3, padding=1),
            nn.GroupNorm(1, 2 * bc),
            nn.GELU(),
        )
        self.up1 = nn.ConvTranspose2d(2 * bc, bc, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * bc, bc, 3, padding=1),
            nn.GroupNorm(1, bc),
            nn.GELU(),
        )
        self.head = nn.Conv2d(bc, out_channels, 1)

        self.skip_proj = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, in_channels, H, W].
        Returns:
            [B, out_channels, H, W] with residual connection.
        """
        residual = self.skip_proj(x)

        # Encoder
        s = self.stem(x)  # [B, bc, H, W]
        e1 = self.enc1(s)  # [B, bc, H, W] (skip1)
        e2 = self.enc2(e1)  # [B, 2*bc, H/2, W/2] (skip2)
        b = self.bottleneck(e2)  # [B, 4*bc, H/4, W/4]

        # Decoder
        d2 = self.up2(b)  # [B, 2*bc, H/2, W/2]
        d2 = self._match_and_cat(d2, e2)  # [B, 4*bc, H/2, W/2]
        d2 = self.dec2(d2)  # [B, 2*bc, H/2, W/2]

        d1 = self.up1(d2)  # [B, bc, H, W]
        d1 = self._match_and_cat(d1, e1)  # [B, 2*bc, H, W]
        d1 = self.dec1(d1)  # [B, bc, H, W]

        out = self.head(d1)  # [B, out_channels, H, W]
        return out + residual

    @staticmethod
    def _match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Crop or pad upsampled tensor to match skip spatial dims, then concatenate."""
        dh = skip.shape[2] - upsampled.shape[2]
        dw = skip.shape[3] - upsampled.shape[3]
        if dh > 0 or dw > 0:
            upsampled = F.pad(upsampled, [0, max(dw, 0), 0, max(dh, 0)])
        elif dh < 0 or dw < 0:
            upsampled = upsampled[:, :, : skip.shape[2], : skip.shape[3]]
        return torch.cat([upsampled, skip], dim=1)


# ---------------------------------------------------------------------------
# Causal Transformer Predictor (ported from LEWM)
# ---------------------------------------------------------------------------


def _ct_modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """AdaLN-Zero modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class _CTFeedForward(nn.Module):
    """Feed-forward block used inside CausalTransformer."""

    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.0, prenorm: bool = False
    ):
        super().__init__()
        layers = []
        if prenorm:
            layers.append(nn.LayerNorm(dim))
        layers += [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CTAttention(nn.Module):
    """Causal multi-head attention for CausalTransformer."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        prenorm: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim) if prenorm else None
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N is sequence length.
            attn_mask: Optional [N, N] boolean mask. True = can attend.
                When None, uses standard causal (lower-triangular) masking.
        Returns:
            [B, N, D]
        """
        drop = self.dropout if self.training else 0.0
        if self.norm is not None:
            x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv
        )  # each [B, heads, N, dim_head]
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, attn_mask=attn_mask
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=True
            )
        out = out.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], -1
        )  # [B, N, inner_dim]
        return self.to_out(out)


class _CTConditionalBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning for CausalTransformer.

    Args:
        dim: Hidden dimension.
        heads: Number of attention heads.
        dim_head: Dimension per attention head.
        mlp_ratio: Ratio of MLP hidden dim to model dim (mlp_hidden = round(mlp_ratio * dim)).
        dropout: Dropout rate.
        adaln_init_scale: Initialization scale for the AdaLN modulation layer.
            Default 0.0 gives standard AdaLN-Zero (all modulations start at zero).
            A small positive value (e.g. 0.02) breaks the zero-init symmetry and
            can help the predictor learn to use action conditioning earlier in
            training. Only used when action_cond_mode="adaln".
        action_cond_mode: Action conditioning mechanism. "adaln" (default) uses
            AdaLN-Zero modulation with 6 gating signals. "additive" directly adds
            a learned projection of the conditioning vector to the visual tokens
            before self-attention, making actions impossible to gate out.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        adaln_init_scale: float = 0.0,
        action_cond_mode: str = "adaln",
        double_norm: bool = False,
    ):
        super().__init__()
        self.action_cond_mode = action_cond_mode
        self.attn = _CTAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout, prenorm=double_norm
        )
        self.mlp = _CTFeedForward(
            dim, int(mlp_ratio * dim), dropout=dropout, prenorm=double_norm
        )

        if action_cond_mode == "adaln":
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
            )
            if adaln_init_scale == 0.0:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                nn.init.normal_(self.adaLN_modulation[-1].weight, std=adaln_init_scale)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        elif action_cond_mode == "additive":
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.cond_proj = nn.Linear(dim, dim)
        elif action_cond_mode == "none":
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        else:
            raise ValueError(f"Unknown action_cond_mode: {action_cond_mode}")

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_cond_mode == "adaln":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )
            x = x + gate_msa * self.attn(
                _ct_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask
            )
            x = x + gate_mlp * self.mlp(
                _ct_modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        elif self.action_cond_mode == "additive":
            x = x + self.cond_proj(c)
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
            x = x + self.mlp(self.norm2(x))
        else:  # none
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
            x = x + self.mlp(self.norm2(x))
        return x


class _CTTransformer(nn.Module):
    """Transformer with AdaLN-Zero conditional blocks for CausalTransformer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        adaln_init_scale: float = 0.0,
        action_cond_mode: str = "adaln",
        double_norm: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.action_cond_mode = action_cond_mode
        if action_cond_mode != "none":
            self.cond_proj = (
                nn.Linear(input_dim, hidden_dim)
                if input_dim != hidden_dim
                else nn.Identity()
            )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                _CTConditionalBlock(
                    hidden_dim,
                    heads,
                    dim_head,
                    mlp_ratio,
                    dropout,
                    adaln_init_scale=adaln_init_scale,
                    action_cond_mode=action_cond_mode,
                    double_norm=double_norm,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None and self.action_cond_mode != "none":
            c = self.cond_proj(c)
        else:
            c = None
        for block in self.layers:
            x = block(x, c, attn_mask=attn_mask)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


def build_frame_causal_mask(
    T: int, H: int, W: int, context_window: Optional[int] = None
) -> torch.Tensor:
    """Build a frame-causal (blockwise lower-triangular) attention mask.

    Each frame's patches can attend to all patches from the same frame and
    from previous frames, but not from future frames.

    Args:
        T: Number of temporal frames.
        H: Spatial grid height (number of patch rows).
        W: Spatial grid width (number of patch columns).
        context_window: If None, full causal (attend to all past frames).
            If int W>0, windowed causal: frame t attends to [max(0, t-W):t].

    Returns:
        Boolean tensor [T*H*W, T*H*W] where True = can attend.
    """
    HW = H * W
    N = T * HW
    mask = torch.zeros(N, N, dtype=torch.bool)
    block = torch.ones(HW, HW, dtype=torch.bool)
    for t1 in range(T):
        t_start = max(0, t1 - context_window) if context_window is not None else 0
        for t2 in range(t_start, t1 + 1):
            mask[t1 * HW : (t1 + 1) * HW, t2 * HW : (t2 + 1) * HW] = block
    return mask
