from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from eb_jepa.models.components import (
    ConvGRUCell,
    ConvNeXtBlock,
    MiniUNet,
    Projector,
    _CTTransformer,
    build_frame_causal_mask,
)
from eb_jepa.models.nn import spatial_layer_norm


def _get_pos_embedding(pe: torch.Tensor, T: int) -> torch.Tensor:
    """Slice or interpolate positional embeddings to match sequence length T.

    Args:
        pe: Positional embedding parameter of shape [1, max_seq_len, D].
        T: Target sequence length.

    Returns:
        [1, T, D] positional embeddings.
    """
    if T <= pe.size(1):
        return pe[:, :T]
    return torch.nn.functional.interpolate(
        pe.permute(0, 2, 1), size=T, mode="linear", align_corners=False
    ).permute(0, 2, 1)


class RNNPredictor(nn.Module):
    """GRU-based predictor for single-step state propagation."""

    def __init__(
        self,
        predictor_dim: int = 512,
        action_dim: int = 2,  # Default for backward compatibility (two_rooms)
        num_layers: int = 1,
        final_ln: bool = True,
        input_dim: int | None = None,
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers
        self.is_rnn = True

        use_proj = input_dim is not None and input_dim != predictor_dim
        self.proj_in = nn.Linear(input_dim, predictor_dim) if use_proj else None
        self.proj_out = nn.Linear(predictor_dim, input_dim) if use_proj else None

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=predictor_dim,
            num_layers=num_layers,
        )

        ln_dim = input_dim if use_proj else predictor_dim
        self.final_ln = nn.LayerNorm(ln_dim) if final_ln else None
        self.context_length = 1

    def forward(self, state, action):
        """
        Propagate one step forward.

        Args:
            state: [B, D, 1, 1, 1]
            action: [B, A, 1]
        Returns:
            next_state: [B, D, 1, 1, 1]

        Note:
            When in eval mode, cuDNN is disabled for the GRU call so that
            backward passes still work. cuDNN's inference path does not
            allocate the reserve space needed for gradient computation.
            With dropout=0 the outputs are identical.
        """
        rnn_state = state.flatten(1, 4)  # [B, D]
        if self.proj_in is not None:
            rnn_state = self.proj_in(rnn_state)  # [B, predictor_dim]
        rnn_state = rnn_state.unsqueeze(0).contiguous()  # [1, B, predictor_dim]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # [1, B, A]

        if not self.training:
            with torch.backends.cudnn.flags(enabled=False):
                next_state, _ = self.rnn(rnn_input, rnn_state)
        else:
            next_state, _ = self.rnn(rnn_input, rnn_state)

        next_state = next_state[0]  # [B, predictor_dim]
        if self.proj_out is not None:
            next_state = self.proj_out(next_state)  # [B, input_dim]
        if self.final_ln is not None:
            next_state = self.final_ln(next_state)

        return next_state.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class CausalTransformerPredictor(nn.Module):
    """Causal transformer predictor with AdaLN-Zero action conditioning.

    Ported from LEWM's ARPredictor. Processes sequences of states with causal
    masking, conditioned on action embeddings via AdaLN-Zero modulation.

    Compatible with jepa.py unroll():
    - is_rnn = False: uses autoregressive mode with context_length history
    - forward() takes [B, D, T, 1, 1] states and [B, A, T] actions

    Args:
        input_dim: Encoder output dimension. The predictor's input/output
            interface matches this dimension. Automatically set to
            encoder_output_dim by the model builder.
        action_dim: Raw action vector dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to predictor dim (default 4).
        dim_head: Dimension per attention head.
        max_seq_len: Maximum sequence length for positional embeddings.
        dropout: Dropout rate in attention and feed-forward.
        emb_dropout: Dropout rate after positional embedding.
        predictor_dim: Internal transformer processing dimension. When set,
            the transformer up-projects from input_dim to predictor_dim
            via input_proj/cond_proj and back via output_proj. Defaults to
            input_dim (no projection) for backward compatibility.
        adaln_init_scale: Initialization scale for the AdaLN modulation layer.
            Default 0.0 gives standard AdaLN-Zero (all modulations start at zero).
            A small positive value (e.g. 0.02) breaks the zero-init symmetry and
            can help the predictor learn to use action conditioning earlier.
        action_cond_mode: Action conditioning mechanism. "adaln" (default) uses
            AdaLN-Zero modulation. "additive" directly adds a learned projection
            of the action embedding to the visual tokens at each block.
        action_mlp_ratio: Ratio of action embedder hidden dim to input_dim.
            Default 1.0 (hidden = input_dim). Set to 4.0 to match le-wm.
        pos_embed_init_scale: Initialization scale for positional embeddings.
            Default 0.02. Set to 1.0 to match le-wm.
        projector_spec: Optional MLP spec string (e.g. ``"192-2048-192"``).
            When set, applies a ``Projector`` as the final output stage.
        projector_activation: Activation for projector hidden layers (default
            ``"gelu"``).
        projector_final_bias: Whether the projector final linear has bias
            (default True).
    """

    def __init__(
        self,
        input_dim: int = 192,
        action_dim: int = 7,
        depth: int = 6,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dim_head: int = 64,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        use_residual: bool = True,
        final_ln: bool = False,
        predictor_dim: Optional[int] = None,
        adaln_init_scale: float = 0.0,
        action_cond_mode: str = "adaln",
        action_mlp_ratio: float = 1.0,
        pos_embed_init_scale: float = 0.02,
        projector_spec: Optional[str] = None,
        projector_activation: str = "gelu",
        projector_final_bias: bool = True,
        double_norm: bool = False,
    ):
        super().__init__()
        self.is_rnn = False
        self.context_length = 1
        pdim = predictor_dim or input_dim
        self.final_ln = nn.LayerNorm(input_dim) if final_ln else None
        self.use_residual = use_residual

        action_hidden = int(action_mlp_ratio * input_dim)
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, input_dim),
        )
        self.pos_embedding = nn.Parameter(
            pos_embed_init_scale * torch.randn(1, max_seq_len, input_dim)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = _CTTransformer(
            input_dim=input_dim,
            hidden_dim=pdim,
            output_dim=input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            adaln_init_scale=adaln_init_scale,
            action_cond_mode=action_cond_mode,
            double_norm=double_norm,
        )

        if projector_spec is not None:
            self.projector = Projector(
                projector_spec,
                activation=projector_activation,
                final_bias=projector_final_bias,
            )
        else:
            self.projector = None

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [B, D, T, 1, 1]
            actions: [B, A, T]
        Returns:
            [B, D, T, 1, 1]
        """
        x = states.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, T, D]
        identity = x  # save for residual
        a = actions.permute(0, 2, 1)  # [B, T, A]
        act_emb = self.action_embedder(a)  # [B, T, D]

        T = x.size(1)
        x = x + _get_pos_embedding(self.pos_embedding, T)
        x = self.emb_dropout(x)
        x = self.transformer(x, act_emb)  # [B, T, D]
        if self.use_residual:
            x = x + identity
        if self.final_ln is not None:
            x = self.final_ln(x)
        if self.projector is not None:
            B, T, D = x.shape
            x = self.projector(x.reshape(B * T, D)).reshape(B, T, D)  # [B, T, D]
        return x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D, T, 1, 1]


class SpatialCausalTransformerPredictor(nn.Module):
    """Causal transformer predictor with frame-causal masking for spatial patch tokens.

    Flattens [B, D, T, H, W] to [B, T*H*W, D] and uses a blockwise
    lower-triangular attention mask: all patches within a frame attend to
    each other and to all patches from previous frames, but not future frames.
    Action conditioning via AdaLN-Zero (same as CausalTransformerPredictor).

    Args:
        input_dim: Encoder output dimension. The predictor's input/output
            interface matches this dimension. Automatically set to
            encoder_output_dim by the model builder.
        action_dim: Raw action vector dimension.
        spatial_size: Spatial grid size (H=W).
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to predictor dim (default 4).
        dim_head: Dimension per attention head.
        max_seq_len: Maximum number of frames for temporal positional embeddings.
        dropout: Dropout rate in attention and feed-forward.
        emb_dropout: Dropout rate after positional embedding.
        predictor_dim: Internal processing dimension for the entire predictor.
            When different from input_dim, linear projections are added at
            entry and exit to map between the encoder's representation space
            and the predictor's operating space. Defaults to input_dim
            (no projection) for backward compatibility.
        adaln_init_scale: Initialization scale for the AdaLN modulation layer.
            Default 0.0 gives standard AdaLN-Zero (all modulations start at zero).
            A small positive value (e.g. 0.02) breaks the zero-init symmetry and
            can help the predictor learn to use action conditioning earlier.
        action_cond_mode: Action conditioning mechanism. "adaln" (default) uses
            AdaLN-Zero modulation. "additive" directly adds a learned projection
            of the action embedding to the visual tokens at each block.
        action_mlp_ratio: Ratio of action embedder hidden dim to pdim.
            Default 1.0 (hidden = pdim). Set to 4.0 to match le-wm.
        pos_embed_init_scale: Initialization scale for positional embeddings.
            Default 0.02. Set to 1.0 to match le-wm.
        projector_spec: Optional MLP spec string (e.g. ``"192-2048-192"``).
            When set, applies a ``Projector`` as the final output stage.
        projector_activation: Activation for projector hidden layers (default
            ``"gelu"``).
        projector_final_bias: Whether the projector final linear has bias
            (default True).
    """

    def __init__(
        self,
        input_dim: int = 32,
        action_dim: int = 7,
        spatial_size: int = 14,
        depth: int = 6,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dim_head: int = 32,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        use_residual: bool = True,
        final_ln: bool = False,
        predictor_dim: Optional[int] = None,
        adaln_init_scale: float = 0.0,
        action_cond_mode: str = "adaln",
        action_mlp_ratio: float = 1.0,
        pos_embed_init_scale: float = 0.02,
        projector_spec: Optional[str] = None,
        projector_activation: str = "gelu",
        projector_final_bias: bool = True,
        double_norm: bool = False,
    ):
        super().__init__()
        self.is_rnn = False
        self.context_length = 1
        self.spatial_size = spatial_size
        self.input_dim = input_dim
        pdim = predictor_dim or input_dim
        self.use_residual = use_residual

        # Entry/exit projections: map between encoder space (input_dim)
        # and predictor operating space (pdim)
        self.repr_proj_in = (
            nn.Linear(input_dim, pdim) if pdim != input_dim else nn.Identity()
        )
        self.repr_proj_out = (
            nn.Linear(pdim, input_dim) if pdim != input_dim else nn.Identity()
        )

        # All internal components operate at pdim
        action_hidden = int(action_mlp_ratio * pdim)
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, pdim),
        )
        self.pos_embedding = nn.Parameter(
            pos_embed_init_scale * torch.randn(1, max_seq_len, pdim)
        )
        self.spatial_pos_embedding = nn.Parameter(
            0.02 * torch.randn(1, spatial_size * spatial_size, pdim)
        )
        self.final_ln = nn.LayerNorm(pdim) if final_ln else None
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = _CTTransformer(
            input_dim=pdim,
            hidden_dim=pdim,
            output_dim=pdim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            adaln_init_scale=adaln_init_scale,
            action_cond_mode=action_cond_mode,
            double_norm=double_norm,
        )

        if projector_spec is not None:
            self.projector = Projector(
                projector_spec,
                activation=projector_activation,
                final_bias=projector_final_bias,
            )
        else:
            self.projector = None

        self._cached_mask = None
        self._cached_T = None

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Get or build the frame-causal attention mask."""
        if self._cached_T != T or self._cached_mask is None:
            self._cached_mask = build_frame_causal_mask(
                T, self.spatial_size, self.spatial_size
            )
            self._cached_T = T
        return self._cached_mask.to(device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass with frame-causal attention over spatial patch tokens.

        Args:
            states: [B, D, T, H, W] spatially-structured encoder states.
            actions: [B, A, T] action sequence.

        Returns:
            [B, D, T, H, W] predicted states.
        """
        B, D, T, H, W = states.shape
        HW = H * W

        # Reshape: [B, D, T, H, W] -> [B, T, H*W, D]
        x = states.permute(0, 2, 3, 4, 1).reshape(B, T, HW, D)  # [B, T, HW, D]

        # Save for outer residual in input_dim space (before projection)
        identity = x.reshape(B, T * HW, D)  # [B, T*HW, D]

        # Project to predictor dimension: D -> pdim
        x = self.repr_proj_in(x)  # [B, T, HW, pdim]

        # Add temporal + spatial positional embeddings (pdim)
        x = (
            x + _get_pos_embedding(self.pos_embedding, T)[:, :, None, :]
        )  # [1, T, 1, pdim]
        x = x + self.spatial_pos_embedding[:, None, :HW, :]  # [1, 1, HW, pdim]

        # Flatten to sequence: [B, T*HW, pdim]
        P = x.shape[-1]
        x = x.reshape(B, T * HW, P)  # [B, T*HW, pdim]
        x = self.emb_dropout(x)

        # Action embeddings: [B, T, A] -> [B, T, pdim] -> expand to [B, T*HW, pdim]
        a = actions.permute(0, 2, 1)  # [B, T, A]
        act_emb = self.action_embedder(a)  # [B, T, pdim]
        act_emb = (
            act_emb.unsqueeze(2).expand(-1, -1, HW, -1).reshape(B, T * HW, P)
        )  # [B, T*HW, pdim]

        # Frame-causal attention mask
        mask = self._get_mask(T, x.device)

        # Run transformer with frame-causal mask
        x = self.transformer(x, act_emb, attn_mask=mask)  # [B, T*HW, pdim]
        if self.final_ln is not None:
            x = self.final_ln(x)  # [B, T*HW, pdim]

        # Project back to input_dim: pdim -> D
        x = self.repr_proj_out(x)  # [B, T*HW, D]

        # Residual in input_dim space
        if self.use_residual:
            x = x + identity

        # Apply projector if configured
        if self.projector is not None:
            N = x.shape[0] * x.shape[1]
            D_proj = x.shape[-1]
            x = self.projector(x.reshape(N, D_proj)).reshape(
                B, T * HW, D_proj
            )  # [B, T*HW, D]

        # Reshape back: [B, T*HW, D] -> [B, D, T, H, W]
        x = x.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # [B, D, T, H, W]
        return x


class ConvGRUPredictor(nn.Module):
    """ConvGRU predictor with the same interface as RNNPredictor.

    Actions are projected via Linear, then broadcast spatially to match
    the hidden state's spatial dims before being fed to the ConvGRU cell.

    Args:
        predictor_dim: Number of channels in the hidden state (encoder output dim).
        spatial_size: Spatial dimension of the hidden state (H=W).
        action_dim: Action vector dimension.
        kernel_size: ConvGRU gate kernel size.
    """

    def __init__(
        self,
        predictor_dim: int = 32,
        spatial_size: int = 16,
        action_dim: int = 2,
        kernel_size: int = 3,
        final_ln: bool = True,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.spatial_size = spatial_size
        self.is_rnn = True
        self.context_length = 1

        use_proj = input_dim is not None and input_dim != predictor_dim
        self.proj_in = nn.Conv2d(input_dim, predictor_dim, 1) if use_proj else None
        self.proj_out = nn.Conv2d(predictor_dim, input_dim, 1) if use_proj else None

        self.action_proj = nn.Linear(action_dim, predictor_dim)
        self.cell = ConvGRUCell(predictor_dim, predictor_dim, kernel_size)
        ln_dim = input_dim if use_proj else predictor_dim
        self.final_ln = nn.LayerNorm(ln_dim) if final_ln else None

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Propagate one step forward.

        Args:
            state: [B, D, 1, H, W] (single timestep with spatial dims).
            action: [B, A, 1].
        Returns:
            next_state: [B, D, 1, H, W].
        """
        B, D, _, H, W = state.shape
        h = state[:, :, 0]  # [B, D, H, W]

        if self.proj_in is not None:
            h = self.proj_in(h)  # [B, predictor_dim, H, W]

        a = action[:, :, 0]  # [B, A]
        a_proj = self.action_proj(a)  # [B, predictor_dim]
        a_spatial = (
            a_proj.unsqueeze(-1).unsqueeze(-1).expand(B, self.predictor_dim, H, W)
        )  # [B, predictor_dim, H, W]

        h_new = self.cell(a_spatial, h)  # [B, predictor_dim, H, W]

        if self.proj_out is not None:
            h_new = self.proj_out(h_new)  # [B, D, H, W]

        D_out = h_new.shape[1]
        if self.final_ln is not None:
            h_new = spatial_layer_norm(h_new, self.final_ln)

        return h_new.unsqueeze(2)  # [B, D_out, 1, H, W]


class _SpatialGRUPredictorBase(nn.Module):
    """Base class for spatial GRU predictors (ConvNeXt / UNet variants).

    Shared logic: proj_in, action_proj, ConvGRU temporal cell, proj_out,
    optional final LayerNorm, and the forward loop.  Subclasses set
    ``self.mixer`` to their specific spatial-mixing module (e.g.
    ``nn.Sequential(ConvNeXtBlocks...)`` or ``MiniUNet``).

    Args:
        input_dim: Per-patch hidden dimension D from the encoder.
        spatial_size: Spatial grid size (H=W).
        action_dim: Action vector dimension.
        predictor_dim: Internal wider dimension (default 64).
        final_ln: Whether to apply LayerNorm per-patch on output.
    """

    def __init__(
        self,
        input_dim: int = 32,
        spatial_size: int = 14,
        action_dim: int = 2,
        predictor_dim: int = 64,
        final_ln: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.spatial_size = spatial_size
        self.is_rnn = False
        self.context_length = 1

        pd = predictor_dim
        self.proj_in = nn.Conv2d(input_dim, pd, 1)
        self.action_proj = nn.Linear(action_dim, pd)

        combined_dim = 2 * pd
        self.temporal_cell = ConvGRUCell(
            hidden_dim=pd, input_dim=combined_dim, kernel_size=1
        )
        self.proj_out = nn.Conv2d(pd, input_dim, 1)
        self.final_ln = nn.LayerNorm(input_dim) if final_ln else None

        # Subclasses must set self.mixer: nn.Module mapping
        # [B, 2*pd, H, W] -> [B, 2*pd, H, W]
        self.mixer: nn.Module

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Unroll over T context frames with spatial mixing + GRU gating.

        Args:
            states: [B, D, T, H, W] context state embeddings.
            actions: [B, A, T] corresponding actions.
        Returns:
            outputs: [B, D, T, H, W] (unroll() uses only [:, :, -1:]).
        """
        B, D, T, H, W = states.shape
        pd = self.proj_in.out_channels

        h = self.proj_in(states[:, :, 0])  # [B, pd, H, W]

        outputs = []
        for t in range(T):
            s_t = self.proj_in(states[:, :, t])  # [B, pd, H, W]
            a_t = self.action_proj(actions[:, :, t])  # [B, pd]
            a_spatial = (
                a_t.unsqueeze(-1).unsqueeze(-1).expand(B, pd, H, W)
            )  # [B, pd, H, W]

            combined = torch.cat([s_t, a_spatial], dim=1)  # [B, 2*pd, H, W]
            mixed = self.mixer(combined)  # [B, 2*pd, H, W]

            h = self.temporal_cell(mixed, h)  # [B, pd, H, W]

            out_t = self.proj_out(h)  # [B, D, H, W]
            if self.final_ln is not None:
                out_t = spatial_layer_norm(out_t, self.final_ln)
            outputs.append(out_t)

        return torch.stack(outputs, dim=2)  # [B, D, T, H, W]


class ConvNeXtGRUPredictor(_SpatialGRUPredictorBase):
    """GRU predictor using stacked ConvNeXt blocks for global spatial mixing.

    Args:
        input_dim: Per-patch hidden dimension D from the encoder.
        spatial_size: Spatial grid size (H=W), e.g. 14 for ViT patch-14.
        action_dim: Action vector dimension.
        predictor_dim: Internal wider dimension (default 64).
        num_blocks: Number of ConvNeXt blocks (default 4).
        expansion_ratio: Inverted bottleneck expansion (default 4).
        final_ln: Whether to apply LayerNorm per-patch on output.
    """

    def __init__(
        self,
        input_dim: int = 32,
        spatial_size: int = 14,
        action_dim: int = 2,
        predictor_dim: int = 64,
        num_blocks: int = 4,
        expansion_ratio: int = 4,
        final_ln: bool = True,
    ):
        super().__init__(input_dim, spatial_size, action_dim, predictor_dim, final_ln)
        combined_dim = 2 * predictor_dim
        self.mixer = nn.Sequential(
            *[
                ConvNeXtBlock(combined_dim, spatial_size, expansion_ratio)
                for _ in range(num_blocks)
            ]
        )


class UNetGRUPredictor(_SpatialGRUPredictorBase):
    """GRU predictor using a mini U-Net for hierarchical spatial mixing.

    Args:
        input_dim: Per-patch hidden dimension D from the encoder.
        spatial_size: Spatial grid size (H=W).
        action_dim: Action vector dimension.
        predictor_dim: Internal wider dimension (default 64).
        base_channels: U-Net base channel width (default 64).
        final_ln: Whether to apply LayerNorm per-patch on output.
    """

    def __init__(
        self,
        input_dim: int = 32,
        spatial_size: int = 14,
        action_dim: int = 2,
        predictor_dim: int = 64,
        base_channels: int = 64,
        final_ln: bool = True,
    ):
        super().__init__(input_dim, spatial_size, action_dim, predictor_dim, final_ln)
        combined_dim = 2 * predictor_dim
        self.mixer = MiniUNet(combined_dim, combined_dim, base_channels)
