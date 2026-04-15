"""Hierarchical JEPA module for multi-scale world modeling and planning."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from eb_jepa.builders import (
    build_action_encoder,
    build_action_regularizer,
    build_cost_module,
    build_encoder,
    build_predcost,
    build_predictor,
    build_regularizer,
)
from eb_jepa.jepa import JEPA, JEPAWithCostModule
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


class HierarchicalJEPA(nn.Module):
    """Hierarchical JEPA with L encoder/predictor levels via composition.

    Holds a single ``nn.ModuleList`` of ``JEPA`` (or ``JEPAWithCostModule``)
    instances, one per hierarchy level. This composition design makes it easy
    to extend a trained L-level model to L+1 levels by appending a new JEPA.

    Architecture:
    - Level 1: Base encoder/predictor (finest, no temporal pooling)
    - Level l > 1: Encoder_l receives strided-subsampled states from level l-1
    - Each level has its own predictor operating at coarser time resolution

    Temporal Resolution:
    - States: Obtained via strided subsampling + neural encoding
    - Actions: Aggregated via learned action encoders over stride-sized windows

    The hierarchy enables:
    - Multi-scale representation learning
    - Efficient long-horizon planning via top-down refinement
    - Abstract reasoning at higher levels, fine-grained control at lower levels
    """

    def __init__(
        self,
        levels: nn.ModuleList,
        temporal_strides: List[int],
        level_weights: Optional[List[float]] = None,
        action_regularizers: Optional[nn.ModuleList] = None,
    ):
        """Initialize HierarchicalJEPA from a list of per-level JEPA instances.

        Args:
            levels: ModuleList of L ``JEPA`` or ``JEPAWithCostModule`` instances.
                Each level[i] has its own ``.encoder``, ``.action_encoder``,
                ``.predictor``, ``.regularizer``, ``.predcost``, and optionally
                ``.cost_module`` (for ``JEPAWithCostModule``).
            temporal_strides: List of L-1 integers, the temporal stride between
                consecutive levels. temporal_strides[0] is the stride from level 1 to level 2.
            level_weights: Optional weights for combining losses across levels.
                Defaults to uniform weighting.
            action_regularizers: Optional ModuleList of L regularizers for action encoder
                outputs. Entry is None for levels without action regularization (e.g. level 1).
        """
        super().__init__()
        self.levels = levels
        self.num_levels = len(levels)
        assert len(temporal_strides) == self.num_levels - 1

        self.temporal_strides = temporal_strides
        self.action_regularizers = action_regularizers

        if level_weights is None:
            level_weights = [1.0] * self.num_levels
        self.level_weights = level_weights

        self.single_unroll = getattr(self.levels[0].predictor, "is_rnn", False)

    # -- Backward-compatibility properties --

    @property
    def encoder(self) -> nn.Module:
        return self.levels[0].encoder

    @property
    def predictor(self) -> nn.Module:
        return self.levels[0].predictor

    @property
    def regularizer(self) -> nn.Module:
        return self.levels[0].regularizer

    @property
    def predcost(self) -> nn.Module:
        return self.levels[0].predcost

    @property
    def encoders(self) -> nn.ModuleList:
        return nn.ModuleList([l.encoder for l in self.levels])

    @property
    def predictors(self) -> nn.ModuleList:
        return nn.ModuleList([l.predictor for l in self.levels])

    @property
    def action_encoders(self) -> nn.ModuleList:
        return nn.ModuleList([l.action_encoder for l in self.levels])

    @property
    def regularizers(self) -> nn.ModuleList:
        return nn.ModuleList([l.regularizer for l in self.levels])

    @property
    def predcosts(self) -> nn.ModuleList:
        return nn.ModuleList([l.predcost for l in self.levels])

    @property
    def cost_modules(self) -> Dict[int, nn.Module]:
        result = {}
        for i, level in enumerate(self.levels):
            if isinstance(level, JEPAWithCostModule) and level.cost_module is not None:
                result[i + 1] = level.cost_module
        return result

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @num_levels.setter
    def num_levels(self, value: int):
        self._num_levels = value

    def get_temporal_scale(self, level: int) -> int:
        """Get the temporal downsampling factor at a given level.

        Args:
            level: Hierarchy level (1 = finest, L = coarsest).

        Returns:
            Total temporal downsampling factor relative to level 1.
        """
        scale = 1
        for l in range(1, level):
            scale *= self.temporal_strides[
                l - 1
            ]  # temporal_strides[0] is stride from level 1->2
        return scale

    @torch.no_grad()
    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations at level 1 (finest level).

        Args:
            observations: Input tensor [B, C, T, H, W].

        Returns:
            Encoded tensor at level 1 [B, D_1, T, H', W'].
        """
        return self.levels[0].encoder(observations)

    def encode_hierarchical(
        self, observations: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Encode observations through all hierarchy levels via strided subsampling.

        Args:
            observations: Input tensor [B, C, T, H, W].

        Returns:
            Dict mapping level (1..L) to encoded tensors [B, D_l, T_l, H_l, W_l].
        """
        z = self.levels[0].encoder(observations)
        encodings = {1: z}

        for level in range(2, self.num_levels + 1):
            stride = self.temporal_strides[level - 2]
            z_subsampled = z[:, :, ::stride, :, :]  # [B, D, T//stride, H, W]
            z = self.levels[level - 1].encoder(z_subsampled)
            encodings[level] = z

        return encodings

    def aggregate_actions(
        self, actions: torch.Tensor, level: int, return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
        """Temporally aggregate actions to match state transitions at a given level.

        Level 1 returns T-1 raw actions. Level ℓ > 1 uses learned action encoders
        to aggregate windows of actions into macro-actions.

        Args:
            actions: Actions tensor [B, A, T].
            level: Target hierarchy level (1 = finest, L = coarsest).
            return_intermediates: If True, return per-level encoded action tensors.

        Returns:
            If return_intermediates is False:
                Aggregated actions [B, A_enc, T_l - 1] where A_enc may differ from A.
            If return_intermediates is True:
                Tuple of (aggregated_actions, intermediates) where intermediates is a
                dict mapping level (2..target_level) to encoded action tensors.
        """

        a_down = actions  # [B, A, T-1]
        intermediates = {} if return_intermediates else None

        for l in range(1, level):
            B, A, T = a_down.shape
            stride = self.temporal_strides[l - 1]  # pools[0] is between level 1 and 2

            num_states = (T + stride) // stride
            num_transitions = num_states - 1

            action_encoder = self.levels[l].action_encoder  # levels[1] for level 2

            aggregated_actions = []
            for i in range(num_transitions):
                start_idx = i * stride
                end_idx = min((i + 1) * stride, T)
                window_actions = a_down[:, :, start_idx:end_idx]  # [B, A, window_size]

                window_size = window_actions.shape[2]
                window_flat = window_actions.reshape(B, -1)  # [B, A * window_size]

                if isinstance(action_encoder, nn.Identity):
                    aggregated = window_actions.mean(dim=2)  # [B, A]
                else:
                    aggregated = action_encoder(window_flat)  # [B, A_enc]

                aggregated_actions.append(aggregated.unsqueeze(2))  # [B, A_enc, 1]

            a_down = torch.cat(aggregated_actions, dim=2)  # [B, A_enc, num_transitions]

            if return_intermediates:
                intermediates[l + 1] = a_down

        if return_intermediates:
            return a_down, intermediates
        return a_down

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        nsteps: int = 1,
        unroll_mode: str = "parallel",
        ctxt_window_time: int = 1,
        compute_loss: bool = True,
        return_all_steps: bool = False,
        levels: Optional[List[int]] = None,
    ):
        """Full forward pass for DDP gradient synchronization.

        Encodes once, unrolls, and computes cost module losses, so all
        encoder gradients flow through DDP's allreduce.

        Returns:
            Tuple of (predicted_states_dict, encodings_dict, losses):
            - predicted_states_dict: Dict[int, Tensor] with predicted states per level.
            - encodings_dict: Dict[int, Tensor] with encoder outputs per level.
            - losses: Tuple (total_loss, rloss, rloss_unweight, rloss_dict, ploss) or None.
        """
        encodings = self.encode_hierarchical(observations)

        pred_states, _, losses = self.unroll(
            observations,
            actions,
            nsteps=nsteps,
            unroll_mode=unroll_mode,
            ctxt_window_time=ctxt_window_time,
            compute_loss=compute_loss,
            return_all_steps=return_all_steps,
            levels=levels,
            _encodings=encodings,
        )

        if compute_loss and losses is not None and self.cost_modules:
            cost_loss, cost_loss_dict = self.compute_cost_losses(encodings)
            total_loss, rloss, rloss_unweight, rloss_dict, ploss = losses
            rloss_dict.update(cost_loss_dict)
            losses = (total_loss + cost_loss, rloss, rloss_unweight, rloss_dict, ploss)

        if not isinstance(pred_states, dict):
            pred_states = {1: pred_states}
        return pred_states, encodings, losses

    def unroll(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        nsteps: int = 1,
        unroll_mode: str = "parallel",
        ctxt_window_time: int = 1,
        compute_loss: bool = True,
        return_all_steps: bool = False,
        levels: Optional[List[int]] = None,
        _encodings: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """Multi-step prediction across hierarchy levels.

        Args:
            observations: Input tensor [B, C, T, H, W].
            actions: Actions tensor [B, A, T_actions], or None.
            nsteps: Number of prediction steps at level 0.
            unroll_mode: "parallel" or "autoregressive".
            ctxt_window_time: Context window size for autoregressive mode.
            compute_loss: Whether to compute losses.
            return_all_steps: If True, return predictions at each step.
              levels: Optional list of levels to unroll. When None, unrolls all
                levels. When provided, only the specified levels are unrolled
                and the return value is always a dict keyed by level.
            _encodings: Pre-computed hierarchical encodings from
                ``encode_hierarchical()``. When provided, skips the internal
                encoding step. Used by ``forward()`` to encode once.

        Returns:
            When levels is None (default): Tuple of (predicted_states_level1,
                encoded_states_level1, losses) for backward compatibility.
            When levels is provided: Tuple of (Dict[int, predicted_states],
                Dict[int, encoded_states], losses).
        """
        max_scale = self.get_temporal_scale(self.num_levels)
        if nsteps % max_scale != 0:
            raise ValueError(
                f"nsteps ({nsteps}) must be divisible by the coarsest temporal "
                f"scale ({max_scale} = product of strides "
                f"{[self.temporal_strides[l] for l in range(self.num_levels - 1)]}). "
                f"Use a multiple of {max_scale}, e.g. {max_scale}, {2 * max_scale}, "
                f"{3 * max_scale}, ..."
            )

        encodings = (
            _encodings
            if _encodings is not None
            else self.encode_hierarchical(observations)
        )

        total_loss = torch.tensor(0.0, device=observations.device)
        total_rloss = torch.tensor(0.0, device=observations.device)
        total_rloss_unweight = torch.tensor(0.0, device=observations.device)
        total_ploss = torch.tensor(0.0, device=observations.device)
        combined_rloss_dict = {}

        all_predicted_states = {}
        all_steps_per_level = {} if return_all_steps else None

        levels_to_unroll = (
            levels if levels is not None else list(range(1, self.num_levels + 1))
        )

        # Pre-compute aggregated actions with intermediates for action regularization
        action_intermediates = None
        if (
            actions is not None
            and compute_loss
            and self.action_regularizers is not None
        ):
            _, action_intermediates = self.aggregate_actions(
                actions, self.num_levels, return_intermediates=True
            )

        for level in levels_to_unroll:
            state_l = encodings[level]
            scale = self.get_temporal_scale(level)
            nsteps_l = max(1, nsteps // scale)

            if actions is not None:
                if action_intermediates is not None and level in action_intermediates:
                    actions_l = action_intermediates[level]
                else:
                    actions_l = self.aggregate_actions(actions, level)
            else:
                actions_l = None

            pred_states_l, _, losses_l = self._unroll_at_level(
                level,
                state_l,
                actions_l,
                nsteps_l,
                unroll_mode,
                max(1, ctxt_window_time // scale),
                compute_loss,
                return_all_steps,
            )

            all_predicted_states[level] = pred_states_l
            if return_all_steps:
                all_steps_per_level[level] = pred_states_l

            if compute_loss and losses_l is not None:
                loss_l, rloss_l, rloss_unweight_l, rloss_dict_l, ploss_l = losses_l
                weight = self.level_weights[level - 1]

                # Action regularization for levels >= 2
                if (
                    actions is not None
                    and self.action_regularizers is not None
                    and level >= 2
                    and self.action_regularizers[level - 1] is not None
                ):
                    actions_encoded = (
                        action_intermediates[level]
                        if action_intermediates is not None
                        and level in action_intermediates
                        else actions_l
                    )
                    action_rloss, action_rloss_unweight, action_rloss_dict = (
                        self.action_regularizers[level - 1](actions_encoded)
                    )
                    loss_l = loss_l + action_rloss
                    rloss_l = rloss_l + action_rloss
                    rloss_unweight_l = rloss_unweight_l + action_rloss_unweight
                    rloss_dict_l.update(action_rloss_dict)

                total_loss += weight * loss_l
                total_rloss += weight * rloss_l
                total_rloss_unweight += weight * rloss_unweight_l
                total_ploss += weight * ploss_l
                for k, v in rloss_dict_l.items():
                    combined_rloss_dict[f"level{level}/{k}"] = v

        if compute_loss:
            losses = (
                total_loss,
                total_rloss,
                total_rloss_unweight,
                combined_rloss_dict,
                total_ploss,
            )
        else:
            losses = None

        # When levels is explicitly provided, return dicts keyed by level
        if levels is not None:
            if return_all_steps:
                return all_steps_per_level, encodings, losses
            else:
                return all_predicted_states, encodings, losses

        # Default backward-compatible behavior: return level 1 only
        if return_all_steps:
            return all_steps_per_level, encodings, losses
        else:
            return all_predicted_states[1], encodings[1], losses

    def unroll_at_level(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        level: int,
        nsteps: Optional[int] = None,
        unroll_mode: str = "autoregressive",
        ctxt_window_time: int = 1,
    ):
        """Unroll at a specific hierarchy level for planning.

        This encodes hierarchically up to the target level and unrolls only at
        that level. Actions are passed directly to the predictor (no aggregation).
        At level 1, the planner provides raw actions. At level > 1, the planner
        samples directly in the encoded action space matching the predictor's input.

        Args:
            observations: Input observations [B, C, T, H, W].
            actions: Actions [B_action, A_l, T_plan] already at the target level's
                action space (raw for level 1, encoded for level > 1).
            level: Hierarchy level (1 = finest, L = coarsest).
            nsteps: Number of steps to predict (inferred from actions if None).
            unroll_mode: "parallel" or "autoregressive".
            ctxt_window_time: Context window size.

        Returns:
            Predicted encodings [B_action, D_l, T_l, H_l, W_l].
        """
        batch_size_obs = observations.shape[0]
        batch_size_action = actions.shape[0]
        if batch_size_obs == 1 and batch_size_action > 1:
            observations = observations.repeat(batch_size_action, 1, 1, 1, 1)

        encodings = self.encode_hierarchical(observations)
        state_l = encodings[level]

        if nsteps is None:
            nsteps_l = actions.shape[2] if actions is not None else 1
        else:
            scale = self.get_temporal_scale(level)
            nsteps_l = max(1, nsteps // scale)

        predicted_states, _, _ = self._unroll_at_level(
            level,
            state_l,
            actions,
            nsteps_l,
            unroll_mode,
            ctxt_window_time,
            compute_loss=False,
            return_all_steps=False,
        )

        return predicted_states

    def _unroll_at_level(
        self,
        level: int,
        state: torch.Tensor,
        actions: Optional[torch.Tensor],
        nsteps: int,
        unroll_mode: str,
        ctxt_window_time: int,
        compute_loss: bool,
        return_all_steps: bool,
    ):
        """Internal unroll for a specific level, delegating to that level's JEPA.

        Uses ``_precomputed_state`` and ``_precomputed_actions`` on the per-level
        JEPA so its ``unroll()`` logic is reused without duplication.
        """
        jepa_level = self.levels[level - 1]
        return jepa_level.unroll(
            observations=None,
            actions=actions,
            nsteps=nsteps,
            unroll_mode=unroll_mode,
            ctxt_window_time=ctxt_window_time,
            compute_loss=compute_loss,
            return_all_steps=return_all_steps,
            _precomputed_state=state,
            _precomputed_actions=actions,
        )

    def compute_cost_losses(self, enc_states_levels: Dict[int, torch.Tensor]):
        """Compute straightening losses for all levels with cost modules.

        Args:
            enc_states_levels: Dict mapping level -> encoded states [B, D_l, T, H_l, W_l].

        Returns:
            Tuple of (total_cost_loss, cost_losses_dict) where:
            - total_cost_loss: Sum of cost losses across all levels
            - cost_losses_dict: Dict with per-level cost loss components
        """
        if not self.cost_modules:
            return torch.tensor(0.0), {}

        total_cost_loss = 0.0
        cost_losses_dict = {}

        for level, cost_module in self.cost_modules.items():
            if level in enc_states_levels:
                cost_loss, cost_loss_dict = cost_module(enc_states_levels[level])
                total_cost_loss += cost_loss
                for k, v in cost_loss_dict.items():
                    cost_losses_dict[f"cost_l{level}_{k}"] = v

        return total_cost_loss, cost_losses_dict


def build_hierarchical_jepa(
    cfg,
    data_config: dict,
    device: torch.device,
) -> "HierarchicalJEPA":
    """Build a HierarchicalJEPA from configuration using shared builders.

    Constructs one ``JEPA`` (or ``JEPAWithCostModule``) per level and wraps
    them in a ``HierarchicalJEPA``.

    Args:
        cfg: Configuration object with model parameters.
        data_config: Plain dict with dataset properties (img_size, action_dim, etc.).
        device: Device to place the model on.

    Returns:
        HierarchicalJEPA instance.
    """
    num_levels = cfg.model.num_levels
    data_action_dim = data_config.get("action_dim", 2)
    prev_action_dim = data_action_dim

    temporal_strides: List[int] = []
    jepa_levels = []
    action_reg_list = []

    prev_output_dim = None
    prev_spatial_size = 1

    for level in range(1, num_levels + 1):
        level_cfg = getattr(cfg.model, f"level_{level}")
        enc_cfg = level_cfg.encoder
        pred_cfg = level_cfg.predictor

        # -- Encoder --
        if level == 1:
            encoder, output_dim, spatial_size = build_encoder(
                enc_cfg,
                input_channels=cfg.model.dobs,
                img_size=data_config["img_size"],
                device=device,
            )
            action_encoder = nn.Identity()
            current_action_dim = data_action_dim
        else:
            stride = level_cfg.temporal_stride
            temporal_strides.append(stride)

            encoder, output_dim, spatial_size = build_encoder(
                enc_cfg,
                input_channels=prev_output_dim,
                img_size=prev_spatial_size,
                device=device,
                input_dim=prev_output_dim,
                prev_spatial_size=prev_spatial_size,
            )

            aenc_cfg = (
                level_cfg.action_encoder
                if hasattr(level_cfg, "action_encoder")
                and level_cfg.action_encoder is not None
                else None
            )
            action_encoder, current_action_dim = build_action_encoder(
                aenc_cfg,
                input_action_dim=prev_action_dim,
                temporal_stride=level_cfg.temporal_stride,
                device=device,
            )

        # -- Predictor --
        if level == 1:
            action_dim_for_predictor = data_action_dim
        else:
            has_action_encoder = (
                hasattr(level_cfg, "action_encoder")
                and level_cfg.action_encoder is not None
            )
            action_dim_for_predictor = (
                level_cfg.action_encoder.get("output_dim", data_action_dim)
                if has_action_encoder
                else prev_action_dim
            )

        predictor = build_predictor(
            pred_cfg,
            input_dim=output_dim,
            action_dim=action_dim_for_predictor,
            spatial_size=spatial_size,
            device=device,
        )

        # -- Regularizer --
        use_proj = cfg.model.get("use_proj", False)
        reg = build_regularizer(
            level_cfg.regularizer,
            encoder_output_dim=output_dim,
            spatial_size=spatial_size,
            action_dim=(data_action_dim if level == 1 else current_action_dim),
            device=device,
            use_proj=use_proj,
        )

        # -- Predcost, cost module --
        predcost = build_predcost()
        cost_module = build_cost_module(
            level_cfg.get("cost"),
            encoder_output_dim=output_dim,
            device=device,
        )

        # -- Assemble per-level JEPA --
        if cost_module is not None:
            jepa_level = JEPAWithCostModule(
                encoder, action_encoder, predictor, reg, predcost, cost_module
            )
        else:
            jepa_level = JEPA(encoder, action_encoder, predictor, reg, predcost)
        jepa_levels.append(jepa_level)

        # -- Action regularizer --
        if level == 1:
            action_reg_list.append(None)
        else:
            action_reg = build_action_regularizer(
                level_cfg.regularizer,
                action_dim=current_action_dim,
                device=device,
            )
            action_reg_list.append(action_reg)

        prev_output_dim = output_dim
        prev_spatial_size = spatial_size[0]
        prev_action_dim = current_action_dim

    level_weights = cfg.model.get("level_weights", [1.0] * num_levels)

    has_action_regs = any(r is not None for r in action_reg_list)
    if has_action_regs:
        action_regularizers = nn.ModuleList(
            [r if r is not None else None for r in action_reg_list]
        )
    else:
        action_regularizers = None

    model = HierarchicalJEPA(
        levels=nn.ModuleList(jepa_levels),
        temporal_strides=temporal_strides,
        level_weights=level_weights,
        action_regularizers=action_regularizers,
    ).to(device)

    return model
