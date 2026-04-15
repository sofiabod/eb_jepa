from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from eb_jepa.planning.objectives import (
    HierarchicalObjective,
    ProjectedDistObjective,
    ReprDistObjective,
)
from eb_jepa.planning.optimizers import (
    AdamPlanner,
    CEMPlanner,
    GradientDescentPlanner,
    H_PlanningResult,
    HierarchicalPlanner,
    MPPIPlanner,
    Planner,
    PlanningResult,
)
from eb_jepa.utils.distributed import unwrap_model
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)

planner_name_map: Dict[str, type] = {
    "cem": CEMPlanner,
    "mppi": MPPIPlanner,
    "gd": GradientDescentPlanner,
    "adam": AdamPlanner,
}
objective_name_map: Dict[str, type] = {
    "repr_dist": ReprDistObjective,
    "projected_dist": ProjectedDistObjective,
    "hierarchical_repr_dist": HierarchicalObjective,
    "hierarchical_projected_dist": HierarchicalObjective,
}


def decode_with_visual_decoder(vd, states):
    """Decode encoder states to uint8 numpy images.

    Uses the visual decoder's ``decode_to_uint8`` method which handles
    inverse channel normalization automatically.

    Args:
        vd: VisualDecoder module (may be DDP-wrapped).
        states: Encoded states [B, D, T, H, W].

    Returns:
        np.ndarray of shape [B, T, H, W, C] uint8.
    """
    vd = unwrap_model(vd)
    return vd.decode_to_uint8(states)


def _infer_model_action_dim(model) -> Optional[int]:
    """Infer the model's expected action_dim from its predictor."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return None
    if hasattr(predictor, "action_embedder"):
        return predictor.action_embedder[0].in_features
    if hasattr(predictor, "rnn"):
        return predictor.rnn.input_size
    if hasattr(predictor, "action_proj"):
        return predictor.action_proj.in_features
    return None


class GCAgent:
    def __init__(
        self,
        model,
        action_dim=2,
        plan_cfg=None,
        preprocessor=None,
        loc_prober: Optional[Callable] = None,
        img_prober: Optional[Callable] = None,
        env: Optional[Callable] = None,
        visual_decoder=None,
        visual_decoders: Optional[Dict[int, Any]] = None,
    ):
        self.plan_cfg = plan_cfg
        self.env = env
        self.model = model
        self.device = next(model.parameters()).device
        self.loc_prober = loc_prober
        self.img_prober = img_prober
        self.preprocessor = preprocessor

        # Detect frameskip-concat: model may expect larger action_dim
        self._env_action_dim = action_dim
        model_action_dim = _infer_model_action_dim(model)
        if model_action_dim is not None and model_action_dim > action_dim:
            self._action_frameskip = model_action_dim // action_dim
            action_dim = model_action_dim
            logger.info(
                f"Detected frameskip-concat: env_action_dim={self._env_action_dim}, "
                f"model_action_dim={action_dim}, frameskip={self._action_frameskip}"
            )
        else:
            self._action_frameskip = 1
        # Support both single visual_decoder (flat) and per-level dict (hierarchical)
        if visual_decoders is not None:
            self.visual_decoder = visual_decoders.get(1)
            self._visual_decoders = visual_decoders
        else:
            self.visual_decoder = visual_decoder
            self._visual_decoders = (
                {1: visual_decoder} if visual_decoder is not None else {}
            )
        self._decode_fn = self._build_decode_fn()

        # Set default values if plan_cfg is None
        if plan_cfg is None:
            self.decode_each_iteration = False
            self.num_act_stepped = 1
            self.planner = None
            self._is_hierarchical = False
            logger.info("No plan_cfg provided in GCAgent, planner not initialized.")
        else:
            self.decode_each_iteration = plan_cfg.planner.get(
                "decode_each_iteration", False
            )
            self.num_act_stepped = plan_cfg.planner.get("num_act_stepped", 1)
            planner_type = plan_cfg.planner.get("type", "flat")

            if planner_type == "hierarchical":
                self._is_hierarchical = True
                self.planner = self._create_hierarchical_planner(plan_cfg, action_dim)
            else:
                self._is_hierarchical = False
                planner_name = plan_cfg.planner.get("planner_name", "cem")
                planner_class = planner_name_map[planner_name]

                action_stats_kwargs = {}
                pp_stats = self._get_preprocessor_action_stats()
                if pp_stats is not None:
                    mean, std = pp_stats
                    if self._action_frameskip > 1:
                        mean = mean.repeat(self._action_frameskip)
                        std = std.repeat(self._action_frameskip)
                    action_stats_kwargs = self._build_action_stats(
                        mean,
                        std,
                        clip_sigma=plan_cfg.planner.get("action_clip_sigma", None),
                        label="Flat planner",
                    )

                if planner_class is not None:
                    self.planner = planner_class(
                        unroll=self.unroll,
                        action_dim=action_dim,
                        decode_fn=self._decode_fn,
                        **plan_cfg.planner,
                        **action_stats_kwargs,
                    )
                else:
                    logger.info("No planner provided in GCAgent.")
                    self.planner = None

        self.goal_state = None
        self.goal_position = None
        self.goal_state_enc = None
        self._prev_losses = None
        self._prev_losses_per_level: Dict[int, dict] = {}

    def _build_decode_fn(self) -> Optional[Callable]:
        """Build unified decode callable for planning iteration visualization.

        Priority:
        1. Visual decoder (works for any dataset).
        2. Location prober + env renderer (two_rooms only).
        3. None (no visualization).

        Returns:
            Callable ``[B, D, T, H, W] -> [B, T, H, W, C] uint8 ndarray``, or None.
        """
        if self.visual_decoder is not None:
            vd = self.visual_decoder

            def _decode_vd(encs):
                return decode_with_visual_decoder(vd, encs)

            return _decode_vd

        if self.loc_prober is not None and hasattr(self.env, "coord_to_pixel"):
            return self.decode_loc_to_pixel

        return None

    def _get_preprocessor_action_stats(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (action_mean, action_std) as float32 tensors, or None."""
        pp = self.preprocessor
        if pp is not None and getattr(pp, "action_mean", None) is not None:
            return (
                torch.as_tensor(pp.action_mean, dtype=torch.float32),
                torch.as_tensor(pp.action_std, dtype=torch.float32),
            )
        return None

    @staticmethod
    def _build_action_stats(
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        clip_sigma: Optional[float] = None,
        data: Optional[torch.Tensor] = None,
        label: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Build action stats kwargs (mean, std, and optional min/max bounds).

        Computes action_min/action_max via clip_sigma (mean +/- sigma*std).
        Falls back to 1st/99th percentiles of ``data`` when clip_sigma is None.
        """
        stats: Dict[str, torch.Tensor] = {
            "action_mean": action_mean,
            "action_std": action_std,
        }
        if clip_sigma is not None:
            stats["action_min"] = action_mean - clip_sigma * action_std
            stats["action_max"] = action_mean + clip_sigma * action_std
        elif data is not None:
            stats["action_min"] = torch.quantile(data, q=0.01, dim=0)
            stats["action_max"] = torch.quantile(data, q=0.99, dim=0)
        logger.info(
            f"{label}: action stats (mean={action_mean.tolist()}, "
            f"std={action_std.tolist()}"
            f"{f', clip_sigma={clip_sigma}' if clip_sigma else ''})"
        )
        return stats

    def _create_hierarchical_planner(
        self, plan_cfg, action_dim: int
    ) -> HierarchicalPlanner:
        """Create a hierarchical planner with level planners using the new convention.

        Level Convention:
        - Level 1: Finest encoder/predictor (stride=1, produces executable actions)
        - Level L: Coarsest encoder/predictor (plans towards final goal)

        Args:
            plan_cfg: Planning configuration.
            action_dim: Action dimension at level 1 (raw action space, e.g., 2).
                        Higher levels automatically use their predictor's action_dim.

        Returns:
            HierarchicalPlanner instance with Dict[int, Planner] keyed from 1 to L.
        """
        base_planner_name = plan_cfg.planner.get("base_planner", "mppi")
        planner_class = planner_name_map[base_planner_name]

        level_configs = plan_cfg.planner.get("level_configs", {})
        num_levels = len(level_configs)

        if num_levels == 0:
            raise ValueError(
                "Hierarchical planner requires at least one level in level_configs"
            )

        # Global defaults from plan_cfg.planner
        global_defaults = {
            k: v
            for k, v in plan_cfg.planner.items()
            if k
            not in [
                "type",
                "base_planner",
                "level_configs",
                "top_down",
                "planning_objective",
                "num_act_stepped",
                "subgoal_mode",
            ]
        }

        # Build dict of level planners with keys 1 to num_levels
        level_planners: Dict[int, Planner] = {}
        for l in range(1, num_levels + 1):
            level_key = f"level_{l}_planner"
            level_cfg = level_configs.get(level_key, {})

            # Merge global defaults with level-specific overrides
            planner_kwargs = {
                k: v
                for k, v in {**global_defaults, **level_cfg}.items()
                if k not in ("latent_action_stats_path", "action_clip_sigma")
            }

            # Get action dimension for this level from the model's predictor
            predictor = self.model.levels[l - 1].predictor
            if hasattr(predictor, "rnn"):
                level_action_dim = predictor.rnn.input_size
            elif hasattr(predictor, "action_embedder"):
                level_action_dim = predictor.action_embedder[0].in_features
            elif hasattr(predictor, "action_proj"):
                level_action_dim = predictor.action_proj.in_features
            else:
                raise AttributeError(
                    f"Cannot determine action_dim from {type(predictor).__name__}"
                )

            # Load action statistics: from file for level > 1, preprocessor for level 1
            action_stats_kwargs = {}
            clip_sigma = level_cfg.get("action_clip_sigma", None)
            if l > 1:
                stats_path = level_cfg.get("latent_action_stats_path", None)
                if stats_path is not None:
                    latent_actions = torch.load(stats_path, weights_only=False).to(
                        self.device
                    )
                    latent_actions = latent_actions.reshape(
                        -1, level_action_dim
                    )  # [N, A_enc]
                    action_stats_kwargs = self._build_action_stats(
                        latent_actions.mean(dim=0),
                        latent_actions.std(dim=0),
                        clip_sigma=clip_sigma,
                        data=latent_actions,
                        label=f"Level {l}",
                    )
            else:
                pp_stats = self._get_preprocessor_action_stats()
                if pp_stats is not None:
                    action_stats_kwargs = self._build_action_stats(
                        *pp_stats,
                        clip_sigma=clip_sigma,
                        label="Level 1",
                    )

            # Build per-level decode function from visual decoders dict
            level_vd = self._visual_decoders.get(l)
            if level_vd is not None:

                def _make_decode_fn(vd):
                    def _decode(encs):
                        return decode_with_visual_decoder(vd, encs)

                    return _decode

                level_decode_fn = _make_decode_fn(level_vd)
            else:
                level_decode_fn = self._decode_fn if l == 1 else None

            level_planner = planner_class(
                unroll=lambda obs, acts, lvl=l: self._unroll_at_level(obs, acts, lvl),
                action_dim=level_action_dim,
                decode_fn=level_decode_fn,
                **planner_kwargs,
                **action_stats_kwargs,
            )
            level_planners[l] = level_planner

        subgoal_mode = plan_cfg.planner.get("subgoal_mode", "single")
        start_level = plan_cfg.planner.get("start_level", None)
        verbose = plan_cfg.logging.get("verbose", False)

        logger.info(
            f"Created hierarchical planner with levels 1 to {num_levels}: "
            f"{[f'level_{l}' for l in range(1, num_levels + 1)]}, "
            f"start_level={start_level if start_level is not None else num_levels}, "
            f"verbose={verbose}"
        )

        return HierarchicalPlanner(
            level_planners=level_planners,
            subgoal_mode=subgoal_mode,
            start_level=start_level,
            verbose=verbose,
        )

    def _unroll_at_level(
        self, obs_init: torch.Tensor, actions: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Unroll the model at a specific hierarchy level.

        Args:
            obs_init: Initial observation [B, C, T, H, W].
            actions: Actions [B, A, T].
            level: Hierarchy level (1 to L).

        Returns:
            Predicted encodings at this level [B, D_l, T_l, H_l, W_l].
        """
        if hasattr(self.model, "unroll_at_level"):
            return self.model.unroll_at_level(obs_init, actions, level=level)
        else:
            return self.unroll(obs_init, actions, repeat_batch=False)

    def set_goal(self, goal_state, goal_position=None, init_state=None):
        self.goal_position = goal_position
        self.goal_state = goal_state
        # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
        self.goal_state_enc = self.model.encode(
            self.preprocessor.normalize_obs(goal_state.to(self.device))
            .unsqueeze(0)
            .unsqueeze(2)
        )
        # Store init_state for diagnostic logging
        self._init_state = init_state
        objective_name = self.plan_cfg.planner.planning_objective.get(
            "objective_type", "repr_dist"
        )
        objective_class = objective_name_map[objective_name]

        objective_kwargs = dict(self.plan_cfg.planner.planning_objective)
        objective_kwargs.pop("objective_type", None)

        if objective_name == "projected_dist":
            cost_module = getattr(self.model, "cost_module", None)
            if cost_module is None or not hasattr(cost_module, "projector"):
                raise ValueError(
                    "ProjectedDistObjective requires model.cost_module with a projector"
                )
            # Set projector to eval mode for planning (avoids BatchNorm issues)
            cost_module.projector.eval()
            objective_kwargs["projector"] = cost_module.projector

        if self._is_hierarchical:
            num_levels = len(self.planner.level_planners)
            # For hierarchical planning, use the coarsest level (max_level) config
            level_key = f"level_{num_levels}_planner"
            level_cfg = self.plan_cfg.planner.level_configs.get(level_key, {})
            objective_kwargs.update(
                {
                    k: v
                    for k, v in level_cfg.items()
                    if k in ["distance", "sum_all_diffs"]
                }
            )

        # Special handling for hierarchical objectives
        if objective_name == "hierarchical_repr_dist":
            # Encode goal at all hierarchy levels
            goal_input = (
                self.preprocessor.normalize_obs(goal_state.to(self.device))
                .unsqueeze(0)
                .unsqueeze(2)
            )
            target_encs = self.model.encode_hierarchical(goal_input)

            # Filter to only use levels up to start_level
            start_level = self.planner.start_level
            target_encs = {k: v for k, v in target_encs.items() if k <= start_level}

            # Construct per-level objectives
            level_objectives = {}
            for level, enc in target_encs.items():
                level_objectives[level] = ReprDistObjective(
                    target_enc=enc,
                    distance=objective_kwargs.get("distance", "l2"),
                    sum_all_diffs=objective_kwargs.get("sum_all_diffs", True),
                )

            self.objective = objective_class(
                level_objectives=level_objectives,
                hierarchical_model=self.model,
                subgoal_weight=objective_kwargs.get("subgoal_weight", 1.0),
                goal_weight=objective_kwargs.get("goal_weight", 1.0),
            )
        elif objective_name == "hierarchical_projected_dist":
            # Get the underlying model (handle torch.compile wrapper)
            model = (
                self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            )

            # Verify model has cost_modules
            if not hasattr(model, "cost_modules") or not model.cost_modules:
                raise ValueError(
                    "hierarchical_projected_dist requires model with cost_modules. "
                    "Train with cost modules enabled in config."
                )

            # Encode goal at all hierarchy levels
            goal_input = (
                self.preprocessor.normalize_obs(goal_state.to(self.device))
                .unsqueeze(0)
                .unsqueeze(2)
            )
            target_encs = self.model.encode_hierarchical(goal_input)

            # Filter to only use levels up to start_level
            start_level = self.planner.start_level
            target_encs = {k: v for k, v in target_encs.items() if k <= start_level}

            # Construct per-level ProjectedDistObjective instances
            level_objectives = {}
            for level, enc in target_encs.items():
                if level not in model.cost_modules:
                    raise ValueError(
                        f"Cost module for level {level} not found in model. "
                        f"Available levels: {list(model.cost_modules.keys())}"
                    )
                projector = model.cost_modules[level].projector
                # Set projector to eval mode for planning (avoids BatchNorm issues)
                projector.eval()
                level_objectives[level] = ProjectedDistObjective(
                    projector=projector,
                    target_enc=enc,
                    distance=objective_kwargs.get("distance", "l2"),
                    sum_all_diffs=objective_kwargs.get("sum_all_diffs", True),
                )

            self.objective = objective_class(
                level_objectives=level_objectives,
                hierarchical_model=self.model,
                subgoal_weight=objective_kwargs.get("subgoal_weight", 1.0),
                goal_weight=objective_kwargs.get("goal_weight", 1.0),
            )
        else:
            self.objective = objective_class(
                target_enc=self.goal_state_enc, **objective_kwargs
            )
        self.planner.set_objective(self.objective)

        # [DIAG] Probe E: Encoding Health Check
        verbose = (
            self.plan_cfg.logging.get("verbose", False) if self.plan_cfg else False
        )
        if self._is_hierarchical and self._init_state is not None and verbose:
            self._log_encoding_health_diagnostics(init_state)

    def _log_encoding_health_diagnostics(self, init_state: torch.Tensor):
        """Log encoding health diagnostics for hierarchical planning.

        Compares goal vs. initial state encodings at each hierarchy level.
        Logs: mean, std, L2 norm, L2 distance, cosine similarity.

        Args:
            init_state: Initial observation [C, H, W].
        """
        # Encode init state at all hierarchy levels
        init_input = (
            self.preprocessor.normalize_obs(init_state.to(self.device))
            .unsqueeze(0)
            .unsqueeze(2)
        )  # [1, C, 1, H, W]

        with torch.no_grad():
            init_encs = self.model.encode_hierarchical(init_input)

        # Get goal encodings from the objective
        goal_encs = {
            level: obj.target_enc
            for level, obj in self.objective.level_objectives.items()
        }

        logger.info("[DIAG] ========== ENCODING HEALTH CHECK ==========")

        for level in sorted(goal_encs.keys()):
            goal_enc = goal_encs[level]  # [1, D, 1, H', W']
            init_enc = init_encs[level]  # [1, D, 1, H', W']

            # Flatten for statistics
            goal_flat = goal_enc.flatten()
            init_flat = init_enc.flatten()

            # Basic stats
            goal_mean = goal_flat.mean().item()
            goal_std = goal_flat.std().item()
            goal_norm = goal_flat.norm().item()

            init_mean = init_flat.mean().item()
            init_std = init_flat.std().item()
            init_norm = init_flat.norm().item()

            # Distance and similarity
            l2_dist = (goal_enc - init_enc).pow(2).sum().sqrt().item()
            cosine_sim = F.cosine_similarity(
                goal_flat.unsqueeze(0), init_flat.unsqueeze(0)
            ).item()

            logger.info(
                f"[DIAG] Level {level}: "
                f"goal(mean={goal_mean:.4f}, std={goal_std:.4f}, norm={goal_norm:.4f}) | "
                f"init(mean={init_mean:.4f}, std={init_std:.4f}, norm={init_norm:.4f}) | "
                f"L2_dist={l2_dist:.4f} | cos_sim={cosine_sim:.4f}"
            )

        logger.info("[DIAG] ==============================================")

    def unroll(self, obs_init, actions, repeat_batch=True, ctxt_window_time=None):
        """Unroll the model autoregressively for planning or evaluation.

        Args:
            obs_init: [B, C, T_ctx, H, W] initial context observations.
            actions: [B, A, T_a] action sequence.
            repeat_batch: Whether to repeat obs_init to match actions batch size.
            ctxt_window_time: Context window size override. If None, uses
                plan_cfg["ctxt_window_time"] or defaults to 1.

        Returns:
            predicted_states: [B, D, T_ctx + nsteps, H', W'] where
                nsteps = T_a - ctxt_window + 1.
        """
        batch_size = actions.shape[0]
        ctxt_window = ctxt_window_time or (
            self.plan_cfg["ctxt_window_time"] if self.plan_cfg else 1
        )
        nsteps = actions.shape[2] - ctxt_window + 1
        if repeat_batch:
            obs_init = obs_init.repeat(batch_size, 1, 1, 1, 1)
        predicted_states, _, _ = self.model.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=ctxt_window,
            compute_loss=False,
            return_all_steps=False,
        )
        return predicted_states

    def unroll_at_levels(
        self, obs_init, actions, levels, repeat_batch=True, ctxt_window_time=None
    ):
        """Unroll the model at specific hierarchy levels.

        Args:
            obs_init: Initial observation [B, C, T_ctx, H, W].
            actions: Actions [B, A, T_a] action sequence.
            levels: List of hierarchy levels to unroll.
            repeat_batch: Whether to repeat obs_init to match actions batch size.
            ctxt_window_time: Context window size override. If None, uses
                plan_cfg["ctxt_window_time"] or defaults to 1.

        Returns:
            Dict mapping level -> predicted states [B, D_l, T_l, H_l, W_l].
        """
        batch_size = actions.shape[0]
        ctxt_window = ctxt_window_time or (
            self.plan_cfg["ctxt_window_time"] if self.plan_cfg else 1
        )
        nsteps = actions.shape[2] - ctxt_window + 1
        if repeat_batch:
            obs_init = obs_init.repeat(batch_size, 1, 1, 1, 1)
        predicted_states_dict, _, _ = self.model.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=ctxt_window,
            compute_loss=False,
            return_all_steps=False,
            levels=levels,
        )
        return predicted_states_dict

    def decode_loc_to_pixel(
        self, predicted_encs, prober=None, wall_x=None, door_y=None
    ):
        """Decode predicted encodings into frames using a position prober.

        Args:
            predicted_encs: Encoded states [B, D, T, H, W].
            prober: JEPAProbe with an apply_head method. If None, uses self.loc_prober.
            wall_x: Optional wall x-coordinates for rendering.
            door_y: Optional door y-coordinates for rendering.

        Returns:
            np.array of shape [B, T, H, W, C] on cpu, or None if prober is None.
        """
        prober = prober if prober is not None else self.loc_prober
        if prober is None:
            return None
        B, D, T, H, W = predicted_encs.shape
        out = (
            prober.apply_head(predicted_encs).permute(0, 2, 1).cpu()
        )  # [B, T, probe_dim]
        if (
            self.preprocessor is not None
            and out.shape[-1] == self.preprocessor.proprio_mean.shape[-1]
        ):
            out = self.preprocessor.denormalize_proprios(out)  # [B, T, probe_dim]
        if hasattr(self.env, "coord_to_pixel"):
            frames = self.env.coord_to_pixel(
                out, wall_x=wall_x, door_y=door_y
            )  # [B, T, C, H, W]
            frames = frames.permute(0, 1, 3, 4, 2).cpu().numpy()  # [B, T, H, W, C]
        else:
            logger.warning(
                "Environment does not support coord_to_pixel; "
                "returning placeholder frames."
            )
            B, T, _ = out.shape
            frames = np.zeros((B, T, 64, 64, 3), dtype=np.uint8)
        return frames

    def plan(self, obs, steps_left=None, t0=False, plan_vis_path=None):
        """Plan a trajectory from the current observation.

        Args:
            obs: Current observation tensor [1, C, 1, H, W].
            steps_left: Number of steps remaining in episode.
            t0: Whether this is the first observation.
            plan_vis_path: Optional path for visualization.

        Returns:
            Planning result (H_PlanningResult for hierarchical, PlanningResult for flat).
        """
        planning_result = self.planner.plan(
            obs,
            steps_left=steps_left,
            eval_mode=True,
            t0=t0,
            plan_vis_path=plan_vis_path,
        )

        if self._is_hierarchical:
            level_results = planning_result.level_results
            self._prev_losses_per_level = {}
            for level, result in level_results.items():
                if result is not None:
                    self._prev_losses_per_level[level] = {
                        "losses": result.losses,
                        "elite_mean": result.prev_elite_losses_mean,
                        "elite_std": result.prev_elite_losses_std,
                    }
        else:
            self._prev_losses = planning_result.losses
            self._prev_elite_losses_mean = planning_result.prev_elite_losses_mean
            self._prev_elite_losses_std = planning_result.prev_elite_losses_std

        return planning_result

    def postprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert planner output to raw env-space actions.

        Pipeline: truncate to num_act_stepped → de-concat frameskip → denormalize.

        Args:
            actions: [T, model_action_dim] normalized planner output.

        Returns:
            [T_env, env_action_dim] denormalized CPU tensor.
        """
        actions = actions[: self.num_act_stepped]
        if self._action_frameskip > 1:
            actions = rearrange(actions, "t (f d) -> (t f) d", d=self._env_action_dim)
        if (
            self.preprocessor is not None
            and getattr(self.preprocessor, "action_mean", None) is not None
        ):
            actions = self.preprocessor.denormalize_actions(actions)
        return actions.detach().cpu()

    def act(self, obs, steps_left=None, t0=False, plan_vis_path=None) -> np.ndarray:
        """Plan and return raw env-space actions as numpy.

        Returns:
            np.ndarray of shape [T_env, env_action_dim] ready for env.step().
        """
        planning_result = self.plan(
            obs,
            steps_left=steps_left,
            t0=t0,
            plan_vis_path=plan_vis_path,
        )

        if self._is_hierarchical:
            level_results = planning_result.level_results
            if 1 in level_results and level_results[1] is not None:
                actions = level_results[1].actions
            else:
                raise ValueError("No Level 1 result available for executable actions")
        else:
            actions = planning_result.actions

        return self.postprocess_actions(actions).numpy()
