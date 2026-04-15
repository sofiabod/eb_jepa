import functools
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, NamedTuple, Optional

import numpy as np
import torch
from einops import rearrange

from eb_jepa.utils.logging import get_logger
from eb_jepa.vis.frames import save_decoded_frames

logger = get_logger(__name__)


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None


class H_PlanningResult(NamedTuple):
    """Result from hierarchical planning.

    Uses the following level convention:
    - Level 0: Groundtruth environment dynamics (mathematical abstraction only, not implemented)
    - Level 1: Finest encoder/predictor (stride=1, produces executable actions)
    - Level L: Coarsest encoder/predictor (plans towards final goal)

    Attributes:
        level_results: Dict mapping level indices (1 to L) to PlanningResult objects.
            Keys are integers from 1 (finest) to L (coarsest). None if level was not planned.
        info: Additional information from the planning process.
    """

    level_results: Dict[int, Optional[PlanningResult]]
    info: dict = None

    @property
    def actions(self) -> torch.Tensor:
        """Return actions from Level 1 (finest level, executable actions)."""
        if self.level_results is None or 1 not in self.level_results:
            raise ValueError("No Level 1 result available for executable actions")
        return self.level_results[1].actions


class Planner(ABC):
    def __init__(self, unroll: Callable, **kwargs):
        self.unroll = unroll
        self.objective = None

    def set_objective(self, objective: Callable):
        self.objective = objective

    @abstractmethod
    def plan(
        self,
        obs_init: torch.Tensor,
        steps_left: Optional[int] = None,
        t0: bool = False,
        eval_mode: bool = False,
    ):
        pass

    def cost_function(
        self, actions: torch.Tensor, obs_init: torch.Tensor
    ) -> torch.Tensor:
        predicted_encs = self.unroll(obs_init, actions)
        return self.objective(predicted_encs)


def _clip_actions_per_group(
    actions: torch.Tensor,
    max_norms: List[float],
    max_norm_dims: List[List[int]],
) -> torch.Tensor:
    """Per-dimension-group element-wise clamping (matches jepa-wms reference)."""
    for dims, maxnorm in zip(max_norm_dims, max_norms):
        actions[..., dims] = torch.clip(actions[..., dims], min=-maxnorm, max=maxnorm)
    return actions


### Specific planning optimizers ###
class GradientDescentPlanner(Planner):
    """Gradient-based planner that optimizes actions via backpropagation."""

    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        lr: float = 1.0,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1.0,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        optimizer_type: str = "gd",
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        decode_each_iteration: bool = False,
        decode_fn: Optional[Callable] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Gradient Descent Planner for action optimization in latent space.

        Args:
            unroll: Function to unroll the world model
            n_iters: Number of optimization iterations
            plan_length: Planning horizon (number of timesteps)
            action_dim: Dimension of the action space
            lr: Learning rate for gradient descent
            action_noise: Standard deviation of Gaussian noise to add after each step
            sample_type: Type of action initialization ("randn" or "zero")
            var_scale: Scale for random initialization
            max_norms: List of max norm values for clipping (None to disable)
            max_norm_dims: List of dimension groups to clip
            optimizer_type: Type of optimizer ("gd" or "adam")
            adam_betas: Betas for Adam optimizer
            adam_eps: Epsilon for Adam optimizer
            decode_each_iteration: Whether to decode predictions at each iteration
            decode_fn: Function to decode latent to pixels
            action_mean: Mean of encoded action distribution for initialization [action_dim]
            action_std: Std of encoded action distribution for initialization [action_dim]
            action_min: Min encoded action values for clamping [action_dim]
            action_max: Max encoded action values for clamping [action_dim]
        """
        super().__init__(unroll)
        self.n_iters = n_iters
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.lr = lr
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.var_scale = var_scale
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.optimizer_type = optimizer_type.lower()
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.decode_each_iteration = decode_each_iteration
        self.decode_fn = decode_fn
        self.action_mean = action_mean
        self.action_std = action_std
        self.action_min = action_min
        self.action_max = action_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_actions(self, plan_length: int) -> torch.Tensor:
        """Initialize actions for planning. Returns shape [T, A]."""
        if self.action_mean is not None:
            actions = self.action_mean.unsqueeze(0).expand(plan_length, -1).clone().to(
                self.device
            ) + self.action_std.unsqueeze(0).expand(plan_length, -1).to(
                self.device
            ) * torch.randn(
                plan_length, self.action_dim, device=self.device
            )
        elif self.sample_type == "randn":
            actions = (
                torch.randn(plan_length, self.action_dim, device=self.device)
                * self.var_scale
            )
        elif self.sample_type == "zero":
            actions = torch.zeros(plan_length, self.action_dim, device=self.device)
        else:
            raise ValueError(f"Unknown sample_type: {self.sample_type}")
        return actions

    def plan(
        self, obs_init, steps_left=None, eval_mode=True, t0=False, plan_vis_path=None
    ):
        """
        Plan a sequence of actions using gradient descent optimization.

        Args:
            obs_init: Initial observation/latent state
            steps_left: Number of steps left in episode (optional)
            eval_mode: Whether in evaluation mode
            t0: Whether this is the first observation in the episode
            plan_vis_path: Path to save visualization (optional)

        Returns:
            PlanningResult with optimized actions and planning metrics
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        # Initialize actions: [T, A]
        actions = self._init_actions(plan_length)
        actions.requires_grad = True

        # Setup optimizer
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                [actions], lr=self.lr, betas=self.adam_betas, eps=self.adam_eps
            )
        else:
            optimizer = torch.optim.SGD([actions], lr=self.lr)

        losses = []
        if self.decode_each_iteration and self.decode_fn is not None:
            pred_frames_over_iterations = []

        for _ in range(self.n_iters):
            optimizer.zero_grad()

            # actions: [T, A] -> [1, A, T] for cost_function
            actions_batched = rearrange(actions, "t a -> 1 a t")
            cost = self.cost_function(actions_batched, obs_init)
            total_loss = cost.mean()
            total_loss.backward()

            losses.append(total_loss.item())

            # Gradient step with optional noise
            with torch.no_grad():
                actions_new = actions - self.lr * actions.grad

                if self.action_noise > 0:
                    actions_new += torch.randn_like(actions_new) * self.action_noise

                # Per-dimension-group clamping
                if self.max_norms is not None and self.max_norm_dims is not None:
                    actions_new = _clip_actions_per_group(
                        actions_new, self.max_norms, self.max_norm_dims
                    )

                actions.copy_(actions_new)

            actions.grad.zero_()

            if self.decode_each_iteration and self.decode_fn is not None:
                with torch.no_grad():
                    predicted_best_encs = self.unroll(
                        obs_init, rearrange(actions, "t a -> 1 a t")
                    )
                    pred_frames = self.decode_fn(predicted_best_encs)
                    pred_frames_over_iterations.append(pred_frames.squeeze(0))

        if self.decode_each_iteration and self.decode_fn is not None:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)

        # Return final actions: [T, A]
        final_actions = actions.detach()

        return PlanningResult(
            actions=final_actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(losses).unsqueeze(-1),
            prev_elite_losses_std=torch.zeros(len(losses)).unsqueeze(-1),
        )


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 15,
        num_samples: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        max_std: float = 2,
        num_elites: int = 64,
        temperature: float = 0.005,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = False,
        decode_fn: Optional[Callable] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_std = max_std
        self.num_elites = num_elites
        self.temperature = temperature
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_fn = decode_fn
        self.action_mean = action_mean
        self.action_std = action_std
        self.action_min = action_min
        self.action_max = action_max
        self._prev_mean = None
        self.local_generator = None

    def _compute_elite_weights(
        self, elite_loss: torch.Tensor, min_cost: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized weights for elite actions.

        Args:
            elite_loss: Cost of elite actions [num_elites, 1].
            min_cost: Minimum cost across all samples [1].

        Returns:
            Normalized weight vector [num_elites].
        """
        score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))
        score /= score.sum(0)
        return score

    def _select_final_action(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        elite_actions: torch.Tensor,
        score: torch.Tensor,
        eval_mode: bool,
    ) -> torch.Tensor:
        """Select the action to return after optimization.

        Args:
            mean: Current mean of action distribution [T, A].
            std: Current std of action distribution [T, A].
            elite_actions: Elite action samples [T, num_elites, A].
            score: Normalized elite weights [num_elites].
            eval_mode: If True, reduce exploration noise.

        Returns:
            Selected action sequence [T, A].
        """
        score_np = score.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(score_np.shape[0]), p=score_np)
        ]  # [T, A]
        self._prev_mean = mean
        if not eval_mode:
            actions += std * torch.randn(
                self.action_dim, device=std.device, generator=self.local_generator
            )
        return actions

    @torch.no_grad()
    def plan(
        self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None
    ):
        """
        Args:
                obs_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        if self.action_mean is not None:
            mean = (
                self.action_mean.unsqueeze(0)
                .expand(plan_length, -1)
                .clone()
                .to(self.device)
            )
            std = (
                self.action_std.unsqueeze(0)
                .expand(plan_length, -1)
                .clone()
                .to(self.device)
            )
        else:
            mean = torch.zeros(plan_length, self.action_dim, device=self.device)
            std = self.max_std * torch.ones(
                plan_length, self.action_dim, device=self.device
            )
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration and self.decode_fn is not None:
            pred_frames_over_iterations = []

        # MPPI iterations
        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A

            # Clamp to latent action bounds if provided
            if self.action_min is not None and self.action_max is not None:
                actions = torch.max(actions, self.action_min.to(self.device))
                actions = torch.min(actions, self.action_max.to(self.device))

            # Per-dimension-group clamping
            if self.max_norms is not None and self.max_norm_dims is not None:
                actions = _clip_actions_per_group(
                    actions, self.max_norms, self.max_norm_dims
                )

            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters with momentum
            min_cost = cost.min(0)[0]
            score = self._compute_elite_weights(elite_loss, min_cost)
            new_mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            ) / (  # T B A
                score.sum(0) + 1e-9
            )
            new_std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - new_mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            mean = new_mean * (1 - self.momentum_mean) + mean * self.momentum_mean
            std = new_std * (1 - self.momentum_std) + std * self.momentum_std
            if self.decode_each_iteration and self.decode_fn is not None:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_fn(predicted_best_encs)
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration and self.decode_fn is not None:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)
        # Select action via hook
        a = self._select_final_action(mean, std, elite_actions, score, eval_mode)

        return PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )


class CEMPlanner(MPPIPlanner):
    """Cross-Entropy Method planner. Special case of MPPI with uniform elite weights."""

    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 30,
        num_samples: int = 300,
        plan_length: int = 15,
        action_dim: int = 2,
        var_scale: float = 1,
        num_elites: int = 10,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = True,
        decode_fn: Optional[Callable] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(
            unroll=unroll,
            n_iters=n_iters,
            num_samples=num_samples,
            plan_length=plan_length,
            action_dim=action_dim,
            max_std=var_scale,
            num_elites=num_elites,
            temperature=0.0,
            momentum_mean=momentum_mean,
            momentum_std=momentum_std,
            max_norms=max_norms,
            max_norm_dims=max_norm_dims,
            decode_each_iteration=decode_each_iteration,
            decode_fn=decode_fn,
            action_mean=action_mean,
            action_std=action_std,
            action_min=action_min,
            action_max=action_max,
        )
        self.var_scale = var_scale

    def _compute_elite_weights(
        self, elite_loss: torch.Tensor, min_cost: torch.Tensor
    ) -> torch.Tensor:
        """Uniform weights over elites (CEM special case)."""
        num_elites = elite_loss.shape[0]
        return torch.ones(num_elites, device=elite_loss.device) / num_elites

    def _select_final_action(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        elite_actions: torch.Tensor,
        score: torch.Tensor,
        eval_mode: bool,
    ) -> torch.Tensor:
        """CEM returns the distribution mean."""
        return mean


AdamPlanner = functools.partial(GradientDescentPlanner, optimizer_type="adam")


class HierarchicalPlanner(Planner):
    """Top-down hierarchical planner for multi-level world models.

    Level Convention:
    - Level 0: Groundtruth environment dynamics (mathematical abstraction only, not implemented)
    - Level 1: Finest encoder/predictor (stride=1, produces executable actions)
    - Level L: Coarsest encoder/predictor (plans towards final goal)

    Algorithm:
    1. Encode goal state at all L levels
    2. At level L (coarsest): plan towards goal_L, get trajectory z_L
    3. For l = L-1 down to 1:
       - Extract subgoals from z_{l+1} (interpolate to finer resolution)
       - Plan at level l towards subgoals in level-(l+1) space
       - Get trajectory z_l
    4. Return finest-level actions from level-1 planning
    """

    def __init__(
        self,
        level_planners: Dict[int, Planner],
        subgoal_mode: str = "single",
        start_level: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize HierarchicalPlanner.

        Args:
            level_planners: Dict mapping level indices (1 to L) to Planner instances.
                - Key 1: Finest level planner (stride=1, produces executable actions)
                - Key L: Coarsest level planner (plans towards final goal)
            subgoal_mode: How to extract subgoals from higher-level trajectories.
                - "single": Use only the first predicted state (index 1) as the subgoal.
                    The agent re-plans at every step (receding horizon).
                - "sequential": Use the full trajectory as subgoals (not yet implemented).
            start_level: Level to start planning from (default: None, meaning use max_level).
                If specified, planning starts from this level instead of max_level.
            verbose: If True, enable detailed [DIAG] logging during planning.
        """
        super().__init__(unroll=None)
        self.level_planners = level_planners
        self.subgoal_mode = subgoal_mode
        self.min_level = 1
        self.max_level = max(level_planners.keys())
        self.start_level = start_level if start_level is not None else self.max_level
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_objective(self, objective: "HierarchicalObjective"):  # noqa: F821
        """Set the planning objective for hierarchical planning.

        For hierarchical planning, this should be a HierarchicalObjective.
        The objective is shared across all levels, with level-specific targeting
        controlled by set_level() calls during planning.
        """
        self.objective = objective
        # Set objective on all level planners (only for levels up to start_level)
        for level in range(1, self.start_level + 1):
            self.level_planners[level].set_objective(objective)

    @torch.no_grad()
    def plan(
        self,
        obs_init: torch.Tensor,
        steps_left: Optional[int] = None,
        t0: bool = False,
        eval_mode: bool = False,
        plan_vis_path: Optional[str] = None,
    ) -> H_PlanningResult:
        """Execute top-down hierarchical planning.

        Args:
            obs_init: Initial observation [1, C, 1, H, W].
            steps_left: Number of steps remaining in episode.
            t0: Whether this is the first observation.
            eval_mode: Whether in evaluation mode.
            plan_vis_path: Optional path for visualization.

        Returns:
            H_PlanningResult with level_results dict (keys 1 to max_level).
            Executable actions are in level_results[1].
        """
        level_results: Dict[int, Optional[PlanningResult]] = {
            l: None for l in range(1, self.max_level + 1)
        }
        all_trajectories: Dict[int, torch.Tensor] = {}
        # Top-down planning: from start_level down to 1
        for level in range(self.start_level, 0, -1):
            planner = self.level_planners[level]

            # Determine effective planning horizon for this level
            if steps_left is None:
                effective_horizon = planner.plan_length
            else:
                effective_horizon = min(planner.plan_length, steps_left)

            if self.verbose:
                logger.info(
                    f"Planning at Level {level}/{self.start_level}: "
                    f"horizon={effective_horizon} (config: {planner.plan_length}, steps_left={steps_left})"
                )
            # For levels below the start level, set subgoals from the level above
            if level < self.start_level and (level + 1) in all_trajectories:
                subgoals = self._extract_subgoals(
                    all_trajectories[level + 1], level + 1, level
                )
                self.objective.set_subgoals(level, subgoals)
                if self.verbose:
                    logger.info(
                        f"Level {level}: Using subgoal from Level {level + 1} "
                        f"trajectory (first predicted state, shape: {list(subgoals.shape)})"
                    )

            # Set the current level for the objective
            self.objective.set_level(level)

            vis_path = f"{plan_vis_path}_level{level}" if plan_vis_path else None
            result = planner.plan(
                obs_init, steps_left=steps_left, t0=t0, plan_vis_path=vis_path
            )
            level_results[level] = result

            # Unroll to get trajectory for subgoal extraction
            actions_at_level = result.actions
            if self.verbose:
                logger.info(
                    f"Level {level}/{self.max_level}: actions {list(actions_at_level.shape)}"
                )
            if actions_at_level.dim() == 2:  # [T, A]
                actions_at_level = rearrange(actions_at_level, "t a -> 1 a t")

            trajectory = planner.unroll(obs_init, actions_at_level)
            all_trajectories[level] = trajectory
            if self.verbose:
                logger.info(
                    f"Level {level}/{self.max_level}: trajectory {list(trajectory.shape)}"
                )

            # [DIAG] Subgoal Quality Analysis at the top planning level
            if self.verbose and level == self.start_level:
                self._log_subgoal_quality_diagnostics(
                    level=level,
                    result=result,
                    trajectory=trajectory,
                    obs_init=obs_init,
                )

        self.objective.clear_subgoals()

        return H_PlanningResult(
            level_results=level_results,
            info={
                "level_trajectories": all_trajectories,
                "active_level": self.max_level,
            },
        )

    def _log_subgoal_quality_diagnostics(
        self,
        level: int,
        result: PlanningResult,
        trajectory: torch.Tensor,
        obs_init: torch.Tensor,
    ):
        """Log subgoal quality diagnostics after top-level planning.

        Analyzes:
        1. MPPI loss convergence (first vs last)
        2. Subgoal vs initial state distance
        3. Subgoal vs goal distance
        4. Init-to-goal distance
        5. Progress ratio
        6. Full trajectory distances to goal

        Args:
            level: Current level (the start_level).
            result: PlanningResult from this level's optimizer.
            trajectory: Predicted trajectory [B, D, T, H', W'].
            obs_init: Initial observation [1, C, 1, H, W].
        """
        losses = result.losses
        if losses is None or len(losses) == 0:
            logger.info("[DIAG] No losses recorded for MPPI convergence analysis")
            return

        # 1. MPPI loss convergence
        first_loss = losses[0].item()
        last_loss = losses[-1].item()
        loss_ratio = last_loss / (first_loss + 1e-8)

        logger.info(
            f"[DIAG] ========== SUBGOAL QUALITY ANALYSIS (Level {level}) =========="
        )
        logger.info(
            f"[DIAG] MPPI Convergence: first_loss={first_loss:.4f}, "
            f"last_loss={last_loss:.4f}, ratio={loss_ratio:.4f}"
        )

        # Get goal encoding at level 2
        goal_enc = self.objective.level_objectives[
            level
        ].target_enc  # [1, D, 1, H', W']

        # trajectory shape: [B, D, T, H', W']
        # Index 0 = init encoding, index 1 = first predicted state (subgoal)
        init_enc = trajectory[:, :, 0:1]  # [1, D, 1, H', W']
        subgoal_enc = trajectory[:, :, 1:2]  # [1, D, 1, H', W']

        # 2. Subgoal vs initial state distance
        subgoal_init_dist = (subgoal_enc - init_enc).pow(2).sum().sqrt().item()

        # 3. Subgoal vs goal distance
        subgoal_goal_dist = (subgoal_enc - goal_enc).pow(2).sum().sqrt().item()

        # 4. Init-to-goal distance
        init_goal_dist = (init_enc - goal_enc).pow(2).sum().sqrt().item()

        # 5. Progress ratio (should be < 1.0 for good subgoal)
        progress_ratio = subgoal_goal_dist / (init_goal_dist + 1e-8)

        logger.info(
            f"[DIAG] Distances: subgoal-init={subgoal_init_dist:.4f}, "
            f"subgoal-goal={subgoal_goal_dist:.4f}, init-goal={init_goal_dist:.4f}"
        )
        logger.info(
            f"[DIAG] Progress ratio (subgoal-goal / init-goal): {progress_ratio:.4f}"
        )

        # 6. Full trajectory progress: distance to goal at each timestep
        T = trajectory.shape[2]
        traj_distances = []
        for t in range(T):
            enc_t = trajectory[:, :, t : t + 1]
            dist_t = (enc_t - goal_enc).pow(2).sum().sqrt().item()
            traj_distances.append(dist_t)

        logger.info(
            f"[DIAG] Trajectory distances to goal: {[f'{d:.2f}' for d in traj_distances]}"
        )

        # Check if trajectory diverges (distance increases over time)
        if len(traj_distances) > 1:
            if traj_distances[-1] > traj_distances[0]:
                logger.info("[DIAG] WARNING: Trajectory DIVERGES from goal!")
            elif traj_distances[-1] < traj_distances[0] * 0.9:
                logger.info("[DIAG] OK: Trajectory converges toward goal")
            else:
                logger.info("[DIAG] NOTE: Trajectory makes minimal progress")

        logger.info("[DIAG] =========================================================")

    def _extract_subgoals(
        self,
        high_level_trajectory: torch.Tensor,
        source_level: int,
        target_level: int,
    ) -> torch.Tensor:
        """Extract subgoal from higher-level trajectory for lower-level planning.

        In single (receding horizon) mode, the subgoal is the first *predicted*
        state from the higher level, i.e. index 1 of the trajectory (index 0 is the
        initial state). The agent re-plans every ``num_act_stepped`` environment
        steps, so only the first subgoal matters.

        Args:
            high_level_trajectory: Predicted trajectory at source level
                [B, D_{ℓ+1}, T, H_{ℓ+1}, W_{ℓ+1}] where T includes the initial state.
            source_level: Level of the trajectory (higher/coarser, ℓ+1).
            target_level: Level to extract subgoals for (lower/finer, ℓ).

        Returns:
            Subgoal [B, D_{ℓ+1}, 1, H_{ℓ+1}, W_{ℓ+1}].
        """
        # Index 0 = initial state, index 1 = first predicted state (ẑ^{ℓ+1}_1)
        return high_level_trajectory[:, :, 1:2]
