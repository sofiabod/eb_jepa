from typing import Callable, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn


### Planning objectives to minimize ###
class ReprDistObjective:
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        target_enc: torch.Tensor,
        distance: Literal["l1", "l2"] = "l2",
        sum_all_diffs: bool = False,
        **kwargs,
    ):
        self.target_enc = target_enc
        self.distance = distance
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        """
        Args:
            encodings: [B, D, T, H, W]
            keepdims: if True, return [B, T], else return [B]

        Returns:
            diff: [B, T] else [B] if sum_all_diffs or not keepdims
        """
        if self.sum_all_diffs:
            keepdims = True
        target = self.target_enc
        if target.shape != encodings.shape:
            target = target.expand(encodings.shape[0], -1, encodings.shape[2], -1, -1)

        if self.distance == "l2":
            diff = (target - encodings).pow(2).mean(dim=(1, 3, 4))  # B T
        elif self.distance == "l1":
            diff = (target - encodings).abs().mean(dim=(1, 3, 4))  # B T
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        if not keepdims:
            diff = diff[:, -1]
        if self.sum_all_diffs:
            diff = diff.sum(dim=1)
        return diff


class ProjectedDistObjective:
    """Objective to minimize distance to the target in a learned projected space.

    This objective projects both the unrolled states and the goal state through a
    learned projector, then computes L1 or L2 distance in the projected space.
    """

    def __init__(
        self,
        projector: nn.Module,
        target_enc: torch.Tensor,
        distance: Literal["l1", "l2"] = "l2",
        sum_all_diffs: bool = False,
        **kwargs,
    ):
        """
        Args:
            projector: Learned projector network. Input: [N, D], Output: [N, D'].
            target_enc: Goal state encoding [1, C, 1, H, W] or [B, C, 1, H, W].
            distance: Distance metric to use ('l1' or 'l2').
            sum_all_diffs: If True, sum distances across all time steps.
                If False, use only the last state.
        """
        self.projector = projector
        self.target_enc = target_enc
        self.distance = distance
        self.sum_all_diffs = sum_all_diffs

        self._projected_target = None

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Project tensor [B, C, T, H, W] -> [B, T, D']."""
        b, c, t, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [B*T*H*W, C]
        x_proj = self.projector(x_flat)  # [B*T*H*W, D']
        d_out = x_proj.shape[-1]
        x_proj = x_proj.view(b, t, h, w, d_out)  # [B, T, H, W, D']
        x_proj = x_proj.mean(dim=(2, 3))  # [B, T, D'] (spatial pooling)
        return x_proj

    @property
    def projected_target(self) -> torch.Tensor:
        """Lazily compute and cache the projected target."""
        if self._projected_target is None:
            with torch.no_grad():
                self._projected_target = self._project(self.target_enc)  # [1, 1, D']
        return self._projected_target

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        """
        Args:
            encodings: Unrolled state encodings [B, C, T, H, W].
            keepdims: If True, return [B, T], else return [B].

        Returns:
            diff: Distance in projected space. Shape [B, T] if keepdims else [B].
        """
        if self.sum_all_diffs:
            keepdims = True

        proj_enc = self._project(encodings)  # [B, T, D']
        proj_target = self.projected_target  # [1, 1, D']

        proj_target = proj_target.expand(proj_enc.shape[0], proj_enc.shape[1], -1)

        if self.distance == "l2":
            diff = (proj_enc - proj_target).pow(2).mean(dim=-1)  # [B, T]
        elif self.distance == "l1":
            diff = (proj_enc - proj_target).abs().mean(dim=-1)  # [B, T]
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        if not keepdims:
            diff = diff[:, -1]

        if self.sum_all_diffs:
            diff = diff.sum(dim=1)

        return diff


class HierarchicalObjective:
    """Generic hierarchical planning objective.

    Composes per-level sub-objectives (ReprDistObjective, ProjectedDistObjective, etc.)
    and handles hierarchical plumbing (level routing, subgoal management, encoding to
    parent level).

    At level L (coarsest / start_level): minimize goal_weight * dist(pred, goal_L).
    At level ℓ < L with subgoals: minimize a mixed objective:
        subgoal_weight * dist(pred_encoded_to_{ℓ+1}, subgoal_{ℓ+1})
        + goal_weight * dist(pred, goal_ℓ)
    This ensures lower levels both follow the subgoal from the level above
    AND stay oriented toward the final goal in their own representation space.
    """

    def __init__(
        self,
        level_objectives: Dict[int, Callable],
        hierarchical_model: "HierarchicalJEPA",  # noqa: F821
        subgoal_weight: float = 1.0,
        goal_weight: float = 1.0,
        **kwargs,
    ):
        """Initialize hierarchical objective.

        Args:
            level_objectives: Dict mapping level -> sub-objective instance.
                Keys are level indices (1, 2, ..., L).
                Values are callable objectives (e.g., ReprDistObjective instances).
            hierarchical_model: The HierarchicalJEPA model for encoding.
            subgoal_weight: Weight for subgoal matching term at sub-levels.
            goal_weight: Weight for final goal matching term at all levels.
        """
        self.level_objectives = level_objectives
        self.model = hierarchical_model
        self.subgoal_weight = subgoal_weight
        self.goal_weight = goal_weight

        self._subgoals: Dict[int, torch.Tensor] = {}
        self._current_level: Optional[int] = None

    @property
    def num_levels(self) -> int:
        return self.model.num_levels

    @property
    def top_level(self) -> int:
        """The top (coarsest) level for planning.

        This is the maximum level in level_objectives, which may be less than
        num_levels if start_level < num_levels.
        """
        return (
            max(self.level_objectives.keys())
            if self.level_objectives
            else self.num_levels
        )

    def set_level(self, level: int):
        """Set the current level for planning.

        Args:
            level: Hierarchy level to plan at.
        """
        self._current_level = level

    def set_subgoals(self, level: int, subgoals: torch.Tensor):
        """Set intermediate subgoals from higher-level planning.

        Args:
            level: The level these subgoals are for.
            subgoals: Subgoal encodings [T_subgoal, D_{level+1}] or [B, D_{level+1}, T_subgoal, H, W].
        """
        self._subgoals[level] = subgoals

    def clear_subgoals(self):
        """Clear all subgoals (call when starting a new planning episode)."""
        self._subgoals = {}

    def _encode_to_parent_level(
        self, encodings: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Encode predictions from level ℓ to level ℓ+1 space.

        This projects predictions from a finer level to the next coarser level
        by applying E^{ℓ+1} (the encoder at level ℓ+1) with strided subsampling.

        Args:
            encodings: Predicted encodings at level ℓ [B, D_ℓ, T, H_ℓ, W_ℓ].
            level: Current level ℓ (must be < num_levels).

        Returns:
            Encodings in level ℓ+1 space [B, D_{ℓ+1}, T', H_{ℓ+1}, W_{ℓ+1}].
        """
        if level >= self.num_levels:
            raise ValueError(
                f"Cannot encode level {level} to parent (max level is {self.num_levels})"
            )

        stride = self.model.temporal_strides[level - 1]
        encodings_strided = encodings[:, :, ::stride]

        encoder_next = self.model.levels[level].encoder
        with torch.no_grad():
            encodings_next = encoder_next(encodings_strided)

        return encodings_next

    def _make_subgoal_objective(
        self, parent_obj: Callable, subgoal: torch.Tensor
    ) -> Callable:
        """Create a temporary objective for subgoal comparison.

        Uses the same class and distance/sum_all_diffs parameters as the
        parent level objective, but targets the subgoal encoding.

        Args:
            parent_obj: The objective instance from the parent (ℓ+1) level.
            subgoal: Subgoal encoding [B, D_{ℓ+1}, T, H_{ℓ+1}, W_{ℓ+1}].

        Returns:
            A callable objective targeting the subgoal.
        """
        distance = getattr(parent_obj, "distance", "l2")
        sum_all_diffs = getattr(parent_obj, "sum_all_diffs", True)

        if isinstance(parent_obj, ReprDistObjective):
            return ReprDistObjective(
                target_enc=subgoal,
                distance=distance,
                sum_all_diffs=sum_all_diffs,
            )
        elif isinstance(parent_obj, ProjectedDistObjective):
            return ProjectedDistObjective(
                projector=parent_obj.projector,
                target_enc=subgoal,
                distance=distance,
                sum_all_diffs=sum_all_diffs,
            )
        else:
            raise NotImplementedError(
                f"Subgoal objective creation not implemented for {type(parent_obj)}"
            )

    def __call__(
        self,
        encodings: torch.Tensor,
        level: Optional[int] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        """Compute objective at specified level.

        At the top level (start_level): goal_weight * dist(pred, goal).
        At sub-levels with subgoals: mixed objective
            subgoal_weight * dist(pred_in_parent_space, subgoal)
            + goal_weight * dist(pred, goal_at_this_level).

        Args:
            encodings: Predicted encodings at current level [B, D_ℓ, T, H_ℓ, W_ℓ].
            level: Level to compute objective for. Uses self._current_level if None.
            keepdims: If True, return [B, T], else return [B].

        Returns:
            Objective value [B, T] if keepdims else [B].
        """
        if level is None:
            level = self._current_level
        if level is None:
            raise ValueError("Level must be specified or set via set_level()")

        # Goal cost at this level (used at all levels)
        goal_cost = self.level_objectives[level](encodings, keepdims=keepdims)

        if level == self.top_level or level not in self._subgoals:
            return self.goal_weight * goal_cost

        # Sub-level with subgoals: mixed objective
        encodings_at_parent = self._encode_to_parent_level(encodings, level)
        parent_obj = self.level_objectives[level + 1]
        subgoal_obj = self._make_subgoal_objective(parent_obj, self._subgoals[level])
        subgoal_cost = subgoal_obj(encodings_at_parent, keepdims=keepdims)

        return self.subgoal_weight * subgoal_cost + self.goal_weight * goal_cost

    def compute_at_level(
        self, encodings: torch.Tensor, level: int, keepdims: bool = False
    ) -> torch.Tensor:
        """Convenience method to compute objective at a specific level."""
        return self(encodings, level=level, keepdims=keepdims)
