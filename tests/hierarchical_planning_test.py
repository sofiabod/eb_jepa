"""Tests for hierarchical JEPA and hierarchical planning."""

from unittest.mock import Mock

import pytest
import torch

from eb_jepa.planning.agent import (
    GCAgent,
)
from eb_jepa.planning.optimizers import (
    H_PlanningResult,
    HierarchicalPlanner,
    Planner,
    PlanningResult,
)

# =============================================================================
# Tests for H_PlanningResult
# =============================================================================


class TestH_PlanningResult:
    """Tests for H_PlanningResult class."""

    def test_creation_with_dict(self):
        """Test creation with Dict[int, PlanningResult] keyed 1 to L."""
        level_results = {
            1: PlanningResult(
                actions=torch.zeros(5, 2),
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            ),
            2: PlanningResult(
                actions=torch.zeros(3, 2),
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            ),
        }
        result = H_PlanningResult(level_results=level_results)

        assert result.level_results is not None
        assert 1 in result.level_results
        assert 2 in result.level_results
        assert result.level_results[1].actions.shape == (5, 2)
        assert result.level_results[2].actions.shape == (3, 2)

    def test_actions_property_returns_level_1(self):
        """Test that actions property returns level_results[1]."""
        level_1_actions = torch.randn(5, 2)
        level_2_actions = torch.randn(3, 2)

        level_results = {
            1: PlanningResult(
                actions=level_1_actions,
                losses=torch.zeros(10),
            ),
            2: PlanningResult(
                actions=level_2_actions,
                losses=torch.zeros(10),
            ),
        }
        result = H_PlanningResult(level_results=level_results)

        assert torch.equal(result.actions, level_1_actions)
        assert not torch.equal(result.actions, level_2_actions)

    def test_error_when_level_1_missing(self):
        """Test error handling when Level 1 is missing."""
        level_results = {
            2: PlanningResult(
                actions=torch.zeros(3, 2),
                losses=torch.zeros(10),
            ),
        }
        result = H_PlanningResult(level_results=level_results)

        with pytest.raises(ValueError, match="No Level 1 result"):
            _ = result.actions

    def test_error_when_level_results_none(self):
        """Test error handling when level_results is None."""
        result = H_PlanningResult(level_results=None)

        with pytest.raises(ValueError, match="No Level 1 result"):
            _ = result.actions


# =============================================================================
# Tests for HierarchicalPlanner Initialization
# =============================================================================


class TestHierarchicalPlannerInitialization:
    """Tests for HierarchicalPlanner initialization."""

    def _create_mock_planner(self, plan_length=5):
        """Create a mock planner for testing."""
        mock_planner = Mock(spec=Planner)
        mock_planner.plan_length = plan_length
        mock_planner.plan = Mock(
            return_value=PlanningResult(
                actions=torch.zeros(plan_length, 2),
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            )
        )
        mock_planner.unroll = Mock(return_value=torch.zeros(1, 8, plan_length, 4, 4))
        mock_planner.set_objective = Mock()
        return mock_planner

    def test_level_planners_dict_keys_1_to_L(self):
        """Test that level_planners is a Dict[int, Planner] with keys 1 to L."""
        level_planners = {
            1: self._create_mock_planner(plan_length=5),
            2: self._create_mock_planner(plan_length=3),
            3: self._create_mock_planner(plan_length=2),
        }
        planner = HierarchicalPlanner(level_planners=level_planners)

        assert isinstance(planner.level_planners, dict)
        assert set(planner.level_planners.keys()) == {1, 2, 3}

    def test_min_level_is_1(self):
        """Test that min_level = 1."""
        level_planners = {
            1: self._create_mock_planner(),
            2: self._create_mock_planner(),
        }
        planner = HierarchicalPlanner(level_planners=level_planners)

        assert planner.min_level == 1

    def test_max_level_property(self):
        """Test max_level property for different configurations."""
        level_planners_2 = {
            1: self._create_mock_planner(),
            2: self._create_mock_planner(),
        }
        planner_2 = HierarchicalPlanner(level_planners=level_planners_2)
        assert planner_2.max_level == 2

        level_planners_3 = {
            1: self._create_mock_planner(),
            2: self._create_mock_planner(),
            3: self._create_mock_planner(),
        }
        planner_3 = HierarchicalPlanner(level_planners=level_planners_3)
        assert planner_3.max_level == 3


# =============================================================================
# Tests for HierarchicalPlanner Plan Method
# =============================================================================


class TestHierarchicalPlannerPlan:
    """Tests for HierarchicalPlanner.plan method."""

    def _create_mock_planner_with_unroll(self, plan_length=5, embed_dim=8):
        """Create a mock planner with unroll capability."""
        mock_planner = Mock(spec=Planner)
        mock_planner.plan_length = plan_length
        mock_planner.plan = Mock(
            return_value=PlanningResult(
                actions=torch.zeros(plan_length, 2),
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            )
        )
        mock_planner.unroll = Mock(
            return_value=torch.zeros(1, embed_dim, plan_length, 4, 4)
        )
        mock_planner.set_objective = Mock()
        return mock_planner

    def _create_mock_objective(self):
        """Create a mock objective that mimics HierarchicalObjective."""
        mock_objective = Mock()
        mock_objective.set_subgoals = Mock()
        mock_objective.clear_subgoals = Mock()
        mock_objective.target_encs = {
            1: torch.zeros(1, 8, 1, 4, 4),
            2: torch.zeros(1, 8, 1, 4, 4),
        }
        return mock_objective

    def test_returns_h_planning_result(self):
        """Test returns H_PlanningResult with correct structure."""
        mock_planner_1 = self._create_mock_planner_with_unroll()
        mock_planner_2 = self._create_mock_planner_with_unroll()

        level_planners = {1: mock_planner_1, 2: mock_planner_2}
        planner = HierarchicalPlanner(level_planners=level_planners)
        planner.set_objective(self._create_mock_objective())

        obs_init = torch.zeros(1, 3, 1, 8, 8)
        result = planner.plan(obs_init, steps_left=100)

        assert isinstance(result, H_PlanningResult)

    def test_level_results_has_keys_1_to_max_level(self):
        """Test level_results has keys 1 to max_level."""
        mock_planner_1 = self._create_mock_planner_with_unroll()
        mock_planner_2 = self._create_mock_planner_with_unroll()

        level_planners = {1: mock_planner_1, 2: mock_planner_2}
        planner = HierarchicalPlanner(level_planners=level_planners)
        planner.set_objective(self._create_mock_objective())

        obs_init = torch.zeros(1, 3, 1, 8, 8)
        result = planner.plan(obs_init, steps_left=100)

        assert set(result.level_results.keys()) == {1, 2}

    def test_top_down_planning_order(self):
        """Test top-down planning order (coarsest to finest)."""
        call_order = []

        mock_planner_1 = self._create_mock_planner_with_unroll()
        mock_planner_1.plan = Mock(
            side_effect=lambda *args, **kwargs: (
                call_order.append(1),
                PlanningResult(
                    actions=torch.zeros(5, 2),
                    losses=torch.zeros(10),
                ),
            )[1]
        )

        mock_planner_2 = self._create_mock_planner_with_unroll()
        mock_planner_2.plan = Mock(
            side_effect=lambda *args, **kwargs: (
                call_order.append(2),
                PlanningResult(
                    actions=torch.zeros(5, 2),
                    losses=torch.zeros(10),
                ),
            )[1]
        )

        level_planners = {1: mock_planner_1, 2: mock_planner_2}
        planner = HierarchicalPlanner(level_planners=level_planners)
        planner.set_objective(self._create_mock_objective())

        obs_init = torch.zeros(1, 3, 1, 8, 8)
        _ = planner.plan(obs_init, steps_left=100)

        # Level 2 (coarsest) should be called first, then Level 1 (finest)
        assert call_order == [2, 1]

    def test_actions_property_accesses_level_1(self):
        """Test actions property accesses Level 1."""
        level_1_actions = torch.randn(5, 2)
        level_2_actions = torch.randn(3, 2)

        mock_planner_1 = self._create_mock_planner_with_unroll()
        mock_planner_1.plan = Mock(
            return_value=PlanningResult(
                actions=level_1_actions,
                losses=torch.zeros(10),
            )
        )

        mock_planner_2 = self._create_mock_planner_with_unroll()
        mock_planner_2.plan = Mock(
            return_value=PlanningResult(
                actions=level_2_actions,
                losses=torch.zeros(10),
            )
        )

        level_planners = {1: mock_planner_1, 2: mock_planner_2}
        planner = HierarchicalPlanner(level_planners=level_planners)
        planner.set_objective(self._create_mock_objective())

        obs_init = torch.zeros(1, 3, 1, 8, 8)
        result = planner.plan(obs_init, steps_left=100)

        assert torch.equal(result.actions, level_1_actions)


# =============================================================================
# Tests for HierarchicalPlanner Extract Subgoals
# =============================================================================


class TestHierarchicalPlannerExtractSubgoals:
    """Tests for HierarchicalPlanner._extract_subgoals."""

    def test_single_subgoal_extracts_index_1(self):
        """Test single mode extracts index 1 (first predicted state, not initial state)."""
        mock_planner = Mock(spec=Planner)
        mock_planner.plan_length = 5

        level_planners = {1: mock_planner, 2: mock_planner}
        planner = HierarchicalPlanner(
            level_planners=level_planners, subgoal_mode="single"
        )

        high_traj = torch.randn(1, 8, 5, 4, 4)  # trajectory with T=5 states
        subgoal = planner._extract_subgoals(high_traj, source_level=2, target_level=1)

        # Single mode returns index 1 (first predicted state) with T=1
        expected = high_traj[:, :, 1:2]
        assert subgoal.shape == (1, 8, 1, 4, 4)
        assert torch.equal(subgoal, expected)


# =============================================================================
# Tests for GCAgent Hierarchical Mode
# =============================================================================


class TestGCAgentHierarchicalMode:
    """Tests for GCAgent hierarchical mode."""

    @staticmethod
    def _make_mock_model_and_preprocessor(num_levels=2):
        """Create a properly mocked model and preprocessor for GCAgent tests."""
        mock_model = Mock()
        mock_model.encode = Mock(return_value=torch.zeros(1, 8, 1, 8, 8))
        mock_model.unroll = Mock(return_value=(torch.zeros(1, 8, 6, 8, 8), None, None))
        mock_model.predictor = None
        param = torch.nn.Parameter(torch.zeros(1))
        mock_model.parameters = Mock(side_effect=lambda: iter([param]))

        # Build level mocks with minimal predictor structure
        class _Pred:
            pass

        levels = []
        for _ in range(num_levels):
            p = _Pred()
            proj = Mock()
            proj.in_features = 2
            p.action_proj = proj
            lvl = Mock()
            lvl.predictor = p
            levels.append(lvl)
        mock_model.levels = levels

        mock_preprocessor = Mock()
        mock_preprocessor.normalize_obs = Mock(side_effect=lambda x: x)
        mock_preprocessor.action_mean = None
        return mock_model, mock_preprocessor

    def test_is_hierarchical_flag_set_correctly(self):
        """Test _is_hierarchical flag is set correctly."""
        from omegaconf import OmegaConf

        mock_model, mock_preprocessor = self._make_mock_model_and_preprocessor(
            num_levels=2
        )

        flat_cfg = OmegaConf.create(
            {
                "planner": {
                    "type": "flat",
                    "planner_name": "cem",
                    "n_iters": 3,
                    "num_samples": 10,
                    "plan_length": 5,
                    "num_elites": 2,
                    "var_scale": 1.0,
                    "decode_each_iteration": False,
                    "num_act_stepped": 1,
                    "planning_objective": {"objective_type": "repr_dist"},
                },
                "ctxt_window_time": 1,
            }
        )
        agent_flat = GCAgent(
            mock_model, action_dim=2, plan_cfg=flat_cfg, preprocessor=mock_preprocessor
        )
        assert agent_flat._is_hierarchical is False

        # Hierarchical planner config
        hierarchical_cfg = OmegaConf.create(
            {
                "planner": {
                    "type": "hierarchical",
                    "base_planner": "mppi",
                    "n_iters": 3,
                    "num_samples": 10,
                    "max_std": 2,
                    "num_elites": 2,
                    "temperature": 0.005,
                    "decode_each_iteration": False,
                    "num_act_stepped": 1,
                    "planning_objective": {"objective_type": "repr_dist"},
                    "level_configs": {
                        "level_1_planner": {"plan_length": 5},
                        "level_2_planner": {"plan_length": 3},
                    },
                },
                "ctxt_window_time": 1,
                "logging": {"verbose": False},
            }
        )
        agent_hierarchical = GCAgent(
            mock_model,
            action_dim=2,
            plan_cfg=hierarchical_cfg,
            preprocessor=mock_preprocessor,
        )
        assert agent_hierarchical._is_hierarchical is True

    def test_create_hierarchical_planner_structure(self):
        """Test _create_hierarchical_planner creates proper dict structure."""
        from omegaconf import OmegaConf

        mock_model, mock_preprocessor = self._make_mock_model_and_preprocessor(
            num_levels=3
        )

        hierarchical_cfg = OmegaConf.create(
            {
                "planner": {
                    "type": "hierarchical",
                    "base_planner": "mppi",
                    "n_iters": 3,
                    "num_samples": 10,
                    "max_std": 2,
                    "num_elites": 2,
                    "temperature": 0.005,
                    "decode_each_iteration": False,
                    "num_act_stepped": 1,
                    "planning_objective": {"objective_type": "repr_dist"},
                    "level_configs": {
                        "level_1_planner": {"plan_length": 5},
                        "level_2_planner": {"plan_length": 3},
                        "level_3_planner": {"plan_length": 2},
                    },
                },
                "ctxt_window_time": 1,
                "logging": {"verbose": False},
            }
        )
        agent = GCAgent(
            mock_model,
            action_dim=2,
            plan_cfg=hierarchical_cfg,
            preprocessor=mock_preprocessor,
        )
        assert isinstance(agent.planner.level_planners, dict)
        assert set(agent.planner.level_planners.keys()) == {1, 2, 3}
        assert agent.planner.min_level == 1
        assert agent.planner.max_level == 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestHierarchicalIntegration:
    """Integration tests for hierarchical planning system."""

    def _create_mock_objective(self, embed_dim=8):
        """Create a mock objective that mimics HierarchicalObjective."""
        mock_objective = Mock()
        mock_objective.set_subgoals = Mock()
        mock_objective.clear_subgoals = Mock()
        mock_objective.target_encs = {
            1: torch.zeros(1, embed_dim, 1, 4, 4),
            2: torch.zeros(1, embed_dim, 1, 4, 4),
        }
        return mock_objective

    def test_full_hierarchical_planning_loop(self):
        """Create mock multi-level planners and run full hierarchical planning loop.

        This test verifies the HierarchicalPlanner structure and planning flow
        using a mock objective that mimics HierarchicalObjective.
        """
        level_1_actions = torch.randn(4, 2)
        level_2_actions = torch.randn(3, 2)

        mock_planner_1 = Mock(spec=Planner)
        mock_planner_1.plan_length = 4
        mock_planner_1.plan = Mock(
            return_value=PlanningResult(
                actions=level_1_actions,
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            )
        )
        mock_planner_1.unroll = Mock(return_value=torch.zeros(1, 8, 4, 4, 4))
        mock_planner_1.set_objective = Mock()

        mock_planner_2 = Mock(spec=Planner)
        mock_planner_2.plan_length = 3
        mock_planner_2.plan = Mock(
            return_value=PlanningResult(
                actions=level_2_actions,
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            )
        )
        mock_planner_2.unroll = Mock(return_value=torch.zeros(1, 8, 3, 4, 4))
        mock_planner_2.set_objective = Mock()

        level_planners = {1: mock_planner_1, 2: mock_planner_2}
        planner = HierarchicalPlanner(level_planners=level_planners)
        planner.set_objective(self._create_mock_objective())

        obs_init = torch.zeros(1, 3, 1, 8, 8)
        result = planner.plan(obs_init, steps_left=20)

        # Verify return structure
        assert isinstance(result, H_PlanningResult)
        assert 1 in result.level_results
        assert 2 in result.level_results

        # Verify executable actions come from Level 1
        assert torch.equal(result.actions, level_1_actions)
        assert result.actions.shape == (4, 2)

    def test_executable_actions_from_level_1(self):
        """Verify executable actions come from Level 1."""
        level_1_actions = torch.randn(5, 2)
        level_2_actions = torch.randn(3, 2)

        level_results = {
            1: PlanningResult(
                actions=level_1_actions,
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            ),
            2: PlanningResult(
                actions=level_2_actions,
                losses=torch.zeros(10),
                prev_elite_losses_mean=torch.zeros(5),
                prev_elite_losses_std=torch.zeros(5),
            ),
        }
        result = H_PlanningResult(level_results=level_results)

        executable = result.actions
        assert torch.equal(executable, level_1_actions)
        assert executable.shape[0] == 5  # Level 1's plan_length


# =============================================================================
# Tests for Hierarchical JEPA Model
# =============================================================================

# TODO: Add comprehensive hierarchical JEPA model tests
# The training integration tests demonstrate that the model works correctly:
# - Model initializes with 3 levels (512, 256, 128 dims)
# - Hierarchical encoding through temporal pooling works
# - Loss computation aggregates contributions from all levels
# - Training completes successfully with finite losses
#
# Future test additions needed:
# 1. TestHierarchicalJEPAInitialization: test model building from config
# 2. TestHierarchicalEncoding: test encode_hierarchical shapes at each level
# 3. TestHierarchicalEncoding: test T=1 goal encoding case
# 4. TestActionDownsampling: test action temporal downsampling
# 5. TestHierarchicalUnroll: test training unroll with loss computation
# 6. TestHierarchicalTrainingStep: integration test for full training step
