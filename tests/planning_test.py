import os
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from eb_jepa.planning.agent import GCAgent
from eb_jepa.planning.evaluation import main_eval
from eb_jepa.planning.objectives import ReprDistObjective
from eb_jepa.planning.optimizers import CEMPlanner, MPPIPlanner, PlanningResult


def test_cem_planner():
    """Test the CEMPlanner class with various scenarios."""

    # Create a mock unroll function
    def mock_unroll(obs_init, actions):
        batch_size = actions.shape[0]
        time_steps = actions.shape[2]
        return torch.zeros(batch_size, 16, time_steps, 8, 8, device=actions.device)

    def mock_objective(predicted_states):
        return torch.sum(predicted_states, dim=(1, 2, 3, 4))

    planner = CEMPlanner(
        unroll=mock_unroll,
        n_iters=3,
        num_samples=10,
        plan_length=5,
        action_dim=2,
        var_scale=1.0,
        num_elites=2,
        decode_each_iteration=False,
    )

    planner.set_objective(mock_objective)

    obs_init = torch.zeros(1, 16, 1, 8, 8, device=planner.device)
    result = planner.plan(obs_init)

    # Check return type and shape
    assert isinstance(result, PlanningResult), "Should return a PlanningResult"
    assert result.actions.shape == (
        5,
        2,
    ), f"Actions should have shape (1, 2) but have shape {result.actions.shape}"
    assert isinstance(result.losses, torch.Tensor), "Losses should be a tensor"
    assert isinstance(
        result.prev_elite_losses_mean, torch.Tensor
    ), "Elite means should be a tensor"
    assert isinstance(
        result.prev_elite_losses_std, torch.Tensor
    ), "Elite stds should be a tensor"

    # Test 2: Planning with steps_left parameter
    result_with_steps = planner.plan(obs_init, steps_left=2)
    assert result_with_steps.actions.shape == (
        2,
        2,
    ), f"Actions should adapt to steps_left but have shape {result_with_steps.actions.shape}"

    # Test 3: Verify cost function behavior
    actions_batch = torch.randn(3, 2, 5)  # B, A, T
    cost = planner.cost_function(actions_batch, obs_init)
    assert cost.shape == (3,), "Cost should have shape (batch_size, 1)"


def test_mppi_planner():
    """Test the MPPIPlanner class with various scenarios."""

    def mock_unroll(obs_init, actions):
        batch_size = actions.shape[0]
        time_steps = actions.shape[2]
        return torch.zeros(batch_size, 16, time_steps, 8, 8, device=actions.device)

    def mock_objective(predicted_states):
        return torch.sum(predicted_states, dim=(1, 2, 3, 4))

    planner = MPPIPlanner(
        unroll=mock_unroll,
        n_iters=3,
        num_samples=10,
        plan_length=5,
        action_dim=2,
        max_std=1.0,
        num_elites=2,
        temperature=0.005,
        decode_each_iteration=False,
    )
    planner.set_objective(mock_objective)

    obs_init = torch.zeros(1, 16, 1, 8, 8, device=planner.device)
    result = planner.plan(obs_init)

    assert isinstance(result, PlanningResult)
    assert result.actions.shape == (5, 2)
    assert isinstance(result.losses, torch.Tensor)

    # Test with steps_left
    result_with_steps = planner.plan(obs_init, steps_left=2)
    assert result_with_steps.actions.shape == (2, 2)


def test_repr_target_dist_objective():
    """Test the ReprDistObjective class."""
    # Create mock target representation
    target_repr = torch.ones(1, 8, 1, 8, 8)  # B, C, T, H, W

    # Initialize objective
    objective = ReprDistObjective(target_repr)

    # Test objective calculation
    predicted_repr = torch.zeros(2, 8, 5, 8, 8)  # B, C, T, H, W
    cost = objective(predicted_repr)

    # Check output
    assert isinstance(cost, torch.Tensor), "Should return a tensor"
    assert cost.shape == (2,), "Should return one cost per batch"
    assert torch.all(cost > 0), "Distance to non-matching target should be positive"

    # Test with matching representation
    matching_repr = torch.ones(1, 8, 5, 8, 8)  # B, C, T, H, W
    matching_cost = objective(matching_repr)
    assert (
        matching_cost.item() < cost[0].item()
    ), "Cost should be lower for matching repr"


@patch("eb_jepa.planning.agent.CEMPlanner")
def test_gc_agent(mock_cem_planner):
    """Test the GCAgent class."""
    mock_model = Mock()
    mock_model.encode = Mock(return_value=torch.zeros(1, 8, 1, 8, 8))
    mock_model.unroll = Mock(return_value=(torch.zeros(10, 8, 6, 8, 8), None, None))
    param = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters = Mock(side_effect=lambda: iter([param]))
    mock_model.predictor = None

    mock_preprocessor = Mock()
    mock_preprocessor.normalize_obs = Mock(side_effect=lambda x: x)
    mock_preprocessor.unnormalize_obs = Mock(side_effect=lambda x: x)
    mock_preprocessor.normalize_proprios = Mock(side_effect=lambda x: x)
    mock_preprocessor.action_mean = None

    planning_result = PlanningResult(
        actions=torch.zeros(6, 2),
        losses=torch.zeros(10),
        prev_elite_losses_mean=torch.zeros(5),
        prev_elite_losses_std=torch.zeros(5),
    )

    mock_planner_instance = Mock()
    mock_planner_instance.plan = Mock(return_value=planning_result)
    mock_planner_instance.objective = None
    mock_planner_instance.set_objective = Mock(
        side_effect=lambda obj: setattr(mock_planner_instance, "objective", obj)
    )
    mock_cem_planner.return_value = mock_planner_instance

    # Patch planner_name_map so the mock planner is actually used
    with patch.dict(
        "eb_jepa.planning.agent.planner_name_map",
        {"cem": mock_cem_planner},
    ):
        from omegaconf import OmegaConf

        plan_cfg = OmegaConf.create(
            {
                "planner": {
                    "planner_name": "cem",
                    "n_iters": 3,
                    "num_samples": 10,
                    "plan_length": 5,
                    "num_elites": 2,
                    "var_scale": 1.0,
                    "decode_each_iteration": False,
                    "num_act_stepped": 1,
                    "planning_objective": {
                        "objective_type": "repr_dist",
                        "sum_all_diffs": True,
                    },
                },
                "ctxt_window_time": 2,
                "logging": {"tqdm_silent": False, "verbose": False},
            }
        )

        agent = GCAgent(
            mock_model,
            action_dim=2,
            plan_cfg=plan_cfg,
            preprocessor=mock_preprocessor,
        )

        # Test: Setting a goal
        goal_state = torch.randn(1, 8, 8)
        goal_position = torch.tensor([4.0, 4.0])
        agent.set_goal(goal_state, goal_position)

        assert agent.goal_position is goal_position, "Goal position should be stored"
        assert (
            mock_model.encode.called
        ), "Model encode should be called when setting goal"
        assert agent.objective is not None, "Objective should be set"
        assert agent.planner.objective is not None, "Planner's objective should be set"

        # Test: Acting
        obs = torch.randn(1, 8, 1, 8, 8)
        action = agent.act(obs, steps_left=10)

        assert mock_planner_instance.plan.called, "Planner's plan should be called"
        assert isinstance(action, np.ndarray), "Should return a numpy array"
        assert action.shape == (1, 2), "Should return an action with shape (1, 2)"

        # Test: Unroll function
        obs_init = torch.randn(1, 8, 1, 8, 8)
        actions = torch.randn(5, 2, 6)
        states = agent.unroll(obs_init, actions)

        assert mock_model.unroll.called, "Model's unroll should be called"
        assert isinstance(states, torch.Tensor), "Should return a tensor"


@patch("eb_jepa.planning.evaluation.GCAgent")
def test_main_eval(mock_gc_agent):
    """Test the main_eval function."""
    num_episodes = 2

    # Create mocks
    mock_model = Mock()
    mock_env = Mock()
    mock_env.reset = Mock(
        return_value=(
            torch.zeros((2, 65, 65)),  # observation
            {
                "target_obs": torch.zeros((2, 65, 65)),
                "target_position": np.array([10.0, 10.0]),
            },  # info
        )
    )
    mock_env.step = Mock(
        return_value=(
            torch.zeros((2, 65, 65)),  # observation
            0.0,  # reward
            False,  # done
            False,  # truncated
            {
                "dot_position": np.array([5.0, 5.0]),
                "target_position": np.array([10.0, 10.0]),
                "target_obs": torch.zeros((2, 65, 65)),
            },  # info
        )
    )
    mock_env.eval_state = Mock(return_value={"success": False, "state_dist": 5.0})
    mock_env.n_steps = 10
    mock_env.n_allowed_steps = 10
    mock_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    mock_env.normalizer = Mock()
    del mock_env.sample_random_init_goal_states

    # Mock env creator function
    def mock_env_creator():
        return mock_env

    # Mock agent instance
    mock_agent_instance = Mock()
    mock_agent_instance.act = Mock(return_value=torch.tensor([[0.1, 0.2]]))
    mock_agent_instance.device = torch.device("cpu")
    mock_agent_instance.decode_each_iteration = False
    mock_agent_instance.num_act_stepped = 1
    # Set up proper tensor values for agent attributes used by analyze_distances
    mock_agent_instance.goal_position = torch.tensor([10.0, 10.0])
    mock_agent_instance.goal_state = torch.zeros(2, 65, 65)
    mock_agent_instance.normalizer = mock_normalizer = Mock()
    mock_normalizer.normalize_state = Mock(side_effect=lambda x: x)
    mock_agent_instance.model = Mock()
    mock_agent_instance.model.encode = Mock(return_value=torch.zeros(1, 8, 1, 4, 4))
    mock_agent_instance.objective = Mock(return_value=torch.zeros(1))
    mock_agent_instance._prev_losses = None
    mock_agent_instance._prev_elite_losses_mean = None
    mock_agent_instance._prev_elite_losses_std = None
    mock_agent_instance._prev_losses_per_level = {}
    mock_agent_instance._is_hierarchical = False
    mock_gc_agent.return_value = mock_agent_instance

    # Create plan config

    plan_cfg = {
        "planner": {
            "planner_name": "cem",
            "n_iters": 3,
            "num_samples": 10,
            "plan_length": 5,
            "num_elites": 20,
            "var_scale": 1.0,
            "num_act_stepped": 1,
            "planning_objective": {"objective_type": "repr_dist"},
        },
        "task_specification": {"goal_source": "random_state", "obs": "rgb"},
        "meta": {"eval_episodes": 2},
        "logging": {"tqdm_silent": False},
    }

    # Run evaluation with fewer episodes for testing
    os.makedirs("./tests/logs/", exist_ok=True)
    results = main_eval(
        plan_cfg=plan_cfg,
        model=mock_model,
        env_creator=mock_env_creator,
        eval_folder=Path("./tests/logs/"),
        num_episodes=num_episodes,
    )

    # Verify results
    assert "success_rate" in results, "Results should include success rate"
    assert "mean_state_dist" in results, "Results should include mean distance"
    assert isinstance(results["success_rate"], float), "Success rate should be a float"
    assert isinstance(
        results["mean_state_dist"], float
    ), "Mean distance should be a float"
    assert (
        mock_env.reset.call_count == 1 + num_episodes
    ), f"Environment should be reset for each episode but {mock_env.reset.call_count=}"
    assert (
        mock_agent_instance.set_goal.call_count == num_episodes
    ), "Goal should be set for each episode"


def test_planning_integration():
    """Test integration of planning components with a simplified model."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones(1))

        def encode(self, x):
            # Simple encoding function that returns a zeroed tensor with correct shape
            B, C, T, H, W = x.shape
            return torch.zeros(B, 8, T, 4, 4, device=x.device)

        def unroll(
            self,
            obs,
            actions,
            nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=False,
            return_all_steps=False,
        ):
            # Simpler unroll function that doesn't depend on complex tensor shapes
            B = obs.shape[0]
            return torch.ones(B, 8, nsteps, 4, 4, device=obs.device), None, None

    # Test full planning episode
    # Create model and move to appropriate device
    model = DummyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a mock preprocessor
    mock_preprocessor = Mock()
    mock_preprocessor.normalize_obs = Mock(side_effect=lambda x: x)
    mock_preprocessor.unnormalize_obs = Mock(side_effect=lambda x: x)
    mock_preprocessor.normalize_proprios = Mock(side_effect=lambda x: x)
    mock_preprocessor.action_mean = None
    mock_preprocessor.action_std = None

    # Create plan config
    from omegaconf import OmegaConf

    plan_cfg = OmegaConf.create(
        {
            "planner": {
                "planner_name": "cem",
                "n_iters": 3,
                "num_samples": 10,
                "plan_length": 4,
                "num_elites": 2,
                "var_scale": 1.0,
                "decode_each_iteration": False,
                "num_act_stepped": 1,
                "planning_objective": {
                    "objective_type": "repr_dist",
                    "sum_all_diffs": True,
                },
            },
            "ctxt_window_time": 2,
            "logging": {"tqdm_silent": False, "verbose": False},
        }
    )

    # Initialize agent
    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_cfg,
        preprocessor=mock_preprocessor,
    )

    # Set goal
    goal_state = torch.ones(1, 4, 4).to(device)
    goal_position = torch.tensor([1.0, 1.0]).to(device)
    agent.set_goal(goal_state, goal_position)

    # Create observation
    obs = torch.zeros(1, 1, 1, 4, 4).to(device)

    # Test planning and action selection
    action = agent.act(obs, steps_left=8)

    # Verify action
    assert isinstance(action, np.ndarray), "Should return a numpy array"
    assert action.shape == (1, 2), "Should return appropriate action shape"

    # Since our dummy model favors larger actions, the planned actions should have magnitude > 0
    assert np.sum(np.abs(action)) > 0, "Agent should select non-zero actions"
