# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2
# licensed under the MIT License

import logging
import warnings
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

from eb_jepa.envs.wrappers.multitask import MultitaskWrapper
from eb_jepa.envs.wrappers.pixels import PixelWrapper
from eb_jepa.envs.wrappers.tensor import TensorWrapper

# Lazy imports for environments with heavy dependencies (mujoco-py, robocasa, etc.)
_lazy_env_cache = {}

_LAZY_ENV_CONFIG = {
    "maze": ("eb_jepa.envs.pointmaze_gym_wrap", "mujoco (>=3.x)"),
    "robocasa": ("eb_jepa.envs.robocasa", "RoboCasa"),
}


def _lazy_make_env(env_key, cfg):
    """Lazily import and call make_env for environments with optional dependencies."""
    if env_key not in _lazy_env_cache:
        module_path, install_name = _LAZY_ENV_CONFIG[env_key]
        try:
            import importlib

            _lazy_env_cache[env_key] = importlib.import_module(module_path).make_env
        except Exception as e:
            raise ImportError(
                f"Missing dependencies for {install_name}. See README.md. Error: {e}"
            ) from e
    return _lazy_env_cache[env_key](cfg)


# These environments have minimal dependencies and can be imported eagerly
from eb_jepa.envs.droid_dset_dummy_env import make_env as make_droid_dset_dummy_env
from eb_jepa.envs.pusht_gym_wrap import make_env as make_pusht_env
from eb_jepa.envs.two_rooms_env import make_env as make_wall_env

warnings.filterwarnings("ignore", category=DeprecationWarning)


class _PlanningEnvAdapter:
    """Wraps PushT/PointMaze domain wrappers for planning eval.

    Renders RGB observations as ``[C, H, W]`` float32 torch tensors and
    ensures ``info["state"]`` is always present.

    Args:
        env: Inner domain wrapper (``PushTWrapper`` or ``PointMazeWrapper``).
        n_allowed_steps: Maximum environment steps per episode.
        render_size: Pixel side length for rendered observations.
        render_fn: Callable that returns ``[H, W, C]`` uint8 numpy from the env.
    """

    def __init__(self, env, n_allowed_steps: int, render_size: int, render_fn=None):
        self.env = env
        self.n_allowed_steps = n_allowed_steps
        self.render_size = render_size
        self._render_fn = render_fn or (lambda: env.render(mode="rgb_array"))

    def _render_tensor(self) -> torch.Tensor:
        img = self._render_fn()  # [H, W, C] uint8 numpy
        return (
            torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
        )  # [C, H, W]

    @property
    def action_space(self):
        return self.env.action_space

    def eval_state(self, goal_state, cur_state):
        return self.env.eval_state(goal_state, cur_state)

    def sample_random_init_goal_states(self, seed):
        return self.env.sample_random_init_goal_states(seed)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        if "state" not in info:
            info["state"] = np.asarray(raw_obs)
        return self._render_tensor(), info

    def step(self, action):
        raw_obs, reward, done, truncated, info = self.env.step(action)
        if "state" not in info:
            info["state"] = np.asarray(raw_obs)
        return self._render_tensor(), reward, done, truncated, info

    def prepare(self, seed, init_state, env_info=None):
        raw_obs, info = self.env.prepare(seed, init_state, env_info=env_info)
        if "state" not in info:
            info["state"] = np.asarray(raw_obs)
        return self._render_tensor(), info

    def update_env(self, env_info):
        if hasattr(self.env, "update_env"):
            self.env.update_env(env_info)

    def render_at_position(self, pos) -> torch.Tensor:
        """Render observation with agent at ``pos`` without modifying env state.

        Delegates to the inner env's ``render_at_position`` and converts the
        result to a ``[C, H, W]`` float tensor in ``[0, 1]``.
        """
        img = self.env.render_at_position(pos)  # [H, W, C] uint8 numpy
        return (
            torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
        )  # [C, H, W]


def make_env_creator(
    env_name: str, env_config: dict, eval_env_cfg: dict = None
) -> Callable:
    """Return a callable that creates the appropriate environment for planning eval.

    Args:
        env_name: Dataset / environment name (e.g. "two_rooms", "pusht").
        env_config: Merged data config dict (from ``init_data``).
        eval_env_cfg: Extra kwargs forwarded to the environment constructor.

    Returns:
        A zero-argument callable that instantiates and returns an env.
    """
    eval_env_cfg = eval_env_cfg or {}

    def _creator():
        if env_name == "two_rooms":
            from eb_jepa.data.two_rooms_dset import (
                WallDatasetConfig,
                update_config_from_yaml,
            )
            from eb_jepa.envs.two_rooms_env import DotWall

            wall_config = update_config_from_yaml(WallDatasetConfig, env_config)
            return DotWall(config=wall_config, **eval_env_cfg)
        elif env_name == "pusht":
            from eb_jepa.envs.pusht_env.pusht_env import PushTEnv
            from eb_jepa.envs.pusht_gym_wrap import PushTWrapper

            render_size = env_config.get("img_size", 96)
            base_env = PushTEnv(
                with_velocity=True, with_target=True, render_size=render_size
            )
            wrapper = PushTWrapper(base_env)
            n_steps = eval_env_cfg.get("n_allowed_steps", 200)
            return _PlanningEnvAdapter(
                wrapper,
                n_allowed_steps=n_steps,
                render_size=render_size,
                render_fn=lambda: wrapper.render(mode="rgb_array"),
            )
        elif env_name == "pointmaze":
            from eb_jepa.envs.pointmaze_env.maze_model import MazeEnv
            from eb_jepa.envs.pointmaze_gym_wrap import PointMazeWrapper

            render_size = env_config.get("img_size", 224)
            base_env = MazeEnv(
                reward_type="sparse",
                reset_target=False,
                ref_min_score=23.85,
                ref_max_score=161.86,
                dataset_url="http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
            )
            wrapper = PointMazeWrapper(base_env)
            n_steps = eval_env_cfg.get("n_allowed_steps", 300)
            return _PlanningEnvAdapter(
                wrapper,
                n_allowed_steps=n_steps,
                render_size=render_size,
                render_fn=lambda: wrapper.render(width=render_size, height=render_size),
            )
        elif env_name in ("droid", "franka_custom"):
            from eb_jepa.envs.droid_dset_dummy_env import DroidDummyWrapper

            return DroidDummyWrapper(**eval_env_cfg)
        else:
            raise ValueError(f"Unknown environment: {env_name}")

    return _creator


def make_multitask_env(cfg):
    """
    Make a multi-task environment.
    """
    logger.info(f"Creating multi-task environment with tasks: {cfg.tasks}")
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.task_specification.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg):
    """
    Flexible interface to build environments.
    """
    import gym

    gym.logger.set_level(40)
    if cfg.task_specification.goal_source in ["dset", "random_action"]:
        if cfg.task_specification.task == "droid-base":
            cfg.task_specification.max_episode_steps = cfg.planner.horizon
        elif cfg.task_specification.task.startswith("robocasa"):
            pass
        else:  # pusht
            cfg.task_specification.max_episode_steps = (
                cfg.frameskip * cfg.task_specification.goal_H
            )
            cfg.task_specification.goal_max_episode_steps = (
                cfg.frameskip * cfg.task_specification.goal_H
            )
    elif cfg.task_specification.goal_source == "random_state":
        # TODO: Hardcoded for now, improve
        cfg.task_specification.max_episode_steps = (
            cfg.frameskip * cfg.task_specification.goal_H
        )
    else:
        if cfg.task_specification.get("max_episode_steps", None) is None:
            cfg.task_specification.max_episode_steps = 100

    if cfg.task_specification.multitask:
        env = make_multitask_env(cfg)

    else:
        env = None
        if cfg.task_specification.task.startswith("pusht-"):
            env = make_pusht_env(cfg)
        elif cfg.task_specification.task.startswith("wall-"):
            env = make_wall_env(cfg)
        elif cfg.task_specification.task.startswith("maze-"):
            env = _lazy_make_env("maze", cfg)
        elif cfg.task_specification.task.startswith("robocasa-"):
            env = _lazy_make_env("robocasa", cfg)
        elif cfg.task_specification.task.startswith("droid-"):
            env = make_droid_dset_dummy_env(cfg)

        env = TensorWrapper(env)
        if cfg.task_specification.get("obs", "state") in ["rgb", "rgb_state"]:
            env = PixelWrapper(cfg, env)
    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        cfg.obs_shape = {
            cfg.task_specification.get("obs", "state"): env.observation_space.shape
        }
    if cfg.task_specification.get("obs", "state") == "rgb_state":
        cfg.obs_shape = {"state": [4], "rgb": cfg.obs_shape["rgb_state"]}

    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.meta.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
