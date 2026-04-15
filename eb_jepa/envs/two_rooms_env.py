# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from eb_jepa.data.preprocessor import Preprocessor
from eb_jepa.data.two_rooms_dset import (
    TWO_ROOMS_LOCATION_MEAN,
    TWO_ROOMS_LOCATION_STD,
    TWO_ROOMS_STATE_MEAN,
    TWO_ROOMS_STATE_STD,
    WallDatasetConfig,
    _normalize_two_rooms_obs,
    _unnormalize_two_rooms_obs,
    check_wall_intersect,
    generate_wall_layouts,
)

InfoType = Dict[str, Any]
ObsType = torch.Tensor


class DotWall(gym.Env):
    def __init__(
        self,
        config: WallDatasetConfig,
        rng: Optional[np.random.Generator] = None,
        cross_wall: bool = True,
        level: str = "medium",
        n_steps: int = 200,
        max_step_norm: float = 2.45,
        normalize: bool = True,
        n_allowed_steps: int = 200,
    ):
        super().__init__()
        self.config = config
        self.border_wall_loc = config.border_wall_loc
        self.dot_std = config.dot_std
        self.padding = self.dot_std * 2
        self.border_padding = self.border_wall_loc - 1 + self.padding
        self.wall_width = config.wall_width
        self.door_space = config.door_space
        self.wall_padding = config.wall_padding
        self.img_size = config.img_size
        self.fix_wall = config.fix_wall
        self.action_step_mean = config.action_step_mean
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = config.device

        self.fix_wall_location = config.fix_wall_location
        self.fix_door_location = config.fix_door_location

        self.cross_wall = cross_wall
        self.level = level

        self.n_allowed_steps = n_allowed_steps
        self.n_steps = n_steps
        self.rng = rng or np.random.default_rng(0)

        self.action_space = gym.spaces.Box(
            low=-max_step_norm, high=max_step_norm, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, config.img_size, config.img_size),
            dtype=np.float32,
        )

        self.normalize = normalize
        if self.normalize:
            self.preprocessor = Preprocessor(
                action_mean=torch.zeros(2),
                action_std=torch.ones(2),
                state_mean=TWO_ROOMS_STATE_MEAN,
                state_std=TWO_ROOMS_STATE_STD,
                proprio_mean=TWO_ROOMS_LOCATION_MEAN,
                proprio_std=TWO_ROOMS_LOCATION_STD,
                normalize_obs_fn=_normalize_two_rooms_obs,
                unnormalize_obs_fn=_unnormalize_two_rooms_obs,
            )
        else:
            self.preprocessor = None

    def _get_normalized_obs(self, obs):
        """Normalize observation if needed."""
        if self.normalize:
            if obs.dtype != torch.float32:
                obs = obs.float()
            return _normalize_two_rooms_obs(obs)
        return obs

    def _get_normalized_location(self, location):
        if self.normalize:
            return self.preprocessor.normalize_proprios(location)
        return location

    @property
    def np_random(self):
        return self.rng

    def render(self):
        return self._get_obs()

    def reset(self, location=None) -> Tuple[ObsType, InfoType]:
        self.wall_x, self.hole_y = self._generate_wall()
        self.left_wall_x = self.wall_x - self.wall_width // 2
        self.right_wall_x = self.wall_x + self.wall_width // 2

        self.wall_img = self._render_walls(self.wall_x, self.hole_y)
        if location is None:
            self._generate_start_and_target()
        else:
            self.dot_position = location

        self.position_history = [self.dot_position]
        obs = self._render_dot_and_wall()
        info = self._build_info()
        return obs, info

    def eval_state(self, goal_dot_position, curr_dot_position, succes_treshold=4.5):
        """Evaluate whether the goal is reached.

        Threshold from https://github.com/gaoyuezhou/dino_wm/blob/main/env/wall/wall_env_wrapper.py
        Expects unnormalized locations.
        """
        state_dist = np.linalg.norm(goal_dot_position - curr_dot_position)
        success = state_dist < succes_treshold
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def _build_info(self) -> InfoType:
        return {
            "dot_position": self.dot_position,
            "target_position": self.target_position,
            "target_obs": self.get_target_obs(),
        }

    def _get_obs(self):
        return self._render_dot_and_wall()

    def get_target_obs(self):
        obs = self._render_dot_and_wall_target(self.target_position)
        return obs

    def step(self, action: np.array) -> Tuple[ObsType, float, bool, bool, InfoType]:
        """Take a step in the environment.

        Args:
            action: [A,] where A is the action dimension (2 for x and y)
        """
        action = torch.tensor(action, device=self.device)
        self.dot_position = self._calculate_next_position(action)
        self.position_history.append(self.dot_position)
        obs = self._render_dot_and_wall()
        done = (self.dot_position - self.target_position).pow(2).mean() < 1.0
        truncated = len(self.position_history) >= self.n_allowed_steps
        return obs, done, done, truncated, self._build_info()

    def step_multiple(self, actions: np.ndarray) -> Tuple[list, list, list, list, list]:
        """Process a sequence of actions at once.

        Args:
            actions: Array of shape [T, A] where T is number of timesteps
                and A is action dimension.

        Returns:
            Tuple of (observations, rewards, dones, truncateds, infos)
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device)

        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []

        for t in range(actions.shape[0]):
            action = actions[t]
            obs, reward, done, truncated, info = self.step(action)

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)

        return observations, rewards, dones, truncateds, infos

    def _calculate_next_position(self, action):
        next_dot_position = self._generate_transition(self.dot_position, action)
        intersect, intersect_w_noise = check_wall_intersect(
            self.dot_position,
            next_dot_position,
            self.wall_x,
            self.hole_y,
            wall_width=self.wall_width,
            door_space=self.door_space,
            border_wall_loc=self.border_wall_loc,
            img_size=self.img_size,
        )
        if intersect is not None:
            next_dot_position = intersect_w_noise
        return next_dot_position

    def _generate_transition(self, location, action):
        next_location = location + action
        return next_location

    def _generate_wall(self):
        if self.fix_wall:
            wall_loc = torch.tensor(self.fix_wall_location, device=self.device)
            door_loc = torch.tensor(self.fix_door_location, device=self.device)
        else:
            layouts, _ = generate_wall_layouts(self.config)

            layout_codes = list(layouts.keys())
            sampled_code = self.rng.choice(layout_codes)
            layout = layouts[sampled_code]

            wall_loc = torch.tensor(
                layout["wall_pos"], device=self.device, dtype=torch.float32
            )
            door_loc = torch.tensor(
                layout["door_pos"], device=self.device, dtype=torch.float32
            )

        return wall_loc, door_loc

    def _generate_start_and_target(self):
        n_steps = self.n_steps
        if self.cross_wall:
            if self.level == "easy":
                avg_dist_n_steps = n_steps * self.action_step_mean

                assert (
                    self.wall_padding - self.wall_width // 2 - self.border_wall_loc
                    >= math.ceil(avg_dist_n_steps * 3 / 4)
                )

                start_min_x = self.left_wall_x - math.ceil(avg_dist_n_steps * 3 / 4)
                start_max_x = self.left_wall_x - math.ceil(avg_dist_n_steps * 1 / 4)
                target_min_x = self.right_wall_x + math.ceil(avg_dist_n_steps * 1 / 4)
                target_max_x = self.right_wall_x + math.ceil(avg_dist_n_steps * 3 / 4)
                min_y = max(
                    self.hole_y - math.ceil(avg_dist_n_steps * 3 / 4),
                    self.border_padding,
                )
                max_y = min(
                    self.hole_y + math.ceil(avg_dist_n_steps * 3 / 4),
                    self.img_size - 1 - self.border_padding,
                )
            else:
                start_min_x = self.border_padding
                start_max_x = self.left_wall_x - self.padding
                target_min_x = self.right_wall_x + self.padding
                target_max_x = self.img_size - 1 - self.border_padding
                min_y, max_y = (
                    self.border_padding,
                    self.img_size - 1 - self.border_padding,
                )

            start_x = start_min_x + self.rng.random() * (start_max_x - start_min_x)
            target_x = target_min_x + self.rng.random() * (target_max_x - target_min_x)

            start_y = torch.tensor(
                min_y + self.rng.random() * (max_y - min_y), device=self.device
            )
            target_y = torch.tensor(
                min_y + self.rng.random() * (max_y - min_y), device=self.device
            )

            if self.rng.random() < 0.5:
                start_x, target_x = target_x, start_x

            self.dot_position = torch.stack([start_x, start_y])
            self.target_position = torch.stack([target_x, target_y])
        else:
            raise NotImplementedError("only cross_wall=True is implemented")

    def _render_walls(self, wall_loc, hole_loc):
        x = torch.arange(0, self.img_size, device=self.device)
        y = torch.arange(0, self.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

        half_width = self.wall_width // 2

        wall_mask = (grid_x >= (wall_loc - half_width)) & (
            grid_x <= (wall_loc + half_width)
        )

        door_mask = (hole_loc - self.door_space <= grid_y) & (
            grid_y <= hole_loc + self.door_space
        )

        res = wall_mask & ~door_mask
        res = res.float()

        border_wall_loc = self.border_wall_loc
        res[:, border_wall_loc - 1] = 1
        res[:, -border_wall_loc] = 1
        res[border_wall_loc - 1, :] = 1
        res[-border_wall_loc, :] = 1

        res = (res * 255.0).clamp(0, 255).to(torch.uint8)

        return res

    def _render_dot(self, location):
        """Render a dot at the given location.

        Args:
            location: Tensor with x,y coordinates of the dot.
                - shape [2,]: single location
                - shape [t, 2]: multiple timesteps

        Returns:
            - If location is [2,]: Tensor of shape [img_size, img_size]
            - If location is [t, 2]: Tensor of shape [t, img_size, img_size]
        """
        x = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)  # [img_size, img_size, 2]

        is_multi_timestep = location.dim() > 1

        if is_multi_timestep:
            t = location.shape[0]
            c = c.unsqueeze(0).expand(t, -1, -1, -1)  # [t, img_size, img_size, 2]
            location = location.unsqueeze(1).unsqueeze(1)  # [t, 1, 1, 2]
        else:
            location = location.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]

        img = (
            (
                torch.exp(
                    -(c - location).norm(dim=-1).pow(2)
                    / (2 * self.dot_std * self.dot_std)
                )
                * 255.0
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )

        if not is_multi_timestep:
            img = img.squeeze(0)

        return img

    def _render_dot_and_wall(self):
        dot_img = self._render_dot(self.dot_position)
        obs_output = torch.stack([dot_img, self.wall_img], dim=0)
        return obs_output

    def _render_dot_and_wall_target(self, location):
        dot_img = self._render_dot(location)
        obs_output = torch.stack([dot_img, self.wall_img], dim=0)
        return obs_output

    def render_at_position(self, pos: torch.Tensor) -> torch.Tensor:
        """Render observation with the dot at ``pos`` without modifying env state.

        Args:
            pos: 2D position tensor ``[2]``.

        Returns:
            Float tensor ``[C, H, W]`` in ``[0, 1]``.
        """
        return self._render_dot_and_wall_target(pos).float() / 255.0

    def coord_to_pixel(
        self, locations: torch.Tensor, wall_x=None, door_y=None
    ) -> torch.Tensor:
        """Render images with the walls and dots at the specified locations
        without modifying the env state.

        Args:
            locations: Tensor of shape (bs, t, 2) with x,y coordinates of the dots
            wall_x: Tensor of shape (bs,)
            door_y: Tensor of shape (bs,)

        Returns:
            Tensor of shape (bs, t, 2, img_size, img_size)
        """
        if not isinstance(locations, torch.Tensor):
            locations = torch.tensor(locations, device=self.device, dtype=torch.float32)

        bs, t, _ = locations.shape
        output = []

        for i in range(bs):
            if wall_x is not None and door_y is not None:
                curr_wall_x = wall_x[i] if wall_x.dim() > 0 else wall_x
                curr_door_y = door_y[i] if door_y.dim() > 0 else door_y
                curr_wall_img = self._render_walls(curr_wall_x, curr_door_y)
            elif not hasattr(self, "wall_img"):
                wall_x, hole_y = self._generate_wall()
                curr_wall_img = self._render_walls(wall_x, hole_y)
            else:
                curr_wall_img = self.wall_img

            dot_imgs = self._render_dot(locations[i])  # [t, img_size, img_size]

            wall_imgs = curr_wall_img.unsqueeze(0).expand(
                t, -1, -1
            )  # [t, img_size, img_size]

            obs = torch.stack(
                [dot_imgs, wall_imgs], dim=1
            )  # [t, 2, img_size, img_size]
            output.append(obs)

        batched_output = torch.stack(output, dim=0)

        return batched_output


def make_env(cfg):
    """Create a DotWall two-rooms environment from a planning config.

    Args:
        cfg: OmegaConf config with task_specification.env fields matching
            WallDatasetConfig parameters.

    Returns:
        TimeLimit-wrapped DotWall environment.
    """
    from eb_jepa.envs.wrappers.time_limit import TimeLimit

    env_cfg = cfg.task_specification.env
    wall_config = WallDatasetConfig(
        img_size=env_cfg.get("img_size", 64),
        dot_std=env_cfg.get("dot_std", 1.3),
        fix_wall=env_cfg.get("fix_wall", True),
        fix_wall_location=env_cfg.get("fix_wall_location", 32),
        fix_door_location=env_cfg.get("fix_door_location", 10),
        wall_width=env_cfg.get("wall_width", 3),
        door_space=env_cfg.get("door_space", 4),
        wall_padding=env_cfg.get("wall_padding", 20),
        door_padding=env_cfg.get("door_padding", 10),
        action_step_mean=env_cfg.get("action_step_mean", 1.0),
        border_wall_loc=env_cfg.get("border_wall_loc", 5),
    )

    env = DotWall(
        config=wall_config,
        cross_wall=env_cfg.get("cross_wall", True),
        level=env_cfg.get("level", "medium"),
        n_steps=env_cfg.get("n_steps", 200),
        max_step_norm=env_cfg.get("max_step_norm", 2.45),
        normalize=env_cfg.get("normalize", True),
        n_allowed_steps=cfg.task_specification.get("max_episode_steps", 200),
    )
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env
