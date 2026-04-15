"""Two-rooms (wall + dot) dataset — consolidated from two_rooms/*.

Generates on-the-fly trajectories of a dot navigating a two-room
environment separated by a wall with a door.  Each ``__getitem__``
call returns a *single* trajectory in the unified 5-tuple format::

    obs, actions, states, reward, info

Where:
    obs     = {"visual": [T, C, H, W], "proprio": [T, 2]}
    actions = [T, 2]
    states  = [T, 2]
    reward  = None
    info    = {"wall_x": Tensor, "door_y": Tensor}
"""

import math
import random
from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch
from scipy.stats import truncnorm

from eb_jepa.data.preprocessor import Preprocessor
from eb_jepa.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Two-rooms normalization constants (computed from training data)
# ---------------------------------------------------------------------------

TWO_ROOMS_STATE_MEAN = torch.tensor([0.0026, 0.0989])
TWO_ROOMS_STATE_STD = torch.tensor([0.0369, 0.2986])
TWO_ROOMS_LOCATION_MEAN = torch.tensor([31.5863, 32.0618])
TWO_ROOMS_LOCATION_STD = torch.tensor([16.1025, 16.1353])


def _min_max_normalize(state: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization to [0, 1]."""
    if len(state.shape) >= 3:
        state = state - state.amin(dim=(-2, -1), keepdim=True)
        state = state / (state.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    else:
        state = state - state.amin(dim=-1, keepdim=True)
        state = state / (state.amax(dim=-1, keepdim=True) + 1e-6)
    return state


def _normalize_two_rooms_obs(state: torch.Tensor) -> torch.Tensor:
    """Min-max then z-score normalization for two-rooms visual observations."""
    state = _min_max_normalize(state)
    adapted_mean = TWO_ROOMS_STATE_MEAN.view(-1, 1, 1).to(state.device)
    adapted_std = TWO_ROOMS_STATE_STD.view(-1, 1, 1).to(state.device) + 1e-6
    ch = state.shape[-3]
    if ch < adapted_mean.shape[0] and not (adapted_mean.shape[0] % ch):
        adapted_mean = adapted_mean[:ch]
        adapted_std = adapted_std[:ch]
    return (state - adapted_mean) / adapted_std


def _unnormalize_two_rooms_obs(state: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_normalize_two_rooms_obs` (z-score only)."""
    adapted_mean = TWO_ROOMS_STATE_MEAN.view(-1, 1, 1).to(state.device)
    adapted_std = TWO_ROOMS_STATE_STD.view(-1, 1, 1).to(state.device)
    ch = state.shape[-3]
    if ch < adapted_mean.shape[0] and not (adapted_mean.shape[0] % ch):
        adapted_mean = adapted_mean[:ch]
        adapted_std = adapted_std[:ch]
    return state * adapted_std + adapted_mean


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def update_config_from_yaml(config_class, yaml_data: dict):
    """Create *config_class* using defaults, overriding with *yaml_data*."""
    config_field_names = {f.name for f in fields(config_class)}
    relevant = {k: v for k, v in yaml_data.items() if k in config_field_names}
    return config_class(**relevant)


# ---------------------------------------------------------------------------
# Geometry / sampling helpers
# ---------------------------------------------------------------------------


def sample_uniformly_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise uniform sample in [a_i, b_i]."""
    r = torch.FloatTensor(a.size()).uniform_(0, 1).to(a.device)
    return a + (b - a) * r


def sample_truncated_norm(
    upper_bound: torch.Tensor,
    lower_bound: torch.Tensor,
    mean: torch.Tensor,
    std: float = 1.4,
) -> torch.Tensor:
    """Element-wise truncated-normal sample."""
    ub = upper_bound.cpu().float().numpy()
    lb = lower_bound.cpu().float().numpy()
    mu = mean.cpu().float().numpy()
    samples = np.zeros_like(mu)
    for i in range(len(mu)):
        a, b = (lb[i] - mu[i]) / std, (ub[i] - mu[i]) / std
        samples[i] = truncnorm.rvs(a, b, loc=mu[i], scale=std)
    return torch.from_numpy(samples)


def generate_wall_layouts(wall_config):
    """Return ``(layouts, None)`` dicts keyed by layout code strings."""
    img_size = wall_config.img_size
    wall_padding = wall_config.wall_padding
    door_padding = wall_config.door_padding
    fix_wall = wall_config.fix_wall
    fix_door_location = wall_config.fix_door_location
    fix_wall_location = wall_config.fix_wall_location

    def extract_min_max(code):
        vals = [int(x) for x in code.split("-") if x]
        if len(vals) == 2:
            return list(range(vals[0], vals[1] + 1))
        if len(vals) == 1:
            return list(range(vals[0], vals[0] + 1))
        return []

    exclude_wall_train = extract_min_max(wall_config.exclude_wall_train)
    exclude_door_train = extract_min_max(wall_config.exclude_door_train)
    only_wall_val = extract_min_max(wall_config.only_wall_val)
    only_door_val = extract_min_max(wall_config.only_door_val)

    assert all(
        len(a) == 0
        for a in [exclude_wall_train, exclude_door_train, only_wall_val, only_door_val]
    ) or all(
        len(a) > 0
        for a in [exclude_wall_train, exclude_door_train, only_wall_val, only_door_val]
    ), "Arrays must all be empty or all be non-empty"

    if fix_wall:
        layouts = {
            f"v_wall{fix_wall_location}_door{fix_door_location}": {
                "type": "v",
                "wall_pos": fix_wall_location,
                "door_pos": fix_door_location,
            }
        }
        return layouts, None

    wall_x_values = list(range(wall_padding, img_size - wall_padding))
    door_y_values = list(range(door_padding, img_size - door_padding))
    layouts = {}
    other_layouts = {}

    for wall_pos in wall_x_values:
        for door_pos in door_y_values:
            code = f"wall{wall_pos}_door{door_pos}"
            to_exclude = (
                wall_pos in exclude_wall_train and door_pos in exclude_door_train
            )
            to_include_val = wall_pos in only_wall_val and door_pos in only_door_val

            if not to_exclude:
                layouts[f"v_{code}"] = {
                    "type": "v",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }
                layouts[f"h_{code}"] = {
                    "type": "h",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }
            if to_include_val:
                other_layouts[f"v_{code}"] = {
                    "type": "v",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }
                other_layouts[f"h_{code}"] = {
                    "type": "h",
                    "wall_pos": wall_pos,
                    "door_pos": door_pos,
                }

    if not wall_config.train and exclude_wall_train:
        return other_layouts, None
    return layouts, None


def check_vertical_wall_intersect(pos1, pos2, wall_x, hole_y, door_space):
    check = (torch.sign(pos1[0] - wall_x) * torch.sign(pos2[0] - wall_x)) <= 0.1
    if check:
        d = pos2 - pos1
        a = d[1] / d[0]
        b = pos1[1] - a * pos1[0]
        y = a * wall_x + b
        if hole_y is None or y < hole_y - door_space or y > hole_y + door_space:
            return torch.tensor([wall_x, y]).to(pos1.device)
        return None
    return None


def check_horizontal_wall_intersect(pos1, pos2, wall_y, hole_x, door_space):
    check = (torch.sign(pos1[1] - wall_y) * torch.sign(pos2[1] - wall_y)) <= 0.1
    if check:
        d = pos2 - pos1
        a = d[1] / d[0]
        b = pos1[1] - a * pos1[0]
        x = (wall_y - b) / a
        if hole_x is None or x < hole_x - door_space or x > hole_x + door_space:
            return torch.tensor([x, wall_y]).to(pos1.device)
        return None
    return None


def check_wall_intersect(
    pos1,
    pos2,
    wall_x,
    hole_y,
    wall_width,
    door_space,
    border_wall_loc,
    img_size,
    add_noise=True,
):
    """Check if segment pos1->pos2 intersects any wall.

    Returns:
        ``(intersect, intersect_w_noise)`` or ``(None, None)``.
    """
    left_wall_corner = wall_x - wall_width // 2
    right_wall_corner = wall_x + wall_width // 2
    door_bot, door_top = hole_y - door_space, hole_y + door_space

    if pos2[1] - pos1[1] > 0 and pos2[1] > door_top and pos1[1] < door_top:
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_top, None, door_space
        )
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            noise = torch.randn(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs() * -1
            return intersect, intersect + noise

    if pos2[1] - pos1[1] < 0 and pos2[1] < door_bot and pos1[1] > door_bot:
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_bot, None, door_space
        )
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            noise = torch.randn(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs()
            return intersect, intersect + noise

    left_wall, left_hole = border_wall_loc - 1, None
    right_wall, right_hole = img_size - border_wall_loc, None
    if wall_x > pos1[0]:
        right_wall, right_hole = wall_x - wall_width // 2, hole_y
    else:
        left_wall, left_hole = wall_x + wall_width // 2, hole_y

    top_wall = border_wall_loc - 1
    bot_wall = img_size - border_wall_loc

    vertical_intersect = check_vertical_wall_intersect(
        pos1, pos2, left_wall, left_hole, door_space
    )
    if vertical_intersect is None:
        vertical_intersect = check_vertical_wall_intersect(
            pos1, pos2, right_wall, right_hole, door_space
        )

    horizontal_intersect = check_horizontal_wall_intersect(
        pos1, pos2, top_wall, None, door_space
    )
    if horizontal_intersect is None:
        horizontal_intersect = check_horizontal_wall_intersect(
            pos1, pos2, bot_wall, None, door_space
        )

    if vertical_intersect is not None:
        sign = torch.sign(pos1[0] - vertical_intersect[0])
        vertical_noise = torch.randn(2, device=pos1.device) * 0.5
        vertical_noise[0] = vertical_noise[0].abs() * sign

    if horizontal_intersect is not None:
        sign = torch.sign(pos1[1] - horizontal_intersect[1])
        horizontal_noise = torch.randn(2, device=pos1.device) * 0.5
        horizontal_noise[1] = horizontal_noise[1].abs() * sign

    if vertical_intersect is not None and horizontal_intersect is not None:
        if torch.norm(pos1 - vertical_intersect) < torch.norm(
            pos1 - horizontal_intersect
        ):
            intersect, noise = vertical_intersect, vertical_noise
        else:
            intersect, noise = horizontal_intersect, horizontal_noise
    elif vertical_intersect is not None:
        intersect, noise = vertical_intersect, vertical_noise
    elif horizontal_intersect is not None:
        intersect, noise = horizontal_intersect, horizontal_noise
    else:
        return None, None

    intersect_w_noise = intersect + noise
    intersect_w_noise[0] = torch.clamp(
        intersect_w_noise[0], min=left_wall, max=right_wall
    )
    intersect_w_noise[1] = torch.clamp(intersect_w_noise[1], min=top_wall, max=bot_wall)
    if intersect_w_noise[0] <= left_wall:
        intersect_w_noise[0] = left_wall + 0.3
    if intersect_w_noise[0] >= right_wall:
        intersect_w_noise[0] = right_wall - 0.3
    if intersect_w_noise[1] <= top_wall:
        intersect_w_noise[1] = top_wall + 0.3
    if intersect_w_noise[1] >= bot_wall:
        intersect_w_noise[1] = bot_wall - 0.3
    return intersect, intersect_w_noise


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


@dataclass
class DotDatasetConfig:
    size: int = 10000
    val_size: int = 10000
    batch_size: int = 128
    dot_std: float = 1.3
    action_noise: float = 0.2
    action_angle_noise: float = 0.15
    action_step_mean: float = 1.0
    action_step_std: float = 0.4
    action_lower_bd: float = 0.2
    action_upper_bd: float = 1.8
    max_step: float = 1.0
    n_steps: int = 91
    img_size: int = 64
    train: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    repeat_actions: int = 1
    n_steps_reduce_factor: int = 1
    border_wall_loc: int = 5
    chunked_actions: bool = False
    normalize: bool = True


@dataclass
class WallDatasetConfig(DotDatasetConfig):
    fix_wall: bool = True
    fix_wall_batch_k: Optional[int] = None
    wall_padding: int = 20
    door_padding: int = 10
    wall_width: int = 3
    door_space: int = 4
    cross_wall_rate: float = 0.1
    expert_cross_wall_rate: float = 0.0
    wall_bump_rate: float = 0.0
    dup_traj_rate: float = 0.0
    expert_action_step_mean: float = 0.9
    expert_action_step_std: float = 0
    expert_action_lower_bd: float = 0.9
    expert_action_upper_bd: float = 0.9
    expert_traj_door_padding: float = 2
    exclude_wall_train: str = ""
    exclude_door_train: str = ""
    only_wall_val: str = ""
    only_door_val: str = ""
    fix_wall_location: Optional[int] = 32
    fix_door_location: Optional[int] = 10
    num_train_layouts: Optional[int] = -1
    image_based: bool = True
    sample_length: int = 17


# ---------------------------------------------------------------------------
# DotDataset base
# ---------------------------------------------------------------------------


class DotDataset(torch.utils.data.Dataset):
    """Base class for dot-navigation datasets with on-the-fly generation."""

    def __init__(self, config: DotDatasetConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.padding = config.border_wall_loc - 1
        self._preprocessor = (
            Preprocessor(
                action_mean=torch.zeros(2),
                action_std=torch.ones(2),
                state_mean=TWO_ROOMS_STATE_MEAN,
                state_std=TWO_ROOMS_STATE_STD,
                proprio_mean=TWO_ROOMS_LOCATION_MEAN,
                proprio_std=TWO_ROOMS_LOCATION_STD,
                normalize_obs_fn=_normalize_two_rooms_obs,
                unnormalize_obs_fn=_unnormalize_two_rooms_obs,
            )
            if config.normalize
            else None
        )

        self.action_dim = 2
        self.state_dim = 2
        self.proprio_dim = 2

    @property
    def preprocessor(self):
        """Preprocessor wrapping the normalizer for planning/eval code."""
        return self._preprocessor

    def __len__(self) -> int:
        return self.config.size

    def __getitem__(self, i: int):
        """Return a single trajectory in the unified 5-tuple format.

        Returns:
            ``(obs, actions, states, reward, info)`` where

            - obs: ``{"visual": [T, C, H, W], "proprio": [T, 2]}``
            - actions: ``[T, 2]``
            - states: ``[T, 2]``
            - reward: ``None``
            - info: ``{"wall_x": Tensor, "door_y": Tensor}``
        """
        return self.generate_multistep_sample()

    def render_location(self, locations: torch.Tensor) -> torch.Tensor:
        """Render Gaussian blobs at *locations*.

        Args:
            locations: ``[..., 2]`` (x, y) pixel coordinates.

        Returns:
            Image tensor ``[..., H, W]`` with uint8 Gaussian blobs.
        """
        sz = self.config.img_size
        x = torch.linspace(0, sz - 1, steps=sz, device=self.device)
        y = torch.linspace(0, sz - 1, steps=sz, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        c = c.view(*([1] * (len(locations.shape) - 1)), *c.shape).repeat(
            *locations.shape[:-1], *([1] * len(c.shape))
        )
        locations = locations.unsqueeze(-2).unsqueeze(-2)
        img = (
            (
                torch.exp(
                    -(c - locations).norm(dim=-1).pow(2)
                    / (2 * self.config.dot_std * self.config.dot_std)
                )
                * 255
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )
        return img

    def generate_state_and_actions(
        self, wall_locs=None, door_locs=None, size=None, n_steps: int = 17
    ):
        location = self.generate_state(wall_locs=wall_locs, door_locs=door_locs)
        actions, bias_angle = self.generate_actions(n_steps=n_steps)
        return location, actions, bias_angle

    def generate_state(self, wall_locs=None, door_locs=None, size=None):
        if size is None:
            size = 1
        effective_range = (self.config.img_size - 1) - 2 * self.padding
        location = (
            torch.rand(size=(size, 2), device=self.device) * effective_range
            + self.padding
        )
        left_walls = wall_locs - self.config.wall_width // 2
        right_walls = wall_locs + self.config.wall_width // 2
        door_top = door_locs + self.config.door_space
        door_bot = door_locs - self.config.door_space

        btw_walls = (location[:, 0] >= left_walls) & (location[:, 0] <= right_walls)
        not_btw_doors = (location[:, 1] < door_bot) | (location[:, 1] > door_top)
        inside_walls = btw_walls & not_btw_doors

        min_val = self.config.border_wall_loc - 1
        max_val = self.config.img_size - self.config.border_wall_loc

        if inside_walls.any():
            change_to_ef = (torch.rand(size) < 0.5).to(door_locs.device)
            new_x_left = sample_uniformly_between(
                torch.full((size,), min_val).to(door_locs.device), left_walls
            )
            new_x_right = sample_uniformly_between(
                right_walls, torch.full((size,), max_val).to(door_locs.device)
            )
            location[inside_walls & change_to_ef, 0] = new_x_left[
                inside_walls & change_to_ef
            ]
            location[inside_walls & ~change_to_ef, 0] = new_x_right[
                inside_walls & ~change_to_ef
            ]
        return location

    def sample_walls(self):
        return None

    def generate_multistep_sample(self):
        walls = self.sample_walls()
        start_location, actions, bias_angle = self.generate_state_and_actions(
            wall_locs=walls[0], door_locs=walls[1], n_steps=self.config.n_steps
        )
        if self.config.dup_traj_rate > 0:
            raise NotImplementedError()
        return self.generate_transitions(
            start_location, actions, bias_angle, walls=walls
        )

    @abstractmethod
    def generate_transitions(self, location, actions, bias_angle, walls=None):
        pass

    def generate_transition(self, location, action):
        return location + action

    def generate_actions(self, n_steps: int, bias_angle=None):
        """Generate a correlated random walk action sequence.

        Returns:
            ``(actions, bias_angle)`` with shapes ``[bs, n_steps-1, 2]``
            and ``[bs, 2]``.
        """
        if bias_angle is None:
            bias = torch.rand(1, device=self.device) * 2 * math.pi
            bias_angle = DotDataset.angle_to_vec(bias)
            bs = 1
        else:
            bias = self.vec_to_angle(bias_angle)
            bs = bias_angle.shape[0]

        concentration = 1 / self.config.action_angle_noise
        von_mises = torch.distributions.VonMises(concentration=concentration, loc=0.0)

        angles = [bias]
        for _ in range(1, n_steps - 1):
            noise = von_mises.sample((bs,)).to(self.device)
            angles.append((angles[-1] + noise).fmod(2 * torch.pi))
        angles = torch.stack(angles, dim=1)

        a = (
            self.config.action_lower_bd - self.config.action_step_mean
        ) / self.config.action_step_std
        b = (
            self.config.action_upper_bd - self.config.action_step_mean
        ) / self.config.action_step_std
        tn_dist = truncnorm(
            a,
            b,
            loc=self.config.action_step_mean,
            scale=self.config.action_step_std,
        )
        samples = tn_dist.rvs(size=(bs, n_steps - 1))
        steps = torch.tensor(samples, dtype=torch.float32, device=self.device)
        vecs = DotDataset.angle_to_vec(angles)
        actions = vecs * steps.unsqueeze(-1)
        return actions, bias_angle

    @staticmethod
    def angle_to_vec(a: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)

    @staticmethod
    def vec_to_angle(v: torch.Tensor) -> torch.Tensor:
        return torch.atan2(v[:, 1], v[:, 0])

    @staticmethod
    def xy_to_polar(v: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(v, dim=-1, keepdim=True)
        angle = torch.atan2(v[..., 1], v[..., 0]) % (2 * torch.pi)
        return torch.cat((angle.unsqueeze(-1), norm), dim=-1)

    @staticmethod
    def polar_to_xy(polar: torch.Tensor) -> torch.Tensor:
        angle, norm = polar[..., 0], polar[..., 1]
        return torch.stack((norm * torch.cos(angle), norm * torch.sin(angle)), dim=-1)


# ---------------------------------------------------------------------------
# WallDataset
# ---------------------------------------------------------------------------


class WallDataset(DotDataset):
    """Two-rooms dot + wall dataset with on-the-fly trajectory generation.

    Each ``__getitem__`` returns a single trajectory in the unified format:
    ``(obs, actions, states, reward, info)``.
    """

    def __init__(self, config: WallDatasetConfig):
        layouts, _other = generate_wall_layouts(config)
        self.layouts = layouts
        super().__init__(config)
        log.info(
            f"WallDataset: {len(self.layouts)} layout(s), "
            f"size={config.size}, sample_length={config.sample_length}"
        )

    def render_location(self, locations: torch.Tensor) -> torch.Tensor:
        return super().render_location(locations)

    # -- expert trajectory helpers ------------------------------------------

    def generate_actions_to_goal(self, start, goal, eps=1e-7):
        """Straight-line actions from *start* to *goal*. Returns ``[n, 2]``."""
        direction = goal - start
        direction_norm = direction.norm()
        if direction_norm == 0:
            return torch.empty(0, 2)
        unit_direction = direction / direction_norm

        a_coeff = (
            self.config.expert_action_lower_bd - self.config.expert_action_step_mean
        ) / (self.config.expert_action_step_std + eps)
        b_coeff = (
            self.config.expert_action_upper_bd - self.config.expert_action_step_mean
        ) / (self.config.expert_action_step_std + eps)
        tn_dist = truncnorm(
            a_coeff,
            b_coeff,
            loc=self.config.expert_action_step_mean,
            scale=self.config.expert_action_step_std,
        )

        actions = []
        current_position = start.clone()
        reached_goal = False
        while direction_norm > 0:
            if self.config.expert_action_lower_bd == self.config.expert_action_upper_bd:
                action_norm = self.config.expert_action_step_mean
            else:
                action_norm = tn_dist.rvs()
            if action_norm > direction_norm:
                action_norm = direction_norm
                reached_goal = True
            action = unit_direction * action_norm
            actions.append(action)
            current_position += action
            direction = goal - current_position
            direction_norm = direction.norm()
            if reached_goal:
                break

        if actions[-1].norm() < self.config.action_lower_bd:
            actions.pop()

        if actions:
            return torch.stack(actions)
        return torch.empty(0, 2).to(start.device)

    def generate_cross_wall_points(self, wall_locs, action_padding=0):
        """Points on opposite sides of the wall. Returns ``(left, right)``."""
        bs = wall_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2
        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01

        left_x = sample_uniformly_between(
            torch.full((bs,), min_val, device=wall_locs.device),
            left_wall_locs - action_padding,
        )
        right_x = sample_uniformly_between(
            right_wall_locs + action_padding,
            torch.full((bs,), max_val, device=wall_locs.device),
        )
        bnd_lo = torch.full((bs,), min_val, device=wall_locs.device)
        bnd_hi = torch.full((bs,), max_val, device=wall_locs.device)
        left_y = sample_uniformly_between(bnd_lo, bnd_hi)
        right_y = sample_uniformly_between(bnd_lo, bnd_hi)
        left_pos = torch.stack([left_x, left_y]).transpose(0, 1)
        right_pos = torch.stack([right_x, right_y]).transpose(0, 1)
        return left_pos, right_pos

    def generate_expert_cross_wall_state_and_actions(
        self, wall_locs=None, door_locs=None, n_steps=17
    ):
        bs = wall_locs.size(0)
        door_top = door_locs + self.config.door_space
        door_bot = door_locs - self.config.door_space

        seg_length = self.config.l2_step_skip * self.config.expert_action_step_mean
        seg_left_x = wall_locs - seg_length / 2
        seg_right_x = wall_locs + seg_length / 2
        seg_top_y = door_top - self.config.expert_traj_door_padding
        seg_bot_y = door_bot + self.config.expert_traj_door_padding
        seg_y = sample_uniformly_between(seg_bot_y, seg_top_y)

        seg_left = torch.stack((seg_left_x, seg_y), dim=1)
        seg_right = torch.stack((seg_right_x, seg_y), dim=1)

        start_pos, goal_pos = self.generate_cross_wall_points(wall_locs)

        half_bs = bs // 2
        start_pos[half_bs:], goal_pos[half_bs:] = (
            goal_pos[half_bs:],
            start_pos[half_bs:].clone(),
        )
        seg_left[half_bs:], seg_right[half_bs:] = (
            seg_right[half_bs:],
            seg_left[half_bs:].clone(),
        )

        def in_middle(x, y, left, right, top, bot):
            return left <= x <= right and bot <= y <= top

        expert_actions = torch.zeros((bs, self.config.n_steps - 1, 2)).to(
            wall_locs.device
        )

        for i in range(bs):
            curr_pos = start_pos[i]
            actions = []
            if in_middle(
                start_pos[i][0],
                start_pos[i][1],
                seg_left[i][0],
                seg_right[i][0],
                seg_top_y[i],
                seg_bot_y[i],
            ):
                seg_right[i][1] = curr_pos[1]
            else:
                a2l = self.generate_actions_to_goal(curr_pos, seg_left[i])
                actions.append(a2l)
                curr_pos = seg_left[i]

            if not in_middle(
                goal_pos[i][0],
                goal_pos[i][1],
                seg_left[i][0],
                seg_right[i][0],
                seg_top_y[i],
                seg_bot_y[i],
            ):
                a2r = self.generate_actions_to_goal(curr_pos, seg_right[i])
                actions.append(a2r)
                curr_pos = seg_right[i]

            a2g = self.generate_actions_to_goal(curr_pos, goal_pos[i])
            actions.append(a2g)
            actions = torch.cat(actions)
            expert_actions[i][: actions.shape[0]] = actions

        valid_idxs = torch.tensor(list(range(bs))).to(wall_locs.device)
        return start_pos, expert_actions, valid_idxs

    def generate_expert_cross_wall_state_and_actions_old(
        self, wall_locs=None, door_locs=None, n_steps=17
    ):
        bs = wall_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        x = sample_uniformly_between(left_wall_locs, right_wall_locs)
        y = sample_truncated_norm(
            upper_bound=door_locs + self.config.door_space,
            lower_bound=door_locs - self.config.door_space,
            mean=door_locs,
            std=0.8,
        ).to(door_locs.device)
        loc_at_door = torch.stack([x, y]).transpose(0, 1)

        left_pos, right_pos = self.generate_cross_wall_points(
            wall_locs, action_padding=self.config.action_upper_bd * 2
        )

        cw_actions = torch.zeros((bs, n_steps - 1, 2))
        cw_start_loc = torch.zeros((bs, 2))
        valid_trajs_idxs = []

        for i in range(bs):
            start_pos = loc_at_door[i]
            left_goal = left_pos[i]
            right_goal = right_pos[i]
            left_actions = self.generate_actions_to_goal(start_pos, left_goal)
            right_actions = self.generate_actions_to_goal(start_pos, right_goal)

            if len(left_actions) + len(right_actions) < n_steps - 1:
                continue

            actions = torch.cat(
                [torch.flip(left_actions, dims=[0]) * -1, right_actions]
            )
            start = left_goal
            if random.random() < 0.5:
                actions = torch.flip(actions, dims=[0]) * -1
                start = right_goal

            start_idx = random.randint(0, actions.shape[0] - (n_steps - 1))
            n_step_actions = actions[start_idx : start_idx + n_steps - 1]
            start += actions[:start_idx].sum(dim=0)

            cw_start_loc[i] = start
            cw_actions[i] = n_step_actions
            valid_trajs_idxs.append(i)

        valid_trajs_idxs = torch.tensor(valid_trajs_idxs).to(wall_locs.device)
        return (
            cw_start_loc.to(wall_locs.device),
            cw_actions.to(wall_locs.device),
            valid_trajs_idxs,
        )

    def generate_cross_wall_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        n_steps=17,
    ):
        bs = door_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        x = sample_uniformly_between(left_wall_locs, right_wall_locs)
        y = sample_truncated_norm(
            upper_bound=door_locs + self.config.door_space,
            lower_bound=door_locs - self.config.door_space,
            mean=door_locs,
        ).to(door_locs.device)
        loc_at_door = torch.stack([x, y]).transpose(0, 1)
        step_idxs = torch.randint(1, n_steps, size=x.shape)

        angles = torch.empty(bs)
        for i in range(bs):
            angles[i] = torch.pi + (torch.rand(1) - 0.5) * torch.pi / 2

        angles = self.angle_to_vec(angles).to(self.device)
        actions_dir_left, _ = self.generate_actions(n_steps, bias_angle=angles)
        actions_dir_right, _ = self.generate_actions(n_steps, bias_angle=-1 * angles)

        cw_actions = torch.zeros((bs, n_steps - 1, 2))
        cw_start_loc = torch.zeros((bs, 2))

        for i in range(bs):
            step = step_idxs[i]
            traj = torch.cat(
                [
                    torch.flip(actions_dir_left[i][:step], dims=[0]) * -1,
                    actions_dir_right[i][1 : n_steps - step],
                ]
            )
            step_sum_before_door = traj[:step].sum(dim=0)

            if random.random() < 0.5:
                traj = torch.flip(traj, dims=[0]) * -1
                step_sum_before_door = traj[: n_steps - step].sum(dim=0)

            cw_actions[i] = traj
            cw_start_loc[i] = loc_at_door[i] - step_sum_before_door

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01
        cw_start_loc = torch.clamp(cw_start_loc, min=min_val, max=max_val)

        return cw_start_loc, cw_actions, torch.zeros_like(cw_actions)

    def generate_wall_bump_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        n_steps=17,
    ):
        bs = wall_locs.size(0)
        bump_actions = torch.zeros((bs, n_steps - 1, 2), device=wall_locs.device)
        bump_start_loc = torch.zeros((bs, 2), device=wall_locs.device)

        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2
        door_top = door_locs + self.config.door_space
        door_bot = door_locs - self.config.door_space

        for i in range(bs):
            bump_middle_wall = random.random() < 0.5

            if bump_middle_wall:
                wall_y = torch.zeros(1, device=wall_locs.device)
                while True:
                    wall_y = sample_uniformly_between(
                        torch.tensor(
                            [self.config.border_wall_loc], device=wall_locs.device
                        ),
                        torch.tensor(
                            [self.config.img_size - self.config.border_wall_loc],
                            device=wall_locs.device,
                        ),
                    )
                    if wall_y < door_bot[i] or wall_y > door_top[i]:
                        break

                start_from_left = random.random() < 0.5
                if start_from_left:
                    start_x = sample_uniformly_between(
                        torch.tensor(
                            [self.config.border_wall_loc], device=wall_locs.device
                        ),
                        left_wall_locs[i : i + 1] - 2,
                    )
                    direction = torch.tensor([1.0, 0.0], device=wall_locs.device)
                else:
                    start_x = sample_uniformly_between(
                        right_wall_locs[i : i + 1] + 2,
                        torch.tensor(
                            [self.config.img_size - self.config.border_wall_loc],
                            device=wall_locs.device,
                        ),
                    )
                    direction = torch.tensor([-1.0, 0.0], device=wall_locs.device)
                start_pos = torch.tensor(
                    [start_x.item(), wall_y.item()], device=wall_locs.device
                )

            else:
                border_choice = random.randint(0, 3)
                bwl = self.config.border_wall_loc

                if border_choice == 0:  # Top wall
                    start_y = sample_uniformly_between(
                        torch.tensor([bwl + 5], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl - 5], device=wall_locs.device
                        ),
                    )
                    start_x = sample_uniformly_between(
                        torch.tensor([bwl], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl], device=wall_locs.device
                        ),
                    )
                    direction = torch.tensor([0.0, -1.0], device=wall_locs.device)
                elif border_choice == 1:  # Bottom wall
                    start_y = sample_uniformly_between(
                        torch.tensor([bwl + 5], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl - 5], device=wall_locs.device
                        ),
                    )
                    start_x = sample_uniformly_between(
                        torch.tensor([bwl], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl], device=wall_locs.device
                        ),
                    )
                    direction = torch.tensor([0.0, 1.0], device=wall_locs.device)
                elif border_choice == 2:  # Left wall
                    start_x = sample_uniformly_between(
                        torch.tensor([bwl + 5], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl - 5], device=wall_locs.device
                        ),
                    )
                    start_y = sample_uniformly_between(
                        torch.tensor([bwl], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl], device=wall_locs.device
                        ),
                    )
                    direction = torch.tensor([-1.0, 0.0], device=wall_locs.device)
                else:  # Right wall
                    start_x = sample_uniformly_between(
                        torch.tensor([bwl + 5], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl - 5], device=wall_locs.device
                        ),
                    )
                    start_y = sample_uniformly_between(
                        torch.tensor([bwl], device=wall_locs.device),
                        torch.tensor(
                            [self.config.img_size - bwl], device=wall_locs.device
                        ),
                    )
                    direction = torch.tensor([1.0, 0.0], device=wall_locs.device)

                start_pos = torch.tensor(
                    [start_x.item(), start_y.item()], device=wall_locs.device
                )

            direction_angle = self.vec_to_angle(direction.unsqueeze(0))
            bias_angle = self.angle_to_vec(direction_angle).to(self.device)
            actions, _ = self.generate_actions(n_steps, bias_angle=bias_angle)

            bump_start_loc[i] = start_pos
            bump_actions[i] = actions[0]

        min_val = self.config.border_wall_loc + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01
        bump_start_loc = torch.clamp(bump_start_loc, min=min_val, max=max_val)
        return bump_start_loc, bump_actions

    def generate_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        size=None,
        n_steps=17,
    ):
        location, actions, bias_angle = super().generate_state_and_actions(
            wall_locs=wall_locs, door_locs=door_locs, size=size, n_steps=n_steps
        )
        modified_count = 0

        if self.config.cross_wall_rate:
            cw_count = np.random.rand() < self.config.cross_wall_rate
            if cw_count:
                cw_locations, cw_actions, _ = (
                    self.generate_cross_wall_state_and_actions(
                        wall_locs=wall_locs[:cw_count],
                        door_locs=door_locs[:cw_count],
                        n_steps=n_steps,
                    )
                )
                location[:cw_count] = cw_locations
                actions[:cw_count] = cw_actions
                modified_count = cw_count

        if hasattr(self.config, "wall_bump_rate") and self.config.wall_bump_rate > 0:
            bump_count = math.ceil(self.config.wall_bump_rate)
            bump_count = min(bump_count, 1 - modified_count)
            if bump_count > 0:
                bump_locations, bump_actions = (
                    self.generate_wall_bump_state_and_actions(
                        wall_locs=wall_locs[
                            modified_count : modified_count + bump_count
                        ],
                        door_locs=door_locs[
                            modified_count : modified_count + bump_count
                        ],
                        n_steps=n_steps,
                    )
                )
                location[modified_count : modified_count + bump_count] = bump_locations
                actions[modified_count : modified_count + bump_count] = bump_actions
                modified_count += bump_count

        if self.config.expert_cross_wall_rate:
            ecw_locations, ecw_actions, valid_traj_idxs = (
                self.generate_expert_cross_wall_state_and_actions(
                    wall_locs=wall_locs,
                    door_locs=door_locs,
                    n_steps=n_steps,
                )
            )
            max_ecw_count = math.ceil(self.config.expert_cross_wall_rate)
            valid_traj_idxs = valid_traj_idxs[:max_ecw_count]
            if valid_traj_idxs.shape[0]:
                location[valid_traj_idxs] = ecw_locations[valid_traj_idxs]
                actions[valid_traj_idxs] = ecw_actions[valid_traj_idxs]

        bs = location.size(0)
        perm = torch.randperm(bs)
        location = location[perm]
        actions = actions[perm]
        bias_angle = bias_angle[perm]
        return location, actions, bias_angle

    # -- collision detection ------------------------------------------------

    def check_wall_intersection(self, current_location, next_location, walls):
        half_width = self.config.wall_width // 2
        wall_left = walls - half_width
        wall_right = walls + half_width

        current_right = current_location[:, 0] <= wall_right
        next_right = next_location[:, 0] <= wall_right
        current_left = current_location[:, 0] >= wall_left
        next_left = next_location[:, 0] >= wall_left

        inside_wall = (current_right & current_left) != (next_right & next_left)
        across_wall = (current_right != next_right) & (current_left != next_left)
        return inside_wall | across_wall

    def check_pass_through_door(
        self, current_location, next_location, wall_loc, door_loc
    ):
        half_width = self.config.wall_width // 2
        left_wall = wall_loc - half_width
        right_wall = wall_loc + half_width

        d = next_location - current_location
        a = d[1] / d[0]
        b = current_location[1] - a * current_location[0]

        if (
            torch.sign(left_wall - current_location[0])
            * torch.sign(left_wall - next_location[0])
            < 0
        ):
            y_left = a * left_wall + b
            pass_left_wall = (
                door_loc - self.config.door_space
                <= y_left
                <= door_loc + self.config.door_space
            )
        else:
            pass_left_wall = True

        if (
            torch.sign(right_wall - current_location[0])
            * torch.sign(right_wall - next_location[0])
            < 0
        ):
            y_right = a * right_wall + b
            pass_right_wall = (
                door_loc - self.config.door_space
                <= y_right
                <= door_loc + self.config.door_space
            )
        else:
            pass_right_wall = True

        return pass_left_wall and pass_right_wall

    @staticmethod
    def segments_intersect(A, B):
        """Test batch of 2D segment pairs for intersection.

        Args:
            A: ``[bs, 2, 2]`` — start/end of segment A.
            B: ``[bs, 2, 2]`` — start/end of segment B.

        Returns:
            LongTensor ``[bs]`` with 1 where segments intersect.
        """
        A0, A1 = A[:, 0], A[:, 1]
        B0, B1 = B[:, 0], B[:, 1]
        dA = A1 - A0
        dB = B1 - B0

        def cross_2d(v, w):
            return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

        cross_A_B0 = cross_2d(dA, B0 - A0)
        cross_A_B1 = cross_2d(dA, B1 - A0)
        cross_B_A0 = cross_2d(dB, A0 - B0)
        cross_B_A1 = cross_2d(dB, A1 - B0)

        intersect_A = cross_A_B0 * cross_A_B1 < 0
        intersect_B = cross_B_A0 * cross_B_A1 < 0
        return (intersect_A & intersect_B).long()

    def check_wall_width_intersection(
        self,
        locations,
        next_locations,
        walls,
        doors,
    ):
        disp = torch.stack([locations, next_locations], dim=1)
        deltas = next_locations - locations
        upwards = deltas[:, 1] > 0
        downwards = deltas[:, 1] < 0

        left_wall = walls - self.config.wall_width // 2
        right_wall = walls + self.config.wall_width // 2
        door_bot = doors - self.config.door_space
        door_top = doors + self.config.door_space

        top_left = torch.stack([left_wall, door_top], dim=1)
        top_right = torch.stack([right_wall, door_top], dim=1)
        bot_left = torch.stack([left_wall, door_bot], dim=1)
        bot_right = torch.stack([right_wall, door_bot], dim=1)

        top_seg = torch.stack([top_left, top_right], dim=1)
        bot_seg = torch.stack([bot_left, bot_right], dim=1)

        top_intersect = self.segments_intersect(disp, top_seg)
        bot_intersect = self.segments_intersect(disp, bot_seg)
        return (top_intersect & upwards) | (bot_intersect & downwards)

    # -- trajectory generation (with collision handling) --------------------

    def generate_transitions(
        self,
        location,
        actions,
        bias_angle,
        walls,
    ):
        """Generate full trajectory with wall collision handling.

        Args:
            location: ``[bs, 2]`` starting positions.
            actions: ``[bs, n_steps-1, 2]`` action sequences.
            bias_angle: ``[bs, 2]`` bias angle vectors.
            walls: ``(wall_x, door_y)`` each ``[bs]``.

        Returns:
            ``(obs, actions, states, reward, info)`` in unified format:

            - obs: ``{"visual": [T, C, H, W], "proprio": [T, 2]}``
            - actions: ``[T, 2]``
            - states: ``[T, 2]``
            - reward: ``None``
            - info: ``{"wall_x": Tensor, "door_y": Tensor}``
        """
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])

            left_border = torch.zeros_like(walls[0])
            left_border[:] = self.config.border_wall_loc - 1
            right_border = torch.zeros_like(walls[0])
            right_border[:] = self.config.img_size - self.config.border_wall_loc
            top_border, bot_border = left_border, right_border

            check_border_intersection = (
                (
                    (
                        torch.sign(locations[-1][:, 0] - left_border)
                        * torch.sign(next_location[:, 0] - left_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 0] - right_border)
                        * torch.sign(next_location[:, 0] - right_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 1] - top_border)
                        * torch.sign(next_location[:, 1] - top_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 1] - bot_border)
                        * torch.sign(next_location[:, 1] - bot_border)
                    )
                    <= 0
                )
            )

            check_wall_inter = self.check_wall_intersection(
                locations[-1], next_location, walls[0]
            )

            check_wall_width_inter = self.check_wall_width_intersection(
                locations=locations[-1],
                next_locations=next_location,
                walls=walls[0],
                doors=walls[1],
            )

            check_intersection = (
                check_border_intersection | check_wall_inter | check_wall_width_inter
            )

            for j in check_intersection.nonzero():
                if check_border_intersection[j] or check_wall_width_inter[j]:
                    next_location[j] = locations[-1][j].clone()
                else:
                    if not self.check_pass_through_door(
                        current_location=locations[-1][j][0],
                        next_location=next_location[j][0],
                        wall_loc=walls[0][j],
                        door_loc=walls[1][j],
                    ):
                        next_location[j] = locations[-1][j].clone()

            locations.append(next_location)

        wall_x = walls[0]  # [bs]
        door_y = walls[1]  # [bs]

        # Stack locations: [bs, T, 1, 2] (unsqueeze for multi-dot compat)
        locations_stacked = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)

        # Render observations
        states_img = self.render_location(locations_stacked)
        walls_img = self.render_walls(*walls).unsqueeze(1).unsqueeze(1)
        walls_img = walls_img.repeat(1, states_img.shape[1], 1, 1, 1)
        states_with_walls = torch.cat([states_img, walls_img], dim=-3)

        # Temporal downsampling
        if self.config.n_steps_reduce_factor > 1:
            states_with_walls = states_with_walls[
                :, :: self.config.n_steps_reduce_factor
            ]
            locations_stacked = locations_stacked[
                :, :: self.config.n_steps_reduce_factor
            ]
            reduced_chunks = actions.shape[1] // self.config.n_steps_reduce_factor
            action_chunks = torch.chunk(actions, chunks=reduced_chunks, dim=1)
            actions = torch.cat(
                [torch.sum(chunk, dim=1, keepdim=True) for chunk in action_chunks],
                dim=1,
            )

        # Remove agent dimension
        locations_stacked = locations_stacked.squeeze(2)  # [bs, T, 2]
        actions = actions.squeeze(2)  # [bs, T-1, 2]
        states_with_walls = states_with_walls.float()  # [bs, T, C, H, W]

        # Normalize
        if self.config.normalize:
            states_with_walls = _normalize_two_rooms_obs(states_with_walls)
            locations_stacked = (
                locations_stacked - TWO_ROOMS_LOCATION_MEAN.to(locations_stacked.device)
            ) / (TWO_ROOMS_LOCATION_STD.to(locations_stacked.device) + 1e-6)

        # Drop last timestep of visual/locations to align with actions
        # states_with_walls: [bs, T, C, H, W] -> [bs, T-1, C, H, W]
        states_with_walls = states_with_walls[:, :-1]
        # locations_stacked: [bs, T, 2] -> [bs, T-1, 2]
        locations_stacked = locations_stacked[:, :-1]

        # Sample a sub-sequence of length sample_length
        max_start = self.config.n_steps - self.config.sample_length
        start = np.random.randint(0, max_start) if max_start > 0 else 0
        end = start + self.config.sample_length
        states_with_walls = states_with_walls[:, start:end]  # [bs, T', C, H, W]
        actions = actions[:, start:end]  # [bs, T', 2]
        locations_stacked = locations_stacked[:, start:end]  # [bs, T', 2]

        # Squeeze batch dim (bs=1 per __getitem__)
        visual = states_with_walls.squeeze(0)  # [T, C, H, W]
        act = actions.squeeze(0)  # [T, 2]
        locs = locations_stacked.squeeze(0)  # [T, 2]

        obs = {"visual": visual, "proprio": locs}
        reward = None
        info = {"wall_x": wall_x, "door_y": door_y}
        return obs, act, locs, reward, info

    # -- wall sampling & rendering ------------------------------------------

    def sample_walls(self):
        """Sample a wall layout for a single trajectory.

        Returns:
            ``(wall_x, door_y)`` each ``[1]`` tensor.
        """
        layout_codes = list(self.layouts.keys())
        if self.config.fix_wall_batch_k is not None:
            layout_codes = random.sample(layout_codes, self.config.fix_wall_batch_k)

        weights = [1] * len(layout_codes)
        sampled_codes = random.choices(layout_codes, weights=weights, k=1)
        wall_locs = []
        door_locs = []

        for code in sampled_codes:
            attr = self.layouts[code]
            wall_locs.append(attr["wall_pos"])
            door_locs.append(attr["door_pos"])

        wall_locs = torch.tensor(wall_locs, device=self.device)
        door_locs = torch.tensor(door_locs, device=self.device)
        return (wall_locs, door_locs)

    def render_walls(self, wall_locs, hole_locs):
        """Render wall + door as a binary image.

        Args:
            wall_locs: ``[bs]`` x-coordinates.
            hole_locs: ``[bs]`` y-coordinates.

        Returns:
            uint8 image ``[bs, H, W]``.
        """
        sz = self.config.img_size
        x = torch.arange(0, sz, device=self.device)
        y = torch.arange(0, sz, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)

        wall_locs_r = wall_locs.view(1, 1, 1).repeat(1, sz, sz)
        hole_locs_r = hole_locs.view(1, 1, 1).repeat(1, sz, sz)

        offset = self.config.wall_width // 2
        wall_mask = (wall_locs_r - offset <= grid_x) & (grid_x <= wall_locs_r + offset)

        res = (
            wall_mask
            * (
                (hole_locs_r < grid_y - self.config.door_space)
                + (hole_locs_r > grid_y + self.config.door_space)
            )
        ).float()

        bwl = self.config.border_wall_loc
        res[:, :, bwl - 1] = 1
        res[:, :, -bwl] = 1
        res[:, bwl - 1, :] = 1
        res[:, -bwl, :] = 1

        return (res * 255).clamp(0, 255).to(torch.uint8)
