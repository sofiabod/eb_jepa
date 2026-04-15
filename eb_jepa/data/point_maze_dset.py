# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from eb_jepa.utils.logging import get_logger

from .traj_dset import TrajDataset
from .utils import register_dataset

log = get_logger(__name__)


@register_dataset("pointmaze")
class PointMazeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = None,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        states = torch.load(self.data_path / "states.pth").float()
        self.states = states
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth")

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        log.info(f"✅ Loaded {n} PointMaze rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.compute_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self.compute_mean_std(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = self.compute_mean_std(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        obs_dir = self.data_path / "obses"
        image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        image = image[frames]  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, None, {}  # env_info

    def __getitem__(self, idx, **kwargs):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)
