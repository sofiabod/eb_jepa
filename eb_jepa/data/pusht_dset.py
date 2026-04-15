# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License

import pickle
from pathlib import Path
from typing import Callable, Optional

import decord
import torch
from decord import VideoReader
from einops import rearrange

from eb_jepa.utils.logging import get_logger

from .traj_dset import TrajDataset
from .utils import register_dataset

log = get_logger(__name__)


@register_dataset("pusht")
class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = None,
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True,  # agent's velocity
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ["T"] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[
            ..., :2
        ].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        log.info(f"✅ Loaded {n} PushT rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            # Precomputed stats (matching DATA_STATS in __init__.py)
            self.action_mean = torch.tensor([-0.0087, 0.0068])
            self.action_std = torch.tensor([0.2019, 0.2002])
            _state_mean = torch.tensor(
                [
                    236.6155,
                    264.5674,
                    255.1307,
                    266.3721,
                    1.9584,
                    -2.93032027,
                    2.54307914,
                ]
            )
            _state_std = torch.tensor(
                [101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094]
            )
            _proprio_mean = torch.tensor([236.6155, 264.5674, -2.93032027, 2.54307914])
            _proprio_std = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])
            self.state_mean = _state_mean[: self.state_dim]
            self.state_std = _state_std[: self.state_dim]
            self.proprio_mean = _proprio_mean[: self.proprio_dim]
            self.proprio_std = _proprio_std[: self.proprio_dim]
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
        vid_dir = self.data_path / "obses"
        decord.bridge.set_bridge("torch")
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, None, {"shape": shape}

    def __getitem__(self, idx, **kwargs):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)
