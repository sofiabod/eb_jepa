# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DROID video dataset for robot manipulation."""

import json
import os
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Optional, Sequence

import decord
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from einops import repeat
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from eb_jepa.data.traj_dset import TrajDataset
from eb_jepa.data.utils import register_dataset

_GLOBAL_SEED = 0
logger = getLogger(__name__)

decord.bridge.set_bridge("native")


# ==================== DROID-specific utility functions ====================


def poses_to_diffs(poses):
    """
    Convert poses to delta actions (differences between consecutive poses).

    Args:
        poses: numpy array of shape [T, 7] where each pose is [xyz(3), euler(3), gripper(1)]

    Returns:
        numpy array of shape [T-1, 7] containing delta actions
    """
    xyz = poses[:, :3]  # shape [T, 3]
    thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
    matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas
    ]

    # Compute delta xyz
    xyz_diff = xyz[1:] - xyz[:-1]

    # Compute delta rotation
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [
        Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff
    ]
    angle_diff = np.stack([d for d in angle_diff], axis=0)

    # Compute delta gripper
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]

    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)


def compute_new_pose(pose, action):
    """
    Compute new pose by applying delta action to current pose.

    Args:
        pose: torch.Tensor of shape [B, T=1, 7]
        action: torch.Tensor of shape [B, T=1, 7] (delta action)

    Returns:
        torch.Tensor of shape [B, T=1, 7] (new pose)
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()

    # Compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]

    # Compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas
    ]
    delta_matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
        for theta in delta_thetas
    ]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [
        Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff
    ]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]

    # Compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)

    # New pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


# ==================== Dataset class ====================


def get_json(directory):
    """Find and load the first JSON file in a directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while processing {filename}: {e}"
                )
    return None


@register_dataset("droid")
class DROIDVideoDataset(TrajDataset):
    """
    DROID robot manipulation dataset with video observations.

    Args:
        data_path: Path to dataset (CSV file with video paths or directory with manifest patterns)
        camera_views: List of camera view keys to use (e.g., ["wrist_mp4_path", "left_mp4_path"])
        frameskip: Skip every N frames when loading
        action_skip: Skip every N frames for actions
        num_frames: Number of frames per video clip
        fps: Target FPS for video loading
        transform: Optional transform to apply to video frames
        camera_frame: If True, transform poses to camera frame
        normalize_action: If True, normalize actions/states to zero mean unit variance
        mpk_dset: If True, load from MPK-format dataset
        mpk_manifest_patterns: List of glob patterns for MPK dataset files
        droid_to_rcasa_action_format: Action repeat factor for RoboCasa format
        local: If True, load from local filesystem
        seed: Random seed for reproducibility
        droid_fraction: Fraction of dataset to use (for debugging/ablation)

    Returns:
        __getitem__ returns (obs, actions, states, reward) tuple where:
        - obs: dict with keys "visual" [T, C, H, W] and "proprio" [T, D]
        - actions: [T-1, A] delta actions
        - states: [T, D] robot states
        - reward: scalar tensor (always 0.0 for DROID)
    """

    def __init__(
        self,
        data_path: str,
        camera_views: Sequence[str] = ["wrist_mp4_path"],
        frameskip: int = 1,
        action_skip: int = 1,
        num_frames: int = 16,
        fps: int = 5,
        transform: Optional[torch.nn.Module] = None,
        camera_frame: bool = False,
        normalize_action: bool = False,
        mpk_dset: bool = False,
        mpk_manifest_patterns: Optional[Sequence[str]] = None,
        droid_to_rcasa_action_format: int = 1,
        local: bool = True,
        seed: Optional[int] = None,
        droid_fraction: float = 1.0,
    ):
        self.data_path = data_path
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.action_skip = action_skip
        self.fps = fps
        self.transform = transform
        self.normalize_action = normalize_action
        self.camera_frame = camera_frame
        self.mpk_dset = mpk_dset
        self.mpk_manifest_patterns = mpk_manifest_patterns
        self.droid_to_rcasa_action_format = droid_to_rcasa_action_format
        self.local = local
        self.seed = seed

        # Same sample-level across workers because same self.rng pickled across workers
        # This randomness only affects clip slicing and camera viewpoint sampling.
        self.rng = np.random.RandomState(seed)

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.'
            )

        # Camera views
        self.camera_views = camera_views
        logger.info(f"Using DROID with camera views: {self.camera_views}")

        # Load samples
        if self.mpk_dset:
            self._action_type = "delta-state"
            self.samples = self._load()
            num_samples_stored = len(self.samples) if normalize_action else 1
            debug = False
        else:
            self.h5_name = "trajectory.h5"
            self.samples = list(
                pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0]
            )
            num_samples_stored = 50 if normalize_action else 1
            debug = False

        # Apply dataset fraction slicing
        if droid_fraction < 1.0:
            original_len = len(self.samples)
            num_samples = max(1, int(original_len * droid_fraction))
            self.samples = self.samples[:num_samples]
            logger.info(
                f"Slicing dataset from {original_len} to {num_samples} samples ({droid_fraction*100:.1f}%)"
            )
        else:
            logger.info(
                f"Not slicing DROID dataset, using {len(self.samples)} samples, 100% of video paths"
            )

        # Compute normalization statistics
        states = []
        actions = []
        proprio_states = []
        seq_lengths = []
        for i in tqdm(
            range(num_samples_stored),
            desc=f"Loading {num_samples_stored} DROID eps to compute mean/std",
        ):
            buffer, action, state, _, _ = self.__getitem__(i, debug=debug)
            states.append(torch.tensor(state))
            actions.append(torch.tensor(action))
            proprio_states.append(torch.tensor(state))
            seq_lengths.append(len(state))

        self.states = torch.stack(states)
        self.actions = torch.stack(actions)
        self.proprios = torch.stack(proprio_states)
        self.seq_lengths = torch.tensor(seq_lengths)
        self.rewards = None

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = TrajDataset.compute_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = TrajDataset.compute_mean_std(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = TrajDataset.compute_mean_std(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        # Note: self.actions / self.proprios are only used above to compute
        # mean/std statistics.  Actual data is loaded from disk in __getitem__
        # and normalized there.

    def get_seq_length(self, idx: int) -> int:
        """Return the length of the idx-th trajectory."""
        return self.num_frames

    def __getitem__(self, idx: int, debug: bool = False, **kwargs):
        """
        Load a video clip and its corresponding actions/states.

        Returns:
            Tuple of (obs, actions, states, reward) where:
            - obs: dict with "visual" [T, C, H, W] and "proprio" [T, D]
            - actions: numpy array [T-1, A]
            - states: numpy array [T, D]
            - reward: torch.Tensor scalar (0.0)
        """
        max_retries = 10
        path = self.samples[idx]

        for attempt in range(max_retries):
            try:
                if self.mpk_dset:
                    buffer, actions, states, extrinsics, indices = self.loadvideo_hf(
                        path
                    )
                else:
                    buffer, actions, states, extrinsics, indices = (
                        self.loadvideo_decord(path)
                    )
                break
            except Exception as e:
                if debug or attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load video after {max_retries} attempts. "
                        f"Last path: {path}, error: {e}"
                    ) from e
                idx = self.rng.randint(0, self.__len__())
                path = self.samples[idx]

        if self.droid_to_rcasa_action_format > 1:
            actions = self.repeat_divide_action(
                actions, act_repeat=self.droid_to_rcasa_action_format
            )

        # Pad actions with dummy last action so that it has the same length as obs
        if len(actions) < len(states):
            actions = np.concatenate(
                [actions, np.zeros((1, actions.shape[-1]))], axis=0
            )

        actions = torch.tensor(actions, dtype=torch.float32)
        states = torch.tensor(states, dtype=torch.float32)

        if hasattr(self, "action_mean"):
            actions = (actions - self.action_mean) / self.action_std
            states = (states - self.state_mean) / self.state_std

        obs = {
            "visual": buffer,
            "proprio": states,
        }

        # buffer: [T, C, H, W]
        return obs, actions, states, torch.tensor(0.0), None

    def repeat_divide_action(
        self, action: np.ndarray, act_repeat: int = 5
    ) -> np.ndarray:
        """
        Action repeat and divide. Used when a model is used to concatenated "small" actions
        and we want to feed it DROID actions that are big and 7-dimensional.

        Args:
            action: [T, A] action array
            act_repeat: Repeat factor

        Returns:
            [T, F*A] repeated and divided actions
        """
        return repeat(action, "t a -> t (f a)", f=act_repeat) / float(act_repeat)

    def transform_frame(self, poses: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """Transform poses from world frame to camera frame."""
        from scipy.spatial.transform import Rotation

        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def _load(self) -> list[str]:
        """Load file paths from manifest patterns."""
        paths = []
        for pattern in self.mpk_manifest_patterns:
            # Remove leading '**/' if present
            cleaned_pattern = pattern.lstrip("/")
            # Use glob with the full pattern, relative to data_path
            found = list(Path(self.data_path).glob(cleaned_pattern))
            if not found:
                logger.warning(f"No files found for pattern {cleaned_pattern}")
            paths.extend(found)
        return [str(p) for p in paths]

    def loadvideo_hf(self, path: str):
        """
        Load video from HF/MPK format dataset.

        Returns:
            buffer: torch.Tensor [T, C, H, W] with video frames
            actions: np.ndarray [T-1, 7] with robot actions
            states: np.ndarray [T, 7] with robot states
            extrinsics: None (not used in this dataset)
            indices: np.ndarray [T] with sampled frame indices
        """
        trajectory = h5py.File(path)
        camera_view = self.camera_views[self.rng.randint(0, len(self.camera_views))]
        states = np.concatenate(
            [
                np.array(
                    trajectory["episode_data"]["observation"]["cartesian_position"]
                ),
                np.array(trajectory["episode_data"]["observation"]["gripper_position"])[
                    :, None
                ],
            ],
            axis=1,
        )  # [T, 7]

        # Sample a random window of nframes
        vfps = 30
        fpc = self.num_frames
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(states)

        if vlen < nframes:
            raise Exception(f"Video is too short {path=}, {nframes=}, {vlen=}")

        ef = self.rng.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]
        actions = poses_to_diffs(states[:: self.action_skip])

        buffer = trajectory["episode_data"]["observation"][camera_view][indices, :][
            :: self.frameskip
        ]
        buffer = buffer / 255.0
        buffer = torch.tensor(buffer, dtype=torch.float32).permute(
            0, 3, 1, 2
        )  # T H W C -> T C H W
        if self.transform is not None:
            buffer = self.transform(buffer)
        return buffer, actions, states, None, indices

    def loadvideo_decord(self, path: str):
        """
        Load video from DROID standard format with decord.

        Returns:
            buffer: torch.Tensor [T, C, H, W] with video frames
            actions: np.ndarray [T-1, 7] with robot actions
            states: np.ndarray [T, 7] with robot states
            extrinsics: np.ndarray [T, 6] with camera extrinsics
            indices: np.ndarray [T] with sampled frame indices
        """
        # Load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # Load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath)

        # Randomly sample a camera view
        camera_view = self.camera_views[self.rng.randint(0, len(self.camera_views))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][
            f"{camera_name}_left"
        ]

        states = np.concatenate(
            [
                np.array(
                    trajectory["observation"]["robot_state"]["cartesian_position"]
                ),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[
                    :, None
                ],
            ],
            axis=1,
        )  # [T, 7]

        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video file missing: {vpath}")
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))

        # Sample frames
        vfps = vr.get_avg_fps()
        fpc = self.num_frames
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # Sample a random window of nframes
        ef = self.rng.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]

        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)

        actions = poses_to_diffs(states[:: self.action_skip])

        # Load video frames
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        buffer = buffer / 255.0
        buffer = torch.tensor(buffer, dtype=torch.float32).permute(
            0, 3, 1, 2
        )  # T H W C -> T C H W

        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples)


register_dataset("franka_custom")(DROIDVideoDataset)
