# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License

import numpy as np
import torch
from einops import rearrange


class Preprocessor:
    def __init__(
        self,
        action_mean,
        action_std,
        state_mean,
        state_std,
        proprio_mean,
        proprio_std,
        transform=None,
        normalize_obs_fn=None,
        unnormalize_obs_fn=None,
        channel_mean=None,
        channel_std=None,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform
        self._normalize_obs = normalize_obs_fn or (lambda x: x)
        self._unnormalize_obs = unnormalize_obs_fn or (lambda x: x)
        self.channel_mean = channel_mean  # [C] or None
        self.channel_std = channel_std  # [C] or None

    def normalize_obs(self, obs):
        """Normalize visual observations for model input.

        Chains the pluggable ``normalize_obs_fn`` (dataset-specific, e.g.
        two-rooms z-score) with channel normalization (e.g. ImageNet mean/std).

        Args:
            obs: Visual observation tensor ``[..., C, H, W]`` in raw scale.

        Returns:
            Normalized observation, same shape as input.
        """
        obs = self._normalize_obs(obs)
        return self._normalize_channel(obs)

    def unnormalize_obs(self, obs):
        """Inverse of :meth:`normalize_obs`.

        Undoes channel normalization first, then dataset-specific normalization.

        Args:
            obs: Normalized observation tensor.

        Returns:
            Unnormalized observation in raw ``[0, 1]`` scale, same shape.
        """
        obs = self.unnormalize_visual(obs)
        return self._unnormalize_obs(obs)

    def to_uint8_frames(self, images: torch.Tensor) -> np.ndarray:
        """Convert normalized observations to uint8 frames for visualization.

        Uses :meth:`unnormalize_obs` (which chains channel and dataset-specific
        denormalization), then scales to ``[0, 255]`` uint8.

        Args:
            images: ``[..., C, H, W]`` normalized tensor.

        Returns:
            ``[..., H, W, C]`` uint8 numpy array.
        """
        images = self.unnormalize_obs(images)
        # [..., C, H, W] -> [..., H, W, C]
        images = images.movedim(-3, -1)
        return (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    def unnormalize_visual(self, images):
        """Invert channel normalization for visualization.

        If ``channel_mean`` / ``channel_std`` were provided at construction
        time, undoes ``(x - mean) / std``.  Otherwise returns images unchanged.

        Args:
            images: ``[..., C, H, W]`` tensor (channel-first, e.g. ``[B, C, H, W]``
                or ``[C, H, W]``).

        Returns:
            Unnormalized images in ``[0, 1]`` range, same shape.
        """
        if self.channel_mean is None:
            return images
        mean = self.channel_mean.to(images.device)
        std = self.channel_std.to(images.device)
        # Reshape [C] -> [1, ..., 1, C, 1, 1] for broadcasting with [..., C, H, W]
        shape = [1] * (images.dim() - 3) + [mean.shape[0], 1, 1]
        mean = mean.view(*shape)
        std = std.view(*shape)
        return images * std + mean

    def _normalize_channel(self, images):
        """Apply channel normalization (e.g. ImageNet mean/std).

        Inverse of :meth:`unnormalize_visual`.

        Args:
            images: ``[..., C, H, W]`` tensor in ``[0, 1]`` range.

        Returns:
            Channel-normalized images, same shape.
        """
        if self.channel_mean is None:
            return images
        mean = self.channel_mean.to(images.device)
        std = self.channel_std.to(images.device)
        shape = [1] * (images.dim() - 3) + [mean.shape[0], 1, 1]
        return (images - mean.view(*shape)) / std.view(*shape)

    def normalize_actions(self, actions):
        """Z-score normalize actions. Works for any shape ``[..., action_dim]``."""
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(
            actions.device
        )

    def denormalize_actions(self, actions):
        """Invert z-score normalization. Works for any shape ``[..., action_dim]``."""
        return actions * self.action_std.to(actions.device) + self.action_mean.to(
            actions.device
        )

    def denormalize_proprios(self, proprio):
        """Invert z-score normalization. Works for any shape ``[..., proprio_dim]``."""
        return proprio * self.proprio_std.to(proprio.device) + self.proprio_mean.to(
            proprio.device
        )

    def normalize_proprios(self, proprio):
        """
        input shape (..., proprio_dim)
        """
        return (proprio - self.proprio_mean.to(proprio.device)) / self.proprio_std.to(
            proprio.device
        )

    def normalize_states(self, state):
        """
        input shape (..., state_dim)
        """
        return (state - self.state_mean) / self.state_std

    def denormalize_mse(self, mse):
        """Convert normalized MSE back to raw-scale MSE.

        Args:
            mse: MSE computed in normalized proprio space.

        Returns:
            MSE in raw (unnormalized) proprio space.
        """
        return mse * (self.proprio_std.mean().to(mse.device) ** 2)

    def transform_obs_visual(self, obs_visual):
        transformed_obs_visual = torch.tensor(obs_visual)
        transformed_obs_visual = (
            rearrange(transformed_obs_visual, "b t h w c -> b t c h w") / 255.0
        )
        transformed_obs_visual = self.transform(transformed_obs_visual)
        return transformed_obs_visual

    def transform_obs(self, obs):
        """
        np arrays to tensors
        """
        transformed_obs = {}
        transformed_obs["visual"] = self.transform_obs_visual(obs["visual"])
        transformed_obs["proprio"] = self.normalize_proprios(
            torch.tensor(obs["proprio"])
        )
        return transformed_obs
