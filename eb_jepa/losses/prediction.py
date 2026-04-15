from __future__ import annotations

from typing import List

import lpips as lpips_lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class SquareLossSeq(nn.Module):
    """Square loss over a sequence [B, C, T, H, W] (feature dim at dim 1)."""

    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        # state, predi: [B, D, T, H, W] → [D, B*T*H*W] → [B*T*H*W, D]
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return F.mse_loss(state, predi)


class TemporalSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        Temporal Similarity Loss.
        Encourages consecutive frames to have similar representations by penalizing
        the squared difference between consecutive time steps.
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t


class TemporalStraighteningLoss(torch.nn.Module):
    def __init__(self, projector: nn.Module = None, detach: bool = True):
        """
        Temporal Straightening Loss for learning trajectory projectors.

        Encourages the trajectory in the projected latent space to be "straight"
        by minimizing the angle between successive velocity vectors.

        For triplets (z_{t-1}, z_t, z_{t+1}), computes:
            loss = -dot((z_{t+1} - z_t), (z_t - z_{t-1}))

        This loss is minimized when consecutive displacement vectors are aligned,
        encouraging straight (geodesic-like) trajectories.

        Args:
            projector: Optional projection network applied before computing the loss.
            detach: If True, detach input features (no gradient to encoder).
        """
        super().__init__()
        self.projector = nn.Identity() if projector is None else projector
        self.detach = detach

    def forward(self, x: torch.Tensor):
        """
        Compute temporal straightening loss over a trajectory.

        Args:
            x: Trajectory states [B, C, T, H, W] or [T, B, D].
                If 5D, will be reshaped to [T, B, C*H*W] before projection.

        Returns:
            Tuple of (loss, loss_dict) where loss_dict contains individual loss components.
        """
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(2, 0, 1, 3, 4).reshape(t, b, c * h * w)  # [T, B, C*H*W]
        else:
            t, b, d = x.shape

        if t <= 2:
            zero = torch.tensor(0.0, device=x.device)
            return zero, {"straightening": 0.0}

        if self.detach:
            x = x.detach()

        x_flat = x.reshape(-1, x.shape[-1])  # [T*B, D]
        z = self.projector(x_flat)  # [T*B, D']
        d_out = z.shape[-1]
        z = z.view(t, b, d_out)  # [T, B, D']

        v_forward = z[2:] - z[1:-1]  # [T-2, B, D'] (z_{t+1} - z_t)
        v_backward = z[1:-1] - z[:-2]  # [T-2, B, D'] (z_t - z_{t-1})

        v_forward = F.normalize(v_forward, dim=-1)
        v_backward = F.normalize(v_backward, dim=-1)

        dot_products = (v_forward * v_backward).sum(dim=-1)  # [T-2, B]
        loss = -dot_products.mean()

        return loss, {"straightening": loss.detach()}


class InverseDynamicsLoss(torch.nn.Module):
    def __init__(self, idm: nn.Module, spatial: bool = False):
        """Predicts actions from consecutive states and compares with ground truth actions.

        Args:
            idm: Inverse dynamics model. If ``spatial=False``, expects
                ``(state_t, state_t+1)`` each ``[B, D]``. If ``spatial=True``,
                expects ``[B, N, C]`` spatial tokens.
            spatial: Whether the IDM expects spatial token sequences
                (AttentiveInverseDynamicsModel) or flat vectors
                (InverseDynamicsModel).
        """
        super().__init__()
        self.idm = idm
        self.spatial = spatial

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        """Compute IDM loss.

        Args:
            x: States across time steps.
                If ``spatial=False``: ``[T, B, D]`` (flattened).
                If ``spatial=True``: ``[T, B, N, C]`` (spatial tokens).
            actions: ``[B, A, T]`` or ``[B, A, T-1]`` ground truth actions.
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)

        t = x.shape[0]
        b = x.shape[1]

        # T states -> T-1 transitions
        states_t = x[:-1].transpose(0, 1)  # [B, T-1, ...] (... = D or N,C)
        states_tp1 = x[1:].transpose(0, 1)  # [B, T-1, ...]

        if self.spatial:
            # Spatial path: [B, T-1, N, C] -> [B*(T-1), N, C]
            states_t_flat = states_t.reshape(-1, *states_t.shape[2:])
            states_tp1_flat = states_tp1.reshape(-1, *states_tp1.shape[2:])
        else:
            # Flat path: [B, T-1, D] -> [B*(T-1), D]
            d = x.shape[2]
            states_t_flat = states_t.reshape(-1, d)
            states_tp1_flat = states_tp1.reshape(-1, d)

        pred_actions = self.idm(states_t_flat, states_tp1_flat)  # [B*(T-1), A]

        # Backwards compatibility: handle both [B, A, T] and [B, A, T-1] actions
        actions_transposed = actions.transpose(1, 2)  # [B, T or T-1, A]
        if actions_transposed.shape[1] == t:
            target_actions = actions_transposed[:, :-1].reshape(-1, actions.size(1))
        else:
            target_actions = actions_transposed.reshape(-1, actions.size(1))

        return F.mse_loss(pred_actions, target_actions.detach())


class LPIPSLoss(nn.Module):
    """Combined MSE + LPIPS perceptual loss for visual decoder training.

    Wraps lpips.LPIPS(net="vgg") in eval mode (frozen VGG weights).
    Expects inputs in channel-normalized space; applies inverse normalization
    before computing LPIPS (which expects [0, 1] range).

    Memory optimization: processes images in chunks through VGG to avoid
    the massive memory spike from feeding B*T images simultaneously.
    With B=32, T=8 (256 images at 224x224), unchunked LPIPS uses ~63 GB;
    chunking to 32 images reduces this to ~8 GB.

    Args:
        pixel_weight: Weight for the MSE term (default: 10.0).
        perceptual_weight: Weight for the LPIPS term (default: 1.0).
        normalize_mean: Per-channel mean used by the data pipeline.
        normalize_std: Per-channel std used by the data pipeline.
        lpips_chunk_size: Number of images to process through VGG at once.
            Lower values use less memory but are slightly slower.
            Set to -1 to disable chunking (original behavior). Default: 32.
    """

    def __init__(
        self,
        pixel_weight: float = 10.0,
        perceptual_weight: float = 1.0,
        normalize_mean: List[float] = (0.485, 0.456, 0.406),
        normalize_std: List[float] = (0.229, 0.224, 0.225),
        lpips_chunk_size: int = 32,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_chunk_size = lpips_chunk_size

        self.lpips_fn = lpips_lib.LPIPS(net="vgg").eval()
        self.lpips_fn.requires_grad_(False)

        self.register_buffer(
            "normalize_mean",
            torch.tensor(normalize_mean, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "normalize_std",
            torch.tensor(normalize_std, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def _inverse_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undo channel normalization and clamp to [0, 1]."""
        return (x * self.normalize_std + self.normalize_mean).clamp(0.0, 1.0)

    def forward(self, decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pixel_weight * MSE + perceptual_weight * LPIPS.

        Args:
            decoded: Decoded images [B, C, T, H, W] in normalized space.
            target: Ground-truth images [B, C, T, H, W] in normalized space.

        Returns:
            Combined scalar loss.
        """
        B, C, T, H, W = decoded.shape
        mse = F.mse_loss(decoded, target)

        decoded_bt = decoded.permute(0, 2, 1, 3, 4).reshape(
            B * T, C, H, W
        )  # [B*T, C, H, W]
        target_bt = target.permute(0, 2, 1, 3, 4).reshape(
            B * T, C, H, W
        )  # [B*T, C, H, W]

        with torch.amp.autocast("cuda", enabled=False):
            unnorm_dec = self._inverse_normalize(decoded_bt.float())
            unnorm_tgt = self._inverse_normalize(target_bt.float())

            N = unnorm_dec.shape[0]
            chunk = self.lpips_chunk_size
            if chunk <= 0 or chunk >= N:
                lpips_val = self.lpips_fn(unnorm_dec, unnorm_tgt).mean()
            else:
                lpips_chunks = []
                for i in range(0, N, chunk):
                    lp = self.lpips_fn(
                        unnorm_dec[i : i + chunk], unnorm_tgt[i : i + chunk]
                    )
                    lpips_chunks.append(lp)
                lpips_val = torch.cat(lpips_chunks, dim=0).mean()

        return self.pixel_weight * mse + self.perceptual_weight * lpips_val
