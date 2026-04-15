from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.losses.anticollapse import (
    CovarianceLoss,
    EppsPulley,
    HingeStdLoss,
    _sliced_epps_pulley,
    _total_batch_size,
)
from eb_jepa.losses.prediction import InverseDynamicsLoss, TemporalSimilarityLoss


class _IDM_Sim_Regularizer_Base(torch.nn.Module):
    """Base class for IDM+Sim regularizers with shared preprocessing logic.

    Subclasses override ``_compute_anticollapse`` to provide the anti-collapse
    term (VC std+cov or SIGReg BCS).
    """

    def __init__(
        self,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        pool_time: bool = False,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        reg_per_patch: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False,
    ):
        super().__init__()
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        self.pool_time = pool_time
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.reg_per_patch = reg_per_patch
        self.sim_t_after_proj = sim_t_after_proj
        self.idm_after_proj = idm_after_proj

        self.sim_loss_fn = TemporalSimilarityLoss()
        if idm is not None:
            from eb_jepa.models.components import AttentiveInverseDynamicsModel

            spatial_idm = isinstance(idm, AttentiveInverseDynamicsModel)
            self.idm_loss_fn = InverseDynamicsLoss(idm, spatial=spatial_idm)
        else:
            self.idm_loss_fn = None

    def _preprocess(self, x):
        """Shared projection and reshaping logic.

        Returns:
            Tuple of (x_unprojected, x_spatial, x_projected, b, c, t, h, w, c_out).
            ``x_unprojected``: ``[T, B, C*H*W]`` flat (for sim loss and flat IDM).
            ``x_spatial``: ``[T, B, H*W, C]`` spatial tokens (for attentive IDM).
            ``x_projected``: ``[B, T, H, W, C_out]`` (for anti-collapse).
        """
        b, c, t, h, w = x.shape
        x_unprojected = x.permute(2, 0, 1, 3, 4).reshape(t, b, -1)  # [T, B, C*H*W]
        x_spatial = x.permute(2, 0, 3, 4, 1).reshape(t, b, h * w, c)  # [T, B, H*W, C]

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [B*T*H*W, C]
        x_proj = self.projector(x_flat)  # [B*T*H*W, C_out]
        c_out = x_proj.shape[-1]
        x_projected = x_proj.view(b, t, h, w, c_out)  # [B, T, H, W, C_out]

        return x_unprojected, x_spatial, x_projected, b, c, t, h, w, c_out

    def _compute_sim_idm(
        self, x, x_unprojected, x_spatial, x_projected, actions, b, t, h, w, c_out
    ):
        """Compute shared sim_t and IDM losses."""
        x_projected_reshaped = x_projected.permute(2, 0, 1, 3, 4).reshape(
            t, b, -1
        )  # [T, B, C_out*H*W]

        if self.sim_t_after_proj:
            sim_loss_t = self.sim_loss_fn(x_projected_reshaped)
        else:
            sim_loss_t = self.sim_loss_fn(x_unprojected)

        idm_loss = torch.tensor(0.0, device=x.device)
        if self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None:
            if self.idm_loss_fn.spatial:
                idm_loss = self.idm_loss_fn(x_spatial, actions)
            elif self.idm_after_proj:
                idm_loss = self.idm_loss_fn(x_projected_reshaped, actions)
            else:
                idm_loss = self.idm_loss_fn(x_unprojected, actions)

        return sim_loss_t, idm_loss

    def _get_x_for_anticollapse(self, x_projected, b, t, h, w, c_out):
        """Reshape projected features for anti-collapse computation.

        Returns a ``[G, N, D]`` tensor where G independent groups of N samples
        are regularized then averaged. When ``pool_time=True`` timesteps are
        pooled into the sample dimension (legacy behaviour); when ``False``
        (default, le-wm convention) each timestep forms its own group.
        """
        if self.reg_per_patch and h > 1:
            if self.pool_time:
                # [H*W, B*T, C_out]
                return x_projected.permute(2, 3, 0, 1, 4).reshape(h * w, b * t, c_out)
            # [T*H*W, B, C_out]
            return x_projected.permute(1, 2, 3, 0, 4).reshape(t * h * w, b, c_out)

        if self.spatial_as_samples:
            if self.pool_time:
                return x_projected.reshape(
                    1, b * t * h * w, c_out
                )  # [1, B*T*H*W, C_out]
            return x_projected.permute(1, 0, 2, 3, 4).reshape(
                t, b * h * w, c_out
            )  # [T, B*H*W, C_out]

        x_flat = x_projected.permute(0, 1, 4, 2, 3).reshape(
            b, t, -1
        )  # [B, T, C_out*H*W]
        d = x_flat.shape[-1]
        if self.pool_time:
            return x_flat.reshape(1, b * t, d)  # [1, B*T, C_out*H*W]
        return x_flat.permute(1, 0, 2)  # [T, B, C_out*H*W]

    def _compute_anticollapse(self, x_projected, b, t, h, w, c_out):
        """Compute anti-collapse loss. Returns (weighted_loss, unweighted_loss, loss_dict)."""
        raise NotImplementedError

    def forward(self, x, actions=None):
        x_unprojected, x_spatial, x_projected, b, c, t, h, w, c_out = self._preprocess(
            x
        )
        sim_loss_t, idm_loss = self._compute_sim_idm(
            x, x_unprojected, x_spatial, x_projected, actions, b, t, h, w, c_out
        )
        ac_weighted, ac_unweighted, ac_dict = self._compute_anticollapse(
            x_projected, b, t, h, w, c_out
        )

        total_weighted_loss = (
            ac_weighted + self.sim_coeff_t * sim_loss_t + self.idm_coeff * idm_loss
        )
        total_unweighted_loss = ac_unweighted + sim_loss_t + idm_loss

        loss_dict = {
            **ac_dict,
            "sim_loss_t": sim_loss_t.detach(),
            "idm_loss": idm_loss if isinstance(idm_loss, float) else idm_loss.detach(),
        }

        return total_weighted_loss, total_unweighted_loss, loss_dict


class VC_IDM_Sim_Regularizer(_IDM_Sim_Regularizer_Base):
    """Composite regularizer with Variance-Covariance anti-collapse + IDM + temporal similarity."""

    def __init__(
        self,
        cov_coeff: float,
        std_coeff: float,
        std_margin: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()

    def _compute_anticollapse(self, x_projected, b, t, h, w, c_out):
        groups = self._get_x_for_anticollapse(
            x_projected, b, t, h, w, c_out
        )  # [G, N, D]
        g_std = []
        g_cov = []
        for g in range(groups.shape[0]):
            g_std.append(self.std_loss_fn(groups[g]))
            g_cov.append(self.cov_loss_fn(groups[g]))
        std_loss = torch.stack(g_std).mean()
        cov_loss = torch.stack(g_cov).mean()

        weighted = self.cov_coeff * cov_loss + self.std_coeff * std_loss
        unweighted = cov_loss + std_loss
        loss_dict = {
            "cov_loss": cov_loss.detach(),
            "std_loss": std_loss.detach(),
        }
        return weighted, unweighted, loss_dict


class SIGReg_IDM_Sim_Regularizer(_IDM_Sim_Regularizer_Base):
    """Composite regularizer with SIGReg (BCS) anti-collapse + IDM + temporal similarity."""

    def __init__(
        self,
        sigreg_coeff: float,
        num_slices: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigreg_coeff = sigreg_coeff
        self.num_slices = num_slices
        self.step = 0
        self._total_n = None
        self.epps = EppsPulley()

    def _compute_anticollapse(self, x_projected, b, t, h, w, c_out):
        groups = self._get_x_for_anticollapse(
            x_projected, b, t, h, w, c_out
        )  # [G, N, D]
        n_samples = groups.shape[1]

        if self._total_n is None:
            self._total_n = _total_batch_size(n_samples)

        group_losses = []
        for g in range(groups.shape[0]):
            loss_g, self.step = _sliced_epps_pulley(
                groups[g], self.step, self.num_slices, self._total_n, self.epps
            )
            group_losses.append(loss_g)
        sigreg_loss = torch.stack(group_losses).mean()

        weighted = self.sigreg_coeff * sigreg_loss
        loss_dict = {"sigreg_loss": sigreg_loss.detach()}
        return weighted, sigreg_loss, loss_dict


class ActionVCRegularizer(nn.Module):
    """Variance-Covariance regularizer for encoded action outputs.

    Wraps HingeStdLoss + CovarianceLoss to prevent action encoder collapse
    at hierarchy levels >= 2. Follows the same return signature as state
    regularizers: (weighted_loss, unweighted_loss, loss_dict).
    """

    def __init__(
        self,
        std_coeff: float,
        cov_coeff: float,
        std_margin: float = 1.0,
    ):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()

    def forward(
        self, actions_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute variance-covariance regularization on encoded actions.

        Args:
            actions_encoded: Encoded actions [B, A_enc, T].

        Returns:
            Tuple of (total_weighted_loss, total_unweighted_loss, loss_dict).
        """
        B, A_enc, T = actions_encoded.shape
        x = actions_encoded.permute(0, 2, 1).reshape(B * T, A_enc)  # [B*T, A_enc]

        std_loss = self.std_loss_fn(x)
        cov_loss = self.cov_loss_fn(x)

        total_weighted = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted = std_loss + cov_loss
        loss_dict = {
            "action_std_loss": std_loss.detach(),
            "action_cov_loss": cov_loss.detach(),
        }
        return total_weighted, total_unweighted, loss_dict


class ActionSIGRegRegularizer(nn.Module):
    """SIGReg (BCS) regularizer for encoded action outputs.

    Computes Epps-Pulley Gaussianity statistic per timestep (each with
    N=B samples) and averages over T, consistent with the state regularizer
    convention.
    """

    def __init__(self, sigreg_coeff: float, num_slices: int = 1024):
        super().__init__()
        self.sigreg_coeff = sigreg_coeff
        self.num_slices = num_slices
        self.step = 0
        self._total_n = None
        self.epps = EppsPulley()

    def forward(
        self, actions_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute SIGReg regularization on encoded actions.

        Args:
            actions_encoded: Encoded actions [B, A_enc, T].

        Returns:
            Tuple of (total_weighted_loss, total_unweighted_loss, loss_dict).
        """
        B, A_enc, T = actions_encoded.shape
        x = actions_encoded.permute(2, 0, 1)  # [T, B, A_enc]

        if self._total_n is None:
            self._total_n = _total_batch_size(B)

        t_losses = []
        for ti in range(T):
            loss_t, self.step = _sliced_epps_pulley(
                x[ti], self.step, self.num_slices, self._total_n, self.epps
            )
            t_losses.append(loss_t)
        sigreg_loss = torch.stack(t_losses).mean()

        total_weighted = self.sigreg_coeff * sigreg_loss
        loss_dict = {"action_sigreg_loss": sigreg_loss.detach()}
        return total_weighted, sigreg_loss, loss_dict
