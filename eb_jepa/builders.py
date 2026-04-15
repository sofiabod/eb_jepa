"""Shared 1-level builder functions for JEPA model components.

Each builder takes a sub-config (e.g., the encoder config dict), not the
top-level cfg, so they are config-shape-agnostic. Both flat (ac_video_jepa)
and hierarchical (h_ac_video_jepa) examples use these builders.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from eb_jepa.jepa import JEPAProbe
from eb_jepa.losses.prediction import LPIPSLoss, SquareLossSeq
from eb_jepa.losses.regularizers import (
    SIGReg_IDM_Sim_Regularizer,
    VC_IDM_Sim_Regularizer,
)
from eb_jepa.models.components import (
    ActionMLP,
    AttentiveInverseDynamicsModel,
    ConvEncoder,
    CostModule,
    InverseDynamicsModel,
    MLPEncoder,
    Projector,
)
from eb_jepa.models.decoders import (
    ResNetVisualDecoder,
    SpatialVisualDecoder,
    VisualDecoder,
    ViTVisualDecoder,
)
from eb_jepa.models.encoders import (
    DinoEncoder,
    ImpalaEncoder,
    TorchVisionEncoder,
    ViTCLSEncoder,
    ViTEncoder,
)
from eb_jepa.models.predictors import (
    CausalTransformerPredictor,
    ConvGRUPredictor,
    ConvNeXtGRUPredictor,
    RNNPredictor,
    SpatialCausalTransformerPredictor,
    UNetGRUPredictor,
)
from eb_jepa.models.probes import AttentiveXYHead, MLPXYHead
from eb_jepa.utils.logging import get_logger
from eb_jepa.utils.optimizers import LARS

logger = get_logger(__name__)


def build_encoder(
    cfg_encoder,
    *,
    input_channels: int,
    img_size: int,
    device: torch.device,
    input_dim: Optional[int] = None,
    prev_spatial_size: int = 1,
) -> Tuple[nn.Module, int, Tuple[int, int]]:
    """Build an encoder from a sub-config dict.

    Dispatches on ``cfg_encoder.architecture`` to build the appropriate
    encoder. Supports both image-level encoders (level 1) and embedding-level
    encoders (level > 1).

    Args:
        cfg_encoder: Encoder sub-config (OmegaConf or dict).
        input_channels: Number of input channels (image channels for level 1,
            or previous encoder output_dim for level > 1).
        img_size: Image size for level-1 encoders, or previous spatial size
            for level > 1 encoders.
        device: Torch device.
        input_dim: Previous encoder output dim for level > 1 encoders. When
            provided, overrides ``input_channels`` for MLPEncoder/ConvEncoder.
        prev_spatial_size: Previous spatial size for level > 1 encoders.

    Returns:
        Tuple of (encoder, output_dim, (spatial_h, spatial_w)).
    """
    enc_params = dict(cfg_encoder)
    architecture = enc_params.pop("architecture", "impala")
    enc_params.pop("freeze", None)

    if architecture == "impala":
        enc_params["stack_sizes"] = tuple(enc_params.get("stack_sizes", [16, 32, 32]))
        output_dim = enc_params.pop("output_dim", 512)
        encoder = ImpalaEncoder(
            mlp_output_dim=output_dim,
            input_channels=input_channels,
            input_shape=(input_channels, img_size, img_size),
            **enc_params,
        )
        spatial_size = (1, 1)

    elif architecture == "vit_cls":
        encoder = ViTCLSEncoder(
            image_size=img_size, input_channels=input_channels, **enc_params
        )
        output_dim = encoder.output_dim
        spatial_size = (1, 1)

    elif architecture == "vit":
        patch_size = enc_params.get("patch_size", 16)
        encoder = ViTEncoder(
            image_size=img_size, input_channels=input_channels, **enc_params
        )
        output_dim = encoder.output_dim
        grid = img_size // patch_size
        spatial_size = (grid, grid)

    elif architecture == "torchvision":
        encoder = TorchVisionEncoder(input_channels=input_channels, **enc_params)
        output_dim = encoder.output_dim
        out_spatial = enc_params.get("output_spatial", 1) or 1
        spatial_size = (out_spatial, out_spatial)

    elif architecture == "dino":
        encoder = DinoEncoder(image_size=img_size, **enc_params)
        output_dim = encoder.output_dim
        grid = encoder.grid_size
        spatial_size = (grid, grid)

    elif architecture == "mlp":
        assert input_dim is not None, "input_dim required for MLPEncoder"
        encoder = MLPEncoder(
            input_dim=input_dim,
            final_ln=enc_params.pop("final_ln", True),
            **enc_params,
        )
        output_dim = encoder.output_dim
        spatial_size = (prev_spatial_size, prev_spatial_size)

    elif architecture == "conv":
        assert input_dim is not None, "input_dim required for ConvEncoder"
        stride = enc_params.pop("stride", 2)
        encoder = ConvEncoder(
            input_dim=input_dim,
            stride=stride,
            final_ln=enc_params.pop("final_ln", True),
            **enc_params,
        )
        output_dim = encoder.output_dim
        s = max(1, prev_spatial_size // stride)
        spatial_size = (s, s)

    else:
        raise ValueError(f"Unknown encoder architecture: {architecture}")

    output_dim = getattr(encoder, "output_dim", None) or getattr(
        encoder, "mlp_output_dim", output_dim
    )
    return encoder, output_dim, spatial_size


def build_predictor(
    cfg_predictor,
    *,
    input_dim: int,
    action_dim: int,
    spatial_size: Tuple[int, int],
    device: torch.device,
    train_autoenc_only: bool = False,
) -> nn.Module:
    """Build a predictor from a sub-config dict.

    Dispatches on ``cfg_predictor.type`` to build the appropriate predictor.

    Args:
        cfg_predictor: Predictor sub-config (OmegaConf or dict).
        input_dim: Encoder output dimension.
        action_dim: Action vector dimension.
        spatial_size: Spatial size (H, W) of encoder output.
        device: Torch device.
        train_autoenc_only: If True, returns nn.Identity() (no prediction).

    Returns:
        Predictor module.
    """
    if train_autoenc_only:
        logger.info(
            "Autoencoder-only mode: using Identity predictor (no prediction loss)"
        )
        return nn.Identity()

    pred_cfg = dict(cfg_predictor)
    pred_type = pred_cfg.pop("type", "rnn")
    h = spatial_size[0]

    if pred_type == "rnn":
        rnn_kw = dict(
            predictor_dim=pred_cfg.get("predictor_dim", input_dim),
            action_dim=action_dim,
            num_layers=pred_cfg.get("num_layers", 1),
        )
        if "final_ln" in pred_cfg:
            rnn_kw["final_ln"] = pred_cfg["final_ln"]
        return RNNPredictor(**rnn_kw)

    elif pred_type == "causal_transformer":
        use_residual = pred_cfg.pop("use_residual", False)
        return CausalTransformerPredictor(
            input_dim=input_dim,
            action_dim=action_dim,
            use_residual=use_residual,
            **pred_cfg,
        )

    elif pred_type == "spatial_causal_transformer":
        use_residual = pred_cfg.pop("use_residual", False)
        return SpatialCausalTransformerPredictor(
            input_dim=input_dim,
            action_dim=action_dim,
            spatial_size=h,
            use_residual=use_residual,
            **pred_cfg,
        )

    elif pred_type == "conv_gru":
        gru_kw = dict(
            predictor_dim=pred_cfg.get("predictor_dim", input_dim),
            spatial_size=pred_cfg.get("spatial_size", h),
            action_dim=action_dim,
            kernel_size=pred_cfg.get("kernel_size", 3),
        )
        if "final_ln" in pred_cfg:
            gru_kw["final_ln"] = pred_cfg["final_ln"]
        return ConvGRUPredictor(**gru_kw)

    elif pred_type == "convnext_gru":
        cnext_kw = dict(
            input_dim=input_dim,
            spatial_size=h,
            action_dim=action_dim,
            predictor_dim=pred_cfg.get("predictor_dim", 64),
            num_blocks=pred_cfg.get("num_blocks", 4),
            expansion_ratio=pred_cfg.get("expansion_ratio", 4),
        )
        if "final_ln" in pred_cfg:
            cnext_kw["final_ln"] = pred_cfg["final_ln"]
        return ConvNeXtGRUPredictor(**cnext_kw)

    elif pred_type == "unet_gru":
        unet_kw = dict(
            input_dim=input_dim,
            spatial_size=h,
            action_dim=action_dim,
            predictor_dim=pred_cfg.get("predictor_dim", 64),
            base_channels=pred_cfg.get("base_channels", 64),
        )
        if "final_ln" in pred_cfg:
            unet_kw["final_ln"] = pred_cfg["final_ln"]
        return UNetGRUPredictor(**unet_kw)

    else:
        raise ValueError(f"Unknown predictor type: {pred_type}")


def build_regularizer(
    cfg_reg,
    *,
    encoder_output_dim: int,
    spatial_size: Tuple[int, int],
    action_dim: int,
    device: torch.device,
    use_proj: bool = False,
) -> nn.Module:
    """Build a regularizer from a sub-config dict.

    Args:
        cfg_reg: Regularizer sub-config (OmegaConf or dict).
        encoder_output_dim: Channel dimension of encoder output.
        spatial_size: Spatial size (H, W) of encoder output.
        action_dim: Action dimension for the IDM.
        device: Torch device.
        use_proj: Whether to use a projector in the regularizer.

    Returns:
        Regularizer module.
    """
    h, w = spatial_size
    is_spatial = h > 1 or w > 1

    projector = None
    if use_proj:
        flat_dim = encoder_output_dim * h * w
        projector = Projector(f"{flat_dim}-2048-{flat_dim}")

    idm_coeff = cfg_reg.get("idm_coeff", 0.1)
    idm = None
    if idm_coeff > 0:
        idm_cfg = cfg_reg.get("idm", {})
        idm_type = idm_cfg.get("type", "attentive" if is_spatial else "mlp")

        if idm_type == "attentive" and is_spatial:
            idm = AttentiveInverseDynamicsModel(
                input_dim=encoder_output_dim,
                action_dim=action_dim,
                embed_dim=idm_cfg.get("embed_dim", min(encoder_output_dim, 384)),
                depth=idm_cfg.get("depth", 3),
                num_heads=idm_cfg.get("num_heads", None),
                mlp_ratio=idm_cfg.get("mlp_ratio", 4.0),
                grid_size=h if h == w else None,
                hidden_dim=idm_cfg.get("hidden_dim", 256),
            ).to(device)
        else:
            idm_state_dim = encoder_output_dim * h * w
            idm = InverseDynamicsModel(
                state_dim=idm_state_dim,
                hidden_dim=idm_cfg.get("hidden_dim", 256),
                action_dim=action_dim,
            ).to(device)

    reg_type = cfg_reg.get("type", "vc")
    shared_kwargs = dict(
        sim_coeff_t=cfg_reg.get("sim_coeff_t", 12),
        idm_coeff=idm_coeff,
        idm=idm,
        pool_time=cfg_reg.get("pool_time", False),
        projector=projector,
        spatial_as_samples=cfg_reg.get("spatial_as_samples", False),
        reg_per_patch=cfg_reg.get("reg_per_patch", False),
        idm_after_proj=cfg_reg.get("idm_after_proj", False),
        sim_t_after_proj=cfg_reg.get("sim_t_after_proj", False),
    )

    if reg_type == "sigreg":
        return SIGReg_IDM_Sim_Regularizer(
            sigreg_coeff=cfg_reg.get("sigreg_coeff", 10.0),
            num_slices=cfg_reg.get("num_slices", 1024),
            **shared_kwargs,
        )
    return VC_IDM_Sim_Regularizer(
        cov_coeff=cfg_reg.get("cov_coeff", 8),
        std_coeff=cfg_reg.get("std_coeff", 16),
        **shared_kwargs,
    )


def build_predcost() -> nn.Module:
    """Build prediction cost (``SquareLossSeq``)."""
    return SquareLossSeq()


def build_cost_module(
    cfg_cost,
    *,
    encoder_output_dim: int,
    device: torch.device,
) -> Optional[CostModule]:
    """Build an optional ``CostModule`` (projector + straightening loss).

    Args:
        cfg_cost: Cost sub-config (OmegaConf, dict, or None).
        encoder_output_dim: Channel dimension of encoder output.
        device: Torch device.

    Returns:
        CostModule or None if not configured.
    """
    from eb_jepa.losses.prediction import TemporalStraighteningLoss

    if cfg_cost is None or not cfg_cost.get("projector", {}).get("enabled", False):
        return None

    proj_cfg = cfg_cost.projector
    loss_cfg = cfg_cost.get("loss", {})
    mlp_spec = proj_cfg.get(
        "mlp_spec",
        f"{encoder_output_dim}-{encoder_output_dim // 2}-{encoder_output_dim // 4}",
    )

    projector = Projector(mlp_spec).to(device)
    loss_fn = TemporalStraighteningLoss(
        projector=projector,
        detach=loss_cfg.get("detach_encoder", True),
    )
    cost_module = CostModule(projector=projector, loss=loss_fn)
    logger.info(
        f"Cost module: projector={mlp_spec}, "
        f"detach_encoder={loss_cfg.get('detach_encoder', True)}"
    )
    return cost_module


def build_action_encoder(
    cfg_aenc,
    *,
    input_action_dim: int,
    temporal_stride: int,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    """Build an action encoder for a hierarchy level.

    Args:
        cfg_aenc: Action encoder sub-config, or None for nn.Identity().
        input_action_dim: Action dim entering this level (raw for level 2,
            or previous action encoder output_dim for level > 2).
        temporal_stride: Temporal stride at this level (window size).
        device: Torch device.

    Returns:
        Tuple of (action_encoder, output_action_dim).
    """
    if cfg_aenc is None:
        return nn.Identity(), input_action_dim

    return (
        ActionMLP(
            input_dim=temporal_stride * input_action_dim,
            hidden_dims=cfg_aenc.get("hidden_dims", [64]),
            output_dim=cfg_aenc.get("output_dim", input_action_dim),
            final_ln=cfg_aenc.get("final_ln", False),
        ),
        cfg_aenc.get("output_dim", input_action_dim),
    )


def build_visual_decoder(
    vd_cfg,
    *,
    encoder_output_dim: int,
    spatial_h: int,
    num_channels: int,
    img_size: int,
    cfg_data,
    device: torch.device,
    encoder_scale: str = "tiny",
    encoder_patch_size: int = 16,
) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
    """Build an optional visual decoder with LPIPS loss components.

    Dispatches on ``vd_cfg.type``: ``"vit"`` → ViTVisualDecoder,
    ``"resnet"`` → ResNetVisualDecoder, spatial (``spatial_h > 1``) →
    SpatialVisualDecoder, else → VisualDecoder.

    Args:
        vd_cfg: Visual decoder sub-config (OmegaConf or dict). Must have
            ``enabled=True`` to build a decoder.
        encoder_output_dim: Channel dimension of encoder output.
        spatial_h: Spatial height of encoder output (1 for non-spatial).
        num_channels: Number of image channels (e.g. 3 for RGB).
        img_size: Target image size (square).
        cfg_data: Data sub-config for extracting normalization params.
        device: Torch device.
        encoder_scale: ViT encoder scale for ViTVisualDecoder (default
            ``"tiny"``).
        encoder_patch_size: Encoder patch size for ViTVisualDecoder
            (default 16).

    Returns:
        Tuple of ``(visual_decoder, lpips_loss, lpips_fn)``. All ``None``
        when the decoder is not enabled.
    """
    if not vd_cfg or not vd_cfg.get("enabled", False):
        return None, None, None

    # Normalization config with ImageNet fallback
    normalize_cfg = cfg_data.get("transform", {}).get("normalize") if cfg_data else None
    if normalize_cfg is None:
        normalize_cfg = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize_mean = list(normalize_cfg[0])
    normalize_std = list(normalize_cfg[1])

    vd_type = vd_cfg.get("type", "conv")
    if vd_type == "vit":
        visual_decoder = ViTVisualDecoder(
            input_dim=encoder_output_dim,
            scale=vd_cfg.get("vit_scale", encoder_scale),
            patch_size=vd_cfg.get("vit_patch_size", encoder_patch_size),
            image_size=img_size,
            output_channels=num_channels,
            depth_override=vd_cfg.get("vit_depth", None),
            use_refine=vd_cfg.get("use_refine", False),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        ).to(device)
        logger.info(
            f"ViT visual decoder built "
            f"(scale={vd_cfg.get('vit_scale', encoder_scale)})"
        )
    elif vd_type == "resnet":
        visual_decoder = ResNetVisualDecoder(
            input_dim=encoder_output_dim,
            input_spatial=spatial_h,
            output_channels=num_channels,
            target_h=img_size,
            target_w=img_size,
            base_channels=vd_cfg.get("base_channels", 256),
            min_channels=vd_cfg.get("min_channels", 64),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        ).to(device)
        logger.info(
            f"ResNet visual decoder built (input_spatial={spatial_h}, "
            f"base_channels={vd_cfg.get('base_channels', 256)})"
        )
    elif spatial_h > 1:
        visual_decoder = SpatialVisualDecoder(
            input_dim=encoder_output_dim,
            input_spatial=spatial_h,
            output_channels=num_channels,
            target_h=img_size,
            target_w=img_size,
            base_channels=vd_cfg.get("base_channels", 256),
            min_channels=vd_cfg.get("min_channels", 64),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        ).to(device)
        logger.info(
            f"Spatial visual decoder built (input_spatial={spatial_h}, "
            f"base_channels={vd_cfg.get('base_channels', 256)})"
        )
    else:
        visual_decoder = VisualDecoder(
            input_dim=encoder_output_dim,
            output_channels=num_channels,
            target_h=img_size,
            target_w=img_size,
            base_channels=vd_cfg.get("base_channels", 256),
            init_spatial=vd_cfg.get("init_spatial", 1),
            min_channels=vd_cfg.get("min_channels", 64),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        ).to(device)
        logger.info(
            f"Visual decoder built "
            f"(base_channels={vd_cfg.get('base_channels', 256)})"
        )

    import lpips as lpips_lib

    lpips_fn = lpips_lib.LPIPS(net="vgg").eval().to(device)
    lpips_fn.requires_grad_(False)

    lpips_loss = None
    if vd_cfg.get("use_lpips", False):
        lpips_loss = LPIPSLoss(
            pixel_weight=vd_cfg.get("pixel_weight", 10.0),
            perceptual_weight=vd_cfg.get("perceptual_weight", 1.0),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            lpips_chunk_size=vd_cfg.get("lpips_chunk_size", 32),
        ).to(device)

    return visual_decoder, lpips_loss, lpips_fn


def build_xy_prober(
    probe_cfg,
    *,
    jepa: nn.Module,
    encoder_output_dim: int,
    spatial_h: int,
    normalizer,
    device: torch.device,
    pos_mean: Optional[torch.Tensor] = None,
    pos_std: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, JEPAProbe]:
    """Build an XY prober (head + JEPAProbe wrapper).

    Dispatches on ``probe_cfg.type``: ``"attentive"`` → AttentiveXYHead,
    else → MLPXYHead. Wraps in a ``JEPAProbe`` with MSE loss.

    Args:
        probe_cfg: Probe sub-config (OmegaConf or dict).
        jepa: JEPA model instance to wrap.
        encoder_output_dim: Channel dimension of encoder output.
        spatial_h: Spatial height of encoder output (used for
            AttentiveXYHead ``grid_size``).
        normalizer: Optional normalizer for the head.
        device: Torch device.
        pos_mean: Per-dim position mean ``[output_dim]`` for Z-score
            normalization. None disables normalization.
        pos_std: Per-dim position std ``[output_dim]`` for Z-score
            normalization.

    Returns:
        Tuple of ``(xy_head, xy_prober)``.
    """
    state_dims = list(probe_cfg.get("state_dims", [0, 1]))
    output_dim = len(state_dims)
    probe_type = probe_cfg.get("type", "mlp")

    if probe_type == "attentive":
        grid_size = spatial_h if spatial_h > 1 else None
        xy_head = AttentiveXYHead(
            input_shape=encoder_output_dim,
            output_dim=output_dim,
            decoder_embed_dim=probe_cfg.get("decoder_embed_dim", 384),
            depth=probe_cfg.get("depth", 3),
            grid_size=grid_size,
            normalizer=normalizer,
            pos_mean=pos_mean,
            pos_std=pos_std,
        ).to(device)
    else:
        hidden_dims = list(probe_cfg.get("hidden_dims", [512]))
        xy_head = MLPXYHead(
            input_shape=encoder_output_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            normalizer=normalizer,
            pos_mean=pos_mean,
            pos_std=pos_std,
        ).to(device)

    xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())
    return xy_head, xy_prober


def build_optimizer(
    cfg_optim,
    param_groups: list,
) -> torch.optim.Optimizer:
    """Build an optimizer from a sub-config dict.

    Dispatches on ``cfg_optim.type`` (default ``"adamw"``). Each example
    constructs its own ``param_groups`` (inherently example-specific) and
    passes them here for unified optimizer creation.

    Args:
        cfg_optim: Optimizer sub-config (OmegaConf or dict). Recognised
            keys: ``type``, ``weight_decay``, ``momentum``, ``lars_eta``,
            ``lars_clip_lr``, ``lars_exclude_bias_n_norm``.
        param_groups: List of parameter group dicts, each with at least
            a ``"params"`` key and typically a ``"lr"`` key.

    Returns:
        Configured ``torch.optim.Optimizer``.

    Raises:
        ValueError: If ``cfg_optim.type`` is not one of
            ``{"adamw", "adam", "lars"}``.
    """
    optim_type = cfg_optim.get("type", "adamw")

    if optim_type == "adamw":
        return torch.optim.AdamW(
            param_groups,
            weight_decay=cfg_optim.get("weight_decay", 0),
        )
    elif optim_type == "adam":
        return torch.optim.Adam(param_groups)
    elif optim_type == "lars":
        return LARS(
            param_groups,
            weight_decay=cfg_optim.get("weight_decay", 0),
            momentum=cfg_optim.get("momentum", 0.9),
            eta=cfg_optim.get("lars_eta", 0.001),
            clip_lr=cfg_optim.get("lars_clip_lr", True),
            exclude_bias_n_norm=cfg_optim.get("lars_exclude_bias_n_norm", True),
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {optim_type!r}. "
            f"Supported: 'adamw', 'adam', 'lars'."
        )


def build_action_regularizer(
    cfg_reg,
    *,
    action_dim: int,
    device: torch.device,
) -> Optional[nn.Module]:
    """Build an action regularizer for a hierarchy level.

    Dispatches to ``ActionVCRegularizer`` (default / VC) or
    ``ActionSIGRegRegularizer`` (when regularizer type is ``sigreg``).

    Args:
        cfg_reg: Regularizer sub-config containing action reg coefficients.
        action_dim: Dimension of encoded actions at this level.
        device: Device to place the module on.

    Returns:
        Action regularizer module, or None if not configured.
    """
    from eb_jepa.losses.regularizers import ActionSIGRegRegularizer, ActionVCRegularizer

    if cfg_reg is None:
        return None

    reg_type = cfg_reg.get("type", "vc")

    if reg_type == "sigreg":
        sigreg_coeff = cfg_reg.get("action_sigreg_coeff", 0.0)
        if sigreg_coeff == 0.0:
            return None
        return ActionSIGRegRegularizer(
            sigreg_coeff=sigreg_coeff,
            num_slices=cfg_reg.get("num_slices", 1024),
        ).to(device)

    std_coeff = cfg_reg.get("action_std_coeff", 0.0)
    cov_coeff = cfg_reg.get("action_cov_coeff", 0.0)
    if std_coeff == 0.0 and cov_coeff == 0.0:
        return None
    return ActionVCRegularizer(
        std_coeff=std_coeff,
        cov_coeff=cov_coeff,
    ).to(device)
