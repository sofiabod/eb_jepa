from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


# Mapping from dot-notation config keys to short abbreviations for folder naming.
# Used by launch_sbatch.py when building experiment folder names from swept params.
# Add new abbreviations here when adding new sweep parameters.
# IMPORTANT: Abbreviations must be unique to avoid folder naming conflicts.

SWEEP_PARAM_ABBREV = {
    # Meta
    "meta.seed": "s",
    # Regularizer coefficients (ac_video_jepa) - VC
    "model.regularizer.cov_coeff": "cov",
    "model.regularizer.std_coeff": "std",
    "model.regularizer.sim_coeff_t": "simt",
    "model.regularizer.idm_coeff": "idm",
    # Regularizer coefficients (ac_video_jepa) - SIGReg
    "model.regularizer.sigreg_coeff": "sigr",
    # Hierarchical regularizer coefficients (h_ac_video_jepa) - Level 1
    "model.level_1.regularizer.cov_coeff": "lvl1cov",
    "model.level_1.regularizer.std_coeff": "lvl1std",
    "model.level_1.regularizer.sim_coeff_t": "lvl1simt",
    "model.level_1.regularizer.idm_coeff": "lvl1idm",
    "model.level_1.regularizer.sigreg_coeff": "lvl1sigr",
    # Hierarchical regularizer coefficients (h_ac_video_jepa) - Level 2
    "model.level_2.regularizer.cov_coeff": "lvl2cov",
    "model.level_2.regularizer.std_coeff": "lvl2std",
    "model.level_2.regularizer.sim_coeff_t": "lvl2simt",
    "model.level_2.regularizer.idm_coeff": "lvl2idm",
    "model.level_2.regularizer.sigreg_coeff": "lvl2sigr",
    # Hierarchical regularizer coefficients (h_ac_video_jepa) - Level 3
    "model.level_3.regularizer.cov_coeff": "lvl3cov",
    "model.level_3.regularizer.std_coeff": "lvl3std",
    "model.level_3.regularizer.sim_coeff_t": "lvl3simt",
    "model.level_3.regularizer.idm_coeff": "lvl3idm",
    "model.level_3.regularizer.sigreg_coeff": "lvl3sigr",
    # Hierarchical action regularizer coefficients
    "model.level_2.regularizer.action_std_coeff": "lvl2astd",
    "model.level_3.regularizer.action_std_coeff": "lvl3astd",
    "model.level_2.regularizer.action_sigreg_coeff": "lvl2asigr",
    "model.level_3.regularizer.action_sigreg_coeff": "lvl3asigr",
    # Hierarchical level weights
    "model.level_weights": "lw",
    # Hierarchical architecture
    "model.level_2.action_encoder.output_dim": "lvl2aed",
    "model.level_3.action_encoder.output_dim": "lvl3aed",
    "model.level_3.encoder.output_dim": "lvl3ed",
    "model.nsteps": "ns",
    # Cost module (for planning objectives)
    "model.cost.projector.mlp_spec": "mlp",
    "model.cost.loss.detach_encoder": "det",
    # Optimizer
    "optim.lr": "lr",
    "optim.epochs": "ep",
    "optim.weight_decay": "wd",
    "optim.lr_scales.level_2": "lrs2",
    "optim.lr_scales.level_3": "lrs3",
    # Data
    "data.batch_size": "bs",
    # Loss (image_jepa, video_jepa)
    "loss.cov_coeff": "lcov",
    "loss.std_coeff": "lstd",
    "loss.lmbd": "lmbd",
}


def _cfg_get(cfg, dotted_key: str, default=None):
    """Retrieve value from nested config using dot notation.

    Returns ``default`` if any intermediate key is missing or ``None``."""
    val = cfg
    for k in dotted_key.split("."):
        try:
            val = val.get(k) if hasattr(val, "get") else getattr(val, k)
        except (KeyError, AttributeError):
            return default
        if val is None:
            return default
    return val


def _encode_params(cfg, param_keys: List[str]) -> str:
    """Encode config values as 'abbrev1val1_abbrev2val2_...' using SWEEP_PARAM_ABBREV."""
    parts = []
    for key in param_keys:
        abbrev = SWEEP_PARAM_ABBREV.get(key, key.split(".")[-1][:4])
        parts.append(f"{abbrev}{_cfg_get(cfg, key)}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Experiment naming: build unique exp names from config
# Format: {dataset}_{encoder}_{predictor}_{regularizer}[_flags]
# ---------------------------------------------------------------------------

_PRED_ABBREV = {
    "rnn": "rnn",
    "causal_transformer": "causalT",
    "spatial_causal_transformer": "spatialCT",
    "conv_gru": "convGRU",
    "convnext_gru": "cnxGRU",
    "unet_gru": "unetGRU",
}


def _enc_tag(cfg, prefix: str = "model") -> str:
    """Short encoder id: 'imp', 'vit-s-p14', 'dino-v2s14-frozen', 'tv-rn18'."""
    arch = _cfg_get(cfg, f"{prefix}.encoder.architecture", "impala")
    g = lambda k, d=None: _cfg_get(cfg, f"{prefix}.{k}", d)
    if arch == "impala":
        s = "imp"
    elif arch in ("vit", "vit_cls"):
        tag = "vitcls" if arch == "vit_cls" else "vit"
        s = f"{tag}-{str(g('encoder.scale', 'small'))[0]}-p{g('encoder.patch_size', 16)}"
    elif arch == "dino":
        m = g("encoder.name", "dinov2_vits14")
        s = f"dino-{m.replace('dino', '').replace('_vit', '')}"
    elif arch == "torchvision":
        bb = g("encoder.backbone", "resnet18")
        s = f"tv-{bb.replace('resnet', 'rn').replace('efficientnet_', 'eff')}"
    else:
        s = arch
    if g("encoder.freeze", False):
        s += "-frozen"
    return s


def _pred_tag(cfg, prefix: str = "model") -> str:
    """Short predictor id: 'rnn', 'spatialCT-d6-p96', 'causalT-d6-h16'."""
    pt = _cfg_get(cfg, f"{prefix}.predictor.type", "rnn")
    s = _PRED_ABBREV.get(pt, pt)
    g = lambda k: _cfg_get(cfg, f"{prefix}.predictor.{k}")
    if pt == "spatial_causal_transformer" and g("depth") and g("predictor_dim"):
        s += f"-d{g('depth')}-p{g('predictor_dim')}"
    elif pt == "causal_transformer" and g("depth") and g("heads"):
        s += f"-d{g('depth')}-h{g('heads')}"
    return s


def _reg_keys(reg_type: str, prefix: str) -> List[str]:
    """Regularizer coeff keys: sigreg uses sigreg_coeff, VC uses cov+std."""
    shared = [f"{prefix}.sim_coeff_t", f"{prefix}.idm_coeff"]
    if reg_type == "sigreg":
        return [f"{prefix}.sigreg_coeff"] + shared
    return [f"{prefix}.cov_coeff", f"{prefix}.std_coeff"] + shared


def _ac_base(cfg):
    """Name: {dataset}_{enc}_{pred}_{reg}[_detPT] + reg coeff keys."""
    ds = _cfg_get(cfg, "data.env_name") or _cfg_get(cfg, "data.dataset") or "unk"
    rt = _cfg_get(cfg, "model.regularizer.type", "vc")
    prefix = f"{ds}_{_enc_tag(cfg)}_{_pred_tag(cfg)}_{rt}"
    if _cfg_get(cfg, "model.rollout.detach_pred_target", False):
        prefix += "_detPT"
    return prefix, _reg_keys(rt, "model.regularizer")


def _hac_base(cfg):
    """Name: {dataset}_h{N}lvl-{enc}-{pred}-{reg} + per-level reg coeff keys."""
    ds = _cfg_get(cfg, "data.env_name") or _cfg_get(cfg, "data.dataset") or "unk"
    rt = _cfg_get(cfg, "model.level_1.regularizer.type", "vc")
    n = cfg.model.num_levels
    prefix = (
        f"{ds}_h{n}lvl-{_enc_tag(cfg, 'model.level_1')}"
        f"-{_pred_tag(cfg, 'model.level_1')}-{rt}"
    )
    keys = []
    for lvl in range(1, n + 1):
        keys.extend(_reg_keys(rt, f"model.level_{lvl}.regularizer"))
    return prefix, keys


_BASE_PARAMS = {"ac_video_jepa": _ac_base, "h_ac_video_jepa": _hac_base}


def get_exp_name(example_name: str, cfg, param_grid: Optional[dict] = None) -> str:
    """Get short experiment name encoding key hyperparameters (seed appended separately).

    If param_grid is provided, appends a suffix for swept params not already
    in the base name.
    """
    base_keys: List[str] = []

    if example_name in _BASE_PARAMS:
        prefix, base_keys = _BASE_PARAMS[example_name](cfg)
        base_name = f"{prefix}_{_encode_params(cfg, base_keys)}"
    elif example_name == "image_jepa":
        ds = _cfg_get(cfg, "data.dataset") or "unk"
        proj = "proj" if cfg.model.use_projector else "noproj"
        parts = [
            ds,
            cfg.model.type,
            cfg.loss.type,
            cfg.optim.get("type", "lars"),
            proj,
            f"bs{cfg.data.batch_size}",
            f"ep{cfg.optim.epochs}",
        ]
        base_keys = [
            "data.dataset",
            "model.type",
            "loss.type",
            "optim.type",
            "model.use_projector",
            "data.batch_size",
            "optim.epochs",
        ]
        if cfg.model.use_projector:
            parts.append(f"ph{cfg.model.proj_hidden_dim}")
            parts.append(f"po{cfg.model.proj_output_dim}")
            base_keys += ["model.proj_hidden_dim", "model.proj_output_dim"]
        if cfg.loss.type == "vicreg":
            parts.append(f"std{cfg.loss.std_coeff}")
            parts.append(f"cov{cfg.loss.cov_coeff}")
            base_keys += ["loss.std_coeff", "loss.cov_coeff"]
        elif cfg.loss.type == "bcs":
            parts.append(f"lmbd{cfg.loss.lmbd}")
            base_keys.append("loss.lmbd")
        base_name = "_".join(str(p) for p in parts)
    elif example_name == "video_jepa":
        base_name = (
            f"resnet_bs{cfg.data.batch_size}"
            f"_lr{cfg.optim.lr}"
            f"_std{cfg.loss.std_coeff}"
            f"_cov{cfg.loss.cov_coeff}"
        )
    else:
        base_name = "exp"

    if param_grid:
        sweep_keys = [
            k
            for k in sorted(param_grid)
            if k != "meta.seed" and k not in set(base_keys)
        ]
        if sweep_keys:
            base_name = f"{base_name}_{_encode_params(cfg, sweep_keys)}"

    return base_name


def extract_prefixed_overrides(
    overrides: Dict[str, Any],
    prefix: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split *overrides* into entries matching *prefix* and the rest.

    Keys matching ``"{prefix}.some.key"`` are returned (with the prefix
    stripped) in the first dict; all other keys go in the second.
    """
    dot_prefix = prefix + "."
    prefixed: Dict[str, Any] = {}
    remaining: Dict[str, Any] = {}
    for key, value in overrides.items():
        if key.startswith(dot_prefix):
            prefixed[key[len(dot_prefix) :]] = value
        else:
            remaining[key] = value
    return prefixed, remaining


def load_config(
    config_path: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> DictConfig:
    """Load YAML config with optional dot-notation overrides (e.g., 'model.lr': 0.001)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if not quiet:
        logger.info(f"Loaded config from {config_path}")

    if cli_overrides:
        # Convert dot notation to nested dict
        override_dict = {}
        for key, value in cli_overrides.items():
            keys = key.split(".")
            current = override_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value

        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_dict))
        if not quiet:
            logger.info(f"Applied {len(cli_overrides)} config overrides")

    return cfg


def load_config_with_prefixed_overrides(
    fname: str,
    cfg: Optional[DictConfig],
    prefixes: List[str],
    **overrides,
) -> tuple[DictConfig, Dict[str, Dict]]:
    """Load config and extract per-prefix overrides in one call.

    Handles both the CLI path (``cfg is None``: extract prefixed overrides
    from ``**overrides``, then ``load_config``) and the ``launch_sbatch``
    path (``cfg`` is pre-built: pop each prefix key and convert via
    ``OmegaConf.to_container``).

    Args:
        fname: Path to the YAML config file (used only when ``cfg is None``).
        cfg: Pre-loaded config, or ``None`` for the CLI path.
        prefixes: Prefixes to extract (e.g. ``["plan_cfg", "eval_cfg"]``).
        **overrides: CLI dot-notation overrides (used only when ``cfg is None``).

    Returns:
        ``(cfg, prefix_overrides)`` where *prefix_overrides* maps each prefix
        to its extracted override dict (may be empty).
    """
    prefix_overrides: Dict[str, Dict] = {}
    if cfg is None:
        remaining = overrides
        for prefix in prefixes:
            extracted, remaining = extract_prefixed_overrides(remaining, prefix)
            prefix_overrides[prefix] = extracted
        cfg = load_config(fname, remaining if remaining else None)
    else:
        for prefix in prefixes:
            raw = cfg.pop(prefix, None)
            prefix_overrides[prefix] = (
                OmegaConf.to_container(raw) if raw is not None else {}
            )
    return cfg, prefix_overrides


def get_checkpoints_dir() -> Path:
    """Get the base checkpoints directory from EBJEPA_CKPTS env variable."""
    return Path(os.environ.get("EBJEPA_CKPTS", "checkpoints"))


def get_unified_experiment_dir(
    example_name: str,
    sweep_name: str,
    exp_name: str,
    seed: int,
    dataset_name: str = "",
    base_dir: Union[str, Path, None] = None,
    create: bool = True,
) -> Path:
    """Create experiment dir: {base_dir}/{example_name}/{dataset_name}/{sweep_name}/{exp_name}_seed{seed}."""
    if base_dir is None:
        base_dir = get_checkpoints_dir()

    parts = [Path(base_dir), example_name]
    if dataset_name:
        parts.append(dataset_name)
    parts.extend([sweep_name, f"{exp_name}_seed{seed}"])

    exp_dir = Path(*parts).absolute()

    if create:
        exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def get_default_run_name(prefix: str = "dev") -> str:
    """Generate a timestamped run name with the given prefix."""
    return datetime.now().strftime(f"{prefix}_%Y-%m-%d_%H-%M")


def get_dataset_name(cfg) -> str:
    """Extract dataset name from config (env_name or dataset field).

    Raises:
        ValueError: If neither cfg.data.env_name nor cfg.data.dataset is set.
    """
    name = cfg.data.get("env_name") or cfg.data.get("dataset")
    if not name:
        raise ValueError("Config must set data.env_name or data.dataset")
    return name


def resolve_experiment_folder(
    example_name: str, cfg, folder: Union[str, Path, None] = None
) -> tuple[Path, str]:
    """Resolve experiment folder and name from config, creating the directory.

    Args:
        example_name: Which example (e.g. "ac_video_jepa").
        cfg: Full config object.
        folder: Explicit folder path; if None, auto-generated.

    Returns:
        (folder, exp_name) tuple.
    """
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        exp_name = folder.name.rsplit("_seed", 1)[0]
        return folder, exp_name

    if cfg.meta.get("model_folder"):
        folder = Path(cfg.meta.model_folder)
        folder.mkdir(parents=True, exist_ok=True)
        exp_name = folder.name.rsplit("_seed", 1)[0]
        return folder, exp_name

    sweep_name = get_default_run_name("dev")
    exp_name = get_exp_name(example_name, cfg)
    try:
        dataset_name = get_dataset_name(cfg)
    except ValueError:
        dataset_name = ""
    folder = get_unified_experiment_dir(
        example_name=example_name,
        sweep_name=sweep_name,
        exp_name=exp_name,
        seed=cfg.meta.seed,
        dataset_name=dataset_name,
    )
    return folder, exp_name


def log_config(cfg: Union[Dict, DictConfig], title: str = "Run Configuration") -> None:
    """Log configuration in a readable format."""
    logger.info("=" * 60)
    logger.info(f"⚙️  {title}:")
    logger.info("=" * 60)

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    for section, values in cfg.items():
        if isinstance(values, dict):
            for key, value in values.items():
                logger.info(f"  {section}.{key}={value}")
        else:
            logger.info(f"  {section}={values}")
    logger.info("=" * 60)


def log_data_info(
    dataset_name: str,
    num_batches: int,
    batch_size: int,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
) -> None:
    """Log dataset information.

    ``batch_size`` is the **global** (config-level) batch size.  When running
    multi-GPU, the per-GPU batch size is also displayed.
    """
    ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    bs_str = (
        f"{batch_size}" if ws == 1 else f"{batch_size} global ({batch_size // ws}/gpu)"
    )
    parts = [f"📦 Data: {dataset_name} | {num_batches} batches x {bs_str}"]
    if train_samples is not None and val_samples is not None:
        parts.append(f"train={train_samples:,} | val={val_samples:,}")
    logger.info(" | ".join(parts))


def log_model_info(model: nn.Module, param_counts: Dict[str, int]) -> None:
    """Log model structure and parameter counts."""
    logger.info(f"🧠 Model:\n{model}")
    param_str = " | ".join(f"{k}={v:,}" for k, v in param_counts.items())
    logger.info(f"🔢 Parameters: {param_str}")


def log_epoch(
    epoch: int,
    metrics: Dict[str, float],
    total_epochs: Optional[int] = None,
    elapsed_time: Optional[float] = None,
) -> None:
    """Log epoch summary: 📊 [Epoch 001/100] metric1=val1 | metric2=val2 | time=123.4s."""
    if total_epochs:
        prefix = f"[Epoch {epoch:03d}/{total_epochs}]"
    else:
        prefix = f"[Epoch {epoch:03d}]"

    metrics_str = format_metrics(metrics)

    if elapsed_time is not None:
        logger.info(f"📊 {prefix} {metrics_str} | time={elapsed_time:.1f}s")
    else:
        logger.info(f"📊 {prefix} {metrics_str}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dict as 'loss=0.1234 | acc=95.12'."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
