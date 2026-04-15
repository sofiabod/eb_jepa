from __future__ import annotations

import copy
import os
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer

from eb_jepa.utils.config import load_config
from eb_jepa.utils.logging import get_logger

if TYPE_CHECKING:
    from eb_jepa.h_jepa import HierarchicalJEPA

logger = get_logger(__name__)


def setup_amp(cfg, device: torch.device) -> tuple[torch.dtype, bool, GradScaler]:
    """Configure mixed-precision training from config.

    Returns:
        (dtype, use_amp, scaler) tuple.
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    use_scaler = use_amp and dtype == torch.float16
    scaler = GradScaler(device.type, enabled=use_scaler)
    logger.info(
        f"Using AMP with {dtype=}, scaler={'on' if use_scaler else 'off'}"
        if use_amp
        else "AMP disabled"
    )
    return dtype, use_amp, scaler


def optimizer_step(
    scaler: GradScaler,
    optimizers: Union[Optimizer, List[Optimizer]],
    model: Optional[nn.Module] = None,
    grad_clip: Optional[float] = None,
) -> Optional[float]:
    """Unscale, clip, step, and update scaler in one call.

    Returns the pre-clip gradient norm when *grad_clip* is set, else ``None``.
    """
    if isinstance(optimizers, Optimizer):
        optimizers = [optimizers]
    grad_norm = None
    if grad_clip is not None:
        for opt in optimizers:
            scaler.unscale_(opt)
        if model is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            ).item()
    for opt in optimizers:
        scaler.step(opt)
    scaler.update()
    return grad_norm


def setup_device(device: str = "auto") -> torch.device:
    """Set up the compute device. Options: 'auto', 'cuda', or 'cpu'.

    In distributed mode, CUDA_VISIBLE_DEVICES is set at module level so
    ``cuda:0`` always refers to the correct physical GPU.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device, 0) if device == "cuda" else torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    logger.info(f"Using device: {dev}")
    return dev


def setup_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Base random seed.
        rank: Distributed rank offset for data diversity across processes.
    """
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_wandb(
    project: str,
    config: Union[Dict, DictConfig],
    run_dir: Union[str, Path],
    run_name: Optional[str] = None,
    resume: bool = True,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    enabled: bool = True,
    sweep_id: Optional[str] = None,
):
    """Initialize W&B with safe resume (preserves existing run metadata on resume)."""
    # Respect WANDB_DISABLED environment variable (used by wandb itself)
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        logger.info("W&B logging disabled via WANDB_DISABLED environment variable")
        return None

    if not enabled:
        logger.info("W&B logging disabled")
        return None

    import wandb

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id_file = run_dir / "wandb_run_id.txt"

    # Convert OmegaConf to dict if needed
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Handle wandb sweep registration via environment variables
    # This is how wandb associates runs with sweeps
    if sweep_id:
        os.environ["WANDB_SWEEP_ID"] = sweep_id
        logger.info(f"Registering run with wandb sweep: {sweep_id}")
        if tags:
            tags = list(tags) + [f"sweep_{sweep_id}"]
        else:
            tags = [f"sweep_{sweep_id}"]

    # Check if we should resume an existing run
    if resume and run_id_file.exists():
        with open(run_id_file, "r") as f:
            existing_run_id = f.read().strip()

        # For sweep runs, use environment variables for resume
        if sweep_id:
            os.environ["WANDB_RUN_ID"] = existing_run_id
            os.environ["WANDB_RESUME"] = "allow"
            wandb_config = {
                "project": project,
                "dir": str(run_dir),
                "config": config,
            }
            if run_name:
                wandb_config["name"] = run_name
            if tags:
                wandb_config["tags"] = tags
            if group:
                wandb_config["group"] = group
            run = wandb.init(**wandb_config)
            logger.info(f"Resumed W&B run: {existing_run_id} in sweep {sweep_id}")
            return run

        # SAFE RESUME: Only pass id and resume flag - do NOT pass name/config/tags
        # This prevents overwriting existing run metadata on W&B
        wandb_config = {
            "project": project,
            "dir": str(run_dir),
            "id": existing_run_id,
            "resume": "must",  # "must" = fail if run doesn't exist (safer than "allow")
        }
        if group:
            wandb_config["group"] = group

        try:
            run = wandb.init(**wandb_config)
            logger.info(
                f"Resumed W&B run: {existing_run_id} (existing config preserved)"
            )
            return run
        except wandb.errors.UsageError:
            # Run doesn't exist anymore on W&B, create new one
            logger.warning(f"W&B run {existing_run_id} not found, creating new run")
            run_id_file.unlink()  # Remove stale run ID file

    # NEW RUN: Pass all configuration
    wandb_config = {
        "project": project,
        "dir": str(run_dir),
        "config": config,
    }
    if run_name:
        wandb_config["name"] = run_name
    if tags:
        wandb_config["tags"] = tags
    if group:
        wandb_config["group"] = group

    run = wandb.init(**wandb_config)
    with open(run_id_file, "w") as f:
        f.write(run.id)
    logger.info(f"Created W&B run: {run.id}")

    return run


def setup_eval_env(
    cfg: DictConfig,
    plan_cfg_overrides: Dict[str, Any],
    eval_cfg_overrides: Dict[str, Any],
    num_batches: int,
) -> tuple[Optional[Dict], Optional[Callable], int, Optional[Any]]:
    """Set up the planning/eval environment if ``enable_plan_eval`` is on.

    Args:
        cfg: Main training config (must contain ``meta``, ``eval``, ``logging``).
        plan_cfg_overrides: Overrides to apply to the plan config YAML.
        eval_cfg_overrides: Overrides to apply to the eval config YAML.
        num_batches: Total training batches per epoch (``len(loader)``),
            used as fallback when ``eval_every_itr <= 0``.

    Returns:
        ``(plan_cfg, env_creator, num_eval_episodes, eval_val_loader)``.
        All ``None / 10 / None`` when eval is disabled.
    """
    if not cfg.meta.get("enable_plan_eval", False):
        return None, None, 10, None

    if cfg.meta.eval_every_itr <= 0:
        cfg.meta.eval_every_itr = num_batches

    plan_cfg = OmegaConf.to_container(
        load_config(cfg.eval.plan_cfg_path, plan_cfg_overrides, quiet=True)
    )
    plan_cfg.setdefault("logging", {})
    plan_cfg["logging"].update(copy.deepcopy(dict(cfg.logging)))

    eval_cfg_dict = OmegaConf.to_container(
        load_config(cfg.eval.eval_cfg_path, eval_cfg_overrides, quiet=True)
    )

    from eb_jepa.data.utils import init_data
    from eb_jepa.envs import make_env_creator

    eval_env_name = eval_cfg_dict.get("data", {}).get("env_name", cfg.data.env_name)
    _, eval_val_loader, env_config = init_data(
        env_name=eval_env_name, cfg_data=dict(eval_cfg_dict.get("data", {}))
    )
    num_eval_episodes = eval_cfg_dict.get("meta", {}).get("num_eval_episodes", 10)
    cfg_eval_env = eval_cfg_dict.get("env", {})

    env_creator = make_env_creator(
        env_name=eval_env_name,
        env_config=env_config,
        eval_env_cfg=cfg_eval_env,
    )

    return plan_cfg, env_creator, num_eval_episodes, eval_val_loader


def train_visual_decoder_temporal(
    visual_decoder: nn.Module,
    enc_states: torch.Tensor,
    target_images: torch.Tensor,
    lpips_loss,
    scaler: GradScaler,
    use_amp: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Train visual decoder with per-timestep backward to save GPU memory.

    Encodes each timestep independently, computes LPIPS or MSE loss, and
    immediately backward()s to avoid storing decoder+LPIPS activations for
    all B*T images simultaneously (reduces peak from ~46 GB to ~6 GB).

    Both LPIPS and MSE branches normalize by T_min for a consistent
    per-timestep average.

    Args:
        visual_decoder: Decoder module (possibly DDP-wrapped).
        enc_states: Encoder outputs ``[B, D, T, ...]``, detached from encoder.
        target_images: Ground-truth images ``[B, C, T, H, W]``.
        lpips_loss: LPIPS loss module, or ``None`` for MSE fallback.
        scaler: AMP GradScaler.
        use_amp: Whether AMP is enabled.
        dtype: AMP dtype (float16 / bfloat16).
        device: Torch device.

    Returns:
        Accumulated (detached) visual decoder loss scalar.
    """
    T_min = min(enc_states.shape[2], target_images.shape[2])
    vd_loss_total = torch.tensor(0.0, device=device)
    for t in range(T_min):
        with autocast(device.type, enabled=use_amp, dtype=dtype):
            dec_t = visual_decoder(enc_states[:, :, t : t + 1].detach())
            tgt_t = target_images[:, :, t : t + 1]
            if lpips_loss is not None:
                vd_loss_t = lpips_loss(dec_t, tgt_t) / T_min
            else:
                vd_loss_t = nn.MSELoss()(dec_t, tgt_t) / T_min
        scaler.scale(vd_loss_t).backward()
        vd_loss_total = vd_loss_total + vd_loss_t.detach()
    return vd_loss_total


def compute_effective_rank(matrix: torch.Tensor) -> float:
    """Compute entropy-based effective rank of a matrix.

    Args:
        matrix: Input tensor [N, D].

    Returns:
        Effective rank (float between 1.0 and D).
    """
    s = torch.linalg.svdvals(matrix.float())
    p = s / s.sum()
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


class EffectiveRankAccumulator:
    """Accumulates embeddings over multiple batches for robust effective rank estimation.

    Uses a reset-after-read pattern: ``compute()`` returns the metric and clears
    internal buffers so the next logging window starts fresh.

    Args:
        max_batches: Number of batches to aggregate per name before dropping oldest.
    """

    def __init__(self, max_batches: int = 10) -> None:
        self.max_batches = max_batches
        self._buffers: Dict[str, deque] = {}

    def accumulate(self, name: str, embeddings: torch.Tensor) -> None:
        """Store one [N, D] chunk of embeddings under *name*.

        Args:
            name: Metric key (e.g. ``"train/visual_effective_rank"``).
            embeddings: Tensor of shape [N, D].
        """
        if name not in self._buffers:
            self._buffers[name] = deque(maxlen=self.max_batches)
        self._buffers[name].append(embeddings.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Concatenate stored chunks, compute effective rank per name, and reset.

        Returns:
            Dict mapping each accumulated name to its effective rank value.
        """
        results: Dict[str, float] = {}
        for name, chunks in self._buffers.items():
            if chunks:
                cat = torch.cat(list(chunks), dim=0)  # [N_total, D]
                results[name] = compute_effective_rank(cat)
        self._buffers.clear()
        return results


def build_hierarchical_param_groups(
    model: nn.Module,
    base_lr: float,
    lr_scales: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Build per-level optimizer param groups for a HierarchicalJEPA model.

    Args:
        model: HierarchicalJEPA instance.
        base_lr: Base learning rate.
        lr_scales: Optional dict mapping ``"level_{i}"`` to a float LR multiplier.
            Levels not listed default to 1.0.

    Returns:
        List of param group dicts ready for ``torch.optim.AdamW``.
    """
    lr_scales = lr_scales or {}
    assigned: set[int] = set()
    param_groups: List[Dict[str, Any]] = []

    for i in range(model.num_levels):
        level = i + 1
        level_jepa = model.levels[i]
        modules = [level_jepa]
        if (
            model.action_regularizers is not None
            and model.action_regularizers[i] is not None
        ):
            modules.append(model.action_regularizers[i])
        params = [
            p
            for m in modules
            for p in m.parameters()
            if id(p) not in assigned and not assigned.add(id(p))
        ]
        if params:
            param_groups.append(
                {
                    "params": params,
                    "name": f"level_{level}",
                    "lr": base_lr * lr_scales.get(f"level_{level}", 1.0),
                }
            )

    remaining = [p for p in model.parameters() if id(p) not in assigned]
    if remaining:
        param_groups.append({"params": remaining, "lr": base_lr, "name": "other"})

    assert sum(len(g["params"]) for g in param_groups) == sum(
        1 for _ in model.parameters()
    )
    return param_groups


def compute_and_save_action_stats(
    h_jepa: "HierarchicalJEPA",
    loader,
    folder: Path,
    epoch: int,
    device: torch.device,
    num_batches: int = 20,
    force: bool = False,
) -> None:
    """Compute latent action statistics and save them for planning initialization.

    Iterates through data, encodes actions at each level with a non-Identity
    action encoder, and saves the results in the format expected by the planner.

    Args:
        h_jepa: Trained HierarchicalJEPA model.
        loader: Training data loader.
        folder: Experiment folder to save stats into.
        epoch: Current epoch number.
        device: Torch device.
        num_batches: Number of batches to collect.
        force: If False (default), skip levels whose stats files already exist.
    """
    h_jepa.eval()
    for level in range(2, h_jepa.num_levels + 1):
        action_encoder = h_jepa.levels[level - 1].action_encoder
        if isinstance(action_encoder, nn.Identity):
            continue

        latest_path = folder / f"level_{level}_action_stats.pt"
        if not force and latest_path.exists():
            logger.info(
                f"Action stats L{level}: skipped (already exists at {latest_path})"
            )
            continue

        all_actions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                _, a, _, _, _ = batch
                a = a[:, :, :-1].to(device)  # [B, A, T-1]
                actions_l = h_jepa.aggregate_actions(a, level)  # [B, A_enc, T_l-1]
                all_actions.append(
                    actions_l.permute(0, 2, 1).cpu()
                )  # [B, T_l-1, A_enc]

        all_actions = torch.cat(all_actions, dim=0)  # [N, T_l, A_enc]

        epoch_path = folder / f"level_{level}_action_stats_epoch_{epoch}.pt"
        latest_path = folder / f"level_{level}_action_stats.pt"
        torch.save(all_actions, epoch_path)
        torch.save(all_actions, latest_path)

        flat = all_actions.reshape(-1, all_actions.shape[-1])
        logger.info(
            f"Action stats L{level} (epoch {epoch}): "
            f"mean={flat.mean(0).tolist()}, std={flat.std(0).tolist()}, "
            f"Q01={torch.quantile(flat, 0.01, dim=0).tolist()}, "
            f"Q99={torch.quantile(flat, 0.99, dim=0).tolist()}"
        )
    h_jepa.train()
