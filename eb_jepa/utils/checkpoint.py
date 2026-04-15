from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

from eb_jepa.utils.distributed import unwrap_model, unwrap_state_dict
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    scaler: Optional[Any] = None,
    **extra_state,
) -> None:
    """Save a training checkpoint (model, optimizer, scheduler, scaler, extra_state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    checkpoint.update(extra_state)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint: {path}")


def _migrate_hjepa_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remap old HierarchicalJEPA keys (parallel ModuleLists) to new format (levels list).

    Old format: ``encoders.0.weight``, ``predictors.1.weight``, etc.
    New format: ``levels.0.encoder.weight``, ``levels.1.predictor.weight``, etc.

    If no old-format keys are found, the state dict is returned unchanged.
    """
    component_map = {
        "encoders": "encoder",
        "predictors": "predictor",
        "action_encoders": "action_encoder",
        "regularizers": "regularizer",
        "predcosts": "predcost",
    }
    needs_migration = any(
        k.startswith(prefix + ".") for k in state_dict for prefix in component_map
    )
    if not needs_migration:
        return state_dict

    new_state: Dict[str, Any] = {}
    migrated = 0
    for k, v in state_dict.items():
        new_key = k
        for plural, singular in component_map.items():
            if k.startswith(f"{plural}."):
                parts = k.split(".", 2)
                idx = parts[1]
                rest = parts[2] if len(parts) > 2 else ""
                new_key = (
                    f"levels.{idx}.{singular}.{rest}"
                    if rest
                    else f"levels.{idx}.{singular}"
                )
                migrated += 1
                break
        new_state[new_key] = v

    if migrated > 0:
        logger.info(
            f"Migrated {migrated} old-format HierarchicalJEPA state dict keys to new levels-based format"
        )
    return new_state


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    load_optimizer: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint. Returns dict with epoch, step, and extra_state.

    The returned 'epoch' is the epoch to resume training from (0-indexed).
    If no checkpoint exists, returns epoch=0 to start fresh.
    If a checkpoint exists with epoch=N, returns epoch=N+1 to resume from the next epoch.

    Args:
        load_optimizer: If False, skip loading optimizer/scheduler state (useful for eval_only_mode
                        when model architecture doesn't match checkpoint optimizer state).
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}")
        return {"epoch": 0, "step": 0, "resumed": False}

    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Handle compiled/DDP model state dicts
    state_dict = unwrap_state_dict(checkpoint.get("model_state_dict", {}))

    # Migrate old HierarchicalJEPA state dict format (parallel ModuleLists)
    # to new format (composition of JEPA instances via levels list)
    state_dict = _migrate_hjepa_state_dict(state_dict)

    msg = model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded model state from: {path} with msg: {msg}")

    if (
        load_optimizer
        and optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        msg = optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Restored optimizer state with msg: {msg}")
    elif not load_optimizer and optimizer is not None:
        logger.info("Skipped loading optimizer state (load_optimizer=False)")

    if (
        load_optimizer
        and scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        msg = scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Restored scheduler state with msg: {msg}")
    elif not load_optimizer and scheduler is not None:
        logger.info("Skipped loading scheduler state (load_optimizer=False)")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        msg = scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(f"Restored scaler state with msg: {msg}")

    return {
        "epoch": checkpoint.get("epoch", 0) + 1,  # Resume from next epoch
        "step": checkpoint.get("step", 0),
        "resumed": True,
        **{
            k: v
            for k, v in checkpoint.items()
            if k
            not in [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "epoch",
                "step",
            ]
        },
    }


def resume_training(
    folder: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    load_checkpoint_name: str = "latest.pth.tar",
    model_folder: Optional[Union[str, Path]] = None,
    load_optimizer: bool = True,
    strict: bool = True,
) -> tuple[int, Dict[str, Any]]:
    """Auto-resume from ``(model_folder or folder) / load_checkpoint_name``.

    No checkpoint = start from epoch 0 (no-op).  Returns ``(start_epoch, ckpt_info)``.
    """
    base = Path(model_folder) if model_folder else Path(folder)
    ckpt_path = base / load_checkpoint_name
    ckpt_info = load_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        scaler,
        device=device,
        strict=strict,
        load_optimizer=load_optimizer,
    )
    start_epoch = ckpt_info.get("epoch", 0)
    return start_epoch, ckpt_info


def save_training_state(
    folder: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    save_every: int = 0,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    step: int = 0,
    **extra_state,
) -> None:
    """Save ``latest.pth.tar`` always; also ``e-{epoch}.pth.tar`` every *save_every* epochs."""
    folder = Path(folder)
    save_checkpoint(
        folder / "latest.pth.tar",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step=step,
        scaler=scaler,
        **extra_state,
    )
    if save_every > 0 and epoch % save_every == 0 and epoch > 0:
        save_checkpoint(
            folder / f"e-{epoch}.pth.tar",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            scaler=scaler,
            **extra_state,
        )
