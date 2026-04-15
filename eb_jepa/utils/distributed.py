from __future__ import annotations

import logging
import os
import socket
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def _get_port(world_size: int, default_port: int = 37129) -> int:
    """Pick a port for distributed training.

    For single-GPU: pick any free port.
    For multi-GPU: derive from SLURM_JOB_ID so all ranks in the same job
    agree on the port, but different jobs on the same node don't collide.
    """
    if world_size == 1:
        from torch.distributed.elastic.utils.distributed import get_free_port

        return get_free_port()
    if "SLURM_JOB_ID" in os.environ:
        return 10000 + int(os.environ["SLURM_JOB_ID"]) % 50000
    return default_port


def setup_distributed() -> tuple[int, int, bool]:
    """Initialize distributed training from SLURM or torchrun env vars.

    Follows the reference init_distributed() pattern:
    - Set all env vars *before* calling init_process_group to avoid race conditions.
    - Use mixed backend ``cpu:gloo,cuda:nccl``.
    - Re-raise on init_process_group failure with context.

    Returns:
        (local_rank, world_size, is_main) tuple.
    """
    # Set TMPDIR for SLURM jobs (before any distributed init)
    if "SLURM_JOB_ID" in os.environ:
        from pathlib import Path

        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), dist.get_rank() == 0

    # Default MASTER_ADDR (overridden in SLURM path below)
    os.environ.setdefault("MASTER_ADDR", "localhost")

    # torchrun sets these
    dist_keys = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    dist_env_set = all(key in os.environ for key in dist_keys)

    # SLURM path: translate SLURM env vars
    if not dist_env_set:
        try:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            os.environ["MASTER_ADDR"] = (
                os.environ["HOSTNAME"]
                if "HOSTNAME" in os.environ
                else socket.gethostname()
            )
        except Exception:
            return 0, 1, True

    if "LOCAL_RANK" not in os.environ:
        return 0, 1, True

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size == 1:
        return 0, 1, True

    # Set MASTER_PORT via helper
    os.environ["MASTER_PORT"] = str(_get_port(world_size))

    try:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize distributed training: {e}. "
            f"Rank={rank}, World={world_size}, Master={os.environ.get('MASTER_ADDR')}"
        ) from e

    is_main = rank == 0
    if not is_main:
        logging.getLogger("eb_jepa").setLevel(logging.WARNING)

    logger.info(
        f"Distributed: rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )
    return local_rank, world_size, is_main


def wrap_ddp(
    model: nn.Module,
    device: torch.device,
    sync_batchnorm: bool = False,
    compile: bool = False,
) -> nn.Module:
    """Wrap model with DistributedDataParallel if distributed is initialized.

    DDP only synchronizes gradients through its ``forward()`` hook.
    Model classes must define ``forward()`` (e.g. delegating to ``unroll()``)
    and callers must use ``model(...)`` not ``model.unroll(...)``.

    For attribute/method access on the raw module (e.g. ``encode_hierarchical``,
    ``cost_modules``), use ``unwrap_model(model)`` or keep a reference to the
    raw module before wrapping.

    Args:
        model: Model to wrap.
        device: CUDA device for DDP.
        sync_batchnorm: Convert BatchNorm to SyncBatchNorm before wrapping.
        compile: Apply ``torch.compile`` before DDP wrapping.
    """
    if compile:
        model = torch.compile(model)
    if not dist.is_initialized():
        return model
    if sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return DDP(model, device_ids=[device.index or 0])


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip torch.compile and DDP wrappers to get the raw model."""
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def unwrap_state_dict(sd: dict) -> dict:
    """Strip ``module.`` and ``_orig_mod.`` prefixes from state-dict keys."""
    return {
        k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()
    }


def make_sampler(dataset) -> Optional[DistributedSampler]:
    """Return a DistributedSampler if distributed is initialized, else None."""
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(dataset)
    return None


def local_batch_size(global_batch_size: int) -> int:
    """Divide a global (total) batch size by world_size for per-GPU batching.

    All configs specify the **global** batch size.  When running with DDP on
    *N* GPUs the DataLoader on each rank should use ``global_batch_size // N``.
    On a single GPU (or without distributed init) this is a no-op.
    """
    world_size = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"global batch_size ({global_batch_size}) must be divisible by "
            f"world_size ({world_size})"
        )
    return global_batch_size // world_size


def cleanup_distributed() -> None:
    """Destroy the distributed process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
