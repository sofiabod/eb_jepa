"""Dataset factory for action-conditioned trajectory datasets.

Provides ``init_data(env_name, cfg_data)`` which:

1. Loads base YAML config from ``eb_jepa/data/cfgs/{env_name}.yaml``
2. Merges runtime overrides from ``cfg_data``
3. Instantiates the appropriate ``TrajDataset`` subclass
4. Slices trajectories + creates train/val ``DataLoader``s

To register a new dataset, decorate the class with
``@register_dataset("my_env")`` and add a ``cfgs/my_env.yaml``.
"""

import inspect
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import default_collate

from eb_jepa.data.two_rooms_dset import (
    WallDataset,
    WallDatasetConfig,
    update_config_from_yaml,
)
from eb_jepa.utils.yaml import expand_env_vars

DATASETS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    """Class decorator that adds a ``TrajDataset`` subclass to the registry."""

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


# Register two_rooms (defined in two_rooms_dset.py, special-cased in init_data)
register_dataset("two_rooms")(WallDataset)

# Lazy-register other datasets so imports only happen when needed.
# Each *_dset.py file uses @register_dataset at class definition.
_LAZY_MODULES = {
    "droid": "eb_jepa.data.droid_dset",
    "franka_custom": "eb_jepa.data.droid_dset",
    "pusht": "eb_jepa.data.pusht_dset",
    "pointmaze": "eb_jepa.data.point_maze_dset",
    "robocasa": "eb_jepa.data.robocasa_dset",
}


def _ensure_registered(env_name: str) -> None:
    """Import the dataset module if *env_name* is not yet in the registry."""
    if env_name not in DATASET_REGISTRY and env_name in _LAZY_MODULES:
        __import__(_LAZY_MODULES[env_name])


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def traj_collate_fn(batch):
    """Collate + permute to model convention: [B,C,T,H,W] and [B,A,T].

    Datasets return per-sample tuples ``(obs, actions, states, reward, info)``
    or ``(obs, actions, states, reward)`` (4 elements, no info).
    ``obs["visual"]`` is in ``[T,C,H,W]`` and ``actions`` in ``[T,A]``.
    The model expects ``[B,C,T,H,W]`` and ``[B,A,T]``.
    """
    first_sample = batch[0]
    has_info = len(first_sample) == 5

    if has_info:
        obs_list, act_list, state_list, reward_list, info_list = zip(*batch)
    else:
        obs_list, act_list, state_list, reward_list = zip(*batch)
        info_list = [{}] * len(batch)

    T = obs_list[0]["visual"].shape[0]
    reward_list = [r if r is not None else torch.zeros(T) for r in reward_list]

    visual = default_collate([o["visual"] for o in obs_list])  # [B, T, C, H, W]
    proprio = default_collate([o["proprio"] for o in obs_list])  # [B, T, D]
    actions = default_collate(act_list)  # [B, T, A]
    states = default_collate(state_list)  # [B, T, D]
    rewards = default_collate(reward_list)  # [B, T]

    obs = {
        "visual": visual.permute(0, 2, 1, 3, 4),  # [B, C, T, H, W]
        "proprio": proprio,  # [B, T, D]
    }
    actions = actions.permute(0, 2, 1)  # [B, A, T]

    return obs, actions, states, rewards, list(info_list)


# ---------------------------------------------------------------------------
# Config & transform helpers
# ---------------------------------------------------------------------------


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / "cfgs" / f"{env_name}.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    base_config = expand_env_vars(base_config)
    if overrides:
        base_config.update(overrides)
    return base_config


def make_transform_from_config(merged_cfg: dict):
    """Build a VideoTransform from the ``transform`` sub-dict of a merged config.

    Returns ``None`` when no ``transform`` key is present (backward-compatible).
    """
    tcfg = merged_cfg.get("transform")
    if tcfg is None:
        return None

    from eb_jepa.data.transforms import make_transforms

    normalize = tcfg.get("normalize")
    if normalize is None:
        normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    return make_transforms(
        random_horizontal_flip=tcfg.get("random_horizontal_flip", False),
        random_resize_aspect_ratio=tuple(
            tcfg.get("random_resize_aspect_ratio", (1.0, 1.0))
        ),
        random_resize_scale=tuple(tcfg.get("random_resize_scale", (1.0, 1.0))),
        reprob=tcfg.get("reprob", 0.0),
        auto_augment=tcfg.get("auto_augment", False),
        motion_shift=tcfg.get("motion_shift", False),
        img_size=merged_cfg.get("img_size", 224),
        normalize=normalize,
        do_255_to_1=tcfg.get("do_255_to_1", False),
    )


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------


def _make_loaders(train_dset, val_dset, merged_cfg):
    """Create train/val DataLoaders from datasets and merged config."""
    from eb_jepa.utils.distributed import local_batch_size

    batch_size = local_batch_size(merged_cfg.get("batch_size", 64))
    val_batch_size = merged_cfg.get("val_batch_size", 4)
    num_workers = merged_cfg.get("num_workers", 4)
    pin_mem = merged_cfg.get("pin_mem", True)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

    if len(train_dset) == 0:
        train_loader = None
    else:
        sampler = None
        shuffle = True
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(train_dset)
            shuffle = False

        train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persistent_workers,
            collate_fn=traj_collate_fn,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
        collate_fn=traj_collate_fn,
    )
    return train_loader, val_loader


def _filter_constructor_kwargs(cls, kwargs: dict) -> dict:
    """Keep only kwargs accepted by *cls.__init__*."""
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs  # cls accepts **kwargs
    return {k: v for k, v in kwargs.items() if k in valid}


# ---------------------------------------------------------------------------
# Common config keys consumed by init_data (not forwarded to dataset ctor)
# ---------------------------------------------------------------------------

_INIT_DATA_KEYS = {
    "batch_size",
    "val_batch_size",
    "num_workers",
    "pin_mem",
    "persistent_workers",
    "num_frames_val",
    "frameskip",
    "action_skip",
    "train_fraction",
    "process_actions",
    "seed",
    "transform",
    "env_name",
    # dataset YAML metadata (not constructor args)
    "action_dim",
    "proprio_dim",
    "state_dim",
    "num_channels",
    "img_size",
    "size",
    "val_size",
}


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def init_data(env_name: str, cfg_data: dict = None, **kwargs):
    """Initialize data loaders for the specified environment.

    Loads base config from ``eb_jepa/data/cfgs/{env_name}.yaml``
    and merges with any overrides from *cfg_data*.

    Args:
        env_name: Name of the environment (e.g., ``"two_rooms"``, ``"droid"``).
        cfg_data: Configuration overrides for the dataset.

    Returns:
        Tuple of ``(train_loader, val_loader, config_dict)``.
    """
    _ensure_registered(env_name)
    if env_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown env: {env_name}. "
            f"Registered datasets: {sorted(DATASET_REGISTRY.keys())}"
        )

    merged_cfg = load_env_data_config(env_name, cfg_data)

    # --- Special case: two_rooms (on-the-fly generation, no slicing) ---
    if env_name == "two_rooms":
        return _init_two_rooms(merged_cfg)

    # --- Special case: pusht (pre-split train/val directories) ---
    if env_name == "pusht":
        return _init_pusht(merged_cfg)

    # --- Generic path: instantiate → split → slice → loaders ---
    transform = make_transform_from_config(merged_cfg)

    # DROID fallback: if no transform config, use a plain Resize
    if transform is None and env_name in ("droid", "franka_custom"):
        import torchvision.transforms as T

        target_size = merged_cfg.get("img_size", 224)
        transform = T.Resize((target_size, target_size), antialias=True)

    # Build dataset constructor kwargs (everything except init_data keys)
    dataset_kwargs = {k: v for k, v in merged_cfg.items() if k not in _INIT_DATA_KEYS}
    dataset_kwargs["transform"] = transform

    cls = DATASET_REGISTRY[env_name]
    dataset_kwargs = _filter_constructor_kwargs(cls, dataset_kwargs)
    base_dataset = cls(**dataset_kwargs)

    # Slice into train/val
    from eb_jepa.data.traj_dset import get_train_val_sliced

    seed = merged_cfg.get("seed", 42)
    num_frames = merged_cfg.get("num_frames", 16)
    num_frames_val = merged_cfg.get("num_frames_val")
    _, _, train_slices, val_slices = get_train_val_sliced(
        base_dataset,
        train_fraction=merged_cfg.get("train_fraction", 0.9),
        random_seed=seed,
        num_frames=num_frames,
        num_frames_val=num_frames_val,
        frameskip=merged_cfg.get("frameskip", 1),
        action_skip=merged_cfg.get("action_skip", 1),
        process_actions=merged_cfg.get("process_actions", "concat"),
    )

    merged_cfg["action_dim"] = train_slices.action_dim
    merged_cfg.setdefault("size", len(train_slices))
    merged_cfg.setdefault("val_size", len(val_slices))
    train_loader, val_loader = _make_loaders(train_slices, val_slices, merged_cfg)
    return train_loader, val_loader, merged_cfg


# ---------------------------------------------------------------------------
# Special-case helpers
# ---------------------------------------------------------------------------


def _init_two_rooms(merged_cfg: dict):
    """two_rooms: on-the-fly generation, no trajectory slicing."""
    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)

    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

    from eb_jepa.utils.distributed import local_batch_size

    dset = WallDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=local_batch_size(config.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
        collate_fn=traj_collate_fn,
    )

    val_dset = WallDataset(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
        collate_fn=traj_collate_fn,
    )
    return loader, val_loader, merged_cfg


def _init_pusht(merged_cfg: dict):
    """pusht: data is pre-split into train/ and val/ directories."""
    from eb_jepa.data.pusht_dset import PushTDataset
    from eb_jepa.data.traj_dset import TrajSlicerDataset

    transform = make_transform_from_config(merged_cfg)
    seed = merged_cfg.get("seed", 42)
    num_frames = merged_cfg.get("num_frames", 16)

    common = dict(
        n_rollout=merged_cfg.get("n_rollout", None),
        transform=transform,
        normalize_action=merged_cfg.get("normalize_action", True),
        with_velocity=merged_cfg.get("with_velocity", True),
    )
    train_dset = PushTDataset(data_path=merged_cfg["data_path"] + "/train", **common)
    val_dset = PushTDataset(data_path=merged_cfg["data_path"] + "/val", **common)

    frameskip = merged_cfg.get("frameskip", 1)
    action_skip = merged_cfg.get("action_skip", 1)
    process_actions = merged_cfg.get("process_actions", "concat")
    gen = torch.Generator().manual_seed(seed)

    train_slices = TrajSlicerDataset(
        train_dset,
        num_frames,
        frameskip,
        action_skip,
        process_actions=process_actions,
        generator=gen,
    )
    val_slices = TrajSlicerDataset(
        val_dset,
        merged_cfg.get("num_frames_val") or num_frames,
        frameskip,
        action_skip,
        process_actions=process_actions,
        generator=torch.Generator().manual_seed(seed),
    )

    merged_cfg["action_dim"] = train_slices.action_dim
    merged_cfg.setdefault("size", len(train_slices))
    merged_cfg.setdefault("val_size", len(val_slices))
    train_loader, val_loader = _make_loaders(train_slices, val_slices, merged_cfg)
    return train_loader, val_loader, merged_cfg
