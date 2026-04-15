#!/usr/bin/env python3
"""CLI wrapper: visualize planning cost as 2D heatmaps over the environment.

Core visualization logic lives in ``eb_jepa.vis.heatmaps``. This script
provides a CLI entry point for standalone use with both flat and hierarchical
checkpoints.

Usage::

    python scripts/visualize_planning_cost_heatmap.py \
        --checkpoint_path /path/to/latest.pth.tar \
        --config_path /path/to/config.yaml
"""

from pathlib import Path

import fire
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.builders import build_encoder
from eb_jepa.jepa import JEPAbase
from eb_jepa.utils.checkpoint import load_checkpoint
from eb_jepa.utils.logging import get_logger
from eb_jepa.utils.training import setup_device
from eb_jepa.vis.heatmaps import (
    _DISPLAY_ORIGINS,
    _collect_heatmap_data,
    _edge_goals,
    _get_normalize_fn,
    _level_desc,
    _load_data_config,
    _setup_env,
    visualize_heatmaps_grid,
)

logger = get_logger(__name__)


def _build_model_for_viz(cfg, data_cfg, device):
    """Build a minimal model sufficient for heatmap visualization.

    For hierarchical configs, builds a full HierarchicalJEPA.
    For flat configs (both old and new format), builds a JEPAbase with
    the encoder only.

    Returns:
        ``(model, num_levels)`` tuple.
    """
    is_hierarchical = hasattr(cfg.model, "num_levels")

    if is_hierarchical:
        from eb_jepa.h_jepa import build_hierarchical_jepa

        model = build_hierarchical_jepa(
            cfg,
            OmegaConf.to_container(data_cfg, resolve=True),
            device,
        )
        return model, model.num_levels

    data_dict = OmegaConf.to_container(data_cfg, resolve=True)
    num_channels = data_dict.get("num_channels", cfg.model.get("dobs", 3))
    img_size = data_dict["img_size"]

    if hasattr(cfg.model, "encoder"):
        enc_cfg = cfg.model.encoder
    else:
        enc_cfg = OmegaConf.create(
            {
                "architecture": cfg.model.get("encoder_architecture", "impala"),
            }
        )

    encoder, _, _ = build_encoder(
        enc_cfg,
        input_channels=num_channels,
        img_size=img_size,
        device=device,
    )
    model = JEPAbase(encoder, nn.Identity(), nn.Identity()).to(device)
    return model, 1


def main(
    checkpoint_path: str,
    config_path: str,
    output_filename: str = "planning_cost_heatmap.pdf",
    resolution: int = 65,
    num_goals: int = 4,
    device: str = "auto",
    env_name: str = "two_rooms",
):
    """Visualize planning cost as a 2D heatmap.

    Works with both flat (ac_video_jepa) and hierarchical (h_ac_video_jepa)
    checkpoints.

    Args:
        checkpoint_path: Path to checkpoint file (``.pth.tar``).
        config_path: Path to training config YAML.
        output_filename: Output filename (saved in ``checkpoint_dir/heatmaps/``).
        resolution: Grid resolution (points per dimension).
        num_goals: Number of edge goal positions to visualize.
        device: Device to use (``"auto"``, ``"cuda"``, or ``"cpu"``).
        env_name: Environment name (``"two_rooms"``, ``"pusht"``, or ``"pointmaze"``).
    """
    device = setup_device(device)

    checkpoint_dir = Path(checkpoint_path).parent
    heatmaps_dir = checkpoint_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    data_cfg = _load_data_config(env_name)

    model, num_levels = _build_model_for_viz(cfg, data_cfg, device)
    load_checkpoint(checkpoint_path, model, device=device, strict=False)
    model.eval()
    logger.info(f"Loaded {num_levels}-level model from {checkpoint_path}")

    normalize_fn = _get_normalize_fn(env_name)
    origin = _DISPLAY_ORIGINS.get(env_name, "lower")
    env, bounds = _setup_env(env_name, device=device)
    goal_positions = _edge_goals(env_name, bounds, n=num_goals, device=device)

    wall_img = getattr(env, "wall_img", None)

    for level in range(1, num_levels + 1):
        heatmap_data = _collect_heatmap_data(
            model,
            env,
            bounds,
            goal_positions,
            level,
            normalize_fn=normalize_fn,
            resolution=resolution,
        )

        desc = _level_desc(level, num_levels)
        level_suffix = f"_level{level}" if desc else ""
        output_path = str(
            heatmaps_dir / output_filename.replace(".pdf", f"{level_suffix}.pdf")
        )
        suptitle = desc or None

        visualize_heatmaps_grid(
            heatmap_data,
            output_path,
            wall_img=wall_img,
            suptitle=suptitle,
            origin=origin,
        )
        gradient_path = output_path.replace(".pdf", "_gradient.pdf")
        visualize_heatmaps_grid(
            heatmap_data,
            gradient_path,
            wall_img=wall_img,
            suptitle=suptitle,
            gradient=True,
            origin=origin,
        )

    logger.info(
        f"Done! Generated heatmaps for {num_levels} level(s) "
        f"with {num_goals} goals in {heatmaps_dir}"
    )


if __name__ == "__main__":
    fire.Fire(main)
