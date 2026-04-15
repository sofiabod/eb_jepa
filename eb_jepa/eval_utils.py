"""Shared evaluation utilities for ac_video_jepa and h_ac_video_jepa."""

import os
from pathlib import Path

import torch
import yaml

from eb_jepa.planning.evaluation import main_eval, main_unroll_eval
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def launch_plan_eval(
    model,
    env_creator,
    folder,
    epoch,
    global_step,
    suffix="",
    num_eval_episodes=10,
    loader=None,
    prober=None,
    plan_cfg=None,
    preprocessor=None,
    visual_decoders=None,
):
    """Evaluate planning capabilities of a JEPA or H-JEPA model.

    Args:
        model: Trained JEPA or HierarchicalJEPA model.
        env_creator: Function to create the evaluation environment.
        folder: Path to save evaluation results.
        epoch: Current training epoch.
        global_step: Current global training step.
        suffix: Suffix for the evaluation folder name.
        num_eval_episodes: Number of evaluation episodes to run.
        loader: Data loader for validation.
        prober: Position probing head for decoding.
        plan_cfg: Planning configuration dictionary.
        preprocessor: Optional Preprocessor for obs/location normalization.
        visual_decoders: Optional dict mapping level (int) -> VisualDecoder.

    Returns:
        Dictionary of evaluation metrics (success_rate, mean_state_dist, or ATE).
    """
    logger.info(f"Planning eval: epoch={epoch} step={global_step}")
    model.eval()
    folder = Path(folder)

    eval_tag = plan_cfg.get("eval_tag", "") if plan_cfg else ""
    if eval_tag:
        eval_folder = folder / "plan_eval" / eval_tag / f"step-{global_step}{suffix}"
    else:
        eval_folder = folder / "plan_eval" / f"step-{global_step}{suffix}"
    os.makedirs(eval_folder, exist_ok=True)

    if plan_cfg is not None:
        plan_cfg_file = eval_folder / "plan_config.yaml"
        with open(plan_cfg_file, "w") as f:
            yaml.dump(plan_cfg, f)

    eval_results = main_eval(
        plan_cfg=plan_cfg,
        model=model,
        env_creator=env_creator,
        eval_folder=eval_folder,
        num_episodes=num_eval_episodes,
        loader=loader,
        prober=prober,
        model_folder=folder,
        preprocessor=preprocessor,
        visual_decoders=visual_decoders,
    )

    if "success_rate" in eval_results:
        logger.info(
            f"   success_rate={eval_results['success_rate']:.2f} | mean_dist={eval_results['mean_state_dist']:.4f}"
        )
    elif "ate/end_distance" in eval_results:
        logger.info(
            f"   ATE={eval_results['ate/end_distance']:.4f} | "
            f"xyz={eval_results['ate/end_distance_xyz']:.4f} | "
            f"ori={eval_results['ate/end_distance_orientation']:.4f} | "
            f"grip={eval_results['ate/end_distance_closure']:.4f}"
        )
    model.train()

    return eval_results


@torch.no_grad()
def launch_unroll_eval(
    model,
    env_creator,
    folder,
    epoch,
    global_step,
    suffix="",
    loader=None,
    probers=None,
    cfg=None,
    visual_decoders=None,
    lpips_fn=None,
    preprocessor=None,
):
    """Evaluate unrolling (prediction) capabilities of a JEPA or H-JEPA model.

    Dynamically parses result keys to log per-level, per-timestep metrics,
    working for both flat (ac_video_jepa) and hierarchical (h_ac_video_jepa).

    Args:
        model: Trained JEPA or HierarchicalJEPA model.
        env_creator: Function to create the evaluation environment.
        folder: Path to save evaluation results.
        epoch: Current training epoch.
        global_step: Current global training step.
        suffix: Suffix for the evaluation folder name.
        loader: Data loader for validation.
        probers: Dict mapping level (int) -> JEPAProbe, or a single JEPAProbe.
        cfg: Full configuration object.
        visual_decoders: Optional dict mapping level (int) -> VisualDecoder.
        lpips_fn: Optional LPIPS function for perceptual metrics.
        preprocessor: Optional Preprocessor for obs/location normalization.

    Returns:
        Dictionary of per-level, per-timestep MSE/LPIPS metrics.
    """
    model.eval()
    logger.info(f"Unroll eval: epoch={epoch} step={global_step}")
    folder = Path(folder)
    eval_folder = folder / "unroll_eval" / f"step-{global_step}{suffix}"
    os.makedirs(eval_folder, exist_ok=True)
    eval_results = main_unroll_eval(
        model,
        env_creator,
        eval_folder,
        loader=loader,
        probers=probers,
        cfg=cfg,
        visual_decoders=visual_decoders,
        lpips_fn=lpips_fn,
        preprocessor=preprocessor,
    )

    unroll_levels = cfg.get("unroll_levels", None)
    if unroll_levels is None:
        unroll_levels = (
            cfg.eval.get("unroll_levels", [1]) if hasattr(cfg, "eval") else [1]
        )
    unroll_levels = list(unroll_levels)

    for level in unroll_levels:
        prefix = f"val_rollout/level{level}"
        steps = sorted(
            int(k.split("/")[-1])
            for k in eval_results
            if k.startswith(f"{prefix}/mean_mse/")
        )
        if steps:
            mean_values = " | ".join(
                [f"t{i}={eval_results[f'{prefix}/mean_mse/{i}']:.2f}" for i in steps]
            )
            std_values = " | ".join(
                [f"{i}: {eval_results[f'{prefix}/std_mse/{i}']:.2f}" for i in steps]
            )
            logger.info(
                f"Unroll eval level {level} - mean_mse: {mean_values} | std_mse: {std_values}"
            )

        lpips_steps = sorted(
            int(k.split("/")[-1])
            for k in eval_results
            if k.startswith(f"{prefix}/mean_lpips/")
        )
        if lpips_steps:
            lpips_values = " | ".join(
                [
                    f"t{i}={eval_results[f'{prefix}/mean_lpips/{i}']:.4f}"
                    for i in lpips_steps
                ]
            )
            logger.info(f"Unroll eval level {level} - mean_lpips: {lpips_values}")

    model.train()

    return eval_results
