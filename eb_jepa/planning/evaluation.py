from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from eb_jepa.data.utils import traj_collate_fn
from eb_jepa.planning.agent import GCAgent, decode_with_visual_decoder
from eb_jepa.utils.distributed import unwrap_model
from eb_jepa.utils.logging import get_logger
from eb_jepa.vis.frames import save_gif, show_images
from eb_jepa.vis.plots import (
    analyze_distances,
    create_comparison_gif,
    plot_actions,
    plot_losses,
)

logger = get_logger(__name__)


def main_unroll_eval(
    model,
    env_creator,
    eval_folder,
    num_samples=4,
    loader=None,
    probers=None,
    cfg=None,
    visual_decoders=None,
    lpips_fn=None,
    preprocessor=None,
):
    """
    Evaluate the model's unrolling capabilities at each configured hierarchy level.

    Args:
        model: HierarchicalJEPA model.
        env_creator: Factory function to create the evaluation environment.
        eval_folder: Path to save evaluation artifacts.
        num_samples: Number of validation batches to evaluate.
        loader: Validation data loader.
        probers: Dict mapping level (int) -> JEPAProbe for position decoding,
            or a single JEPAProbe (backward compat, treated as level 1).
        cfg: Full training configuration object.
        visual_decoders: Optional dict mapping level (int) -> VisualDecoder
            for image-based unroll visualization.

    Returns:
        Dict of per-level, per-timestep MSE metrics.
    """
    # Normalize probers to a dict keyed by level
    if probers is None:
        probers_dict = {}
    elif isinstance(probers, dict):
        probers_dict = probers
    else:
        probers_dict = {1: probers}

    vd_dict = visual_decoders if isinstance(visual_decoders, dict) else {}

    unroll_levels = list(cfg.eval.get("unroll_levels", [1]))
    is_hierarchical = hasattr(model, "encode_hierarchical")
    env = None
    if env_creator is not None:
        try:
            env = env_creator()
            env.reset()
        except Exception as e:
            logger.warning(f"Could not create eval env: {e}")
            env = None
    device = next(model.parameters()).device
    if preprocessor is None:
        preprocessor = getattr(loader.dataset, "preprocessor", None)
    agent = GCAgent(
        model=model,
        plan_cfg=None,
        preprocessor=preprocessor,
        env=env,
        loc_prober=probers_dict.get(1),
    )

    # Per-level accumulators
    mse_values = {level: [] for level in unroll_levels}
    position_mse_values = {level: [] for level in unroll_levels}
    probe_accuracy_values = {level: [] for level in unroll_levels}
    lpips_values = {level: [] for level in unroll_levels}
    lpips_recon_values = {level: [] for level in unroll_levels}
    action_sensitivity_values = {level: [] for level in unroll_levels}
    action_cosine_sim_values = {level: [] for level in unroll_levels}
    unroll_times = []
    loader_iter = iter(loader)

    for idx in tqdm(
        range(num_samples), desc="Evaluating unroll", disable=cfg.logging.tqdm_silent
    ):
        try:
            batch = next(loader_iter)
        except StopIteration:
            logger.warning(
                f"Loader exhausted after {idx} samples (requested {num_samples})"
            )
            break

        obs_dict, a, loc, _, info = batch
        x = obs_dict["visual"] if isinstance(obs_dict, dict) else obs_dict
        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
            wall_x = (
                torch.stack([d["wall_x"] for d in info])
                if "wall_x" in info[0]
                else None
            )
            door_y = (
                torch.stack([d["door_y"] for d in info])
                if "door_y" in info[0]
                else None
            )
        elif isinstance(info, dict):
            wall_x = info.get("wall_x")
            door_y = info.get("door_y")
        else:
            wall_x = None
            door_y = None

        x = x.to(device)
        a = a[:, :, :-1].to(device)
        val_nsteps = (
            cfg.model.get("rollout", {}).get("val_nsteps", None) if cfg else None
        )
        if val_nsteps is not None:
            eval_nsteps = min(val_nsteps, a.shape[2])
            a = a[:, :, :eval_nsteps]
        else:
            eval_nsteps = a.shape[2]
        x = x[:, :, : eval_nsteps + 1]
        probe_state_dims = list(
            cfg.get("probe", {}).get("state_dims", list(range(loc.shape[-1])))
        )
        loc = loc[
            :, : eval_nsteps + 1, probe_state_dims
        ]  # [B, eval_nsteps+1, probe_output_dim]
        with torch.no_grad():
            rollout_cfg = cfg.model.get("rollout", {})
            val_ctxt = rollout_cfg.get(
                "val_ctxt_window_time", rollout_cfg.get("ctxt_window_time", 1)
            )
            obs_init = x[:, :, :val_ctxt]  # [B, C, val_ctxt, H, W]

            # Encode GT
            B, C, T, H, W = x.shape
            if is_hierarchical:
                gt_all_levels = model.encode_hierarchical(x)
            else:
                gt_all_levels = {1: model.encode(x)}

            autoenc_only = cfg.meta.get("train_autoenc_only", False) if cfg else False
            if autoenc_only:
                predicted_states_dict = gt_all_levels
                rand_predicted_dict = gt_all_levels
            else:
                start_time = time.time()
                if is_hierarchical:
                    predicted_states_dict = agent.unroll_at_levels(
                        obs_init,
                        a,
                        levels=unroll_levels,
                        repeat_batch=False,
                        ctxt_window_time=val_ctxt,
                    )
                else:
                    predicted_states_dict = {
                        1: agent.unroll(
                            obs_init,
                            a,
                            repeat_batch=False,
                            ctxt_window_time=val_ctxt,
                        )
                    }
                end_time = time.time()
                unroll_times.append(end_time - start_time)

                # Also compute random-action predictions for GIF visualization
                if is_hierarchical:
                    rand_predicted_dict = agent.unroll_at_levels(
                        obs_init,
                        torch.randn_like(a),
                        levels=unroll_levels,
                        repeat_batch=False,
                        ctxt_window_time=val_ctxt,
                    )
                else:
                    rand_predicted_dict = {
                        1: agent.unroll(
                            obs_init,
                            torch.randn_like(a),
                            repeat_batch=False,
                            ctxt_window_time=val_ctxt,
                        )
                    }

            # GT frames for GIF visualization (only needed once, not per-level)
            gt_frames = None
            if preprocessor is not None:
                gt_frames = preprocessor.to_uint8_frames(
                    x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                )  # [B, T, H, W, C]

            for level in unroll_levels:
                predicted_states = predicted_states_dict[level]
                scale = model.get_temporal_scale(level) if is_hierarchical else 1

                gt_encoded_l = (
                    model.encode_hierarchical(x)
                    if is_hierarchical
                    else {1: model.encode(x)}
                )[
                    level
                ]  # [B, D_l, T_l, H_l, W_l]

                latent_mse = (
                    ((gt_encoded_l - predicted_states) ** 2)
                    .mean(dim=(1, 3, 4))
                    .cpu()
                    .numpy()
                )  # [B, T_l]
                mse_values[level].append(latent_mse)

                # Action sensitivity: compare GT-action vs random-action predictions
                rand_predicted_states = rand_predicted_dict[level]
                diff = predicted_states - rand_predicted_states  # [B, D, T, H, W]
                diff_norm = (diff**2).mean(dim=(1, 3, 4)).sqrt()  # [B, T]
                pred_norm = (predicted_states**2).mean(dim=(1, 3, 4)).sqrt()  # [B, T]
                rel_diff = (diff_norm / (pred_norm + 1e-8)).cpu().numpy()  # [B, T]
                action_sensitivity_values[level].append(rel_diff)

                # Cosine similarity between GT-action and random-action predictions
                flat_gt = predicted_states.permute(0, 2, 3, 4, 1).reshape(
                    predicted_states.shape[0], predicted_states.shape[2], -1
                )  # [B, T, D*H*W]
                flat_rand = rand_predicted_states.permute(0, 2, 3, 4, 1).reshape(
                    rand_predicted_states.shape[0], rand_predicted_states.shape[2], -1
                )  # [B, T, D*H*W]
                cos_sim = (
                    F.cosine_similarity(flat_gt, flat_rand, dim=-1).cpu().numpy()
                )  # [B, T]
                action_cosine_sim_values[level].append(cos_sim)

                prober_l = probers_dict.get(level)
                if (
                    prober_l is not None
                    and unwrap_model(prober_l.head).output_dim == loc.shape[-1]
                ):
                    # Decode predicted positions
                    pred_positions = (
                        prober_l.apply_head(predicted_states).permute(0, 2, 1).cpu()
                    )  # [B, T_l, 2]

                    # Subsample GT positions to match this level's temporal resolution
                    gt_positions = loc[:, ::scale, :]  # [B, T_l, 2]

                    position_mse = (
                        ((pred_positions - gt_positions.cpu()) ** 2)
                        .mean(dim=-1)
                        .cpu()
                        .numpy()
                    )  # [B, T_l]
                    position_mse_values[level].append(position_mse)

                    # Probe accuracy: decode GT encodings and compare to GT positions
                    gt_decoded_positions = (
                        prober_l.apply_head(gt_encoded_l).permute(0, 2, 1).cpu()
                    )  # [B, T_l, 2]
                    probe_accuracy = (
                        ((gt_decoded_positions - gt_positions.cpu()) ** 2)
                        .mean(dim=-1)
                        .cpu()
                        .numpy()
                    )  # [B, T_l]
                    probe_accuracy_values[level].append(probe_accuracy)

                    # Generate per-level comparison GIFs (requires env with coord_to_pixel)
                    if (
                        preprocessor is not None
                        and gt_frames is not None
                        and hasattr(agent.env, "coord_to_pixel")
                    ):
                        rand_predicted_states = rand_predicted_dict[level]
                        pred_decoded = agent.decode_loc_to_pixel(
                            predicted_states,
                            prober=prober_l,
                            wall_x=wall_x,
                            door_y=door_y,
                        )
                        rand_pred_decoded = agent.decode_loc_to_pixel(
                            rand_predicted_states,
                            prober=prober_l,
                            wall_x=wall_x,
                            door_y=door_y,
                        )
                        gt_decoded = agent.decode_loc_to_pixel(
                            gt_encoded_l, prober=prober_l, wall_x=wall_x, door_y=door_y
                        )

                        gt_frames_l = gt_frames[:, ::scale]
                        T_l = gt_frames_l.shape[1]
                        raw_indices = list(range(0, T, scale))[:T_l]
                        frame_labels = [f"t={ri}" for ri in raw_indices]
                        create_comparison_gif(
                            gt_frames_l,
                            pred_decoded,
                            rand_pred_decoded,
                            gt_dec=gt_decoded,
                            save_path=f"{eval_folder}/b{idx}_level{level}.gif",
                            frame_labels=frame_labels,
                            ctxt_frames=val_ctxt,
                        )

                # Visual decoder path (works for any dataset)
                vd_l = vd_dict.get(level)
                if vd_l is not None and gt_frames is not None:
                    rand_predicted_states = rand_predicted_dict[level]
                    pred_frames = decode_with_visual_decoder(vd_l, predicted_states)
                    rand_frames = decode_with_visual_decoder(
                        vd_l, rand_predicted_states
                    )
                    gt_dec_frames = decode_with_visual_decoder(vd_l, gt_encoded_l)
                    gt_frames_l = gt_frames[:, ::scale]
                    T_l = gt_frames_l.shape[1]
                    raw_indices = list(range(0, T, scale))[:T_l]
                    frame_labels_vd = [f"t={ri}" for ri in raw_indices]
                    create_comparison_gif(
                        gt_frames_l,
                        pred_frames,
                        rand_frames,
                        gt_dec=gt_dec_frames,
                        save_path=f"{eval_folder}/b{idx}_level{level}_vd.gif",
                        frame_labels=frame_labels_vd,
                        ctxt_frames=val_ctxt,
                    )

                    # Per-timestep LPIPS metrics
                    if lpips_fn is not None:
                        B_cur = pred_frames.shape[0]
                        per_t_lpips = []
                        per_t_lpips_recon = []
                        for t_idx in range(T_l):
                            pred_t = (
                                torch.from_numpy(pred_frames[:, t_idx])
                                .float()
                                .permute(0, 3, 1, 2)
                                / 255.0
                            ).to(
                                device
                            )  # [B, C, H, W]
                            gt_t = (
                                torch.from_numpy(gt_frames_l[:, t_idx])
                                .float()
                                .permute(0, 3, 1, 2)
                                / 255.0
                            ).to(
                                device
                            )  # [B, C, H, W]
                            with torch.amp.autocast("cuda", enabled=False):
                                lp = lpips_fn(pred_t, gt_t).mean().item()
                            per_t_lpips.append(lp)

                            recon_t = (
                                torch.from_numpy(gt_dec_frames[:, t_idx])
                                .float()
                                .permute(0, 3, 1, 2)
                                / 255.0
                            ).to(
                                device
                            )  # [B, C, H, W]
                            with torch.amp.autocast("cuda", enabled=False):
                                lp_recon = lpips_fn(recon_t, gt_t).mean().item()
                            per_t_lpips_recon.append(lp_recon)
                        lpips_values[level].append(per_t_lpips)
                        lpips_recon_values[level].append(per_t_lpips_recon)

    results = {}
    results["avg_unroll_time"] = np.mean(unroll_times)

    for level in unroll_levels:
        if len(mse_values[level]) == 0:
            continue
        all_mse = np.vstack(mse_values[level])  # [num_batches, T_l]
        mean_mse = np.mean(all_mse, axis=0)  # [T_l]
        std_mse = np.std(all_mse, axis=0)  # [T_l]
        for t in range(mean_mse.shape[0]):
            results[f"val_rollout/level{level}/mean_mse/{t}"] = mean_mse[t]
            results[f"val_rollout/level{level}/std_mse/{t}"] = std_mse[t]
        results[f"val_rollout/level{level}/mean_mse_avg"] = float(np.mean(mean_mse))

        if len(position_mse_values[level]) > 0:
            all_pos_mse = np.vstack(position_mse_values[level])  # [num_batches, T_l]
            mean_pos_mse = np.mean(all_pos_mse, axis=0)
            std_pos_mse = np.std(all_pos_mse, axis=0)
            for t in range(mean_pos_mse.shape[0]):
                results[f"val_rollout/level{level}/mean_pos_mse/{t}"] = mean_pos_mse[t]
                results[f"val_rollout/level{level}/std_pos_mse/{t}"] = std_pos_mse[t]
            results[f"val_rollout/level{level}/mean_pos_mse_avg"] = float(
                np.mean(mean_pos_mse)
            )

        if len(probe_accuracy_values[level]) > 0:
            all_probe_acc = np.vstack(probe_accuracy_values[level])
            mean_probe_acc = np.mean(all_probe_acc, axis=0)
            for t in range(mean_probe_acc.shape[0]):
                results[f"val_rollout/level{level}/probe_accuracy_mse/{t}"] = (
                    mean_probe_acc[t]
                )

        if len(lpips_values[level]) > 0:
            all_lpips = np.array(lpips_values[level])  # [num_batches, T_l]
            mean_lpips = np.mean(all_lpips, axis=0)
            for t in range(mean_lpips.shape[0]):
                results[f"val_rollout/level{level}/mean_lpips/{t}"] = mean_lpips[t]
            results[f"val_rollout/level{level}/mean_lpips_avg"] = float(
                np.mean(mean_lpips)
            )

        if len(lpips_recon_values[level]) > 0:
            all_lpips_recon = np.array(lpips_recon_values[level])
            mean_lpips_recon = np.mean(all_lpips_recon, axis=0)
            for t in range(mean_lpips_recon.shape[0]):
                results[f"val_rollout/level{level}/mean_lpips_recon/{t}"] = (
                    mean_lpips_recon[t]
                )
            results[f"val_rollout/level{level}/mean_lpips_recon_avg"] = float(
                np.mean(mean_lpips_recon)
            )

        if len(lpips_values[level]) > 0 and len(lpips_recon_values[level]) > 0:
            prediction_lpips = mean_lpips - mean_lpips_recon
            results[f"val_rollout/level{level}/prediction_lpips_avg"] = float(
                np.mean(prediction_lpips)
            )

        if len(action_sensitivity_values[level]) > 0:
            all_act_sens = np.vstack(
                action_sensitivity_values[level]
            )  # [num_batches, T_l]
            mean_act_sens = np.mean(all_act_sens, axis=0)  # [T_l]
            for t in range(mean_act_sens.shape[0]):
                results[f"val_rollout/level{level}/action_sensitivity/{t}"] = (
                    mean_act_sens[t]
                )
            results[f"val_rollout/level{level}/action_sensitivity_avg"] = float(
                np.mean(mean_act_sens)
            )

        if len(action_cosine_sim_values[level]) > 0:
            all_cos_sim = np.vstack(
                action_cosine_sim_values[level]
            )  # [num_batches, T_l]
            mean_cos_sim = np.mean(all_cos_sim, axis=0)  # [T_l]
            for t in range(mean_cos_sim.shape[0]):
                results[f"val_rollout/level{level}/action_cosine_sim/{t}"] = (
                    mean_cos_sim[t]
                )
            results[f"val_rollout/level{level}/action_cosine_sim_avg"] = float(
                np.mean(mean_cos_sim)
            )

    pd.DataFrame([results]).to_csv(f"{eval_folder}/eval.csv", index=None)
    return results


### Main planning eval loop ###
def main_eval(
    plan_cfg,
    model,
    env_creator,
    eval_folder,
    num_episodes=10,
    loader=None,
    prober=None,
    model_folder=None,
    preprocessor=None,
    visual_decoders=None,
):
    plan_cfg = OmegaConf.create(plan_cfg)

    # Modify relative latent_action_stats_path to use model_folder prefix
    if model_folder is not None and plan_cfg.planner.get("type") == "hierarchical":
        level_configs = plan_cfg.planner.get("level_configs", {})
        for level_name, level_cfg in level_configs.items():
            stats_path = level_cfg.get("latent_action_stats_path", None)
            if stats_path is not None and not os.path.isabs(stats_path):
                level_cfg["latent_action_stats_path"] = os.path.join(
                    model_folder, stats_path
                )

    env = env_creator()
    env.reset()

    action_dim = env.action_space.shape[0]
    if preprocessor is None:
        preprocessor = env.preprocessor if hasattr(env, "preprocessor") else None
    if preprocessor is None and loader is not None:
        preprocessor = getattr(loader.dataset, "preprocessor", None)
        if preprocessor is None:
            base_ds = getattr(loader.dataset, "dataset", None)
            if base_ds is not None:
                preprocessor = getattr(base_ds, "preprocessor", None)
    agent = GCAgent(
        model,
        action_dim=action_dim,
        plan_cfg=plan_cfg,
        preprocessor=preprocessor,
        loc_prober=prober,
        env=env,
        visual_decoders=visual_decoders if isinstance(visual_decoders, dict) else None,
    )
    logger.info(f"Agent created with planner {agent.planner.__class__.__name__}")
    logger.info(f"Planning with {plan_cfg=}")

    # Determine eval mode
    offline = plan_cfg.task_specification.get("eval_mode", "online") == "offline"
    goal_source = plan_cfg.task_specification.goal_source

    successes = []
    distances = []
    episode_times = []
    episode_observations = []
    episode_infos = []
    ate_results = []
    eval_dataset = loader.dataset if loader is not None else None
    eval_seed = plan_cfg.get("seed", 1)
    eval_rng = torch.Generator().manual_seed(eval_seed)

    for ep in range(num_episodes):
        episode_start_time = time.time()
        ep_folder = eval_folder / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)
        if agent.decode_each_iteration:
            ep_plan_vis_dir = ep_folder / "plan_vis"
            os.makedirs(ep_plan_vis_dir, exist_ok=True)

        goal_position = None
        info = {}

        if goal_source == "dset":
            idx = torch.randint(0, len(eval_dataset), (1,), generator=eval_rng).item()

            # Access raw dataset for un-concatenated actions/states/env_info
            traj_idx, start, end = eval_dataset.slices[idx]
            raw_dset = eval_dataset.dataset
            obs_raw, act_raw, state_raw, _, env_info_raw = raw_dset[traj_idx]

            init_state = state_raw[start].numpy()
            segment_actions = act_raw[start : end - 1]  # normalized per-step actions

            if not offline:
                env.update_env(env_info_raw)

                # Denormalize actions for env replay
                if preprocessor is not None:
                    exec_actions = preprocessor.denormalize_actions(segment_actions)
                else:
                    exec_actions = segment_actions

                # Replay actions in live env to get properly rendered frames
                obs, info = env.prepare(ep, init_state)
                expert_obses = [obs]
                for a_step in exec_actions:
                    obs_step, _, _, _, step_info = env.step(a_step.numpy())
                    expert_obses.append(obs_step)

                obs = expert_obses[0]
                goal_img = expert_obses[-1]
                goal_position = step_info.get("state", state_raw[end - 1])

                # Re-prepare env to init state for planning
                obs, info = env.prepare(ep, init_state)

                # Get gt_actions (normalized, frame-skipped) for potential offline ATE
                sample = eval_dataset[idx]
                batch = traj_collate_fn([sample])
                _, a, _, _, _ = batch
                gt_actions = a[0, :, :-1]
            else:
                sample = eval_dataset[idx]
                batch = traj_collate_fn([sample])
                obs_dict, a, loc, _, _ = batch
                x = obs_dict["visual"][0]  # [C, T, H, W]
                gt_actions = a[0, :, :-1]  # [A, T-1]
                obs = preprocessor.unnormalize_visual(x[:, 0])  # [C, H, W] → raw [0,1]
                goal_img = preprocessor.unnormalize_visual(
                    x[:, -1]
                )  # [C, H, W] → raw [0,1]
                goal_position = state_raw[end - 1]
        elif goal_source == "random_state":
            if hasattr(env, "sample_random_init_goal_states"):
                init_state, goal_state = env.sample_random_init_goal_states(seed=ep)
                goal_img, goal_info = env.prepare(ep, goal_state)
                goal_position = goal_info.get("state")
                obs, info = env.prepare(ep, init_state)
            else:
                obs, info = env.reset()
                obs, reward, done, truncated, info = env.step(
                    np.zeros(env.action_space.shape[0])
                )
                goal_img = info["target_obs"]
                goal_position = info.get("target_position")

        if plan_cfg.logging.get("optional_plots", True):
            combined = torch.stack([obs, goal_img], dim=0)
            show_images(
                combined,
                nrow=2,
                titles=["Init", "Goal"],
                save_path=f"{ep_folder}/state.pdf",
                close_fig=True,
                first_channel_only=False,
                clamp=True,
            )
        agent.set_goal(
            goal_img.detach().clone().to(dtype=torch.float32),
            goal_position,
            init_state=obs.detach().clone().to(dtype=torch.float32),
        )

        if offline:
            # ---- Offline: single-shot plan + ATE ----
            obs_tensor = (
                preprocessor.normalize_obs(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )  # [1, C, 1, H, W]
            plan_vis_path = (
                f"{ep_plan_vis_dir}/step0" if agent.decode_each_iteration else None
            )
            planning_result = agent.plan(
                obs_tensor, t0=True, plan_vis_path=plan_vis_path
            )

            if agent._is_hierarchical:
                level_results = planning_result.level_results
                planned_actions = (
                    level_results[1].actions
                    if 1 in level_results and level_results[1] is not None
                    else planning_result.actions
                )
            else:
                planned_actions = planning_result.actions  # [T, A]

            plan_len = planned_actions.shape[0]
            gt_trunc = gt_actions[:, :plan_len].permute(1, 0)  # [T, A]
            gt_trunc = gt_trunc.to(planned_actions.device)

            # Denormalize to raw action space for standardized ATE
            planned_raw = agent.postprocess_actions(planned_actions)
            gt_raw = agent.postprocess_actions(gt_trunc)

            delta = torch.abs(planned_raw.sum(0) - gt_raw.sum(0))
            end_distance = delta.sum().item()
            end_distance_xyz = (
                delta[:3].sum().item() if delta.shape[0] >= 3 else end_distance
            )
            end_distance_orientation = (
                delta[3:6].sum().item() if delta.shape[0] >= 6 else 0.0
            )
            end_distance_closure = delta[6:].sum().item() if delta.shape[0] > 6 else 0.0

            ate_results.append(
                {
                    "ate/end_distance": end_distance,
                    "ate/end_distance_xyz": end_distance_xyz,
                    "ate/end_distance_orientation": end_distance_orientation,
                    "ate/end_distance_closure": end_distance_closure,
                }
            )
            logger.info(
                f"Episode {ep}: ATE={end_distance:.4f} "
                f"(xyz={end_distance_xyz:.4f}, ori={end_distance_orientation:.4f}, "
                f"grip={end_distance_closure:.4f})"
            )

            if plan_cfg.logging.get("optional_plots", True):
                # Save expert (ground-truth) trajectory from the dataset
                if goal_source == "dset":
                    x_vis = x.clone()  # [C, T, H, W]
                    if preprocessor is not None:
                        expert_frames = preprocessor.to_uint8_frames(
                            x_vis.permute(1, 0, 2, 3)  # [T, C, H, W]
                        )  # [T, H, W, C] numpy uint8
                    else:
                        expert_frames = (
                            (x_vis.permute(1, 0, 2, 3) * 255)
                            .clamp(0, 255)
                            .to(torch.uint8)
                        )  # [T, C, H, W]
                    save_gif(
                        expert_frames,
                        save_path=f"{ep_folder}/expert_trajectory.gif",
                        show_frame_numbers=True,
                        fps=20,
                    )

            if plan_cfg.logging.get("optional_plots", True):
                plot_losses(
                    (
                        [planning_result.losses]
                        if hasattr(planning_result, "losses")
                        and planning_result.losses is not None
                        else []
                    ),
                    (
                        [planning_result.prev_elite_losses_mean]
                        if hasattr(planning_result, "prev_elite_losses_mean")
                        and planning_result.prev_elite_losses_mean is not None
                        else []
                    ),
                    (
                        [planning_result.prev_elite_losses_std]
                        if hasattr(planning_result, "prev_elite_losses_std")
                        and planning_result.prev_elite_losses_std is not None
                        else []
                    ),
                    work_dir=ep_folder,
                    num_act_stepped=agent.num_act_stepped,
                )

                torch.save(
                    {"planned": planned_raw, "gt": gt_raw},
                    Path(ep_folder) / "actions.pt",
                )
                dim_labels = (
                    ["x", "y", "z", "rx", "ry", "rz", "grip"]
                    if planned_raw.shape[-1] == 7
                    else None
                )
                plot_actions(
                    planned_raw,
                    gt_raw,
                    work_dir=ep_folder,
                    action_mean=preprocessor.action_mean if preprocessor else None,
                    action_std=preprocessor.action_std if preprocessor else None,
                    dim_labels=dim_labels,
                )
        else:
            # ---- Online: MPC loop ----
            done = False
            steps_left = env.n_allowed_steps
            pbar = tqdm(
                desc="executing agent",
                total=steps_left,
                leave=True,
                disable=plan_cfg.logging.tqdm_silent,
            )
            t0 = True

            observations = [obs]
            infos = [info]

            prev_losses = []
            prev_elite_losses_mean = []
            prev_elite_losses_std = []
            prev_losses_per_level: Dict[int, list] = {}
            prev_elite_losses_mean_per_level: Dict[int, list] = {}
            prev_elite_losses_std_per_level: Dict[int, list] = {}

            while steps_left > 0:
                plan_vis_path = (
                    f"{ep_plan_vis_dir}/step{env.n_allowed_steps - steps_left}"
                    if agent.decode_each_iteration
                    else None
                )
                obs_tensor = (
                    preprocessor.normalize_obs(
                        obs.detach()
                        .clone()
                        .to(dtype=torch.float32, device=agent.device)
                    )
                    .unsqueeze(0)
                    .unsqueeze(2)
                )  # [1, C, 1, H, W]
                action = agent.act(
                    obs_tensor,
                    steps_left=steps_left,
                    t0=t0,
                    plan_vis_path=plan_vis_path,
                )
                # action is already raw env-space numpy [T_env, env_action_dim]
                if agent._prev_losses_per_level:
                    for level, data in agent._prev_losses_per_level.items():
                        prev_losses_per_level.setdefault(level, []).append(
                            data["losses"]
                        )
                        prev_elite_losses_mean_per_level.setdefault(level, []).append(
                            data["elite_mean"]
                        )
                        prev_elite_losses_std_per_level.setdefault(level, []).append(
                            data["elite_std"]
                        )
                elif agent._prev_losses is not None:
                    prev_losses.append(agent._prev_losses)
                    prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
                    prev_elite_losses_std.append(agent._prev_elite_losses_std)
                for a_step in action:
                    obs, reward, done, truncated, info = env.step(a_step)
                    t0 = False
                    observations.append(obs)
                    infos.append(info)
                    steps_left -= 1
                    pbar.update(1)
                    eval_results = env.eval_state(
                        info.get("target_position", goal_position),
                        info.get("dot_position", info.get("state")),
                    )
                    success = eval_results["success"]
                    state_dist = eval_results["state_dist"]
                pbar.set_postfix({"success": success, "state_dist": state_dist})
            pbar.close()

            episode_observations.append(torch.stack(observations))
            episode_infos.append(infos)
            successes.append(success)
            distances.append(state_dist)

            raw_normalizer = getattr(env, "normalizer", None)
            if (
                plan_cfg.logging.get("optional_plots", True)
                and raw_normalizer is not None
            ):
                analyze_distances(
                    episode_observations[-1],
                    episode_infos[-1],
                    str(ep_folder / "agent"),
                    goal_position=agent.goal_position,
                    goal_state=agent.goal_state,
                    normalizer=raw_normalizer,
                    model=agent.model,
                    objective=agent.objective,
                    device=agent.device,
                )
            if plan_cfg.logging.get("optional_plots", True):
                if prev_losses_per_level:
                    for level in sorted(prev_losses_per_level.keys()):
                        plot_losses(
                            prev_losses_per_level[level],
                            prev_elite_losses_mean_per_level[level],
                            prev_elite_losses_std_per_level[level],
                            work_dir=ep_folder,
                            num_act_stepped=agent.num_act_stepped,
                            level=level,
                        )
                else:
                    plot_losses(
                        prev_losses,
                        prev_elite_losses_mean,
                        prev_elite_losses_std,
                        work_dir=ep_folder,
                        num_act_stepped=agent.num_act_stepped,
                    )
            save_path = f"{ep_folder}/agent_steps_{'succ' if success else 'fail'}.gif"
            gif_obs = episode_observations[-1]
            gif_init = observations[0]
            gif_goal = goal_img
            save_gif(
                gif_obs,
                save_path=save_path,
                show_frame_numbers=True,
                fps=20,
                init_frame=gif_init,
                goal_frame=gif_goal,
            )
            logger.info(f"GIF saved to {save_path}")

        episode_end_time = time.time()
        episode_times.append(episode_end_time - episode_start_time)

    if offline:
        task_data = {
            k: np.mean([r[k] for r in ate_results]) for k in ate_results[0].keys()
        }
        task_data["avg_episode_time"] = np.mean(episode_times)
    else:
        task_data = {
            "success_rate": np.mean(successes),
            "mean_state_dist": np.mean(distances),
            "avg_episode_time": np.mean(episode_times),
        }
    pd.DataFrame([task_data]).to_csv(f"{eval_folder}/eval.csv", mode="a", index=None)
    return task_data
