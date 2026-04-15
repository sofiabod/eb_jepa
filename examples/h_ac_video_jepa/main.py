"""Training script for Hierarchical Action Conditioned Video JEPA."""

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

from itertools import islice
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.amp import autocast
from tqdm import tqdm

from eb_jepa.builders import build_optimizer, build_visual_decoder, build_xy_prober
from eb_jepa.data.utils import init_data
from eb_jepa.eval_utils import launch_plan_eval, launch_unroll_eval
from eb_jepa.h_jepa import (
    HierarchicalJEPA,
    build_hierarchical_jepa,
)
from eb_jepa.utils.checkpoint import resume_training, save_training_state
from eb_jepa.utils.config import (
    load_config_with_prefixed_overrides,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    resolve_experiment_folder,
)
from eb_jepa.utils.distributed import (
    cleanup_distributed,
    setup_distributed,
    unwrap_model,
    unwrap_state_dict,
    wrap_ddp,
)
from eb_jepa.utils.logging import get_logger
from eb_jepa.utils.schedulers import CosineWithWarmup
from eb_jepa.utils.training import (
    EffectiveRankAccumulator,
    build_hierarchical_param_groups,
    compute_and_save_action_stats,
    optimizer_step,
    setup_amp,
    setup_device,
    setup_eval_env,
    setup_seed,
    setup_wandb,
    train_visual_decoder_temporal,
)

logger = get_logger(__name__)


from eb_jepa.vis.heatmaps import generate_cost_heatmaps


def run(
    fname: str = "examples/h_ac_video_jepa/cfgs/train/two_rooms/vc.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train a Hierarchical Action-Conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file.
        cfg: Pre-loaded config object (optional, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., model.num_levels=4).
    """
    cfg, prefix_ovr = load_config_with_prefixed_overrides(
        fname, cfg, ["plan_cfg", "eval_cfg"], **overrides
    )
    plan_cfg_overrides = prefix_ovr["plan_cfg"]
    eval_cfg_overrides = prefix_ovr["eval_cfg"]

    # Distributed setup
    local_rank, world_size, is_main = setup_distributed()

    folder, exp_name = resolve_experiment_folder("h_ac_video_jepa", cfg, folder)

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )
    if is_main:
        diag_loader, _, _ = init_data(
            env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
        )
    else:
        diag_loader = None

    device = setup_device("auto")
    setup_seed(cfg.meta.seed, rank=local_rank)

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "h_ac_video_jepa",
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[
            f"seed_{cfg.meta.seed}",
            "h_ac_video_jepa",
            f"levels_{cfg.model.num_levels}",
        ],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False) and is_main,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        data_config["batch_size"],
        train_samples=data_config["size"],
        val_samples=data_config.get("val_size", data_config["size"]),
    )

    dtype, use_amp, scaler = setup_amp(cfg, device)

    steps_per_epoch = data_config["size"] // data_config["batch_size"]
    iterations_per_epoch = cfg.optim.get("iterations_per_epoch", None)
    if iterations_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, iterations_per_epoch)
    total_steps = cfg.optim.epochs * steps_per_epoch

    enable_eval = cfg.meta.get("enable_plan_eval", False)
    plan_cfg, env_creator, num_eval_episodes, eval_val_loader = setup_eval_env(
        cfg, plan_cfg_overrides, eval_cfg_overrides, num_batches=steps_per_epoch
    )
    if is_main:
        config_path = folder / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, config_path)
        logger.info(f"Saved complete config to {config_path}")

    h_jepa = build_hierarchical_jepa(
        cfg=cfg,
        data_config=data_config,
        device=device,
    )

    total_params = sum(p.numel() for p in h_jepa.parameters())
    encoder_params = sum(
        sum(p.numel() for p in level.encoder.parameters()) for level in h_jepa.levels
    )
    predictor_params = sum(
        sum(p.numel() for p in level.predictor.parameters()) for level in h_jepa.levels
    )

    logger.info(f"Hierarchical JEPA with {cfg.model.num_levels} levels")
    log_model_info(
        h_jepa,
        {
            "total": total_params,
            "encoders": encoder_params,
            "predictors": predictor_params,
        },
    )

    log_config(cfg)

    num_channels = data_config.get("num_channels", cfg.model.dobs)
    img_size = data_config["img_size"]
    probe_cfg = cfg.get("probe", {})
    probe_state_dims = list(probe_cfg.get("state_dims", [0, 1]))
    unroll_levels = list(cfg.eval.get("unroll_levels", [1]))

    # Extract position normalization stats from dataset (Z-score targets)
    dset = loader.dataset
    pos_mean = getattr(dset, "state_mean", None)
    pos_std = getattr(dset, "state_std", None)
    if pos_mean is not None:
        pos_mean = pos_mean[probe_state_dims]
        pos_std = pos_std[probe_state_dims]
        logger.info(
            f"Probe Z-score stats: mean={pos_mean.tolist()}, " f"std={pos_std.tolist()}"
        )

    # Build per-level probe heads by detecting each level's output dimension
    test_input = torch.rand(1, num_channels, 4, img_size, img_size).to(device)
    with torch.no_grad():
        test_encodings = h_jepa.encode_hierarchical(test_input)
    xy_heads = {}
    xy_probers = {}
    for level in unroll_levels:
        output_dim_l = test_encodings[level].shape[1]
        spatial_h_l = test_encodings[level].shape[3]
        head_l, prober_l = build_xy_prober(
            probe_cfg,
            jepa=h_jepa,
            encoder_output_dim=output_dim_l,
            spatial_h=spatial_h_l,
            normalizer=None,
            device=device,
            pos_mean=pos_mean,
            pos_std=pos_std,
        )
        xy_heads[level] = head_l
        xy_probers[level] = prober_l
    # Backward-compatible aliases for level-1 probe (used by launch_plan_eval)
    xy_head = xy_heads[1]
    xy_prober = xy_probers[1]

    # Build per-level visual decoders
    vd_cfg = cfg.get("visual_decoder", {})
    visual_decoders = {}
    lpips_loss = None
    lpips_fn = None
    for level in unroll_levels:
        output_dim_l = test_encodings[level].shape[1]
        spatial_h_l = test_encodings[level].shape[3]
        vd_l, lpips_loss, lpips_fn = build_visual_decoder(
            vd_cfg,
            encoder_output_dim=output_dim_l,
            spatial_h=spatial_h_l,
            num_channels=num_channels,
            img_size=img_size,
            cfg_data=cfg.get("data", {}),
            device=device,
        )
        if vd_l is not None:
            visual_decoders[level] = vd_l
    if visual_decoders:
        logger.info(f"Visual decoders built for levels {list(visual_decoders.keys())}")

    # Build per-level param groups so each hierarchy level can have its own LR
    base_lr = cfg.optim.lr
    lr_scales = dict(cfg.optim.get("lr_scales", {}))
    param_groups = build_hierarchical_param_groups(h_jepa, base_lr, lr_scales)
    jepa_optimizer = build_optimizer(cfg.optim, param_groups)
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    all_probe_params = []
    for head in xy_heads.values():
        all_probe_params.extend(head.parameters())
    for vd in visual_decoders.values():
        all_probe_params.extend(vd.parameters())
    probe_optimizer = torch.optim.AdamW(all_probe_params, lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    eval_only = cfg.meta.get("eval_only_mode", False)
    train_decoder_only = cfg.meta.get("train_decoder_only", False)
    model_folder = (
        cfg.meta.get("model_folder")
        if (eval_only or train_decoder_only) and cfg.meta.get("model_folder")
        else None
    )
    start_epoch, ckpt_info = resume_training(
        folder,
        h_jepa,
        jepa_optimizer,
        scheduler=jepa_scheduler,
        scaler=scaler,
        device=device,
        load_checkpoint_name=cfg.meta.get("load_checkpoint", "latest.pth.tar"),
        model_folder=model_folder,
        load_optimizer=not eval_only and not train_decoder_only,
        strict=not eval_only,
    )
    if ckpt_info.get("resumed", False):
        if "xy_heads_state_dict" in ckpt_info:
            for lvl_str, sd in ckpt_info["xy_heads_state_dict"].items():
                lvl = int(lvl_str) if isinstance(lvl_str, str) else lvl_str
                if lvl in xy_heads:
                    xy_heads[lvl].load_state_dict(
                        unwrap_state_dict(sd), strict=not eval_only
                    )
        elif "xy_head_state_dict" in ckpt_info:
            xy_heads[1].load_state_dict(
                unwrap_state_dict(ckpt_info["xy_head_state_dict"]),
                strict=not eval_only,
            )
        if "visual_decoders_state_dict" in ckpt_info:
            for lvl_str, sd in ckpt_info["visual_decoders_state_dict"].items():
                lvl = int(lvl_str) if isinstance(lvl_str, str) else lvl_str
                if lvl in visual_decoders:
                    visual_decoders[lvl].load_state_dict(
                        unwrap_state_dict(sd), strict=not eval_only
                    )
        if not eval_only and not train_decoder_only:
            if "probe_optimizer_state_dict" in ckpt_info:
                probe_optimizer.load_state_dict(ckpt_info["probe_optimizer_state_dict"])
            if "probe_scheduler_state_dict" in ckpt_info:
                probe_scheduler.load_state_dict(ckpt_info["probe_scheduler_state_dict"])

    if train_decoder_only:
        if not ckpt_info.get("resumed", False):
            raise RuntimeError(
                "train_decoder_only requires a loaded checkpoint. "
                "Set meta.model_folder to a folder containing latest.pth.tar."
            )
        h_jepa.requires_grad_(False)
        start_epoch = 0
        logger.info("Decoder-only mode: froze all h_jepa parameters, reset epoch to 0")

    if cfg.meta.get("eval_only_mode", False):
        if not is_main:
            cleanup_distributed()
            return {}
        if not enable_eval:
            raise ValueError("eval_only_mode requires enable_plan_eval=True")
        logger.info("Running evaluation only (no training)")
        eval_loader = eval_val_loader if eval_val_loader is not None else val_loader
        action_stats_batches = cfg.logging.get("action_stats_num_batches", 20)
        compute_and_save_action_stats(
            h_jepa,
            loader,
            folder,
            start_epoch,
            device,
            num_batches=action_stats_batches,
        )
        eval_results = launch_unroll_eval(
            h_jepa,
            env_creator,
            folder,
            start_epoch,
            ckpt_info.get("step", 0),
            "_eval_only",
            eval_loader,
            xy_probers,
            cfg,
            visual_decoders=visual_decoders,
            lpips_fn=lpips_fn,
        )
        eval_results.update(
            launch_plan_eval(
                h_jepa,
                env_creator,
                folder,
                start_epoch,
                global_step=ckpt_info.get("step", 0),
                suffix="_eval_only",
                num_eval_episodes=num_eval_episodes,
                loader=eval_loader,
                prober=xy_prober,
                plan_cfg=plan_cfg,
                visual_decoders=visual_decoders or None,
            )
        )
        if "success_rate" in eval_results:
            logger.info(
                f"Evaluation complete. Success rate: {eval_results['success_rate']:.2%}"
            )
        elif "ate/end_distance" in eval_results:
            logger.info(
                f"Evaluation complete. ATE: {eval_results['ate/end_distance']:.4f}"
            )
        return eval_results

    rank_acc = EffectiveRankAccumulator()

    h_jepa_module = h_jepa
    if not train_decoder_only:
        h_jepa = wrap_ddp(h_jepa, device, compile=cfg.model.get("compile", False))
    for lvl in visual_decoders:
        visual_decoders[lvl] = wrap_ddp(visual_decoders[lvl], device)
    for lvl in xy_heads:
        xy_heads[lvl] = wrap_ddp(xy_heads[lvl], device)
        xy_probers[lvl].head = xy_heads[lvl]
    xy_head = xy_heads[1]
    xy_prober = xy_probers[1]

    for epoch in range(start_epoch, cfg.optim.epochs):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_start_time = time()
        pbar = tqdm(
            islice(enumerate(loader), steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False) or not is_main,
        )
        for idx, batch in pbar:
            itr_start_time = time()
            global_step = epoch * steps_per_epoch + idx
            obs, a, loc, _, _ = batch
            x = obs["visual"]  # [B, C, T, H, W]
            x = x.to(device)
            a = a[:, :, :-1].to(device)  # [B, A, T-1]
            loc = loc.permute(0, 2, 1).to(device)  # [B, T, D] -> [B, D, T]
            total_loss = torch.tensor(0.0, device=device)

            if train_decoder_only:
                with torch.no_grad():
                    encodings = h_jepa_module.encode_hierarchical(x)
                jepa_loss = torch.tensor(0.0, device=device)
                regl = torch.tensor(0.0, device=device)
                regl_unweight = torch.tensor(0.0, device=device)
                regldict = {}
                pl = torch.tensor(0.0, device=device)
            else:
                jepa_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    _, encodings, (jepa_loss, regl, regl_unweight, regldict, pl) = (
                        h_jepa(
                            x,
                            a,
                            nsteps=cfg.model.get("rollout", {}).get("nsteps", 8),
                            unroll_mode="autoregressive",
                            ctxt_window_time=1,
                            compute_loss=True,
                            return_all_steps=False,
                        )
                    )
                    total_loss += jepa_loss

                scaler.scale(total_loss).backward()
                grad_clip = cfg.optim.get("grad_clip")
                jepa_grad_norm = optimizer_step(
                    scaler, jepa_optimizer, h_jepa_module, grad_clip
                )
                jepa_scheduler.step()

            probe_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                probe_loss_total = torch.tensor(0.0, device=device)
                for probe_level in unroll_levels:
                    scale = h_jepa_module.get_temporal_scale(probe_level)
                    loc_l = loc[:, probe_state_dims, ::scale]
                    state_l = encodings[probe_level].detach()
                    output_l = xy_heads[probe_level](state_l)
                    loc_l_norm = unwrap_model(xy_heads[probe_level]).normalize_targets(
                        loc_l
                    )
                    xy_loss_l = nn.MSELoss()(output_l, loc_l_norm)
                    if dset.preprocessor is not None:
                        xy_loss_l = dset.preprocessor.denormalize_mse(xy_loss_l)
                    probe_loss_total += xy_loss_l
                total_loss += probe_loss_total

            # Visual decoder training (temporal chunking with per-timestep backward)
            vd_loss_total = torch.tensor(0.0, device=device)
            if visual_decoders:
                for vd_level in unroll_levels:
                    if vd_level not in visual_decoders:
                        continue
                    scale = h_jepa_module.get_temporal_scale(vd_level)
                    gt_images_l = x[:, :, ::scale]  # [B, C, T_l, H, W]
                    enc_detached = encodings[vd_level].detach()
                    vd_loss_total = vd_loss_total + train_visual_decoder_temporal(
                        visual_decoders[vd_level],
                        enc_detached,
                        gt_images_l,
                        lpips_loss,
                        scaler,
                        use_amp,
                        dtype,
                        device,
                    )
                probe_loss_total = probe_loss_total + vd_loss_total

            scaler.scale(probe_loss_total).backward()
            optimizer_step(scaler, probe_optimizer)
            probe_scheduler.step()

            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "reg": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

            # Accumulate effective rank over the last N steps before a log step
            if is_main and global_step % cfg.logging.log_every < rank_acc.max_batches:
                with torch.no_grad():
                    for level in range(1, h_jepa_module.num_levels + 1):
                        enc_l = encodings[level]  # [B, D, T_l, H', W']
                        D = enc_l.shape[1]
                        flat_enc = enc_l.permute(0, 2, 3, 4, 1).reshape(
                            -1, D
                        )  # [B*T*H'*W', D]
                        rank_acc.accumulate(
                            f"train/visual_collapse/level_{level}/effective_rank",
                            flat_enc,
                        )
                    for level in range(2, h_jepa_module.num_levels + 1):
                        actions_l = h_jepa_module.aggregate_actions(a, level)
                        A_enc = actions_l.shape[1]
                        flat_actions = actions_l.permute(0, 2, 1).reshape(
                            -1, A_enc
                        )  # [B*T, A_enc]
                        rank_acc.accumulate(
                            f"train/action_collapse/level_{level}/effective_rank",
                            flat_actions,
                        )

            itr_time = time() - itr_start_time
            if global_step % cfg.logging.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/reg_loss": regl.item(),
                    "train/reg_loss_unweight": regl_unweight.item(),
                    "train/pred_loss": pl.item(),
                    "train/probe_loss": probe_loss_total.item(),
                    **(
                        {"train/visual_decoder_loss": vd_loss_total.item()}
                        if visual_decoders
                        else {}
                    ),
                    "global_step": global_step,
                    "epoch": epoch,
                    "itr_time": itr_time,
                    **{
                        f"optim/jepa_lr/{pg['name']}": pg["lr"]
                        for pg in jepa_optimizer.param_groups
                    },
                    "optim/probe_lr": probe_optimizer.param_groups[0]["lr"],
                    **(
                        {"optim/grad_norm": jepa_grad_norm}
                        if jepa_grad_norm is not None
                        else {}
                    ),
                }
                for loss_name, loss_value in regldict.items():
                    log_data[f"train/regl/{loss_name}"] = loss_value

                # Per-feature collapse diagnostics (single batch, cheap, rank 0 only)
                if is_main:
                    with torch.no_grad():
                        for level in range(2, h_jepa_module.num_levels + 1):
                            actions_l = h_jepa_module.aggregate_actions(a, level)
                            action_norm = torch.norm(actions_l, dim=(1, 2)).mean()
                            log_data[f"train/action_encoder_norm/level_{level}"] = (
                                action_norm.item()
                            )
                            action_std = actions_l.std()
                            log_data[f"train/action_encoder_std/level_{level}"] = (
                                action_std.item()
                            )

                            A_enc = actions_l.shape[1]
                            flat_actions = actions_l.permute(0, 2, 1).reshape(
                                -1, A_enc
                            )  # [B*T, A_enc]
                            per_feat_std = flat_actions.std(dim=0)  # [A_enc]
                            log_data[
                                f"train/action_collapse/level_{level}/per_feat_std_min"
                            ] = per_feat_std.min().item()
                            log_data[
                                f"train/action_collapse/level_{level}/per_feat_std_mean"
                            ] = per_feat_std.mean().item()
                            log_data[
                                f"train/action_collapse/level_{level}/per_feat_std_max"
                            ] = per_feat_std.max().item()

                        for level in range(1, h_jepa_module.num_levels + 1):
                            enc_l = encodings[level]  # [B, D, T_l, H', W']
                            D = enc_l.shape[1]
                            flat_enc = enc_l.permute(0, 2, 3, 4, 1).reshape(
                                -1, D
                            )  # [B*T*H'*W', D]
                            vis_std = flat_enc.std(dim=0)  # [D]
                            log_data[
                                f"train/visual_collapse/level_{level}/per_feat_std_min"
                            ] = vis_std.min().item()
                            log_data[
                                f"train/visual_collapse/level_{level}/per_feat_std_mean"
                            ] = vis_std.mean().item()

                # Multi-batch effective rank metrics
                log_data.update(rank_acc.compute())

                # Add cost losses if available
                if h_jepa_module.cost_modules:
                    cost_keys = [k for k in regldict if k.startswith("cost_")]
                    for k in cost_keys:
                        log_data[f"train/{k}"] = regldict[k]

                if is_main and cfg.logging.get("log_wandb"):
                    wandb.log(log_data, step=global_step)

            if (
                is_main
                and enable_eval
                and (global_step + 1) % cfg.meta.eval_every_itr == 0
                and global_step > 0
            ):
                action_stats_batches = cfg.logging.get("action_stats_num_batches", 20)
                compute_and_save_action_stats(
                    h_jepa_module,
                    diag_loader,
                    folder,
                    epoch,
                    device,
                    num_batches=action_stats_batches,
                    force=True,
                )
                eval_results = launch_plan_eval(
                    h_jepa_module,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    num_eval_episodes=num_eval_episodes,
                    loader=val_loader,
                    prober=xy_prober,
                    plan_cfg=plan_cfg,
                    visual_decoders=visual_decoders or None,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

            if (
                is_main
                and (global_step + 1) % cfg.meta.light_eval_freq == 0
                and global_step > 0
            ):
                eval_results = launch_unroll_eval(
                    h_jepa_module,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    loader=val_loader,
                    probers=xy_probers,
                    cfg=cfg,
                    visual_decoders=visual_decoders,
                    lpips_fn=lpips_fn,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

        epoch_time = time() - epoch_start_time

        log_epoch(
            epoch,
            {
                "loss": total_loss.item(),
                "reg": regl.item(),
                "pred": pl.item(),
                "probe": probe_loss_total.item(),
            },
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        if is_main and cfg.logging.get("log_wandb"):
            wandb.log(
                {"epoch": epoch, "epoch_time": epoch_time},
                step=global_step,
            )

        if is_main:
            ckpt_extra = {
                "xy_heads_state_dict": {
                    level: unwrap_model(head).state_dict()
                    for level, head in xy_heads.items()
                },
                "probe_optimizer_state_dict": probe_optimizer.state_dict(),
                "probe_scheduler_state_dict": probe_scheduler.state_dict(),
            }
            if visual_decoders:
                ckpt_extra["visual_decoders_state_dict"] = {
                    level: unwrap_model(vd).state_dict()
                    for level, vd in visual_decoders.items()
                }

            save_training_state(
                folder,
                h_jepa,
                jepa_optimizer,
                epoch,
                save_every=cfg.logging.save_every,
                scheduler=jepa_scheduler,
                scaler=scaler,
                step=global_step,
                **ckpt_extra,
            )

        # Epoch-end diagnostics: action stats (must run before plan eval)
        if is_main:
            action_stats_batches = cfg.logging.get("action_stats_num_batches", 20)
            compute_and_save_action_stats(
                h_jepa_module,
                diag_loader,
                folder,
                epoch,
                device,
                num_batches=action_stats_batches,
                force=True,
            )

        # Epoch-end diagnostics: cost heatmaps (2D environments)
        heatmap_freq = cfg.logging.get("heatmap_every_n_epochs")
        if (
            is_main
            and heatmap_freq
            and epoch % heatmap_freq == 0
            and cfg.data.env_name in ("two_rooms", "pusht", "pointmaze")
        ):
            try:
                generate_cost_heatmaps(h_jepa_module, folder, epoch, cfg, device)
            except Exception as e:
                logger.warning(f"Cost heatmap generation failed (epoch {epoch}): {e}")

    cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(run)
