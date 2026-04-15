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

from eb_jepa.builders import (
    build_cost_module,
    build_encoder,
    build_optimizer,
    build_predcost,
    build_predictor,
    build_regularizer,
    build_visual_decoder,
    build_xy_prober,
)
from eb_jepa.data.utils import init_data
from eb_jepa.eval_utils import launch_plan_eval, launch_unroll_eval
from eb_jepa.jepa import JEPAWithCostModule
from eb_jepa.models.encoders import (
    DinoEncoder,
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
    fname: str = "examples/ac_video_jepa/cfgs/train/two_rooms/vc.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an action-conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file.
        cfg: Pre-loaded config object (optional, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., model.henc=64).
    """
    cfg, prefix_ovr = load_config_with_prefixed_overrides(
        fname, cfg, ["plan_cfg", "eval_cfg"], **overrides
    )
    plan_cfg_overrides = prefix_ovr["plan_cfg"]
    eval_cfg_overrides = prefix_ovr["eval_cfg"]

    # Distributed setup
    local_rank, world_size, is_main = setup_distributed()

    folder, exp_name = resolve_experiment_folder("ac_video_jepa", cfg, folder)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Folder: {folder}")

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )

    # -- SETUP
    device = setup_device("auto")
    setup_seed(cfg.meta.seed, rank=local_rank)

    # -- WANDB
    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "ac_video_jepa",
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "ac_video_jepa"],
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

    # -- ENV (for plan/unroll eval)
    enable_eval = cfg.meta.get("enable_plan_eval", False)
    plan_cfg, env_creator, num_eval_episodes, eval_val_loader = setup_eval_env(
        cfg, plan_cfg_overrides, eval_cfg_overrides, num_batches=steps_per_epoch
    )
    if is_main:
        config_path = folder / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, config_path)
        logger.info(f"Saved complete config to {config_path}")

    # -- MODEL
    num_channels = data_config.get("num_channels", cfg.model.dobs)
    img_size = data_config["img_size"]
    action_dim = data_config["action_dim"]
    train_autoenc_only = cfg.meta.get("train_autoenc_only", False)
    freeze_encoder = cfg.model.get("encoder", {}).get("freeze", False)

    encoder, encoder_output_dim, spatial_size = build_encoder(
        cfg.model.encoder,
        input_channels=num_channels,
        img_size=img_size,
        device=device,
    )
    h, w = spatial_size

    test_input = torch.rand(1, num_channels, 1, img_size, img_size)
    encoder.eval()
    test_output = encoder(test_input)
    encoder.train()
    logger.info(f"Encoder output: {tuple(test_output.shape)}")

    predictor = build_predictor(
        cfg.model.get("predictor", {}),
        input_dim=encoder_output_dim,
        action_dim=action_dim,
        spatial_size=spatial_size,
        device=device,
        train_autoenc_only=train_autoenc_only,
    )

    aencoder = nn.Identity()
    if h > 1 and isinstance(predictor, nn.Identity):
        pass
    elif h > 1:
        from eb_jepa.models.predictors import (
            ConvGRUPredictor,
            ConvNeXtGRUPredictor,
            UNetGRUPredictor,
        )

        predictor_type = cfg.model.get("predictor", {}).get("type", "rnn")
        if not isinstance(
            predictor, (ConvGRUPredictor, ConvNeXtGRUPredictor, UNetGRUPredictor)
        ):
            logger.warning(
                f"Encoder has spatial output ({h}x{w}) but predictor '{predictor_type}' "
                f"has no spatial interaction. Consider using predictor_type: conv_gru."
            )

    use_proj = cfg.model.regularizer.get("use_proj", False)
    regularizer = build_regularizer(
        cfg.model.regularizer,
        encoder_output_dim=encoder_output_dim,
        spatial_size=spatial_size,
        action_dim=action_dim,
        device=device,
        use_proj=use_proj,
    )
    ploss = build_predcost()

    # -- COST MODULE (optional, for planning objectives)
    cost_module = None
    skip_cost_module = False
    if cfg.meta.get("eval_only_mode", False) and cfg.meta.get(
        "skip_cost_module_at_eval", False
    ):
        skip_cost_module = True
        logger.info(
            "Skipping cost module creation: skip_cost_module_at_eval=True in eval_only_mode"
        )

    if not skip_cost_module:
        cost_module = build_cost_module(
            cfg.model.get("cost"),
            encoder_output_dim=encoder_output_dim,
            device=device,
        )
    elif skip_cost_module:
        logger.info("Cost module NOT created (skipped for evaluation)")

    jepa = JEPAWithCostModule(
        encoder,
        aencoder,
        predictor,
        regularizer,
        ploss,
        cost_module=cost_module,
    ).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    log_config(cfg)

    # -- PROBER
    probe_cfg = cfg.get("probe", {})
    probe_state_dims = list(probe_cfg.get("state_dims", [0, 1]))

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

    xy_head, xy_prober = build_xy_prober(
        probe_cfg,
        jepa=jepa,
        encoder_output_dim=encoder_output_dim,
        spatial_h=h,
        normalizer=None,
        device=device,
        pos_mean=pos_mean,
        pos_std=pos_std,
    )

    # -- VISUAL DECODER (optional)
    enc_cfg = cfg.model.get("encoder", {})
    visual_decoder, lpips_loss, lpips_fn = build_visual_decoder(
        cfg.get("visual_decoder", {}),
        encoder_output_dim=encoder_output_dim,
        spatial_h=h,
        num_channels=num_channels,
        img_size=img_size,
        cfg_data=cfg.get("data", {}),
        device=device,
        encoder_scale=enc_cfg.get("scale", "tiny"),
        encoder_patch_size=enc_cfg.get("patch_size", 16),
    )

    # Encoder freezing (diagnostic for action-ignoring)
    # freeze_encoder was already popped from enc_cfg above
    if freeze_encoder:
        jepa.encoder.requires_grad_(False)
        # For DinoEncoder, unfreeze the learned projection on top of the
        # pretrained backbone so it can be trained (base_model stays frozen).
        if isinstance(jepa.encoder, DinoEncoder):
            if hasattr(jepa.encoder, "proj") and not isinstance(
                jepa.encoder.proj, nn.Identity
            ):
                jepa.encoder.proj.requires_grad_(True)
            jepa.encoder.final_ln.requires_grad_(True)
        frozen_params = sum(
            p.numel() for p in jepa.encoder.parameters() if not p.requires_grad
        )
        trainable_params = sum(
            p.numel() for p in jepa.encoder.parameters() if p.requires_grad
        )
        logger.info(
            f"Encoder frozen: {frozen_params} params frozen, "
            f"{trainable_params} params trainable (projection)"
        )

    jepa_train_params = [p for p in jepa.parameters() if p.requires_grad]
    param_groups = [{"params": jepa_train_params, "lr": cfg.optim.lr}]
    jepa_optimizer = build_optimizer(cfg.optim, param_groups)
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    all_probe_params = list(xy_head.parameters())
    if visual_decoder is not None:
        all_probe_params.extend(visual_decoder.parameters())
    probe_optimizer = torch.optim.AdamW(all_probe_params, lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    # -- LOAD CKPT
    train_decoder_only = cfg.meta.get("train_decoder_only", False)
    if train_autoenc_only and visual_decoder is None:
        raise RuntimeError(
            "train_autoenc_only requires visual_decoder.enabled=true in config."
        )
    model_folder = (
        cfg.meta.get("model_folder")
        if train_decoder_only and cfg.meta.get("model_folder")
        else None
    )
    eval_only = cfg.meta.get("eval_only_mode", False)
    start_epoch, ckpt_info = resume_training(
        folder,
        jepa,
        jepa_optimizer,
        scheduler=jepa_scheduler,
        scaler=scaler,
        device=device,
        load_checkpoint_name=cfg.meta.get("load_checkpoint", "latest.pth.tar"),
        model_folder=model_folder,
        load_optimizer=(not train_decoder_only and not eval_only),
        strict=(not skip_cost_module and not eval_only),
    )
    if ckpt_info.get("resumed", False):
        if "xy_head_state_dict" in ckpt_info:
            xy_head.load_state_dict(unwrap_state_dict(ckpt_info["xy_head_state_dict"]))
        if "visual_decoder_state_dict" in ckpt_info and visual_decoder is not None:
            visual_decoder.load_state_dict(
                unwrap_state_dict(ckpt_info["visual_decoder_state_dict"])
            )
        if not skip_cost_module and not train_decoder_only:
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
        jepa.requires_grad_(False)
        start_epoch = 0
        logger.info("Decoder-only mode: froze all jepa parameters, reset epoch to 0")

    # Log what happened with cost module
    if skip_cost_module and ckpt_info.get("resumed", False):
        logger.info(
            "Loaded checkpoint without cost module (encoder/predictor weights only)"
        )

    # -- EVAL ONLY MODE
    if cfg.meta.get("eval_only_mode", False):
        if not is_main:
            cleanup_distributed()
            return {}
        logger.info("Running evaluation only (no training)")
        eval_loader = eval_val_loader if eval_val_loader is not None else val_loader
        eval_results = launch_unroll_eval(
            jepa,
            env_creator,
            folder,
            start_epoch,
            ckpt_info.get("step", 0),
            "_eval_only",
            loader=eval_loader,
            probers=xy_prober,
            cfg=cfg,
            visual_decoders={1: visual_decoder} if visual_decoder is not None else None,
            lpips_fn=lpips_fn,
        )
        if enable_eval:
            eval_results.update(
                launch_plan_eval(
                    jepa,
                    env_creator,
                    folder,
                    start_epoch,
                    global_step=ckpt_info.get("step", 0),
                    suffix="_eval_only",
                    num_eval_episodes=num_eval_episodes,
                    loader=eval_loader,
                    prober=xy_prober,
                    plan_cfg=plan_cfg,
                    visual_decoders=(
                        {1: visual_decoder} if visual_decoder is not None else None
                    ),
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
        else:
            logger.info("Unroll evaluation complete (plan eval skipped).")
        return eval_results

    rank_acc = EffectiveRankAccumulator()

    jepa_module = jepa
    if not train_decoder_only:
        jepa = wrap_ddp(jepa, device, compile=cfg.model.get("compile", False))
    if visual_decoder is not None:
        visual_decoder = wrap_ddp(visual_decoder, device)
    xy_head = wrap_ddp(xy_head, device)
    xy_prober.head = xy_head

    # -- TRAINING LOOP
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
            a = a[:, :, :-1].to(device)
            loc = loc.permute(0, 2, 1).to(device)  # [B, T, D] -> [B, D, T]
            loc = loc[:, probe_state_dims, :]  # [B, probe_output_dim, T]
            total_loss = torch.tensor(0.0, device=device)

            if train_autoenc_only:
                # Autoencoder mode: encode + regularizer + reconstruction
                # Reconstruction loss flows gradients into the encoder.
                jepa_optimizer.zero_grad()
                probe_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    _, enc_states, (jepa_loss, regl, regl_unweight, regldict, pl) = (
                        jepa(
                            x,
                            a,
                            nsteps=0,
                            unroll_mode="parallel",
                            compute_loss=True,
                            return_all_steps=False,
                        )
                    )
                    # Reconstruction (NOT detached: gradients flow to encoder)
                    decoded = visual_decoder(enc_states)  # [B, C, T, H, W]
                    T_min = min(decoded.shape[2], x.shape[2])
                    if lpips_loss is not None:
                        vd_loss_total = lpips_loss(
                            decoded[:, :, :T_min], x[:, :, :T_min]
                        )
                    else:
                        vd_loss_total = nn.MSELoss()(
                            decoded[:, :, :T_min], x[:, :, :T_min]
                        )
                    # XY probe on all GT timesteps (detached from encoder)
                    probe_output = xy_head(enc_states.detach())  # [B, output_dim, T]
                    probe_targets = unwrap_model(xy_head).normalize_targets(loc)
                    xy_loss = nn.MSELoss()(probe_output, probe_targets)
                    if dset.preprocessor is not None:
                        xy_loss = dset.preprocessor.denormalize_mse(xy_loss)
                    total_loss = jepa_loss + vd_loss_total + xy_loss

                scaler.scale(total_loss).backward()
                grad_clip = cfg.optim.get("grad_clip")
                jepa_grad_norm = optimizer_step(
                    scaler, [jepa_optimizer, probe_optimizer], jepa_module, grad_clip
                )
                jepa_scheduler.step()
                probe_scheduler.step()
            elif train_decoder_only:
                jepa_loss = torch.tensor(0.0, device=device)
                regl = torch.tensor(0.0, device=device)
                regl_unweight = torch.tensor(0.0, device=device)
                regldict = {}
                pl = torch.tensor(0.0, device=device)
                vd_loss_total = torch.tensor(0.0, device=device)

                # Encode once (reused by probe, visual decoder, and effective rank)
                with torch.no_grad():
                    enc_states = jepa_module.encode(x)

                # Calculate probe loss on all timesteps
                probe_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    probe_output = xy_head(enc_states.detach())  # [B, output_dim, T]
                    probe_targets = unwrap_model(xy_head).normalize_targets(loc)
                    xy_loss = nn.MSELoss()(probe_output, probe_targets)
                    if dset.preprocessor is not None:
                        xy_loss = dset.preprocessor.denormalize_mse(xy_loss)
                    probe_loss_total = xy_loss

                # Visual decoder training (temporal chunking with per-timestep backward)
                if visual_decoder is not None:
                    vd_loss_total = train_visual_decoder_temporal(
                        visual_decoder,
                        enc_states,
                        x,
                        lpips_loss,
                        scaler,
                        use_amp,
                        dtype,
                        device,
                    )
                    probe_loss_total = probe_loss_total + vd_loss_total

                total_loss += probe_loss_total
                scaler.scale(xy_loss).backward()
                optimizer_step(scaler, probe_optimizer)
                probe_scheduler.step()
            else:
                jepa_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    rollout_cfg = cfg.model.get("rollout", {})
                    _, enc_states, (jepa_loss, regl, regl_unweight, regldict, pl) = (
                        jepa(
                            x,
                            a,
                            nsteps=rollout_cfg.nsteps,
                            unroll_mode=rollout_cfg.get(
                                "unroll_mode", "autoregressive"
                            ),
                            ctxt_window_time=rollout_cfg.get("ctxt_window_time", 1),
                            compute_loss=True,
                            return_all_steps=False,
                            stop_gradient=rollout_cfg.get("stop_gradient", False),
                            detach_pred_target=rollout_cfg.get(
                                "detach_pred_target", False
                            ),
                        )
                    )
                    total_loss += jepa_loss

                scaler.scale(jepa_loss).backward()
                grad_clip = cfg.optim.get("grad_clip")
                jepa_grad_norm = optimizer_step(
                    scaler, jepa_optimizer, jepa_module, grad_clip
                )
                jepa_scheduler.step()

                # Calculate probe loss on all timesteps
                probe_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    probe_output = xy_head(enc_states.detach())  # [B, output_dim, T]
                    probe_targets = unwrap_model(xy_head).normalize_targets(loc)
                    xy_loss = nn.MSELoss()(probe_output, probe_targets)
                    if dset.preprocessor is not None:
                        xy_loss = dset.preprocessor.denormalize_mse(xy_loss)
                    probe_loss_total = xy_loss

                # Visual decoder training (reuse cached enc_states from JEPA forward)
                vd_loss_total = torch.tensor(0.0, device=device)
                if visual_decoder is not None:
                    vd_loss_total = train_visual_decoder_temporal(
                        visual_decoder,
                        enc_states,
                        x,
                        lpips_loss,
                        scaler,
                        use_amp,
                        dtype,
                        device,
                    )
                    probe_loss_total = probe_loss_total + vd_loss_total

                total_loss += probe_loss_total
                scaler.scale(xy_loss).backward()
                optimizer_step(scaler, probe_optimizer)
                probe_scheduler.step()

            # Accumulate effective rank (reuse cached enc_states from JEPA forward)
            if is_main and global_step % cfg.logging.log_every < rank_acc.max_batches:
                with torch.no_grad():
                    enc = enc_states.detach()  # [B, D, T, H', W']
                    D = enc.shape[1]
                    flat_enc = enc.permute(0, 2, 3, 4, 1).reshape(
                        -1, D
                    )  # [B*T*H'*W', D]
                    rank_acc.accumulate("train/visual_effective_rank", flat_enc)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "reg": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

            itr_time = time() - itr_start_time
            if global_step % cfg.logging.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/reg_loss": regl.item(),
                    "train/reg_loss_unweight": regl_unweight.item(),
                    "train/pred_loss": pl.item(),
                    "train/probe_loss": xy_loss.item(),
                    **(
                        {"train/visual_decoder_loss": vd_loss_total.item()}
                        if visual_decoder is not None
                        else {}
                    ),
                    "global_step": global_step,
                    "epoch": epoch,
                    "itr_time": itr_time,
                    "optim/jepa_lr": jepa_optimizer.param_groups[0]["lr"],
                    "optim/probe_lr": probe_optimizer.param_groups[0]["lr"],
                    **(
                        {
                            "optim/grad_norm": jepa_grad_norm,
                        }
                        if jepa_grad_norm is not None
                        else {}
                    ),
                }
                for loss_name, loss_value in regldict.items():
                    log_data[f"train/regl/{loss_name}"] = loss_value

                log_data.update(rank_acc.compute())

                if is_main and cfg.logging.get("log_wandb"):
                    wandb.log(log_data, step=global_step)

            # Planning eval (only if eval is enabled, rank 0 only)
            if (
                is_main
                and enable_eval
                and not train_autoenc_only
                and (global_step + 1) % cfg.meta.eval_every_itr == 0
                and global_step > 0
            ):
                eval_loader = (
                    eval_val_loader if eval_val_loader is not None else val_loader
                )
                eval_results = launch_plan_eval(
                    jepa_module,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    num_eval_episodes=num_eval_episodes,
                    loader=eval_loader,
                    prober=xy_prober,
                    plan_cfg=plan_cfg,
                    visual_decoders=(
                        {1: visual_decoder} if visual_decoder is not None else None
                    ),
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

            # Light eval (only if eval is enabled, rank 0 only)
            if (
                is_main
                and (global_step + 1) % cfg.meta.light_eval_freq == 0
                and global_step > 0
            ):
                eval_loader = (
                    eval_val_loader if eval_val_loader is not None else val_loader
                )
                eval_results = launch_unroll_eval(
                    jepa_module,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    loader=eval_loader,
                    probers=xy_prober,
                    cfg=cfg,
                    visual_decoders=(
                        {1: visual_decoder} if visual_decoder is not None else None
                    ),
                    lpips_fn=lpips_fn,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

        epoch_time = time() - epoch_start_time

        # Log epoch summary
        log_epoch(
            epoch,
            {
                "loss": total_loss.item(),
                "reg": regl.item(),
                "pred": pl.item(),
                "probe": xy_loss.item(),
            },
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        if is_main and cfg.logging.get("log_wandb"):
            wandb.log(
                {"epoch": epoch, "epoch_time": epoch_time},
                step=global_step,
            )

        # Save checkpoint
        if is_main:
            ckpt_extra = {
                "xy_head_state_dict": unwrap_model(xy_head).state_dict(),
                "probe_optimizer_state_dict": probe_optimizer.state_dict(),
                "probe_scheduler_state_dict": probe_scheduler.state_dict(),
            }
            if visual_decoder is not None:
                ckpt_extra["visual_decoder_state_dict"] = unwrap_model(
                    visual_decoder
                ).state_dict()

            save_training_state(
                folder,
                jepa,
                jepa_optimizer,
                epoch,
                save_every=cfg.logging.save_every,
                scheduler=jepa_scheduler,
                scaler=scaler,
                step=global_step,
                **ckpt_extra,
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
                generate_cost_heatmaps(jepa_module, folder, epoch, cfg, device)
            except Exception as e:
                logger.warning(f"Cost heatmap generation failed (epoch {epoch}): {e}")

    cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(run)
