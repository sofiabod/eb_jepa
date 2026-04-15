"""
Video JEPA Training Script

Train a self-supervised video prediction model on Moving MNIST using
Joint Embedding Predictive Architecture (JEPA) with VC regularization.
"""

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

from itertools import islice
from pathlib import Path

import fire
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.builders import build_optimizer
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses.anticollapse import VCLoss
from eb_jepa.losses.prediction import SquareLossSeq
from eb_jepa.models.components import (
    DetHead,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.models.decoders import ImageDecoder
from eb_jepa.utils.checkpoint import resume_training, save_training_state
from eb_jepa.utils.config import (
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    resolve_experiment_folder,
)
from eb_jepa.utils.distributed import (
    cleanup_distributed,
    local_batch_size,
    make_sampler,
    setup_distributed,
    unwrap_model,
    unwrap_state_dict,
    wrap_ddp,
)
from eb_jepa.utils.logging import get_logger
from eb_jepa.utils.training import (
    EffectiveRankAccumulator,
    optimizer_step,
    setup_amp,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.video_jepa.eval import validation_loop
from examples.video_jepa.moving_mnist import MovingMNISTDet

logger = get_logger(__name__)


def run(
    fname: str = "examples/video_jepa/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train a Video JEPA model on Moving MNIST.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., model.lr=0.001)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Distributed setup
    local_rank, world_size, is_main = setup_distributed()

    # Setup
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed, rank=local_rank)

    exp_dir, exp_name = resolve_experiment_folder("video_jepa", cfg, folder)

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "video_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["video_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb and is_main,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    # Load datasets
    train_set = MovingMNISTDet(split="train")
    val_set = MovingMNISTDet(split="val")
    per_gpu_bs = local_batch_size(cfg.data.batch_size)
    sampler = make_sampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=per_gpu_bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=per_gpu_bs,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    log_data_info(
        "MovingMNIST",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_set),
        val_samples=len(val_set),
    )

    # Initialize Video JEPA model
    logger.info("Initializing model...")
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    # Initialize decoder and detection head (for evaluation only)
    decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs)
    dethead = DetHead(cfg.model.dstc, cfg.model.hpre, cfg.model.dobs)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    # Mixed precision (video_jepa doesn't use AMP but we need a scaler for optimizer_step)
    _, _, scaler = setup_amp(cfg, device)

    # Set learning rates for different components
    # Lower learning rate for pixel decoder to prevent overfitting
    param_groups = [
        {"params": jepa.parameters(), "lr": cfg.optim.lr},
        {"params": pixel_decoder.head.parameters(), "lr": cfg.optim.lr / 10},
        {"params": detection_head.head.parameters(), "lr": cfg.optim.lr},
    ]
    optimizer = build_optimizer(cfg.optim, param_groups)

    # Save and log configuration
    if is_main:
        config_path = exp_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        logger.info(f"Saved complete config to {config_path}")
    log_config(cfg)

    # Auto-resume from checkpoint (no-op if no checkpoint exists)
    start_epoch, ckpt_info = resume_training(
        exp_dir,
        jepa,
        optimizer,
        device=device,
        load_checkpoint_name=cfg.meta.get("load_checkpoint", "latest.pth.tar"),
    )
    if ckpt_info.get("resumed", False):
        if "decoder_state_dict" in ckpt_info:
            decoder.load_state_dict(unwrap_state_dict(ckpt_info["decoder_state_dict"]))
        if "dethead_state_dict" in ckpt_info:
            dethead.load_state_dict(unwrap_state_dict(ckpt_info["dethead_state_dict"]))
    global_step = ckpt_info.get("step", 0)

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")

    jepa_module = jepa
    jepa = wrap_ddp(jepa, device, compile=cfg.model.get("compile", False))
    decoder = wrap_ddp(decoder, device)
    dethead = wrap_ddp(dethead, device)
    pixel_decoder.head = decoder
    detection_head.head = dethead

    rank_acc = EffectiveRankAccumulator()
    steps_per_epoch = len(train_loader)
    iterations_per_epoch = cfg.optim.get("iterations_per_epoch", None)
    if iterations_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, iterations_per_epoch)

    for epoch in range(start_epoch, cfg.optim.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(
            islice(enumerate(train_loader), steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False) or not is_main,
        )

        for idx, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["video"]
            loc_map = batch["digit_location"]

            optimizer.zero_grad()
            _, _, (jepa_loss, regl, _, regldict, pl) = jepa(
                x,
                actions=None,
                nsteps=cfg.model.steps,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False,
            )
            recon_loss = pixel_decoder(x, x)
            det_loss = detection_head(x, loc_map)
            total_loss = jepa_loss + recon_loss + det_loss

            scaler.scale(total_loss).backward()
            grad_clip = cfg.optim.get("grad_clip")
            jepa_grad_norm = optimizer_step(scaler, optimizer, jepa_module, grad_clip)

            # Accumulate effective rank every step (deque handles windowing)
            with torch.no_grad():
                enc = jepa_module.encode(x)  # [B, D, T, H', W']
                D = enc.shape[1]
                flat_enc = enc.permute(0, 2, 3, 4, 1).reshape(-1, D)  # [B*T*H'*W', D]
                rank_acc.accumulate("train/visual_effective_rank", flat_enc)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{jepa_loss.item():.4f}",
                    "vc": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

            global_step += 1

        # Validation and logging
        if is_main and epoch % cfg.logging.log_every == 0:
            val_logs = validation_loop(
                val_loader, jepa, detection_head, pixel_decoder, cfg.model.steps, device
            )

            train_metrics = {
                "epoch": epoch,
                "train/loss": jepa_loss.item(),
                "train/vc_loss": regl.item(),
                "train/pred_loss": pl.item(),
                "train/recon_loss": recon_loss.item(),
                "train/det_loss": det_loss.item(),
                **(
                    {
                        "optim/grad_norm": jepa_grad_norm,
                    }
                    if jepa_grad_norm is not None
                    else {}
                ),
            }
            for k, v in regldict.items():
                train_metrics[f"train/{k}"] = float(v)

            all_metrics = {**train_metrics, **val_logs}
            all_metrics.update(rank_acc.compute())

            if wandb_run:
                import wandb

                wandb.log(all_metrics, step=global_step)

            log_epoch(
                epoch,
                {
                    "loss": jepa_loss.item(),
                    "vc": regl.item(),
                    "pred": pl.item(),
                    "val_recon": val_logs.get("val/recon_loss", 0),
                },
                total_epochs=cfg.optim.epochs,
            )

        # Save checkpoint
        if is_main:
            save_training_state(
                exp_dir,
                jepa,
                optimizer,
                epoch,
                save_every=cfg.logging.save_every,
                step=global_step,
                decoder_state_dict=unwrap_model(decoder).state_dict(),
                dethead_state_dict=unwrap_model(dethead).state_dict(),
            )

    if wandb_run:
        import wandb

        wandb.finish()

    cleanup_distributed()
    logger.info("Training complete!")


if __name__ == "__main__":
    fire.Fire(run)
