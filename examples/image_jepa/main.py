"""
Image VICReg Training Script - Native PyTorch Implementation

This script implements VICReg training on CIFAR-10 or ImageNet1k using only PyTorch
and torchvision. Supports both ResNet and Vision Transformer (ViT) backbones.

Usage:
    # CIFAR-10 (default):
    python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml

    # ImageNet1k:
    python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/imagenet1k.yaml

    # With config + overrides:
    python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml optim.epochs=50
"""

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass
import time
from itertools import islice
from pathlib import Path
from typing import Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.models import VisionTransformer
from tqdm import tqdm

from eb_jepa.builders import build_optimizer
from eb_jepa.losses.anticollapse import BCS, VICRegLoss
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
from examples.image_jepa.dataset import (
    DATASET_ABBREV,
    ImageDataset,
    get_dataset_info,
    get_val_transforms,
    make_transforms,
)
from examples.image_jepa.eval import LinearProbe, evaluate_linear_probe

logger = get_logger(__name__)


RESNET_FEATURES_DIM = {"resnet18": 512, "resnet50": 2048}


class TorchvisionResNet(nn.Module):
    """Torchvision ResNet backbone (resnet18 or resnet50).

    Args:
        arch: Architecture name ('resnet18' or 'resnet50').
        image_size: Input image resolution. Uses standard 7x7 conv1 + maxpool
            for image_size >= 64 (e.g. ImageNet), and 3x3 conv1 without maxpool
            for smaller images (e.g. CIFAR-10).
    """

    def __init__(self, arch: str = "resnet18", image_size: int = 32):
        super().__init__()
        build_fn = getattr(torchvision.models, arch)
        self.backbone = build_fn()
        self.backbone.fc = nn.Identity()
        if image_size < 64:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.backbone.maxpool = nn.Identity()
        self.features_dim = RESNET_FEATURES_DIM[arch]

    def forward(self, x):
        return self.backbone(x)


class ImageSSL(nn.Module):
    """Image Self-Supervised Learning model implementation."""

    def __init__(
        self, backbone, features_dim, proj_hidden_dim=2048, proj_output_dim=2048
    ):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.squeeze(-1).squeeze(-1)
        projections = self.projector(features)
        return features, projections


def make_warmup_cosine_scheduler(
    optimizer, warmup_epochs, max_epochs, warmup_start_lr, base_lr, min_lr
):
    """Warmup + cosine annealing using built-in PyTorch schedulers."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup = LinearLR(
        optimizer,
        start_factor=max(warmup_start_lr / base_lr, 1e-8) if base_lr > 0 else 1e-8,
        end_factor=1.0,
        total_iters=max(warmup_epochs - 1, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(max_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    linear_probe,
    scaler,
    device,
    epoch,
    loss_fn,
    use_amp=True,
    dtype=torch.float16,
    tqdm_silent=False,
    rank_acc: Optional["EffectiveRankAccumulator"] = None,
    max_iterations: Optional[int] = None,
    schedule_per_step: bool = False,
    grad_clip: float = 1.0,
):
    """Train for one epoch."""
    model.train()
    linear_probe.train()

    # Dynamic loss accumulator
    loss_totals = {}
    total_linear_loss = 0
    linear_correct = 0
    linear_correct_top5 = 0
    linear_total = 0
    last_grad_norm = None
    num_batches = max_iterations if max_iterations is not None else len(train_loader)

    pbar = tqdm(
        train_loader, desc=f"Epoch {epoch}", disable=tqdm_silent, total=num_batches
    )
    for batch_idx, (views, target) in enumerate(islice(pbar, num_batches)):
        num_views = len(views)
        target = target.to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
            if num_views == 2:
                view1 = views[0].to(device, non_blocking=True)
                view2 = views[1].to(device, non_blocking=True)
                both_views = torch.cat([view1, view2], dim=0)  # [2B, C, H, W]
                all_features, all_z = model(both_views)
                features, _ = all_features.chunk(2)  # [B, D]
                z1, z2 = all_z.chunk(2)  # [B, D]
                loss_dict = loss_fn(z1, z2)
            else:
                B = views[0].shape[0]
                # Group views by spatial resolution for batched forward
                resolution_groups: dict[tuple[int, int], list[int]] = {}
                for i, v in enumerate(views):
                    hw = (v.shape[-2], v.shape[-1])
                    resolution_groups.setdefault(hw, []).append(i)

                all_z_list = [None] * num_views
                features = None
                for hw, indices in resolution_groups.items():
                    group = torch.cat(
                        [views[i].to(device, non_blocking=True) for i in indices],
                        dim=0,
                    )  # [len(indices)*B, C, H, W]
                    group_feat, group_z = model(group)
                    for j, idx in enumerate(indices):
                        all_z_list[idx] = group_z[j * B : (j + 1) * B]
                        if idx == 0:
                            features = group_feat[:B]  # [B, D]

                z_stacked = torch.stack(all_z_list)  # [V, B, D]
                loss_dict = loss_fn(z_stacked)
            loss = loss_dict["loss"]

        with torch.no_grad():
            features_frozen = features.detach().float()

        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)

        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()
        _, top5_pred = linear_outputs.topk(5, dim=1)
        linear_correct_top5_batch = (
            top5_pred.eq(target.unsqueeze(1)).any(1).sum().item()
        )

        total_loss_batch = loss + linear_loss

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        last_grad_norm = optimizer_step(scaler, optimizer, model, grad_clip=grad_clip)

        if schedule_per_step:
            scheduler.step()

        if rank_acc is not None:
            rank_acc.accumulate("train/backbone_effective_rank", features.detach())
            if num_views == 2:
                rank_acc.accumulate("train/projector_effective_rank", z1.detach())
            else:
                rank_acc.accumulate(
                    "train/projector_effective_rank", z_stacked[0].detach()
                )

        # Update metrics dynamically based on loss_dict keys
        for key, value in loss_dict.items():
            if key not in loss_totals:
                loss_totals[key] = 0
            loss_totals[key] += value.item()
        total_linear_loss += linear_loss.item()

        # Update linear probe accuracy (pre-computed under autocast)
        linear_total += target.size(0)
        linear_correct += linear_correct_batch
        linear_correct_top5 += linear_correct_top5_batch

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Linear": f"{linear_loss.item():.4f}",
                "Acc": f"{100.*linear_correct/linear_total:.2f}%",
            }
        )

    # Update learning rate (per-epoch only; per-step is handled in the loop)
    if not schedule_per_step:
        scheduler.step()

    # Build return dict dynamically
    num_batches = batch_idx + 1
    metrics = {key: total / num_batches for key, total in loss_totals.items()}
    metrics["linear_loss"] = total_linear_loss / num_batches
    metrics["linear_acc"] = 100.0 * linear_correct / linear_total
    metrics["linear_acc_top5"] = 100.0 * linear_correct_top5 / linear_total
    if last_grad_norm is not None:
        metrics["grad_norm"] = last_grad_norm

    return metrics


def run(
    fname: str = "examples/image_jepa/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an Image JEPA (VICReg/BCS) model on CIFAR-10 or ImageNet1k.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., optim.epochs=50)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Distributed setup
    local_rank, world_size, is_main = setup_distributed()

    # Setup using shared utilities
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed, rank=local_rank)

    exp_dir, exp_name = resolve_experiment_folder("image_jepa", cfg, folder)

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "image_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["image_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb and is_main,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    dataset_name = cfg.data.get("dataset", "cifar10")
    ds = DATASET_ABBREV.get(dataset_name, dataset_name)
    image_size, num_classes = get_dataset_info(dataset_name)
    logger.info(
        f"Loading {dataset_name} dataset (image_size={image_size}, num_classes={num_classes})..."
    )

    val_transforms = get_val_transforms(dataset_name)

    # Config data_dir takes precedence; fall back to EBJEPA_DSETS env var
    data_dir = cfg.data.get("data_dir") or os.environ.get("EBJEPA_DSETS")
    logger.info(f"Using data directory: {data_dir}")

    if dataset_name == "cifar10":
        base_train_dataset = CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )
        val_dataset = CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transforms
        )
    elif dataset_name == "imagenet1k":
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"ImageNet1k train directory not found at {train_dir}. "
                "Set data.data_dir to the ImageNet root "
                "(e.g. /path/to/ImageNet)."
            )
        base_train_dataset = ImageFolder(root=train_dir, transform=None)
        val_dataset = ImageFolder(root=val_dir, transform=val_transforms)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_dataset = ImageDataset(base_train_dataset, make_transforms(cfg, dataset_name))

    per_gpu_bs = local_batch_size(cfg.data.batch_size)
    sampler = make_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_bs,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    log_data_info(
        dataset_name,
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )

    # Initialize model
    logger.info("Initializing model...")
    patch_size = cfg.model.get("patch_size", 16)
    if cfg.model.type == "resnet":
        resnet_arch = cfg.model.get("resnet_arch", "resnet18")
        backbone = TorchvisionResNet(arch=resnet_arch, image_size=image_size)
        features_dim = backbone.features_dim
    elif cfg.model.type == "vit_s":
        features_dim = 384
        model_kwargs = dict(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=6,
            mlp_dim=4 * features_dim,
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()
    elif cfg.model.type == "vit_cls_s":
        from eb_jepa.models.encoders import ViTCLSEncoder

        features_dim = 384
        backbone = ViTCLSEncoder(
            scale="small", patch_size=patch_size, image_size=image_size
        )
    elif cfg.model.type == "vit_b":
        features_dim = 768
        model_kwargs = dict(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=12,
            mlp_dim=4 * features_dim,
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()

    model = ImageSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
    )

    if not cfg.model.use_projector:
        model.projector = nn.Identity()

    model = model.to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = (
        sum(p.numel() for p in model.projector.parameters())
        if cfg.model.use_projector
        else 0
    )
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})

    # Save and log configuration
    if is_main:
        config_path = exp_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        logger.info(f"Saved complete config to {config_path}")
    log_config(cfg)

    # Initialize linear probe
    linear_probe = LinearProbe(feature_dim=features_dim, num_classes=num_classes).to(
        device
    )

    dtype, use_amp, scaler = setup_amp(cfg, device)

    optim_type = cfg.optim.get("type", "adamw")
    probe_group = {
        "params": linear_probe.parameters(),
        "lr": 1e-3 if optim_type == "adamw" else 0.1,
    }
    if optim_type == "adamw":
        probe_group["weight_decay"] = 1e-7
    param_groups = [
        {"params": model.parameters(), "lr": cfg.optim.lr},
        probe_group,
    ]
    optimizer = build_optimizer(cfg.optim, param_groups)

    schedule_per_step = cfg.optim.get("schedule_per_step", True)
    if schedule_per_step:
        steps_per_epoch_est = len(train_loader)
        iters = cfg.optim.get("iterations_per_epoch", None)
        if iters is not None:
            steps_per_epoch_est = min(steps_per_epoch_est, iters)
        warmup_steps = cfg.optim.warmup_epochs * steps_per_epoch_est
        total_steps = cfg.optim.epochs * steps_per_epoch_est
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=total_steps,
            warmup_start_lr=cfg.optim.warmup_start_lr,
            base_lr=cfg.optim.lr,
            min_lr=cfg.optim.min_lr,
        )
    else:
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=cfg.optim.warmup_epochs,
            max_epochs=cfg.optim.epochs,
            warmup_start_lr=cfg.optim.warmup_start_lr,
            base_lr=cfg.optim.lr,
            min_lr=cfg.optim.min_lr,
        )

    # Initialize loss function
    if cfg.loss.type == "vicreg":
        loss_fn = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff)
    elif cfg.loss.type == "bcs":
        loss_fn = BCS(lmbd=cfg.loss.lmbd)

    loss_fn = loss_fn.to(device)

    # Auto-resume from checkpoint (no-op if no checkpoint exists)
    start_epoch, ckpt_info = resume_training(
        exp_dir,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        load_checkpoint_name=cfg.meta.get("load_checkpoint", "latest.pth.tar"),
    )
    if "linear_probe_state_dict" in ckpt_info:
        linear_probe.load_state_dict(
            unwrap_state_dict(ckpt_info["linear_probe_state_dict"])
        )

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")
    start_time = time.time()
    tqdm_silent = cfg.logging.get("tqdm_silent", False) or not is_main

    # Compile model (before DDP wrapping)
    if torch.cuda.is_available() and cfg.model.get("compile", False):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    model = wrap_ddp(model, device, sync_batchnorm=True)
    linear_probe = wrap_ddp(linear_probe, device)

    rank_acc = EffectiveRankAccumulator()
    grad_clip = cfg.optim.get("grad_clip", 1.0)
    if grad_clip is not None:
        logger.info(f"Gradient clipping enabled: max_norm={grad_clip}")
    steps_per_epoch = len(train_loader)
    iterations_per_epoch = cfg.optim.get("iterations_per_epoch", None)
    if iterations_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, iterations_per_epoch)

    for epoch in range(start_epoch, cfg.optim.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            linear_probe,
            scaler,
            device,
            epoch,
            loss_fn,
            use_amp,
            dtype,
            tqdm_silent,
            rank_acc=rank_acc,
            max_iterations=steps_per_epoch,
            schedule_per_step=schedule_per_step,
            grad_clip=grad_clip,
        )

        # Evaluate linear probe on validation set
        val_acc, val_acc_top5, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, use_amp
        )

        # Log metrics - dynamically add train_ prefix to all train_metrics keys
        log_dict = {"epoch": epoch}
        for key, value in train_metrics.items():
            if "acc" in key:
                log_dict[f"{ds}_train_{key}"] = value
            else:
                log_dict[f"train_{key}"] = value
        log_dict["val_loss"] = val_loss
        log_dict[f"{ds}_val_acc"] = val_acc
        log_dict[f"{ds}_val_acc_top5"] = val_acc_top5
        log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
        log_dict.update(rank_acc.compute())

        if wandb_run:
            wandb.log(log_dict)

        # Log progress
        if is_main and epoch % cfg.logging.log_every == 0:
            elapsed = time.time() - start_time
            log_epoch(
                epoch,
                {
                    "loss": train_metrics["loss"],
                    f"{ds}_val_acc": val_acc,
                    f"{ds}_val_acc_top5": val_acc_top5,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                total_epochs=cfg.optim.epochs,
                elapsed_time=elapsed,
            )

        # Save checkpoint
        if is_main:
            save_training_state(
                exp_dir,
                model,
                optimizer,
                epoch,
                save_every=cfg.logging.save_every,
                scheduler=scheduler,
                scaler=scaler,
                linear_probe_state_dict=unwrap_model(linear_probe).state_dict(),
                linear_val_acc=val_acc,
                linear_val_acc_top5=val_acc_top5,
            )

    logger.info("Training completed!")
    if wandb_run:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(run)
