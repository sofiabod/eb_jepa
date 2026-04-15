from __future__ import annotations

import math
import os
from typing import List, Optional

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from eb_jepa.utils.logging import get_logger
from eb_jepa.vis.frames import (
    FIGSIZE_BASE,
    add_border,
    add_text_overlay,
    prepare_frame,
    save_gif_as_pdf_unroll,
    to_numpy,
)

logger = get_logger(__name__)


# =============================================================================
# Comparison Visualizations
# =============================================================================


def create_comparison_gif(
    gt_seq,
    pred_seq_true,
    pred_seq_random,
    gt_dec=None,
    save_path: str = "comparison.gif",
    fps: int = 15,
    upscale_factor: int = 2,
    frame_labels: List[str] = None,
    ctxt_frames: int = 0,
):
    """
    Create a comparison GIF visualization with multiple sequences.

    Args:
        gt_seq: [B, T, H, W, C] Ground truth sequence
        gt_dec: [B, T, H, W, C] Decoded ground truth (optional)
        pred_seq_true: [B, T, H, W, C] Predictions with true actions
        pred_seq_random: [B, T, H, W, C] Predictions with random actions
        save_path: Output path
        fps: Frames per second
        upscale_factor: Factor to upscale frames for better text readability
        ctxt_frames: Number of initial context frames in pred sequences
            (drawn with cyan border to distinguish from actual predictions).
    """
    b = gt_seq.shape[0]
    num_rows = min(b, 4)

    seqs = [gt_seq, pred_seq_true, pred_seq_random]
    if gt_dec is not None:
        seqs.insert(1, gt_dec)
    seq_length = min(s.shape[1] for s in seqs)

    img_height, img_width = gt_seq.shape[2], gt_seq.shape[3]
    num_cols = len(seqs)

    # Upscaled dimensions for better text rendering
    up_img_height = img_height * upscale_factor
    up_img_width = img_width * upscale_factor
    title_height = 30 * upscale_factor  # Scale title area proportionally

    titles = ["GT"]
    if gt_dec is not None:
        titles.append("Dec GT")
    titles.extend(["GT Act", "Rand Act"])

    frames = []
    for t in range(seq_length):
        canvas = np.zeros(
            (title_height + num_rows * up_img_height, num_cols * up_img_width, 3),
            dtype=np.uint8,
        )

        # Column titles with larger font
        font_scale = 0.4 * upscale_factor
        thickness = max(1, upscale_factor)
        for col, title in enumerate(titles):
            col_x = col * up_img_width + up_img_width // 2
            (tw, _), _ = cv2.getTextSize(
                title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.putText(
                canvas,
                title,
                (col_x - tw // 2, title_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Frames (upscaled)
        pred_col_start = 2 if gt_dec is not None else 1
        for row in range(num_rows):
            base_y = title_height + row * up_img_height
            for col, seq in enumerate(
                seqs
                if gt_dec is None
                else [gt_seq, gt_dec, pred_seq_true, pred_seq_random]
            ):
                frame = prepare_frame(seq[row, t])
                frame_upscaled = cv2.resize(
                    frame,
                    (up_img_width, up_img_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                if ctxt_frames > 0 and col >= pred_col_start:
                    border_w = 3 * upscale_factor
                    if t < ctxt_frames:
                        frame_upscaled = add_border(
                            frame_upscaled,
                            color=(0, 200, 0),
                            width=border_w,
                        )
                    else:
                        frame_upscaled = add_border(
                            frame_upscaled,
                            color=(255, 0, 0),
                            width=border_w,
                        )
                col_x = col * up_img_width
                canvas[
                    base_y : base_y + up_img_height, col_x : col_x + up_img_width
                ] = frame_upscaled

        # Timestep indicator with larger font
        label = (
            frame_labels[t]
            if frame_labels is not None and t < len(frame_labels)
            else f"t={t}"
        )
        add_text_overlay(canvas, label, "bottom_right", font_scale=1.0, thickness=2)
        frames.append(canvas)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    logger.info(f"   ✓ Saved comparison GIF: {os.path.basename(save_path)}")

    # Also save PDF with GT, Dec GT, and GT Act rows
    pdf_path = save_path.replace(".gif", "_unroll.pdf")
    pdf_sequences = [
        [prepare_frame(gt_seq[0, t]) for t in range(seq_length)],
    ]
    row_labels = ["GT"]
    if gt_dec is not None:
        pdf_sequences.append([prepare_frame(gt_dec[0, t]) for t in range(seq_length)])
        row_labels.append("Dec GT")
    pdf_sequences.append(
        [prepare_frame(pred_seq_true[0, t]) for t in range(seq_length)]
    )
    row_labels.append("GT Act")

    # "GT Act" is the last row in the PDF (GT rows don't need annotation)
    gt_act_row_idx = len(row_labels) - 1

    save_gif_as_pdf_unroll(
        pdf_sequences,
        pdf_path,
        num_frames=min(8, seq_length),
        figsize_per_frame=(0.8, 0.8),
        row_labels=row_labels,
        frame_labels=frame_labels,
        ctxt_frames=ctxt_frames,
        ctxt_row_indices=[gt_act_row_idx],
    )

    return frames


# =============================================================================
# Analysis & Plotting
# =============================================================================


def plot_distances(
    data,
    save_path: str,
    figsize: tuple = (4.0, 3.0),
    xlabel: str = "Timesteps",
    ylabel: str = "Distance to goal",
):
    """Plot a line chart and save to file."""
    plt.figure(figsize=figsize, dpi=300)
    sns.lineplot(data=data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def compute_embed_differences(all_encs: torch.Tensor) -> torch.Tensor:
    """Compute MSE differences from goal (last encoding)."""
    sq_diff = (all_encs[:-1] - all_encs[-1:]) ** 2
    return sq_diff.mean(dim=tuple(range(1, all_encs.ndim)))


def analyze_distances(
    obses: torch.Tensor,
    infos: List[dict],
    plot_prefix: str,
    goal_position: torch.Tensor,
    goal_state: torch.Tensor,
    normalizer,
    model,
    objective,
    device: torch.device,
):
    """Analyze distances between observations and goal, generate plots."""
    coords = torch.stack(
        [
            (
                torch.as_tensor(x["dot_position"])
                if not isinstance(x["dot_position"], torch.Tensor)
                else x["dot_position"]
            )
            for x in infos
        ]
    ).unsqueeze(1)

    distances = (
        torch.norm(coords[..., -1, :3] - goal_position[:3].unsqueeze(0), dim=-1)
        .detach()
        .cpu()
    )

    sns.set_theme()
    figsize = (4.0, 3.0)
    plot_distances(distances, plot_prefix + "_distances.pdf", figsize=figsize)

    all_states = (
        normalizer.normalize_state(torch.cat([obses, goal_state.unsqueeze(0)]))
        .unsqueeze(-3)
        .to(device)
    )
    all_encs = model.encode(all_states)
    diffs = compute_embed_differences(all_encs).detach().cpu()

    plot_distances(
        diffs,
        plot_prefix + "_rep_distance_visual.pdf",
        figsize=figsize,
        xlabel="Timesteps",
        ylabel="Rep distance to goal",
    )

    all_objectives = objective(all_encs[:-1]).detach().cpu()
    plot_distances(
        all_objectives,
        plot_prefix + "_objectives.pdf",
        figsize=figsize,
        xlabel="Timesteps",
        ylabel="Objective values",
    )

    return distances, diffs


def plot_losses(
    losses: List[torch.Tensor],
    elite_losses_mean: List[torch.Tensor],
    elite_losses_std: List[torch.Tensor],
    work_dir,
    num_act_stepped: int = 1,
    frameskip: int = 1,
    level: Optional[int] = None,
):
    """Plot losses over optimization steps."""
    if not losses:
        return

    # Pad losses to the same length (max length)
    max_len = max(loss.shape[0] for loss in losses)
    padded_losses = []
    padded_elite_mean = []
    padded_elite_std = []
    for loss, elite_mean, elite_std in zip(losses, elite_losses_mean, elite_losses_std):
        pad_len = max_len - loss.shape[0]
        if pad_len > 0:
            # Pad with last value
            loss = torch.cat([loss, loss[-1:].expand(pad_len, *loss.shape[1:])], dim=0)
            elite_mean = torch.cat(
                [elite_mean, elite_mean[-1:].expand(pad_len, *elite_mean.shape[1:])],
                dim=0,
            )
            elite_std = torch.cat(
                [elite_std, elite_std[-1:].expand(pad_len, *elite_std.shape[1:])], dim=0
            )
        padded_losses.append(loss)
        padded_elite_mean.append(elite_mean)
        padded_elite_std.append(elite_std)

    losses_arr = torch.stack(padded_losses, dim=0).detach().cpu().numpy()
    elite_mean_arr = torch.stack(padded_elite_mean, dim=0).detach().cpu().numpy()
    elite_std_arr = torch.stack(padded_elite_std, dim=0).detach().cpu().numpy()
    n_timesteps, n_opt_steps, n_losses = losses_arr.shape

    sns.set_theme()
    for i in range(n_losses):
        total_plots = min(16, n_timesteps)
        cols = int(np.ceil(total_plots))
        fig_width = FIGSIZE_BASE[0] * cols
        fig_height = FIGSIZE_BASE[1]

        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        steps = np.linspace(0, n_timesteps - 1, total_plots, dtype=int)

        for j, step in enumerate(steps):
            ax = plt.subplot(1, cols, j + 1)
            if n_opt_steps > 1:
                sns.lineplot(data=losses_arr[step, :, i])
                sns.lineplot(data=elite_mean_arr[step, :, i])
                ax.fill_between(
                    range(n_opt_steps),
                    elite_mean_arr[step, :, i] - elite_std_arr[step, :, i],
                    elite_mean_arr[step, :, i] + elite_std_arr[step, :, i],
                    alpha=0.3,
                )
            else:
                ax.bar(0, losses_arr[step, 0, i])
                ax.bar(0, elite_mean_arr[step, 0, i])
                ax.errorbar(
                    0,
                    elite_mean_arr[step, 0, i],
                    yerr=elite_std_arr[step, 0, i],
                    fmt="none",
                    capsize=5,
                )

            ax.set_title(f"Step {step * frameskip * num_act_stepped}")
            ax.tick_params(axis="both")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        prefix = f"level_{level}_" if level is not None else ""
        plt.savefig(work_dir / f"{prefix}losses_{i}.pdf", bbox_inches="tight")
        plt.close()


def plot_actions(
    planned: torch.Tensor,
    gt: torch.Tensor,
    work_dir,
    action_mean: Optional[torch.Tensor] = None,
    action_std: Optional[torch.Tensor] = None,
    dim_labels: Optional[List[str]] = None,
) -> None:
    """Plot per-dimension action trajectories (planned vs GT).

    Args:
        planned: Denormalized planned actions [T, A].
        gt: Denormalized ground-truth actions [T, A].
        work_dir: Episode output folder (Path or str).
        action_mean: Training action mean [A], for ±2σ band.
        action_std: Training action std [A], for ±2σ band.
        dim_labels: Per-dimension labels (e.g. ["x","y","z","rx","ry","rz","grip"]).
    """
    from pathlib import Path

    planned_np = to_numpy(planned)  # [T, A]
    gt_np = to_numpy(gt)  # [T, A]
    T = planned_np.shape[0]
    gt_np = gt_np[:T]  # align to planned length
    A = planned_np.shape[1]
    mean_np = to_numpy(action_mean) if action_mean is not None else None
    std_np = to_numpy(action_std) if action_std is not None else None

    sns.set_theme()
    ncols = min(A, 4)
    nrows = math.ceil(A / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGSIZE_BASE[0] * ncols, FIGSIZE_BASE[1] * nrows),
        squeeze=False,
    )
    timesteps = np.arange(T)

    for d in range(A):
        ax = axes[d // ncols, d % ncols]
        ax.plot(timesteps, planned_np[:, d], "-o", label="planned")
        ax.plot(timesteps, gt_np[:, d], "--s", label="GT")
        if mean_np is not None and std_np is not None:
            ax.axhspan(
                mean_np[d] - 2 * std_np[d],
                mean_np[d] + 2 * std_np[d],
                alpha=0.15,
                color="gray",
            )
        label = dim_labels[d] if dim_labels is not None else f"dim {d}"
        ax.set_title(label)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if d == 0:
            ax.legend()

    for d in range(A, nrows * ncols):
        axes[d // ncols, d % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path(work_dir) / "action_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close()
