from __future__ import annotations

from typing import List, Optional, Union

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)

FIGSIZE_BASE = (4.0, 3.0)


# =============================================================================
# Frame Processing Primitives
# =============================================================================


def to_numpy(frame: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor or array to numpy array."""
    if isinstance(frame, torch.Tensor):
        return frame.detach().cpu().numpy()
    return np.asarray(frame)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert frame to uint8, handling both [0,1] and [0,255] ranges."""
    if frame.dtype == np.uint8:
        return frame
    if frame.max() <= 1.0:
        return (frame * 255).astype(np.uint8)
    return frame.astype(np.uint8)


def to_hwc(frame: np.ndarray) -> np.ndarray:
    """Convert frame from (C, H, W) to (H, W, C) format if needed."""
    if (
        frame.ndim == 3
        and frame.shape[0] in [1, 2, 3]
        and frame.shape[0] < frame.shape[1]
    ):
        return frame.transpose(1, 2, 0)
    return frame


def expand_channels(frame: np.ndarray, target_channels: int = 3) -> np.ndarray:
    """Expand frame to target number of channels (default 3 for RGB)."""
    if frame.ndim == 2:  # Grayscale (H, W)
        return np.stack([frame] * target_channels, axis=-1)
    if frame.ndim == 3 and frame.shape[-1] < target_channels:
        h, w, c = frame.shape
        expanded = np.zeros((h, w, target_channels), dtype=frame.dtype)
        expanded[..., :c] = frame
        return expanded
    return frame


def prepare_frame(frame: Union[torch.Tensor, np.ndarray, None]) -> Optional[np.ndarray]:
    """Convert any frame format to numpy uint8 (H, W, C) with 3 channels."""
    if frame is None:
        return None
    frame = to_numpy(frame)
    frame = to_hwc(frame)
    frame = to_uint8(frame)
    frame = expand_channels(frame)
    return frame


def add_border(
    frame: np.ndarray, color: tuple = (255, 0, 0), width: int = 2
) -> np.ndarray:
    """Add a colored border around a frame."""
    bordered = frame.copy()
    bordered[:width, :] = color
    bordered[-width:, :] = color
    bordered[:, :width] = color
    bordered[:, -width:] = color
    return bordered


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: str = "top_right",
    color: tuple = (255, 255, 255),
    font_scale: float = None,
    thickness: int = None,
) -> np.ndarray:
    """Add text overlay to a frame with auto-scaled font size."""
    h, w = frame.shape[:2]
    scale_factor = min(h, w) / 1000
    if font_scale is None:
        font_scale = max(0.2, scale_factor * 0.5)
    if thickness is None:
        thickness = max(1, int(scale_factor))
    margin = int(h * 0.02)

    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    if position == "top_right":
        text_x = w - text_width - margin
        text_y = text_height + margin
    elif position == "top_left":
        text_x = margin
        text_y = text_height + margin
    elif position == "bottom_right":
        text_x = w - text_width - margin
        text_y = h - margin
    else:  # bottom_left
        text_x = margin
        text_y = h - margin

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


# =============================================================================
# Sequence Processing
# =============================================================================


def frames_to_list(frames) -> List[List[np.ndarray]]:
    """
    Convert various frame input formats to List[List[np.ndarray]].

    Supported inputs:
        - List of (H, W, C) arrays -> single sequence
        - (T, H, W, C) array -> single sequence
        - (B, T, H, W, C) array -> B sequences
        - List of Lists -> B sequences
    """
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:  # (T, H, W, C)
            return [list(frames)]
        elif frames.ndim == 5:  # (B, T, H, W, C)
            return [list(frames[b]) for b in range(frames.shape[0])]
        raise ValueError(f"Unsupported array shape: {frames.shape}")

    if isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("Empty frames list")
        first = frames[0]
        if isinstance(first, (np.ndarray, torch.Tensor)):
            first_np = to_numpy(first) if isinstance(first, torch.Tensor) else first
            if first_np.ndim == 3:  # List of (H, W, C)
                return [
                    [to_numpy(f) if isinstance(f, torch.Tensor) else f for f in frames]
                ]
            elif first_np.ndim == 4:  # List of (T, H, W, C)
                return [list(seq) for seq in frames]
            raise ValueError(f"Unsupported frame shape: {first_np.shape}")
        elif isinstance(first, list):
            return frames
        raise ValueError(f"Unsupported frame type: {type(first)}")

    raise ValueError(f"Unsupported frames type: {type(frames)}")


def select_frame_indices(
    total: int, num_frames: int = None, indices: List[int] = None
) -> List[int]:
    """Select evenly-spaced frame indices or use provided indices."""
    if indices is not None:
        return list(indices)
    if num_frames is None or num_frames >= total:
        return list(range(total))
    return np.linspace(0, total - 1, num_frames, dtype=int).tolist()


# =============================================================================
# GIF/Video Saving
# =============================================================================


def save_gif(
    tensor: torch.Tensor,
    save_path: str,
    fps: int = 10,
    show_frame_numbers: bool = False,
    init_frame=None,
    goal_frame=None,
    upscale_factor: int = 2,
):
    """
    Save a (T, C, H, W) tensor as GIF and PDF with horizontal unrolling.

    Args:
        tensor: Tensor of shape (T, C, H, W) uint8
        save_path: Path to save the GIF file
        fps: Frames per second for the GIF
        show_frame_numbers: Whether to overlay frame numbers on GIF frames
        init_frame: Optional initial state frame for PDF (with red border)
        goal_frame: Optional goal state frame for PDF (with red border)
        upscale_factor: Factor to upscale frames for better text readability in GIF
    """
    total_frames = tensor.shape[0]
    images = []
    images_original = []

    for i in range(total_frames):
        img = prepare_frame(tensor[i])
        images_original.append(img)

        if show_frame_numbers:
            h, w = img.shape[:2]
            img_upscaled = cv2.resize(
                img,
                (w * upscale_factor, h * upscale_factor),
                interpolation=cv2.INTER_NEAREST,
            )
            # Use larger font for better readability
            img_upscaled = add_text_overlay(
                img_upscaled,
                f"Frame {i+1}/{total_frames}",
                "top_right",
                font_scale=0.5,
                thickness=2,
            )
            images.append(img_upscaled)
        else:
            images.append(img)

    imageio.mimsave(save_path, images, fps=fps, loop=0)

    # Also save as PDF with horizontal unrolling (using matplotlib text overlay)
    pdf_path = save_path.replace(".gif", "_unroll.pdf")
    frame_labels = (
        [f"{i+1}/{total_frames}" for i in range(total_frames)]
        if show_frame_numbers
        else None
    )
    save_gif_as_pdf_unroll(
        images_original,
        pdf_path,
        num_frames=min(8, total_frames),
        figsize_per_frame=(0.8, 0.8),
        init_frame=init_frame,
        goal_frame=goal_frame,
        frame_labels=frame_labels,
    )


def save_gif_HWC(frames_list: List, save_path: str, fps: int = 10):
    """Save a list of (H, W, C) frames as a GIF."""
    images = [prepare_frame(f) for f in frames_list]
    imageio.mimsave(save_path, images, fps=fps, loop=0)


# =============================================================================
# PDF Unrolling
# =============================================================================


def save_gif_as_pdf_unroll(
    frames,
    save_path: str,
    num_frames: int = None,
    frame_indices: List[int] = None,
    row_labels: List[str] = None,
    title: str = None,
    figsize_per_frame: tuple = (1.0, 1.0),
    dpi: int = 300,
    init_frame=None,
    goal_frame=None,
    frame_labels: List[str] = None,
    ctxt_frames: int = 0,
    ctxt_row_indices: List[int] = None,
):
    """
    Save frames as a PDF with horizontal unrolling.

    Args:
        frames: Frames in various formats (see frames_to_list for supported formats)
        save_path: Path to save the PDF
        num_frames: Number of evenly-spaced frames to include
        frame_indices: Specific frame indices (overrides num_frames)
        row_labels: Labels for each row
        title: Optional figure title
        figsize_per_frame: Size per frame in inches
        dpi: Resolution
        init_frame: Initial frame with red border (left side)
        goal_frame: Goal frame with red border (right side)
        frame_labels: Labels for each frame (e.g., "1/10", "2/10", ...) rendered as
            high-resolution matplotlib text overlay
        ctxt_frames: Number of initial context frames in pred sequences.
            Context frames get cyan borders to distinguish from predictions.
        ctxt_row_indices: Which row indices contain prediction sequences
            that should be annotated (e.g., the "GT Act" row).
    """
    sequences = frames_to_list(frames)
    num_rows = len(sequences)
    total_frames_per_seq = len(sequences[0])

    selected_indices = select_frame_indices(
        total_frames_per_seq, num_frames, frame_indices
    )

    # Prepare init/goal frames
    init_prepared = prepare_frame(init_frame)
    goal_prepared = prepare_frame(goal_frame)
    if init_prepared is not None:
        init_prepared = add_border(init_prepared)
    if goal_prepared is not None:
        goal_prepared = add_border(goal_prepared)

    # Calculate columns
    num_episode_cols = len(selected_indices)
    has_init = init_prepared is not None
    has_goal = goal_prepared is not None
    num_cols = num_episode_cols + int(has_init) + int(has_goal)

    # Create figure
    fig_width = figsize_per_frame[0] * num_cols
    fig_height = figsize_per_frame[1] * num_rows
    if title:
        fig_height += 0.3

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False
    )
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    for row_idx, sequence in enumerate(sequences):
        col_offset = 0

        # Init frame
        if has_init:
            axes[row_idx, 0].imshow(init_prepared)
            axes[row_idx, 0].axis("off")
            axes[row_idx, 0].set_aspect("equal")
            col_offset = 1

        # Episode frames
        for col_idx, frame_idx in enumerate(selected_indices):
            ax = axes[row_idx, col_offset + col_idx]
            frame = prepare_frame(sequence[frame_idx])

            # Green border on context frames, red on predicted frames
            if ctxt_frames > 0 and ctxt_row_indices and row_idx in ctxt_row_indices:
                if frame_idx < ctxt_frames:
                    frame = add_border(frame, color=(0, 200, 0), width=3)
                else:
                    frame = add_border(frame, color=(255, 0, 0), width=3)

            ax.imshow(frame, cmap="gray" if frame.ndim == 2 else None)
            ax.axis("off")
            ax.set_aspect("equal")

            # Use matplotlib text for high-resolution overlay (row label on first frame)
            if col_idx == 0 and row_labels and row_idx < len(row_labels):
                ax.text(
                    0.02,
                    0.98,
                    row_labels[row_idx],
                    transform=ax.transAxes,
                    fontsize=8,
                    color="white",
                    verticalalignment="top",
                    horizontalalignment="left",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
                )

            # Add frame label using matplotlib text (high-resolution)
            if frame_labels and frame_idx < len(frame_labels):
                ax.text(
                    0.98,
                    0.98,
                    f"{frame_labels[frame_idx]}",
                    transform=ax.transAxes,
                    fontsize=6,
                    color="white",
                    verticalalignment="top",
                    horizontalalignment="right",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
                )

        # Goal frame
        if has_goal:
            axes[row_idx, num_cols - 1].imshow(goal_prepared)
            axes[row_idx, num_cols - 1].axis("off")
            axes[row_idx, num_cols - 1].set_aspect("equal")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.savefig(save_path, bbox_inches="tight", dpi=dpi, format="pdf", pad_inches=0.0)
    plt.close(fig)
    logger.info(f"PDF unroll figure saved to {save_path}")
    return save_path


# =============================================================================
# Image Grid Display
# =============================================================================


def show_images(
    tensor,
    nrow: int = 4,
    titles: List[str] = None,
    labels: List[str] = None,
    save_path: str = None,
    dpi: int = 150,
    close_fig: bool = True,
    first_channel_only: bool = True,
    clamp: bool = True,
):
    """
    Display and optionally save a grid of images from a PyTorch tensor.

    Args:
        tensor: Input tensor of shape (B, C, H, W) or (B, T, C, H, W)
        nrow: Number of images per row
        titles: List of titles for each image
        labels: List of labels for each image
        save_path: Path to save figure
        dpi: Resolution
        close_fig: Whether to close figure after saving
        first_channel_only: Keep only first channel
        clamp: Clamp values to [0, 1]
    """
    tensor = to_numpy(tensor)

    if tensor.ndim == 5:
        tensor = tensor[:, 0]
    if tensor.ndim == 4 and first_channel_only:
        tensor = tensor[:, 0:1]
    if clamp:
        tensor = np.clip(tensor, 0, 1)

    batch_size = tensor.shape[0]
    ncol = min(nrow, batch_size)
    nrow_actual = (batch_size + ncol - 1) // ncol

    fig, axes = plt.subplots(
        nrow_actual, ncol, figsize=(ncol * 2, nrow_actual * 2), dpi=dpi
    )
    if nrow_actual == 1 and ncol == 1:
        axes = [[axes]]

    for i, ax in enumerate(axes.flat):
        if i >= batch_size:
            ax.axis("off")
            continue
        img = tensor[i].squeeze()
        if img.ndim == 3 and img.shape[0] <= 4:
            img = img.transpose(1, 2, 0)
            if img.shape[-1] < 3:
                img = expand_channels(img)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=10)
        if labels:
            ax.text(
                0.5,
                -0.15,
                labels[i],
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if not close_fig or not save_path:
        plt.show()
    if close_fig:
        plt.close(fig)


def save_decoded_frames(
    pred_frames_over_iterations: List,
    costs: List[float],
    plan_vis_path: str,
    overlay: bool = True,
    upscale_size: int = 256,
):
    """
    Save decoded frames from planning iterations as a GIF.

    Args:
        pred_frames_over_iterations: List of (T, H, W, C) arrays
        costs: List of costs per iteration
        plan_vis_path: Path prefix for outputs
        overlay: Whether to add iteration overlay
        upscale_size: Target size for upscaling small frames before overlay
    """
    if pred_frames_over_iterations is None or plan_vis_path is None:
        return

    frames = []
    for i, pred_frames in enumerate(pred_frames_over_iterations):
        for frame in pred_frames:
            frame_copy = frame.copy()
            if overlay:
                h, w = frame_copy.shape[:2]
                if min(h, w) < upscale_size:
                    frame_copy = cv2.resize(
                        frame_copy,
                        (upscale_size, upscale_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                add_text_overlay(
                    frame_copy,
                    f"Iter {i+1}",
                    "top_left",
                    (200, 200, 200),
                    font_scale=0.7,
                    thickness=2,
                )
            frames.append(frame_copy)

    imageio.mimsave(f"{plan_vis_path}.gif", frames, duration=0.1, loop=0)
    logger.info(f"Plan decoding video saved to {plan_vis_path}")

    # Save last iteration as PDF (first frame = context in green, rest = predictions in red)
    last_frames = pred_frames_over_iterations[-1]
    save_gif_as_pdf_unroll(
        list(last_frames),
        f"{plan_vis_path}_unroll.pdf",
        num_frames=min(8, len(last_frames)),
        figsize_per_frame=(0.8, 0.8),
        ctxt_frames=1,
        ctxt_row_indices=[0],
    )
