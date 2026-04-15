"""Planning cost heatmap visualization over 2D environments.

Renders the estimated planning cost between all 2D positions and one or
more reference goal positions as heatmaps overlaid on the environment.

Supports ``two_rooms``, ``pusht``, and ``pointmaze`` environments.
"""

import math
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.envs import make_env_creator
from eb_jepa.planning.objectives import ReprDistObjective
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)

_DATA_CONFIG_DIR = Path(__file__).parent.parent / "data" / "cfgs"

# Per-env heatmap grid bounds in environment coordinates.
_ENV_BOUNDS = {
    "two_rooms": lambda cfg: {
        "x_min": 0,
        "x_max": cfg["img_size"] - 1,
        "y_min": 0,
        "y_max": cfg["img_size"] - 1,
    },
    "pusht": lambda _cfg: {"x_min": 50, "x_max": 450, "y_min": 50, "y_max": 450},
    "pointmaze": lambda _cfg: {
        "x_min": 0.39,
        "x_max": 3.22,
        "y_min": 0.63,
        "y_max": 3.22,
    },
}

_EDGE_OFFSETS = {"two_rooms": 5.0, "pusht": 20.0}

# imshow origin per env. 'upper' for pointmaze so that increasing y points
# downward, matching the MuJoCo camera orientation.
_DISPLAY_ORIGINS = {"two_rooms": "lower", "pusht": "lower", "pointmaze": "upper"}

# Traversable corner cells for the pointmaze U-maze (MuJoCo coordinates).
# U-maze cell centers: (2,2) (2,3) (2,4) | (3,4) | (4,2) (4,3) (4,4).
# Reachable range ~[0.4, 3.2] per axis; use inner positions well within corridors.
_POINTMAZE_GOALS = [
    (1.2, 1.2),  # top-left (near goal cell)
    (1.2, 2.8),  # top-right
    (2.8, 1.2),  # bottom-left
    (2.8, 2.8),  # bottom-right
]


def _load_data_config(env_name: str):
    """Load the data config YAML for *env_name* as an OmegaConf object."""
    path = _DATA_CONFIG_DIR / f"{env_name}.yaml"
    return OmegaConf.load(path)


def _get_normalize_fn(env_name: str) -> Callable:
    """Return the observation normalization function for *env_name*.

    For two_rooms, applies z-score normalization matching the dataset.
    For pusht/pointmaze, applies ImageNet channel normalization (the default
    when ``transform.normalize`` is null in the data config).
    """
    if env_name == "two_rooms":
        from eb_jepa.data.two_rooms_dset import _normalize_two_rooms_obs

        return _normalize_two_rooms_obs

    # PushT / PointMaze: ImageNet channel normalization (mean/std per channel).
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = imagenet_mean.to(x.device)
        std = imagenet_std.to(x.device)
        shape = [1] * (x.dim() - 3) + [3, 1, 1]
        return (x - mean.view(*shape)) / std.view(*shape)

    return _imagenet_normalize


def _setup_env(env_name: str, device: torch.device):
    """Return ``(env, bounds)`` for heatmap rendering.

    Uses ``make_env_creator`` from ``eb_jepa.envs`` to avoid duplicating
    environment construction logic.
    """
    data_cfg = OmegaConf.to_container(_load_data_config(env_name), resolve=False)
    if env_name == "two_rooms":
        data_cfg["device"] = device
    env = make_env_creator(env_name, data_cfg, {})()
    env.reset()

    bounds_fn = _ENV_BOUNDS[env_name]
    bounds = bounds_fn(data_cfg)

    if env_name == "pointmaze":
        inner = env.env if hasattr(env, "env") else env
        while hasattr(inner, "env"):
            inner = inner.env
        inner.prepare_for_render()

    return env, bounds


def _edge_goals(
    env_name: str, bounds: dict, n: int = 4, device: torch.device = torch.device("cpu")
) -> list:
    """Return *n* goal positions in screen-aligned coordinates.

    For pointmaze, returns corners of the traversable U-maze cells.
    For other envs, distributes goals along the environment edges.
    """
    if env_name == "pointmaze":
        return [
            torch.tensor(pos, device=device, dtype=torch.float32)
            for pos in _POINTMAZE_GOALS[:n]
        ]

    offset = _EDGE_OFFSETS[env_name]
    positions: list[torch.Tensor] = []
    edge_specs = [
        (bounds["x_min"] + offset, "y_min", "y_max", True, n // 4 + 1),
        (bounds["x_max"] - offset, "y_min", "y_max", True, n // 4 + 1),
        (bounds["y_min"] + offset, "x_min", "x_max", False, n // 4),
        (bounds["y_max"] - offset, "x_min", "x_max", False, n // 4),
    ]
    for fixed_val, lo_key, hi_key, fixed_is_x, n_slots in edge_specs:
        for i in range(max(1, n_slots)):
            if len(positions) >= n:
                break
            vary = bounds[lo_key] + (i + 1) * (bounds[hi_key] - bounds[lo_key]) / (
                n_slots + 1
            )
            pos = (
                torch.tensor([fixed_val, vary], device=device)
                if fixed_is_x
                else torch.tensor([vary, fixed_val], device=device)
            )
            positions.append(pos)
    return positions[:n]


def _level_desc(level: int, num_levels: int) -> str:
    """Human-readable label for a hierarchy level."""
    if num_levels == 1:
        return ""
    if level == 1:
        return "Level 1 (finest)"
    if level == num_levels:
        return f"Level {level} (coarsest)"
    return f"Level {level}"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_cost_grid(
    model: nn.Module,
    objective,
    env,
    bounds: dict,
    resolution: int,
    normalize_fn: Callable,
    batch_size: int = 256,
    level: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute planning cost over a 2D grid of positions.

    Args:
        model: JEPA model (flat or hierarchical).
        objective: Planning objective callable.
        env: Environment with ``render_at_position`` method.
        bounds: Dict with keys ``x_min``, ``x_max``, ``y_min``, ``y_max``.
        resolution: Grid resolution (points per dimension).
        normalize_fn: Observation normalization function.
        batch_size
        batch_size: Batch size for encoding.
        level: Hierarchy level for encoding (1..L). If None or 1, uses
            level-1 encoding.

    Returns:
        Tuple of ``(x_grid, y_grid, cost_grid)`` as numpy arrays.
    """
    device = next(model.parameters()).device

    x_vals = np.linspace(bounds["x_min"], bounds["x_max"], resolution)
    y_vals = np.linspace(bounds["y_min"], bounds["y_max"], resolution)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="xy")

    positions = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)  # [N, 2]
    n_positions = positions.shape[0]
    logger.info(f"Computing costs for {n_positions} positions...")

    all_costs = []
    is_hierarchical = hasattr(model, "encode_hierarchical")

    with torch.no_grad():
        for i in range(0, n_positions, batch_size):
            batch_pos = positions[i : i + batch_size]  # [B, 2] numpy
            batch_pos_t = torch.tensor(
                batch_pos, device=device, dtype=torch.float32
            )  # [B, 2]
            batch_obs = torch.stack(
                [env.render_at_position(p) for p in batch_pos_t]
            )  # [B, C, H, W]
            batch_obs = normalize_fn(batch_obs.to(device)).unsqueeze(
                2
            )  # [B, C, 1, H, W]

            if is_hierarchical and level is not None and level > 1:
                all_encs = model.encode_hierarchical(batch_obs)
                batch_enc = all_encs[level]
            else:
                batch_enc = model.encode(batch_obs)  # [B, D, 1, H', W']

            all_costs.append(objective(batch_enc, keepdims=False).cpu().numpy())  # [B]

    costs = np.concatenate(all_costs, axis=0)  # [N]
    cost_grid = costs.reshape(resolution, resolution)  # [res_y, res_x]

    logger.info(
        f"Cost grid computed. Min: {costs.min():.4f}, Max: {costs.max():.4f}, "
        f"Mean: {costs.mean():.4f}"
    )

    return x_grid, y_grid, cost_grid


def _collect_heatmap_data(
    model: nn.Module,
    env,
    bounds: dict,
    goal_positions: list,
    level: int,
    normalize_fn: Callable,
    resolution: int = 65,
) -> list:
    """Encode each goal and compute the cost grid at *level*.

    Args:
        model: JEPA model (flat or hierarchical).
        env: Environment with ``render_at_position`` method.
        bounds: Dict with keys ``x_min``, ``x_max``, ``y_min``, ``y_max``.
        goal_positions: List of goal-position tensors ``[2]`` in screen coords.
        level: Hierarchy level (1..L).
        normalize_fn: Observation normalization function.
        resolution
        resolution: Grid resolution per dimension.

    Returns:
        List of dicts with keys ``x_grid``, ``y_grid``, ``cost_grid``,
        ``goal_position``.
    """
    device = next(model.parameters()).device
    is_hierarchical = hasattr(model, "encode_hierarchical")
    heatmap_data: list[dict] = []

    for goal_pos in goal_positions:
        goal_obs = env.render_at_position(goal_pos)  # [C, H, W] float [0, 1]
        goal_obs_norm = normalize_fn(goal_obs.unsqueeze(0).to(device)).unsqueeze(
            2
        )  # [1, C, 1, H, W]

        with torch.no_grad():
            if is_hierarchical:
                goal_encs = model.encode_hierarchical(goal_obs_norm)
                goal_enc = goal_encs[level]
            else:
                goal_enc = model.encode(goal_obs_norm)

        objective = ReprDistObjective(
            target_enc=goal_enc, distance="l2", sum_all_diffs=True
        )
        x_grid, y_grid, cost_grid = compute_cost_grid(
            model,
            objective,
            env,
            bounds,
            resolution=resolution,
            normalize_fn=normalize_fn,
            level=level,
        )
        heatmap_data.append(
            {
                "x_grid": x_grid,
                "y_grid": y_grid,
                "cost_grid": cost_grid,
                "goal_position": goal_pos,
            }
        )

    return heatmap_data


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_single_heatmap(
    ax: plt.Axes,
    data: dict,
    idx: int,
    wall_img_np: Optional[np.ndarray] = None,
    origin: str = "lower",
) -> None:
    """Render a single cost heatmap subplot (heatmap + optional wall overlay + goal).

    Args:
        ax: Matplotlib axes to render into.
        data: Dict with keys ``x_grid``, ``y_grid``, ``cost_grid``,
            ``goal_position``.
        idx: Goal index (0-based), used for subplot title.
        wall_img_np: Optional wall image as numpy array ``[H, W]``.
        origin: ``imshow`` origin (``"lower"`` or ``"upper"``).
    """
    x_grid, y_grid = data["x_grid"], data["y_grid"]
    cost_grid, goal_position = data["cost_grid"], data["goal_position"]

    heatmap_extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]

    heatmap = ax.imshow(
        cost_grid,
        cmap="hot",
        origin=origin,
        extent=heatmap_extent,
        alpha=1.0,
        zorder=0,
    )

    if wall_img_np is not None:
        H, W = wall_img_np.shape
        wall_extent = [0, W - 1, 0, H - 1]
        wall_rgba = np.zeros((*wall_img_np.shape, 4))
        wall_mask = wall_img_np > 128
        wall_rgba[wall_mask, :3] = 0.15
        wall_rgba[wall_mask, 3] = 0.85
        ax.imshow(wall_rgba, origin=origin, extent=wall_extent, zorder=1)

    goal_x, goal_y = goal_position.cpu().numpy()
    ax.plot(
        goal_x,
        goal_y,
        "g*",
        markersize=22,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Goal",
        zorder=5,
    )

    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Planning cost", fontsize=10)
    ax.set_xlabel("X position", fontsize=10)
    ax.set_ylabel("Y position", fontsize=10)
    ax.set_title(
        f"Goal {idx + 1}: ({goal_x:.1f}, {goal_y:.1f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")


def _save_figure(fig, output_path: Union[str, Path]) -> None:
    """Save figure to *output_path* and, if not PNG, also as PNG."""
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved combined heatmap to {output_path}")
    if output_path.suffix != ".png":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved combined heatmap to {png_path}")
    plt.close(fig)


def visualize_heatmaps_grid(
    heatmap_data: list,
    output_path: str,
    wall_img: Optional[torch.Tensor] = None,
    suptitle: Optional[str] = None,
    gradient: bool = False,
    origin: str = "lower",
):
    """Create and save a grid of heatmap visualizations.

    Args:
        heatmap_data: List of dicts with keys ``x_grid``, ``y_grid``,
            ``cost_grid``, ``goal_position``.
        output_path: Path to save figure.
        wall_img: Optional wall image ``[H, W]`` from two_rooms environment.
        suptitle: Optional figure suptitle (e.g., "Level 1 Planning Cost").
        gradient: If True, overlay contour lines and negative-gradient
            (steepest-descent) arrows on each subplot.
        origin: ``imshow`` origin (``"lower"`` or ``"upper"``).
    """
    wall_img_np = wall_img.cpu().numpy() if wall_img is not None else None

    with plt.style.context("default"):
        n = len(heatmap_data)
        ncols = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 5.4 * nrows),
            dpi=150,
            gridspec_kw={"hspace": 0.25, "wspace": 0.30},
        )
        axes = np.atleast_1d(np.array(axes)).flatten()

        for idx, data in enumerate(heatmap_data):
            ax = axes[idx]
            _render_single_heatmap(
                ax, data, idx, wall_img_np=wall_img_np, origin=origin
            )

            if gradient:
                x_grid, y_grid = data["x_grid"], data["y_grid"]
                cost_grid = data["cost_grid"]
                resolution = cost_grid.shape[0]

                x_vals = x_grid[0, :]
                y_vals = y_grid[:, 0]

                dcost_dy, dcost_dx = np.gradient(cost_grid, y_vals, x_vals)

                raw_levels = np.percentile(cost_grid, np.linspace(5, 95, 10))
                levels = np.unique(raw_levels)
                if len(levels) >= 2:
                    cs = ax.contour(
                        x_grid,
                        y_grid,
                        cost_grid,
                        levels=levels,
                        colors="cyan",
                        linewidths=0.6,
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.clabel(cs, cs.levels[::3], inline=True, fontsize=6, fmt="%.1f")

                step = max(1, resolution // 12)
                xs = x_grid[::step, ::step]
                ys = y_grid[::step, ::step]
                dx = -dcost_dx[::step, ::step]
                dy = -dcost_dy[::step, ::step]

                mag = np.sqrt(dx**2 + dy**2)
                mag = np.where(mag == 0, 1.0, mag)
                dx = dx / mag
                dy = dy / mag

                ax.quiver(
                    xs,
                    ys,
                    dx,
                    dy,
                    color="white",
                    edgecolor="black",
                    linewidth=0.3,
                    scale=30,
                    width=0.004,
                    headwidth=3.5,
                    headlength=4,
                    alpha=0.85,
                    zorder=3,
                )

        for idx in range(n, len(axes)):
            axes[idx].axis("off")

        title = suptitle
        if gradient and suptitle is not None:
            title = f"{suptitle} (with gradient field)"
        if title is not None:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.92)
        _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Public API (called by training loops)
# ---------------------------------------------------------------------------


def generate_cost_heatmaps(
    model: nn.Module,
    folder: Union[str, Path],
    epoch: int,
    cfg,
    device: torch.device,
) -> None:
    """Generate and save cost heatmaps for a JEPA model (flat or hierarchical).

    Works for both flat (``JEPAWithCostModule``) and hierarchical
    (``HierarchicalJEPA``) models.  Detects the model type automatically
    via ``hasattr(model, "encode_hierarchical")``.

    Supports ``two_rooms``, ``pusht``, and ``pointmaze`` environments.

    Args:
        model: Trained JEPA model (flat or hierarchical).
        folder: Experiment folder (heatmaps saved under ``folder/heatmaps/``).
        epoch: Current epoch number.
        cfg: Training config (OmegaConf).
        device: Torch device.
    """
    env_name = cfg.data.env_name
    heatmaps_dir = Path(folder) / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    num_levels = model.num_levels if hasattr(model, "encode_hierarchical") else 1
    normalize_fn = _get_normalize_fn(env_name)
    origin = _DISPLAY_ORIGINS.get(env_name, "lower")

    model.eval()
    env, bounds = _setup_env(env_name, device=device)
    goal_positions = _edge_goals(env_name, bounds, n=4, device=device)

    wall_img = getattr(env, "wall_img", None)

    for level in range(1, num_levels + 1):
        heatmap_data = _collect_heatmap_data(
            model, env, bounds, goal_positions, level, normalize_fn
        )

        desc = _level_desc(level, num_levels)
        if desc:
            output_path = heatmaps_dir / f"epoch_{epoch}_level_{level}.pdf"
            suptitle = f"Epoch {epoch} {desc}"
        else:
            output_path = heatmaps_dir / f"epoch_{epoch}.pdf"
            suptitle = f"Epoch {epoch}"

        visualize_heatmaps_grid(
            heatmap_data,
            str(output_path),
            wall_img=wall_img,
            suptitle=suptitle,
            origin=origin,
        )
        gradient_path = str(output_path).replace(".pdf", "_gradient.pdf")
        visualize_heatmaps_grid(
            heatmap_data,
            gradient_path,
            wall_img=wall_img,
            suptitle=suptitle,
            gradient=True,
            origin=origin,
        )
        logger.info(f"Saved heatmap: {output_path}")

    model.train()
