"""
Analyze results from a training hyperparameter sweep.

Reads eval.csv files from disk (plan_eval/ or unroll_eval/ subdirectories),
computes hyperparameter importance and correlations with the chosen metric,
generates figures (PDF) and metrics (CSV), and suggests a new sweep grid.

For success_rate (default): discovers all planning eval tags under plan_eval/,
averages success_rate over the last N training checkpoints, averages over
seeds, then takes the **max across eval tags** so each training config is
judged by its best planning setup (chosen after seed averaging).

For unroll_eval metrics (mean_lpips, prediction_lpips, mean_mse, etc.):
reads eval.csv files from unroll_eval/step-*/ directories, extracts the
specified metric at the given hierarchy level, and averages over the last N
checkpoints. Lower is better for all unroll_eval metrics.

Supports multiple sweep directories: results from all directories are
combined into a single analysis.  When multiple directories are given,
use ``--output-dir`` to specify where CSVs and figures are saved.

USAGE:
------
python -m scripts.analyze_training_sweep \\
  /checkpoint/.../sweep_2026-03-17_18-29

python -m scripts.analyze_training_sweep \\
  /checkpoint/.../sweep_phase1_baseline_validation --top-n 20

python -m scripts.analyze_training_sweep /path/to/sweep_dir \\
     --metric mean_pos_mse --level 1

# Compare two training runs (multiple sweep dirs):
python -m scripts.analyze_training_sweep \\
  /checkpoint/.../train_autoenc_A /checkpoint/.../train_autoenc_B \\
  --metric mean_lpips_recon --output-dir /tmp/comparison
"""

import argparse
import ast
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
logger = logging.getLogger(__name__)

METRIC = "success_rate"
HIGHER_IS_BETTER = True
SEED_KEY = "meta.seed"

_METRIC_INFO: dict[str, dict] = {
    "success_rate": {"higher_is_better": True, "csv_base": None},
    "mean_lpips": {"higher_is_better": False, "csv_base": "mean_lpips"},
    "mean_lpips_recon": {"higher_is_better": False, "csv_base": "mean_lpips_recon"},
    "prediction_lpips": {"higher_is_better": False, "csv_base": "prediction_lpips"},
    "mean_mse": {"higher_is_better": False, "csv_base": "mean_mse"},
    "mean_pos_mse": {"higher_is_better": False, "csv_base": "mean_pos_mse"},
    "action_sensitivity": {"higher_is_better": True, "csv_base": "action_sensitivity"},
    "val_acc": {"higher_is_better": True, "summary_pattern": "val_acc"},
    "val_acc_top5": {"higher_is_better": True, "summary_pattern": "val_acc_top5"},
}


def _numeric_sort(vals: list[str]) -> list[str]:
    """Sort string values numerically if possible, else lexicographically."""
    try:
        return sorted(vals, key=float)
    except ValueError:
        return sorted(vals)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(d: dict, dotkey: str):
    """Get a value from a nested dict using dot notation."""
    for part in dotkey.split("."):
        if isinstance(d, dict):
            d = d.get(part)
        else:
            return None
    return d


def _unwrap_wandb_config(raw: dict) -> dict:
    """Convert wandb config.yaml format to standard nested dict.

    Wandb wraps each top-level key as ``{"key": {"value": actual_value}}``.
    This unwraps to ``{"key": actual_value}`` to match standard config format.
    """
    result = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and "value" in val:
            result[key] = val["value"]
        else:
            result[key] = val
    return result


def _find_config(model_dir: Path) -> dict | None:
    """Find and parse config from model_dir or wandb subdirs.

    Checks ``model_dir/config.yaml`` first (standard sweep layout), then
    ``model_dir/wandb/run-*/files/config.yaml`` (wandb-synced config), then
    falls back to parsing ``log_config()`` output from ``output.log``
    (in-progress runs where wandb hasn't synced files yet).
    """
    cfg_path = model_dir / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    wandb_cfgs = sorted(model_dir.glob("wandb/run-*/files/config.yaml"))
    if wandb_cfgs:
        with open(wandb_cfgs[-1]) as f:
            raw = yaml.safe_load(f)
        return _unwrap_wandb_config(raw)
    # Fallback: parse config from output.log (in-progress runs)
    logs = sorted(model_dir.glob("wandb/run-*/files/output.log"))
    if logs:
        return _parse_config_from_log(logs[-1])
    return None


def _parse_config_from_log(log_path: Path) -> dict | None:
    """Parse config key=value pairs from log_config() output in output.log.

    Returns a nested dict matching the structure that ``_get_nested`` expects,
    or None if no config lines are found.
    """
    flat: dict = {}
    in_config_block = False
    with open(log_path) as f:
        for line in f:
            if "log_config" not in line:
                if in_config_block:
                    break
                continue
            in_config_block = True
            m = re.search(r"\]\s+([\w.]+)=(.+)$", line.strip())
            if not m:
                continue
            key, raw_val = m.group(1), m.group(2)
            try:
                val = ast.literal_eval(raw_val)
            except (ValueError, SyntaxError):
                val = raw_val
            flat[key] = val
    if not flat:
        return None
    # Convert flat {"a.b.c": val} to nested {"a": {"b": {"c": val}}}
    result: dict = {}
    for dotkey, val in flat.items():
        parts = dotkey.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return result


def discover_sweep_config(sweep_dirs: list[Path]) -> tuple[dict, dict]:
    """Auto-discover swept hyperparameters and original param_grid.

    Searches all model subdirectories in every provided sweep directory.
    Supports both standard config.yaml layout and wandb-only configs.

    Returns:
        Tuple of (sweep_params_display_map, original_param_grid).
        sweep_params maps display_name -> config dot-key for every swept
        param (excluding meta.seed). original_param_grid is the raw dict
        from config.yaml. Both are empty dicts if no config is found.
    """
    for sweep_dir in sweep_dirs:
        for subdir in sorted(sweep_dir.iterdir()):
            if not subdir.is_dir():
                continue
            cfg = _find_config(subdir)
            if cfg is None:
                continue
            param_grid = (cfg.get("sweep") or {}).get("param_grid", {})
            if param_grid:
                params = {}
                for key in param_grid:
                    if key == SEED_KEY:
                        continue
                    parts = key.split(".")
                    short = parts[-1]
                    for p in parts[:-1]:
                        if p.startswith("level_") and p[-1].isdigit():
                            short = f"{p.replace('level_', 'L')} {short}"
                            break
                    params[short] = key
                logger.info(f"Discovered {len(params)} sweep params from {subdir}")
                return params, param_grid
    logger.warning("No config.yaml found in run subdirectories; no sweep params found")
    return {}, {}


def _iter_model_runs(
    sweep_dirs: list[Path],
    sweep_params: dict,
) -> Iterator[tuple[Path, int | None, dict]]:
    """Yield (model_dir, seed, hparams) for each valid run directory.

    Shared iteration logic used by all three collectors.
    """
    for sweep_dir in sweep_dirs:
        for model_dir in sorted(sweep_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            cfg = _find_config(model_dir)
            if cfg is None:
                continue
            seed = _get_nested(cfg, SEED_KEY)
            hparams = {}
            for display_name, config_key in sweep_params.items():
                val = _get_nested(cfg, config_key)
                hparams[display_name] = str(val) if isinstance(val, list) else val
            yield model_dir, seed, hparams


def _find_last_n_step_csvs(parent_dir: Path, avg_last_n: int) -> list[Path]:
    """Return last N eval.csv paths sorted by step number.

    Scans ``parent_dir`` for ``step-*/eval.csv``, sorts by step number,
    and returns the last ``avg_last_n`` paths.
    """
    step_csvs = []
    for step_dir in parent_dir.iterdir():
        if not step_dir.is_dir() or not step_dir.name.startswith("step-"):
            continue
        csv_path = step_dir / "eval.csv"
        if csv_path.is_file():
            step_num = _extract_step_number(step_dir.name)
            step_csvs.append((step_num, csv_path))
    step_csvs.sort(key=lambda x: x[0])
    return [csv_path for _, csv_path in step_csvs[-avg_last_n:]]


# ---------------------------------------------------------------------------
# Data collection (disk-based)
# ---------------------------------------------------------------------------


def _extract_step_number(step_dir_name: str) -> int:
    """Extract the numeric step from a directory name like 'step-4679' or 'step-4679_eval_only'."""
    m = re.match(r"step-(\d+)", step_dir_name)
    return int(m.group(1)) if m else -1


def _extract_unroll_metric(
    df_csv: pd.DataFrame, csv_base: str, level: int
) -> float | None:
    """Extract an aggregate unroll metric from an eval.csv DataFrame.

    Tries precomputed aggregate column first, then falls back to averaging
    per-timestep columns for backward compatibility with older sweep data.

    Args:
        df_csv: Single-row DataFrame from reading eval.csv.
        csv_base: Base metric name (e.g. "mean_lpips", "prediction_lpips").
        level: Hierarchy level (e.g. 1).

    Returns:
        Scalar metric value, or None if no relevant columns found.
    """
    prefix = f"val_rollout/level{level}"

    # Try precomputed aggregate
    agg_col = f"{prefix}/{csv_base}_avg"
    if agg_col in df_csv.columns:
        val = df_csv[agg_col].iloc[-1]
        if pd.notna(val):
            return float(val)

    # Fall back to averaging per-timestep columns
    per_t_cols = [c for c in df_csv.columns if c.startswith(f"{prefix}/{csv_base}/")]
    if per_t_cols:
        vals = [
            float(df_csv[c].iloc[-1])
            for c in per_t_cols
            if pd.notna(df_csv[c].iloc[-1])
        ]
        if vals:
            return float(np.mean(vals))

    # Special case: prediction_lpips from per-timestep mean_lpips - mean_lpips_recon
    if csv_base == "prediction_lpips":
        lpips_cols = sorted(
            c for c in df_csv.columns if c.startswith(f"{prefix}/mean_lpips/")
        )
        recon_cols = sorted(
            c for c in df_csv.columns if c.startswith(f"{prefix}/mean_lpips_recon/")
        )
        if lpips_cols and recon_cols and len(lpips_cols) == len(recon_cols):
            diffs = []
            for lc, rc in zip(lpips_cols, recon_cols):
                lv, rv = df_csv[lc].iloc[-1], df_csv[rc].iloc[-1]
                if pd.notna(lv) and pd.notna(rv):
                    diffs.append(float(lv) - float(rv))
            if diffs:
                return float(np.mean(diffs))

    return None


def _score_eval_tag(
    eval_tag_dir: Path,
    avg_last_n: int = 3,
) -> dict | None:
    """Compute averaged metrics for one (model_folder, eval_tag) pair.

    Finds all step-*/eval.csv under ``eval_tag_dir``, sorts by step number,
    and averages the last ``avg_last_n`` checkpoints.

    Returns:
        Dict with success_rate, avg_episode_time, or None if no valid
        eval.csv files are found.
    """
    csv_paths = _find_last_n_step_csvs(eval_tag_dir, avg_last_n)
    if not csv_paths:
        return None

    sr_vals = []
    time_vals = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if METRIC not in df.columns:
            continue
        sr_vals.append(float(df[METRIC].iloc[-1]))
        if "avg_episode_time" in df.columns:
            time_vals.append(float(df["avg_episode_time"].iloc[-1]))

    if not sr_vals:
        return None

    return {
        METRIC: float(np.mean(sr_vals)),
        "avg_episode_time": float(np.mean(time_vals)) if time_vals else float("nan"),
    }


def collect_disk_results(
    sweep_dirs: list[Path],
    sweep_params: dict,
    avg_last_n: int = 3,
) -> pd.DataFrame:
    """Scan plan_eval/ subdirectories on disk for all model folders.

    For each model folder, discovers all eval tags under plan_eval/ and
    averages success_rate over the last ``avg_last_n`` training checkpoints
    per eval tag.  Returns one row per (model_folder, eval_tag) pair so
    that the caller can aggregate over seeds *before* selecting the best
    planning configuration.

    Args:
        sweep_dirs: Top-level sweep directories containing model subdirs.
        sweep_params: Dict mapping display_name -> config dot-key.
        avg_last_n: Number of latest step-dirs to average over.

    Returns:
        DataFrame with one row per (model_folder, eval_tag).
    """
    rows = []
    missing = 0

    for model_dir, seed, hparams in _iter_model_runs(sweep_dirs, sweep_params):
        plan_eval_dir = model_dir / "plan_eval"
        if not plan_eval_dir.is_dir():
            missing += 1
            continue

        found_any = False
        for eval_tag_dir in sorted(plan_eval_dir.iterdir()):
            if not eval_tag_dir.is_dir():
                continue
            result = _score_eval_tag(eval_tag_dir, avg_last_n=avg_last_n)
            if result is None:
                continue
            found_any = True
            rows.append(
                {
                    METRIC: result[METRIC],
                    "avg_episode_time": result["avg_episode_time"],
                    "eval_tag": eval_tag_dir.name,
                    "seed": seed,
                    "model_folder": str(model_dir),
                    **hparams,
                }
            )

        if not found_any:
            missing += 1

    if missing > 0:
        logger.info(f"Skipped {missing} model folders (no plan_eval results)")

    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df)} (model, eval_tag) results from disk")
    return df


def collect_unroll_results(
    sweep_dirs: list[Path],
    sweep_params: dict,
    csv_base: str,
    level: int,
    avg_last_n: int = 3,
) -> pd.DataFrame:
    """Scan unroll_eval/ subdirectories for all model folders.

    For each model folder, reads eval.csv files from unroll_eval/step-*/,
    extracts the specified metric, and averages over the last ``avg_last_n``
    checkpoints.

    Args:
        sweep_dirs: Top-level sweep directories containing model subdirs.
        sweep_params: Dict mapping display_name -> config dot-key.
        csv_base: Base column name (e.g. "mean_lpips", "prediction_lpips").
        level: Hierarchy level to read from (e.g. 1).
        avg_last_n: Number of latest step-dirs to average over.

    Returns:
        DataFrame with one row per model folder (columns: METRIC, seed,
        model_folder, plus hparam columns).
    """
    rows = []
    missing = 0

    for model_dir, seed, hparams in _iter_model_runs(sweep_dirs, sweep_params):
        unroll_dir = model_dir / "unroll_eval"
        if not unroll_dir.is_dir():
            missing += 1
            continue

        csv_paths = _find_last_n_step_csvs(unroll_dir, avg_last_n)
        if not csv_paths:
            missing += 1
            continue

        metric_vals = []
        for csv_path in csv_paths:
            try:
                df_csv = pd.read_csv(csv_path)
            except Exception:
                continue
            val = _extract_unroll_metric(df_csv, csv_base, level)
            if val is not None:
                metric_vals.append(val)

        if not metric_vals:
            missing += 1
            continue

        rows.append(
            {
                METRIC: float(np.mean(metric_vals)),
                "seed": seed,
                "model_folder": str(model_dir),
                **hparams,
            }
        )

    if missing > 0:
        logger.info(f"Skipped {missing} model folders (no unroll_eval results)")

    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df)} unroll_eval results from disk")
    return df


def _parse_latest_metric_from_log(
    log_path: Path,
    pattern: str,
) -> tuple[float | None, int | None]:
    """Extract the latest metric value from log_epoch() lines in output.log.

    Returns:
        Tuple of (metric_value, epoch) or (None, None) if not found.
    """
    metric_re = re.compile(rf"(\w*{re.escape(pattern)})=([\d.]+)")
    epoch_re = re.compile(r"\[Epoch\s+(\d+)")
    last_val: float | None = None
    last_epoch: int | None = None
    with open(log_path) as f:
        for line in f:
            if "log_epoch" not in line:
                continue
            mm = metric_re.search(line)
            if mm:
                last_val = float(mm.group(2))
                em = epoch_re.search(line)
                last_epoch = int(em.group(1)) if em else None
    return last_val, last_epoch


def collect_wandb_summary_results(
    sweep_dirs: list[Path],
    sweep_params: dict,
    summary_pattern: str,
) -> pd.DataFrame:
    """Collect metrics from wandb summary JSON files.

    Used for image_jepa sweeps where metrics are logged to wandb
    rather than saved in plan_eval/ or unroll_eval/ directories.
    Falls back to parsing output.log for in-progress runs.

    Args:
        sweep_dirs: Top-level sweep directories.
        sweep_params: Dict mapping display_name -> config dot-key.
        summary_pattern: Suffix pattern to match in wandb summary keys
            (e.g. ``"_val_acc"`` matches ``"in1k_val_acc"``).

    Returns:
        DataFrame with one row per run.
    """
    rows = []
    missing = 0

    for model_dir, seed, hparams in _iter_model_runs(sweep_dirs, sweep_params):
        metric_val = None
        epoch = None

        # Try output.log first (updated in real-time by the training script)
        log_val, log_epoch = None, None
        logs = sorted(model_dir.glob("wandb/run-*/files/output.log"))
        if logs:
            log_val, log_epoch = _parse_latest_metric_from_log(
                logs[-1], summary_pattern
            )

        # Fall back to wandb-summary.json (synced to disk infrequently)
        summary_val, summary_epoch = None, None
        summaries = sorted(model_dir.glob("wandb/run-*/files/wandb-summary.json"))
        if summaries:
            with open(summaries[-1]) as f:
                summary = json.load(f)
            matching_keys = [k for k in summary if k.endswith(summary_pattern)]
            if matching_keys:
                summary_val = summary[matching_keys[0]]
                summary_epoch = summary.get("epoch")

        # Pick whichever source has a higher epoch (prefer log when tied)
        if log_val is not None and summary_val is not None:
            if (summary_epoch or -1) > (log_epoch or -1):
                metric_val, epoch = summary_val, summary_epoch
            else:
                metric_val, epoch = log_val, log_epoch
        elif log_val is not None:
            metric_val, epoch = log_val, log_epoch
        elif summary_val is not None:
            metric_val, epoch = summary_val, summary_epoch

        if metric_val is None or (
            isinstance(metric_val, float) and np.isnan(metric_val)
        ):
            missing += 1
            continue

        row = {
            METRIC: float(metric_val),
            "seed": seed,
            "model_folder": str(model_dir),
            **hparams,
        }
        if epoch is not None:
            row["epoch"] = epoch
        rows.append(row)

    if missing > 0:
        logger.info(
            f"Skipped {missing} model folders (no wandb summary or matching metric)"
        )

    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df)} wandb summary results from disk")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _aggregate_over_seeds(
    df: pd.DataFrame,
    hparam_cols: list[str],
    extra_agg: dict | None = None,
) -> pd.DataFrame:
    """Group by hparams, aggregate metric over seeds, sort, and reorder columns.

    Args:
        df: Per-run DataFrame with hparam columns and METRIC column.
        hparam_cols: List of hyperparameter column names to group by.
        extra_agg: Optional extra aggregation specs to pass to ``.agg()``.
            For success_rate this includes avg_episode_time and best_eval_tag.

    Returns:
        Aggregated DataFrame with mean/std METRIC columns first.
    """
    mean_col = f"mean_{METRIC}"
    std_col = f"std_{METRIC}"
    agg_spec = {
        mean_col: (METRIC, "mean"),
        std_col: (METRIC, "std"),
    }
    sort_cols = [mean_col]
    sort_ascending = [not HIGHER_IS_BETTER]

    if extra_agg:
        agg_spec.update(extra_agg)
        if "mean_episode_time" in extra_agg:
            sort_cols.append("mean_episode_time")
            sort_ascending.append(True)

    agg = (
        df.groupby(hparam_cols)
        .agg(count=(METRIC, "count"), **agg_spec)
        .reset_index()
        .sort_values(sort_cols, ascending=sort_ascending)
    )
    return agg[
        [mean_col, std_col] + [c for c in agg.columns if c not in (mean_col, std_col)]
    ]


def compute_importance(
    df: pd.DataFrame,
    hparam_cols: list[str],
) -> pd.DataFrame:
    """Compute RF permutation importance and mutual information."""
    # Encode any string/list columns as ordinal
    feature_cols = []
    display_names = []
    for col in hparam_cols:
        if df[col].dtype == object:
            uniq = sorted(df[col].unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            enc_col = f"{col}_enc"
            df[enc_col] = df[col].map(mapping)
            feature_cols.append(enc_col)
        else:
            feature_cols.append(col)
        display_names.append(col)

    X = df[feature_cols].values.astype(float)
    y = df[METRIC].values.astype(float)

    n_samples = len(X)
    if n_samples < 6:
        logger.warning(
            f"Too few samples ({n_samples}) for importance analysis, skipping"
        )
        return pd.DataFrame(
            {
                "Hyperparameter": display_names,
                "RF Importance": [float("nan")] * len(display_names),
                "Permutation Importance": [float("nan")] * len(display_names),
                "Perm. Imp. Std": [float("nan")] * len(display_names),
                "Mutual Information": [float("nan")] * len(display_names),
            }
        )

    rf = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
    rf.fit(X, y)

    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
    n_neighbors = min(5, n_samples - 1)
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=n_neighbors)

    return (
        pd.DataFrame(
            {
                "Hyperparameter": display_names,
                "RF Importance": rf.feature_importances_,
                "Permutation Importance": perm.importances_mean,
                "Perm. Imp. Std": perm.importances_std,
                "Mutual Information": mi,
            }
        )
        .sort_values("Permutation Importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_correlations(
    df: pd.DataFrame,
    hparam_cols: list[str],
) -> pd.DataFrame:
    """Compute Spearman rank correlation of each hparam with success_rate."""
    rows = []
    for col in hparam_cols:
        if df[col].dtype == object:
            enc_col = f"{col}_enc"
            if enc_col not in df.columns:
                uniq = sorted(df[col].unique())
                df[enc_col] = df[col].map({v: i for i, v in enumerate(uniq)})
            vals = df[enc_col].values.astype(float)
        else:
            vals = df[col].values.astype(float)
        if len(set(vals)) < 2:
            rows.append(
                {
                    "Hyperparameter": col,
                    "Spearman rho": float("nan"),
                    "p-value": float("nan"),
                }
            )
        else:
            rho, pval = spearmanr(vals, df[METRIC].values)
            rows.append({"Hyperparameter": col, "Spearman rho": rho, "p-value": pval})

    return (
        pd.DataFrame(rows)
        .sort_values("Spearman rho", key=abs, ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_importance_and_correlation(
    importance_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar charts: permutation importance (left) and Spearman rho (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    imp_sorted = importance_df.sort_values("Permutation Importance")
    colors_imp = sns.color_palette("viridis", n_colors=len(imp_sorted))
    axes[0].barh(
        imp_sorted["Hyperparameter"],
        imp_sorted["Permutation Importance"],
        xerr=imp_sorted["Perm. Imp. Std"],
        color=colors_imp,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Permutation Importance")
    axes[0].set_title("Hyperparameter Importance\n(Random Forest Permutation)")

    corr_sorted = corr_df.sort_values("Spearman rho")
    colors_corr = [
        "#e74c3c" if r < 0 else "#2ecc71" for r in corr_sorted["Spearman rho"]
    ]
    axes[1].barh(
        corr_sorted["Hyperparameter"],
        corr_sorted["Spearman rho"],
        color=colors_corr,
        edgecolor="black",
        linewidth=0.5,
    )
    for i, (_, row) in enumerate(corr_sorted.iterrows()):
        if np.isnan(row["p-value"]):
            marker = ""
        elif row["p-value"] < 0.001:
            marker = "***"
        elif row["p-value"] < 0.01:
            marker = "**"
        elif row["p-value"] < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            x_pos = row["Spearman rho"]
            axes[1].text(
                x_pos + 0.01 * np.sign(x_pos),
                i,
                marker,
                va="center",
                fontsize=12,
                fontweight="bold",
            )
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Spearman Correlation (rho)")
    axes[1].set_title(f"Correlation with {METRIC}\n(* p<0.05, ** p<0.01, *** p<0.001)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_violins(
    df: pd.DataFrame,
    hparam_cols: list[str],
    output_path: Path,
) -> None:
    """Violin plots: metric distribution per hyperparameter value."""
    n_params = len(hparam_cols)
    ncols = min(n_params, 3)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(hparam_cols):
        ax = axes[i]
        plot_df = df[[col, METRIC]].copy()
        plot_df[col] = plot_df[col].astype(str)
        order = _numeric_sort(plot_df[col].unique())
        sns.violinplot(
            data=plot_df,
            x=col,
            y=METRIC,
            order=order,
            ax=ax,
            inner="box",
            cut=0,
        )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC if i % ncols == 0 else "")

    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"{METRIC} Distribution per Hyperparameter Value",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Grid suggestion
# ---------------------------------------------------------------------------


def suggest_next_grid(
    df: pd.DataFrame,
    importance_df: pd.DataFrame,
    original_param_grid: dict,
    sweep_params: dict,
    target_size: int = 288,
) -> dict:
    """Suggest a new sweep grid based on the analysis results.

    Strategy:
        High-importance params: expand range around best value.
        Low-importance params: fix to best value.
        Iteratively un-fix if grid is too small for target.

    Args:
        df: Per-run DataFrame with hparam columns and success_rate.
        importance_df: Output of compute_importance().
        original_param_grid: The original param_grid dict from config.yaml.
        sweep_params: Dict mapping display_name -> config dot-key.
        target_size: Target number of total grid combinations.

    Returns:
        New param_grid dict (config dot-key -> list of values).
    """
    hparam_cols = list(sweep_params.keys())
    inv_map = {v: k for k, v in sweep_params.items()}

    # Build original grid in display-name space
    orig_display = {}
    for config_key, values in original_param_grid.items():
        if config_key == SEED_KEY:
            continue
        dname = inv_map.get(config_key, config_key)
        if dname in hparam_cols:
            orig_display[dname] = [str(v) if isinstance(v, list) else v for v in values]

    # Best value per param
    best_values = {}
    for col in hparam_cols:
        if col in df.columns:
            grouped = df.groupby(col)[METRIC].mean()
            best_values[col] = (
                grouped.idxmax() if HIGHER_IS_BETTER else grouped.idxmin()
            )

    # Sort params by importance ascending (for un-fixing order)
    imp_sorted = importance_df.sort_values("Permutation Importance", ascending=True)
    param_order = list(imp_sorted["Hyperparameter"])
    median_imp = importance_df["Permutation Importance"].median()

    new_grid = {}
    decisions = []

    for param in hparam_cols:
        if param not in orig_display:
            continue
        imp_row = importance_df[importance_df["Hyperparameter"] == param]
        perm_imp = imp_row["Permutation Importance"].values[0]
        best_val = best_values.get(param)
        old_vals = orig_display[param]

        if perm_imp < median_imp:
            new_grid[param] = [best_val]
            decisions.append(
                f"  {param}: FIXED to {best_val} (low importance: {perm_imp:.4f})"
            )
        elif all(isinstance(v, str) and v.startswith("[") for v in old_vals):
            # Categorical (e.g. level_weights): keep top 2
            grouped = (
                df.groupby(param)[METRIC]
                .mean()
                .sort_values(ascending=not HIGHER_IS_BETTER)
            )
            new_grid[param] = list(grouped.index[:2])
            decisions.append(
                f"  {param}: KEEP top 2: {new_grid[param]} (importance: {perm_imp:.4f})"
            )
        else:
            # Numeric: expand around best
            sorted_vals = sorted(float(v) for v in old_vals)
            lo, hi = sorted_vals[0], sorted_vals[-1]
            step = hi - lo
            best_f = float(best_val)
            candidates = set(sorted_vals)
            if best_f == lo:
                candidates.add(max(lo - step * 0.5, max(1, step * 0.25)))
            elif best_f == hi:
                candidates.add(hi + step * 0.5)
            candidates.add((lo + hi) / 2)
            if all(float(v) == int(float(v)) for v in old_vals):
                candidates = sorted(set(int(c) for c in candidates))
            else:
                candidates = sorted(candidates)
            new_grid[param] = candidates
            decisions.append(
                f"  {param}: EXPAND to {candidates} (importance: {perm_imp:.4f}, best={best_val})"
            )

    new_grid["seed"] = original_param_grid.get(SEED_KEY, [1, 1000, 10000])

    def grid_size(g):
        s = 1
        for v in g.values():
            s *= len(v)
        return s

    # Phase 2: un-fix params if grid too small
    for param in param_order:
        if grid_size(new_grid) >= target_size * 0.8:
            break
        if param in new_grid and len(new_grid[param]) <= 1 and param in orig_display:
            new_grid[param] = orig_display[param]
            decisions.append(
                f"  {param}: UN-FIXED to {orig_display[param]} (grid too small)"
            )

    # Phase 3: densify high-importance params if still too small
    for param in reversed(param_order):
        if grid_size(new_grid) >= target_size * 0.8:
            break
        if param not in new_grid or param == "seed":
            continue
        vals = new_grid[param]
        if all(isinstance(v, str) and v.startswith("[") for v in vals):
            if len(vals) < len(orig_display.get(param, [])):
                new_grid[param] = orig_display[param]
                decisions.append(f"  {param}: RESTORED all values (grid expansion)")
        elif len(vals) >= 2:
            new_vals = list(vals)
            for k in range(len(vals) - 1):
                mid = (vals[k] + vals[k + 1]) / 2
                if all(isinstance(v, int) for v in vals):
                    mid = int(mid)
                new_vals.append(mid)
            new_grid[param] = sorted(set(new_vals))
            decisions.append(f"  {param}: DENSIFIED to {new_grid[param]}")

    total = grid_size(new_grid)

    # Print to terminal
    print("=" * 70)
    print("SUGGESTED NEXT SWEEP GRID")
    print("=" * 70)
    print(f"\nDecisions (median importance threshold = {median_imp:.4f}):\n")
    for d in decisions:
        print(d)
    print(f"\nTotal combinations: {total} (target: ~{target_size})")
    if total < target_size * 0.5:
        print(
            f"\nWarning: grid is small ({total} runs). Consider expanding fixed params."
        )
    elif total > target_size * 1.5:
        print(f"\nWarning: grid is large ({total} runs). Consider fixing more params.")

    print("\n" + "=" * 70)
    print("YAML param_grid (paste into train.yaml sweep.param_grid):\n")
    print("sweep:")
    print("  param_grid:")
    for param, vals in new_grid.items():
        yaml_key = sweep_params.get(param, f"meta.{param}")
        if all(isinstance(v, str) and v.startswith("[") for v in vals):
            print(f"    {yaml_key}:")
            for v in vals:
                print(f"      - {v}")
        else:
            clean = [
                (
                    int(v)
                    if isinstance(v, (float, np.integer)) and float(v) == int(float(v))
                    else v
                )
                for v in vals
            ]
            print(f"    {yaml_key}: {clean}")

    return new_grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training hyperparameter sweep results",
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        nargs="+",
        help="Path(s) to sweep folder(s). Multiple directories are combined.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSVs/figures (default: first sweep_dir)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=list(_METRIC_INFO.keys()),
        default="success_rate",
        help="Metric to rank models by (default: success_rate)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Hierarchy level for unroll_eval metrics (default: 1)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=288,
        help="Target number of runs for the suggested next grid (default: 288)",
    )
    parser.add_argument(
        "--avg-last-n",
        type=int,
        default=3,
        help="Number of latest eval checkpoints to average over (default: 3)",
    )
    args = parser.parse_args()

    global METRIC, HIGHER_IS_BETTER
    METRIC = args.metric
    HIGHER_IS_BETTER = _METRIC_INFO[METRIC]["higher_is_better"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sweep_dirs = [Path(d) for d in args.sweep_dir]
    for sd in sweep_dirs:
        if not sd.is_dir():
            parser.error(f"Sweep directory does not exist: {sd}")
    if len(sweep_dirs) > 1 and args.output_dir is None:
        parser.error(
            "--output-dir is required when combining multiple sweep directories"
        )
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover sweep params from config
    sweep_params, original_grid = discover_sweep_config(sweep_dirs)
    hparam_cols = list(sweep_params.keys())
    print(f"\nSweep directories: {[str(d) for d in sweep_dirs]}")
    print(f"Metric: {METRIC} (higher_is_better={HIGHER_IS_BETTER})")
    print(f"Discovered {len(sweep_params)} sweep params: {hparam_cols}")
    print(f"Averaging over last {args.avg_last_n} eval checkpoints\n")

    # 2. Collect data
    metric_info = _METRIC_INFO[METRIC]
    if METRIC == "success_rate":
        # plan_eval/ flow: one row per (model_folder, eval_tag)
        df_all = collect_disk_results(
            sweep_dirs, sweep_params, avg_last_n=args.avg_last_n
        )
        if df_all.empty:
            print("No completed runs found. Exiting.")
            return

        print(f"Collected {len(df_all)} (model, eval_tag) results\n")

        # 3. Average over seeds per (training_config, eval_tag), then pick
        #    the best eval_tag per training config.
        group_cols = hparam_cols + ["eval_tag"]
        per_tag = (
            df_all.groupby(group_cols)
            .agg(
                mean_sr=(METRIC, "mean"),
                mean_time=("avg_episode_time", "mean"),
                n_seeds=(METRIC, "count"),
            )
            .reset_index()
        )
        best_idx = per_tag.groupby(hparam_cols)["mean_sr"].idxmax()
        best_tags = per_tag.loc[best_idx].set_index(hparam_cols)["eval_tag"]

        rows = []
        for _, row in df_all.iterrows():
            key = tuple(row[c] for c in hparam_cols)
            if row["eval_tag"] == best_tags.loc[key]:
                rows.append(row)
        df = pd.DataFrame(rows)
        df = df.rename(columns={"eval_tag": "best_eval_tag"})

        print(f"Selected best eval_tag per training config ({len(df)} runs)\n")

        # 4. Aggregate over seeds
        agg = _aggregate_over_seeds(
            df,
            hparam_cols,
            extra_agg={
                "mean_episode_time": ("avg_episode_time", "mean"),
                "std_episode_time": ("avg_episode_time", "std"),
                "best_eval_tag": ("best_eval_tag", lambda x: x.mode().iloc[0]),
            },
        )
    else:
        # wandb summary flow (image_jepa) or unroll_eval/ flow
        if "summary_pattern" in metric_info:
            df = collect_wandb_summary_results(
                sweep_dirs,
                sweep_params,
                summary_pattern=metric_info["summary_pattern"],
            )
            source_label = "wandb summary"
        else:
            csv_base = metric_info["csv_base"]
            df = collect_unroll_results(
                sweep_dirs,
                sweep_params,
                csv_base=csv_base,
                level=args.level,
                avg_last_n=args.avg_last_n,
            )
            source_label = "unroll_eval"

        if df.empty:
            print("No completed runs found. Exiting.")
            return

        print(f"Collected {len(df)} {source_label} results\n")

        # 4. Aggregate over seeds
        agg = _aggregate_over_seeds(df, hparam_cols)

    print(f"Top {args.top_n} configurations by mean {METRIC}:")
    print(agg.head(args.top_n).to_string(index=False))
    print()

    # 5. Importance
    importance_df = compute_importance(df, hparam_cols)
    print("Hyperparameter importance (sorted by Permutation Importance):")
    print(importance_df.to_string(index=False))
    print()

    # 6. Correlations
    corr_df = compute_correlations(df, hparam_cols)
    print(f"Spearman rank correlation with {METRIC}:")
    print(corr_df.to_string(index=False))
    print()

    # 7. Save CSVs
    agg.to_csv(output_dir / "top_configurations.csv", index=False)
    importance_df.to_csv(output_dir / "hparam_importance.csv", index=False)
    corr_df.to_csv(output_dir / "hparam_correlation.csv", index=False)
    df.to_csv(output_dir / "all_runs.csv", index=False)
    print(f"Saved CSVs to {output_dir}/")

    # 8. Save figures (PDF)
    plot_importance_and_correlation(
        importance_df,
        corr_df,
        output_dir / "hparam_importance.pdf",
    )
    plot_violins(df, hparam_cols, output_dir / "hparam_violins.pdf")
    print(f"Saved PDFs to {output_dir}/\n")

    # 9. Suggest next grid
    if original_grid:
        suggest_next_grid(
            df,
            importance_df,
            original_grid,
            sweep_params,
            target_size=args.target_size,
        )
    else:
        print("Could not read original param_grid; skipping grid suggestion.")


if __name__ == "__main__":
    main()
