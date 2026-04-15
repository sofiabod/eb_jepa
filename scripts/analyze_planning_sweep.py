"""Analyze results from a planning hyperparameter sweep.

Reads eval.csv files from disk (produced by ``launch_planning_sweep.py``),
computes hyperparameter importance and correlations with success_rate,
generates figures (PDF) and metrics (CSV).

Supports multiple manifests with different column schemas (e.g. one sweep
over plan lengths, another over MPPI optimizer params). Columns are
auto-discovered from the union of all manifests.

USAGE:
------
# Analyze sweep results for a single model folder:
python -m scripts.analyze_planning_sweep \
  --model-folder /checkpoint/.../best_model_seed1 \
  --manifest /checkpoint/.../planning_sweep_YYYYMMDD_HHMM/sweep_manifest.csv

# Analyze across multiple seeds:
python -m scripts.analyze_planning_sweep \
  --model-folder /checkpoint/.../best_model_base \
  --seeds 1 1000 10000 \
  --manifest /checkpoint/.../planning_sweep_YYYYMMDD_HHMM/sweep_manifest.csv

# Combine multiple sweeps (requires --output-dir):
python -m scripts.analyze_planning_sweep \
  --model-folder /checkpoint/.../best_model_base \
  --seeds 1 1000 10000 \
  --manifest \
    /checkpoint/.../planning_sweep_20260319_1731/sweep_manifest.csv \
    /checkpoint/.../planning_sweep_20260319_2249/sweep_manifest.csv \
  --output-dir /checkpoint/.../combined_analysis
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_theme(font_scale=1.1)
logger = logging.getLogger(__name__)

METRIC = "success_rate"
HPARAM_COLS = [
    "start_level",
    "num_act_stepped",
    "l1_plan_length",
    "l2_plan_length",
    "l3_plan_length",
]
LEVEL_DEPENDENT_COLS = {
    1: ["l1_plan_length", "l1_momentum_std"],
    2: ["l1_plan_length", "l2_plan_length", "l1_momentum_std", "l2_momentum_std"],
    3: [
        "l1_plan_length",
        "l2_plan_length",
        "l3_plan_length",
        "l1_momentum_std",
        "l2_momentum_std",
        "l3_momentum_std",
    ],
}
ALL_LEVEL_DEPENDENT = {c for cols in LEVEL_DEPENDENT_COLS.values() for c in cols}
NON_HPARAM_COLS = {"eval_tag", "sweep_source"}


def _encode_col(df: pd.DataFrame, col: str) -> str:
    """Encode a categorical column as ordinal integers, return column name to use."""
    if df[col].dtype == object:
        enc_col = f"{col}_enc"
        if enc_col not in df.columns:
            mapping = {v: i for i, v in enumerate(sorted(df[col].unique()))}
            df[enc_col] = df[col].map(mapping)
        return enc_col
    return col


def _numeric_sort(vals: list[str]) -> list[str]:
    """Sort string values numerically if possible, else lexicographically."""
    try:
        return sorted(vals, key=float)
    except ValueError:
        return sorted(vals)


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def collect_results(model_folders: list[Path], manifest: pd.DataFrame) -> pd.DataFrame:
    """Scan eval.csv files on disk and join with manifest hyperparameters."""
    rows = []
    for model_folder in model_folders:
        seed = (
            model_folder.name.rsplit("_seed", 1)[-1]
            if "_seed" in model_folder.name
            else "unknown"
        )
        for _, mrow in manifest.iterrows():
            eval_base = model_folder / "plan_eval" / mrow["eval_tag"]
            if not eval_base.is_dir():
                continue
            csvs = sorted(eval_base.glob("step-*/eval.csv")) + sorted(
                eval_base.glob("step-*_eval_only/eval.csv")
            )
            if not csvs:
                continue
            try:
                edf = pd.read_csv(csvs[-1])
            except Exception as e:
                logger.warning(f"Failed to read {csvs[-1]}: {e}")
                continue
            if METRIC not in edf.columns:
                continue
            row = {
                "model_folder": str(model_folder),
                "seed": seed,
                "eval_tag": mrow["eval_tag"],
                METRIC: edf[METRIC].iloc[-1],
            }
            for col in ["mean_state_dist", "avg_episode_time"]:
                if col in edf.columns:
                    row[col] = edf[col].iloc[-1]
            for col in mrow.index:
                if col != "eval_tag" and col not in row:
                    row[col] = mrow[col]
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df)} eval results")
    return df


def compute_importance(df: pd.DataFrame, hparam_cols: list[str]) -> pd.DataFrame:
    """Compute RF permutation importance and mutual information."""
    feature_cols = [_encode_col(df, c) for c in hparam_cols]
    X = df[feature_cols].fillna(-1).values.astype(float)
    y = df[METRIC].values.astype(float)

    rf = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    return (
        pd.DataFrame(
            {
                "Hyperparameter": hparam_cols,
                "RF Importance": rf.feature_importances_,
                "Permutation Importance": perm.importances_mean,
                "Perm. Imp. Std": perm.importances_std,
                "Mutual Information": mi,
            }
        )
        .sort_values("Permutation Importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_correlations(df: pd.DataFrame, hparam_cols: list[str]) -> pd.DataFrame:
    """Compute Spearman rank correlation of each hparam with success_rate."""
    rows = []
    for col in hparam_cols:
        vals = df[_encode_col(df, col)].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2 or len(set(vals[valid])) < 2:
            rho, pval = float("nan"), float("nan")
        else:
            rho, pval = spearmanr(vals[valid], df[METRIC].values[valid])
        rows.append({"Hyperparameter": col, "Spearman rho": rho, "p-value": pval})

    return (
        pd.DataFrame(rows)
        .sort_values("Spearman rho", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def plot_importance_and_correlation(
    importance_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar charts: permutation importance (left) and Spearman rho (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    imp = importance_df.sort_values("Permutation Importance")
    axes[0].barh(
        imp["Hyperparameter"],
        imp["Permutation Importance"],
        xerr=imp["Perm. Imp. Std"],
        color=sns.color_palette("viridis", n_colors=len(imp)),
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Permutation Importance")
    axes[0].set_title("Hyperparameter Importance\n(Random Forest Permutation)")

    corr = corr_df.sort_values("Spearman rho")
    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in corr["Spearman rho"]]
    axes[1].barh(
        corr["Hyperparameter"],
        corr["Spearman rho"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for i, (_, row) in enumerate(corr.iterrows()):
        p = row["p-value"]
        marker = (
            ""
            if np.isnan(p) or p >= 0.05
            else ("***" if p < 0.001 else "**" if p < 0.01 else "*")
        )
        if marker:
            axes[1].text(
                row["Spearman rho"] + 0.01 * np.sign(row["Spearman rho"]),
                i,
                marker,
                va="center",
                fontsize=12,
                fontweight="bold",
            )
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Spearman Correlation (rho)")
    axes[1].set_title(
        "Correlation with success_rate\n(* p<0.05, ** p<0.01, *** p<0.001)"
    )

    plt.tight_layout()
    _savefig(fig, output_path)


def plot_violins(df: pd.DataFrame, hparam_cols: list[str], output_path: Path) -> None:
    """Violin plots: success_rate distribution per hyperparameter value."""
    n = len(hparam_cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(hparam_cols):
        pdf = df[[col, METRIC]].dropna(subset=[col]).copy()
        pdf[col] = pdf[col].astype(str)
        sns.violinplot(
            data=pdf,
            x=col,
            y=METRIC,
            order=_numeric_sort(pdf[col].unique()),
            ax=axes[i],
            inner="box",
            cut=0,
        )
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("success_rate" if i % ncols == 0 else "")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Success Rate Distribution per Planning Hyperparameter",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, output_path)


def _pareto_front(costs: np.ndarray, benefits: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimize costs, maximize benefits).

    Uses O(n log n) sort-and-sweep: sort by descending benefit, then sweep
    keeping only points where cost strictly decreases.

    Args:
        costs: 1D array of values to minimize (e.g. avg_episode_time).
        benefits: 1D array of values to maximize (e.g. success_rate).

    Returns:
        Boolean mask of length n, True for Pareto-optimal points.
    """
    n = len(costs)
    mask = np.zeros(n, dtype=bool)
    order = np.argsort(-benefits)
    min_cost = np.inf
    for idx in order:
        if costs[idx] < min_cost:
            mask[idx] = True
            min_cost = costs[idx]
    return mask


def plot_pareto(
    agg: pd.DataFrame,
    active_hparams: list[str],
    output_path: Path,
) -> None:
    """Scatter plot of success_rate vs avg_episode_time with Pareto frontier.

    Args:
        agg: Seed-aggregated DataFrame with mean_success_rate and
            mean_avg_episode_time columns.
        active_hparams: List of active hyperparameter column names (used for
            labelling Pareto-optimal points).
        output_path: Where to save the PDF figure.
    """
    if (
        "mean_avg_episode_time" not in agg.columns
        or agg["mean_avg_episode_time"].isna().all()
    ):
        logger.warning("mean_avg_episode_time missing or all NaN; skipping Pareto plot")
        return

    df = agg.dropna(subset=["mean_avg_episode_time", "mean_success_rate"]).copy()
    if len(df) < 2:
        logger.warning("Fewer than 2 valid points for Pareto plot; skipping")
        return

    costs = df["mean_avg_episode_time"].values
    benefits = df["mean_success_rate"].values
    pareto_mask = _pareto_front(costs, benefits)

    with sns.axes_style():
        fig, ax = plt.subplots(figsize=(10, 7))

        sns.scatterplot(
            x=costs,
            y=benefits,
            s=60,
            alpha=0.5,
            color=sns.color_palette("muted")[0],
            edgecolor="white",
            linewidth=0.4,
            label="All configurations",
            ax=ax,
        )

        pareto_costs = costs[pareto_mask]
        pareto_benefits = benefits[pareto_mask]
        sort_idx = np.argsort(pareto_costs)
        ax.step(
            pareto_costs[sort_idx],
            pareto_benefits[sort_idx],
            where="post",
            color=sns.color_palette("bright")[3],
            linewidth=2,
            label="Pareto front",
        )
        ax.scatter(
            pareto_costs,
            pareto_benefits,
            marker="*",
            s=250,
            color=sns.color_palette("bright")[3],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        ax.set_xlabel("Avg Episode Time (s)")
        ax.set_ylabel("Success Rate")
        ax.set_title(
            "Pareto Front: Success Rate vs Avg Episode Time", fontweight="bold"
        )
        ax.legend(frameon=True, fancybox=True, shadow=True)
        sns.despine(ax=ax, left=True, bottom=True)
    _savefig(fig, output_path)


def _level_aware_filter(df: pd.DataFrame, active_hparams: list[str]) -> pd.DataFrame:
    """Drop rows with NaN only in hparam columns relevant to that row's start_level."""
    non_plan = [c for c in active_hparams if c not in ALL_LEVEL_DEPENDENT]
    keep = pd.Series(True, index=df.index)
    if non_plan:
        keep &= df[non_plan].notna().all(axis=1)
    if "start_level" in df.columns:
        for level in [1, 2, 3]:
            req = [c for c in LEVEL_DEPENDENT_COLS[level] if c in active_hparams]
            if req:
                mask = df["start_level"] == level
                keep.loc[mask] &= df.loc[mask, req].notna().all(axis=1)
    return df.loc[keep].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze planning hyperparameter sweep results"
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        required=True,
        help="Base path to trained model folder(s). Seeds appended if --seeds given.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to sweep_manifest.csv. Multiple manifests are "
        "concatenated (requires --output-dir).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to aggregate over (e.g. 1 1000 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: manifest parent dir)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top configurations to display"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    manifest_paths = [Path(m) for m in args.manifest]
    for mp in manifest_paths:
        if not mp.is_file():
            parser.error(f"Manifest file not found: {mp}")
    if len(manifest_paths) > 1 and args.output_dir is None:
        parser.error("--output-dir is required when combining multiple manifests")

    dfs = []
    for mp in manifest_paths:
        mdf = pd.read_csv(mp)
        mdf["sweep_source"] = mp.name
        dfs.append(mdf)
        print(f"Manifest: {mp} ({len(mdf)} configurations)")
    manifest = pd.concat(dfs, ignore_index=True)
    if len(manifest_paths) > 1:
        print(
            f"Combined: {len(manifest)} configurations from {len(manifest_paths)} manifests"
        )

    base = args.model_folder
    if args.seeds:
        model_folders = (
            [Path(base)]
            if any(f"_seed{s}" in base for s in args.seeds)
            else [Path(f"{base}_seed{s}") for s in args.seeds]
        )
    else:
        model_folders = [Path(base)]

    existing = [mf for mf in model_folders if mf.is_dir()]
    if not existing:
        parser.error(f"No model folders found: {model_folders}")
    print(f"Model folders: {len(existing)} of {len(model_folders)} exist")

    output_dir = Path(args.output_dir) if args.output_dir else manifest_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(existing, manifest)
    if df.empty:
        print("No eval results found. Are the jobs finished?")
        return
    print(f"\nCollected {len(df)} eval results")

    all_hparam_cols = [c for c in manifest.columns if c not in NON_HPARAM_COLS]
    active_hparams = [
        c for c in all_hparam_cols if c in df.columns and df[c].dropna().nunique() > 1
    ]
    print(f"Active planning hyperparameters: {active_hparams}\n")
    if not active_hparams:
        print("No varying hyperparameters found. Nothing to analyze.")
        return

    df_analysis = _level_aware_filter(df, active_hparams)

    # Aggregate over seeds
    agg_metrics = [METRIC] + (
        [c for c in ["avg_episode_time"] if c in df_analysis.columns]
    )
    for col in agg_metrics:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors="coerce")
    agg = (
        df_analysis.groupby(active_hparams, dropna=False)[agg_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        f"{stat}_{col}" if stat in ("mean", "std", "count") else col
        for col, stat in agg.columns
    ]
    count_cols = [c for c in agg.columns if c.startswith("count_")]
    if count_cols:
        agg.rename(columns={count_cols[0]: "count"}, inplace=True)
        agg.drop(columns=count_cols[1:], inplace=True)

    sort_cols, sort_asc = ["mean_success_rate"], [False]
    if "mean_avg_episode_time" in agg.columns:
        sort_cols.append("mean_avg_episode_time")
        sort_asc.append(True)
    agg = agg.sort_values(sort_cols, ascending=sort_asc)

    # Reorder columns: metrics first, then hparams
    metric_cols = ["mean_success_rate", "std_success_rate"]
    if "mean_avg_episode_time" in agg.columns:
        metric_cols += ["mean_avg_episode_time", "std_avg_episode_time"]
    metric_cols.append("count")
    remaining = [c for c in agg.columns if c not in metric_cols]
    agg = agg[metric_cols + remaining]

    print(f"Unique configurations: {len(agg)}\n")
    print(f"Top {args.top_n} configurations by mean {METRIC}:")
    print(agg.head(args.top_n).to_string(index=False))
    print()

    importance_df = compute_importance(df_analysis, active_hparams)
    print("Hyperparameter importance (sorted by Permutation Importance):")
    print(importance_df.to_string(index=False))
    print()

    corr_df = compute_correlations(df_analysis, active_hparams)
    print("Spearman rank correlation with success_rate:")
    print(corr_df.to_string(index=False))
    print()

    agg.to_csv(output_dir / "top_configurations.csv", index=False)
    importance_df.to_csv(output_dir / "hparam_importance.csv", index=False)
    corr_df.to_csv(output_dir / "hparam_correlation.csv", index=False)
    df.to_csv(output_dir / "all_runs.csv", index=False)
    print(f"Saved CSVs to {output_dir}/")

    plot_importance_and_correlation(
        importance_df, corr_df, output_dir / "hparam_importance.pdf"
    )
    plot_violins(df_analysis, active_hparams, output_dir / "hparam_violins.pdf")
    if "mean_avg_episode_time" in agg.columns:
        plot_pareto(agg, active_hparams, output_dir / "pareto_front.pdf")
    print(f"Saved PDFs to {output_dir}/\n")


if __name__ == "__main__":
    main()
