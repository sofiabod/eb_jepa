# Scripts

Standalone analysis and visualization tools for EB-JEPA experiments.

## Available Scripts

### `analyze_training_sweep.py`

Analyze results from a training hyperparameter sweep. Reads eval metrics from disk, computes hyperparameter importance and correlations, generates figures (PDF) and metrics (CSV), and suggests a new sweep grid.

```bash
# Analyze a sweep directory
python -m scripts.analyze_training_sweep /path/to/sweep_dir

# Show top 20 configs
python -m scripts.analyze_training_sweep /path/to/sweep_dir --top-n 20

# Analyze unroll eval metrics at a specific hierarchy level
python -m scripts.analyze_training_sweep /path/to/sweep_dir --metric mean_pos_mse --level 1

# Compare two training runs
python -m scripts.analyze_training_sweep /path/to/sweep_A /path/to/sweep_B --output-dir /tmp/comparison
```

### `analyze_planning_sweep.py`

Analyze planning evaluation sweeps. Compares planning configs across checkpoints, generates per-config and aggregated metrics.

```bash
# Analyze a planning sweep
python -m scripts.analyze_planning_sweep \
  --model-folder /path/to/best_model \
  --manifest /path/to/planning_sweep/sweep_manifest.csv

# Compare across multiple planning sweeps
python -m scripts.analyze_planning_sweep \
  --model-folder /path/to/best_model \
  --manifest sweep_A/sweep_manifest.csv sweep_B/sweep_manifest.csv \
  --output-dir /path/to/combined_analysis
```

### `visualize_planning_cost_heatmap.py`

CLI wrapper for planning cost heatmap visualization. Core logic lives in
`eb_jepa.vis.heatmaps`.

```bash
python -m scripts.visualize_planning_cost_heatmap --model-folder /path/to/trained_model
```

## General Usage

All scripts support `--help` for full argument documentation:

```bash
python -m scripts.<script_name> --help
```
