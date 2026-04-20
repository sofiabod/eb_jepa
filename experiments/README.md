# experiments

standalone scripts for reproducing prior work and validating baselines.
these live on the `experiments` branch, separate from the main temporal graph jepa build.

## reproduce_graph_jepa.py

reproduces the graph-jepa (skenderi 2025) results on mutag and proteins using modal gpu.

```
modal run experiments/reproduce_graph_jepa.py               # both datasets
modal run experiments/reproduce_graph_jepa.py --dataset MUTAG
modal run experiments/reproduce_graph_jepa.py --dataset PROTEINS
```

what it does:
- 5 runs x 10-fold cv (same seeds and splits as the paper)
- streams all training output (per-epoch loss, fold accuracy) to your terminal
- after training, fits a logistic regression linear probe on the frozen embeddings
- compares final accuracy to paper_logs/ numbers and writes results to results/graph-jepa-repro/

paper targets:
- mutag:    75.7% +/- 3.8%
- proteins: 76.2% +/- 3.8%

results land in `results/graph-jepa-repro/` as json files.
