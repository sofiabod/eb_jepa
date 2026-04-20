"""
reproduce graph-jepa (skenderi 2025) results on mutag and proteins.

all training output streams to your terminal in real time.
results are written locally to results/graph-jepa-repro/ when each dataset finishes.

usage:
    cd /Users/sonia/Documents/GitHub/eb_jepa
    modal run experiments/reproduce_graph_jepa.py
    modal run experiments/reproduce_graph_jepa.py --dataset MUTAG
    modal run experiments/reproduce_graph_jepa.py --dataset PROTEINS
"""

import modal
import json
from pathlib import Path
from datetime import datetime

GRAPH_JEPA_SRC = Path(__file__).parent.parent / "graph-jepa"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "graph-jepa-repro"

# numbers we are trying to match from paper_logs/
PAPER_RESULTS = {
    "MUTAG":    {"mean": 75.7, "std": 3.8},
    "PROTEINS": {"mean": 76.2, "std": 3.8},
}

# same seeds used in the original run_k_fold
SEEDS = [42, 21, 95, 12, 35]

app = modal.App("reproduce-graph-jepa")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands("apt-get update -qq && apt-get install -y -q metis libmetis-dev")
    .pip_install("torch==2.1.0", extra_index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("torch-geometric")
    .run_commands(
        "pip install torch-scatter torch-sparse torch-cluster "
        "-f https://data.pyg.org/whl/torch-2.1.0+cu118.html"
    )
    .pip_install("yacs", "networkx", "einops", "metis", "scikit-learn", "tensorboard", "numpy")
)

# mount the graph-jepa source directory read-only into the container
graph_jepa_mount = modal.Mount.from_local_dir(
    str(GRAPH_JEPA_SRC),
    remote_path="/root/graph-jepa",
)


@app.function(
    gpu="T4",
    image=image,
    mounts=[graph_jepa_mount],
    timeout=7200,
)
def train_dataset(dataset_name: str) -> dict:
    """
    runs 5 seeds x 10-fold cv for one dataset.
    prints every epoch so you can watch progress in real time.
    returns a dict of results that gets written locally by the entrypoint.
    """
    import os, sys, random, time
    import torch
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from torch_geometric.loader import DataLoader

    # must cd before importing graph-jepa modules (relative paths in configs)
    os.chdir("/root/graph-jepa")
    sys.path.insert(0, "/root/graph-jepa")

    from core.config import cfg
    from core.get_data import create_dataset
    from core.get_model import create_model

    # load yaml config then override a few things for the modal environment
    cfg.merge_from_file(f"train/configs/{dataset_name.lower()}.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    cfg.num_workers = 0  # no multiprocessing inside modal

    print(f"\n{'#'*60}")
    print(f"graph-jepa reproduction: {dataset_name}")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"\nconfig")
    print(f"  epochs:      {cfg.train.epochs}")
    print(f"  runs:        {cfg.train.runs}  (seeds={SEEDS})")
    print(f"  lr:          {cfg.train.lr}")
    print(f"  hidden_size: {cfg.model.hidden_size}")
    print(f"  gnn_layers:  {cfg.model.nlayer_gnn}")
    print(f"  mlpmixer:    {cfg.model.nlayer_mlpmixer}")
    print(f"  patches:     {cfg.metis.n_patches}")
    print(f"  context:     {cfg.jepa.num_context}")
    print(f"  targets:     {cfg.jepa.num_targets}")
    print(f"{'#'*60}")

    # --- dataset ---
    print("\nloading dataset (downloads automatically from tu dortmund if not cached)...")
    dataset, transform, transform_eval = create_dataset(cfg)

    n_graphs = len(dataset)
    avg_nodes = np.mean([d.num_nodes for d in dataset])
    avg_edges = np.mean([d.num_edges for d in dataset])
    n_classes = int(dataset.data.y.max().item()) + 1

    print(f"\ndataset stats: {dataset_name}")
    print(f"  graphs:         {n_graphs}")
    print(f"  avg nodes:      {avg_nodes:.1f}")
    print(f"  avg edges:      {avg_edges:.1f}")
    print(f"  classes:        {n_classes}")
    print(f"  node features:  {dataset.num_node_features}")

    # show model size using a throw-away instance
    _sample = create_model(cfg)
    n_params = sum(p.numel() for p in _sample.parameters() if p.requires_grad)
    print(f"  model params:   {n_params:,}")
    del _sample

    # --- 10-fold splits (fixed seed so folds are identical across runs) ---
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    ys = dataset.data.y
    train_indices, test_indices = [], []
    for tr, te in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(tr).long())
        test_indices.append(torch.from_numpy(te).long())
    print(f"\n10-fold split: {len(train_indices[0])} train / {len(test_indices[0])} test per fold")

    # --- helpers ---

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_epoch(loader, model, optimizer, momentum):
        """
        one training epoch.
        the context encoder gets gradients; the target encoder does not.
        instead, target encoder weights = ema of context encoder weights.
        momentum increases from 0.996 to 1.0 over the full training run —
        early on the target moves a bit, late in training it barely moves.
        """
        model.train()
        criterion = torch.nn.SmoothL1Loss()
        losses, weights = [], []
        for data in loader:
            # sign flip augmentation for laplacian positional encodings
            if model.use_lap:
                flip = torch.rand(data.lap_pos_enc.size(1))
                flip[flip >= 0.5] = 1.0
                flip[flip < 0.5] = -1.0
                data.lap_pos_enc = data.lap_pos_enc * flip.unsqueeze(0)
            data = data.to(device)
            optimizer.zero_grad()
            pred, target = model(data)
            loss = criterion(pred, target)
            losses.append(loss.item())
            weights.append(len(target))
            loss.backward()
            optimizer.step()
            # ema: target encoder slowly tracks context encoder
            with torch.no_grad():
                for p_ctx, p_tgt in zip(
                    model.context_encoder.parameters(),
                    model.target_encoder.parameters(),
                ):
                    p_tgt.data.mul_(momentum).add_((1.0 - momentum) * p_ctx.detach().data)
        return float(np.average(losses, weights=weights))

    @torch.no_grad()
    def eval_epoch(loader, model):
        """jepa loss on a held-out set. no gradients."""
        model.eval()
        criterion = torch.nn.SmoothL1Loss()
        losses, weights = [], []
        for data in loader:
            data = data.to(device)
            pred, target = model(data)
            losses.append(criterion(pred, target).item())
            weights.append(len(target))
        return float(np.average(losses, weights=weights))

    @torch.no_grad()
    def extract_embeddings(loader, model):
        """
        after jepa training, freeze the encoder and extract graph embeddings.
        these are the learned representations we will classify with logistic regression.
        this is the 'linear probe' evaluation: if the embeddings are good,
        a simple linear classifier should work well.
        """
        model.eval()
        X, y = [], []
        for data in loader:
            data = data.to(device)
            X.append(model.encode(data).cpu().numpy())
            y.append(data.y.cpu().numpy())
        return np.concatenate(X), np.concatenate(y)

    # --- main training loop ---

    run_results = []
    total_start = time.time()

    for run_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"run {run_idx + 1}/{len(SEEDS)}  seed={seed}")
        print(f"{'='*60}")
        set_seed(seed)
        fold_accs = []

        for fold, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
            fold_start = time.time()
            print(f"\n  fold {fold + 1}/10")
            print(f"  train graphs: {len(train_idx)}  test graphs: {len(test_idx)}")

            train_ds = dataset[train_idx]
            test_ds = dataset[test_idx]
            train_ds.transform = transform
            test_ds.transform = transform_eval
            test_ds = list(test_ds)

            train_loader = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=0)

            model = create_model(cfg).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=cfg.train.lr_decay,
                patience=cfg.train.lr_patience,
            )

            # ema momentum ramps linearly from 0.996 to 1.0 over all steps
            ipe = len(train_loader)
            total_steps = ipe * cfg.train.epochs
            momentum_schedule = (
                0.996 + i * (1.0 - 0.996) / total_steps
                for i in range(total_steps + 1)
            )

            for epoch in range(cfg.train.epochs):
                t0 = time.time()
                tr_loss = train_epoch(train_loader, model, optimizer, next(momentum_schedule))
                te_loss = eval_epoch(test_loader, model)
                scheduler.step(te_loss)
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]

                print(
                    f"  epoch {epoch:03d}/{cfg.train.epochs}  "
                    f"train={tr_loss:.4f}  test={te_loss:.4f}  "
                    f"{elapsed:.2f}s  lr={lr:.2e}"
                )

                if lr < cfg.train.min_lr:
                    print(f"  lr reached minimum, stopping at epoch {epoch}")
                    break

            # linear probe evaluation
            print(f"\n  extracting embeddings + fitting logistic regression...")
            X_train, y_train = extract_embeddings(train_loader, model)
            X_test, y_test = extract_embeddings(test_loader, model)
            print(f"  embedding shapes: train={X_train.shape}  test={X_test.shape}")

            probe = LogisticRegression(max_iter=10000)
            probe.fit(X_train, y_train)
            acc = accuracy_score(y_test, probe.predict(X_test))
            fold_accs.append(acc)

            fold_elapsed = time.time() - fold_start
            print(f"  fold {fold + 1} accuracy: {acc * 100:.1f}%  ({fold_elapsed:.0f}s total)")

        run_mean = float(np.mean(fold_accs)) * 100
        run_std = float(np.std(fold_accs)) * 100
        fold_str = "  ".join(f"{a * 100:.1f}" for a in fold_accs)
        print(f"\nrun {run_idx + 1} result: {run_mean:.1f}% +/- {run_std:.1f}%")
        print(f"  folds: {fold_str}")
        run_results.append({"mean": run_mean, "std": run_std, "fold_accs": [a * 100 for a in fold_accs]})

    # aggregate across all 5 runs
    means = [r["mean"] for r in run_results]
    stds = [r["std"] for r in run_results]
    final_mean = float(np.mean(means))
    final_std = float(np.mean(stds))
    paper = PAPER_RESULTS[dataset_name]
    diff = final_mean - paper["mean"]
    within = abs(diff) <= paper["std"]
    total_elapsed = (time.time() - total_start) / 3600

    print(f"\n{'#'*60}")
    print(f"final results: {dataset_name}")
    print(f"  reproduced:  {final_mean:.1f}% +/- {final_std:.1f}%")
    print(f"  paper:       {paper['mean']:.1f}% +/- {paper['std']:.1f}%")
    print(f"  diff:        {diff:+.1f}%  ({'within variance' if within else 'outside variance'})")
    print(f"  total time:  {total_elapsed:.2f}h")
    print(f"{'#'*60}")

    return {
        "dataset": dataset_name,
        "run_results": run_results,
        "final_mean": final_mean,
        "final_std": final_std,
        "paper_mean": paper["mean"],
        "paper_std": paper["std"],
        "within_variance": within,
        "total_hours": total_elapsed,
    }


@app.local_entrypoint()
def main(dataset: str = "both"):
    """
    default runs both datasets sequentially.
    results are written to results/graph-jepa-repro/ on your local machine.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["MUTAG", "PROTEINS"] if dataset == "both" else [dataset.upper()]
    all_results = {}

    for ds in datasets:
        print(f"\nsubmitting {ds} to modal...")
        result = train_dataset.remote(ds)
        all_results[ds] = result

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{ds.lower()}_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"result written to {out_path}")

    print(f"\n{'='*60}")
    print("reproduction summary")
    print(f"{'='*60}")
    for ds, r in all_results.items():
        status = "PASS" if r["within_variance"] else "OUTSIDE VARIANCE"
        print(
            f"  {ds:10s}  reproduced={r['final_mean']:.1f}% +/- {r['final_std']:.1f}%"
            f"  paper={r['paper_mean']:.1f}% +/- {r['paper_std']:.1f}%  [{status}]"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"summary_{ts}.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nfull summary written to {summary_path}")
