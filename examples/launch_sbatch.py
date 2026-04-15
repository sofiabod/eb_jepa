import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

"""
Unified SLURM launcher for all EB-JEPA examples.

Provides seed averaging, sweep name filtering, and wandb sweep features for all examples.

USAGE:
------
# Launch 3 seeds of a single configuration (default sweep name: sweep_YYYYMMDD_HHMM):
python -m examples.launch_sbatch --example ac_video_jepa

# Launch 3 seeds with custom sweep name:
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment

# Launch full hyperparameter sweep (ac_video_jepa only):
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment --full-sweep

# With wandb sweep UI for hyperparameter analysis:
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment --use-wandb-sweep

# Override config values:
python -m examples.launch_sbatch --example ac_video_jepa --optim.lr 0.0005

SEED AVERAGING IN WANDB UI:
---------------------------
Runs with the same hyperparameters but different seeds share the same wandb run name.

To view averaged metrics:
1. Go to wandb web UI -> Runs table
2. Click "Group by" -> select "Name"
   -> This groups runs with identical hyperparameters (different seeds) together

To filter runs from a specific sweep:
3. Click "Filter" -> "Group" -> select your sweep name (e.g., 'my_experiment')
   -> This shows only runs from that sweep, grouped by name (see above)

WANDB SWEEP ANALYSIS UI (requires --use-wandb-sweep):
-----------------------------------------------------
When using --use-wandb-sweep, wandb creates a sweep object that enables advanced
hyperparameter analysis.

To access the sweep analysis:
1. Go to wandb web UI -> left pane -> click "Sweeps"
2. Click on your sweep name
3. Wandb automatically generates plots linking hyperparameters to the metric
   (success_rate), including:
   - Parallel coordinates plot
   - Hyperparameter importance
   - Parameter vs. metric scatter plots
"""

import argparse
import importlib
import json
import shutil
from itertools import product
from pathlib import Path

import submitit
from omegaconf import OmegaConf

from eb_jepa.utils.config import (
    get_checkpoints_dir,
    get_dataset_name,
    get_default_run_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
)

# Cluster-agnostic defaults; override via local/slurm.yaml (see examples/slurm.yaml.example)
_SLURM_DEFAULTS = {
    "mem_per_gpu": "80G",
    "cpus_per_task": 16,
    "timeout_min": 24 * 60,
    "partition": None,
    "gpus_per_node": 1,
    "qos": None,
    "account": None,
}


def _load_slurm_config() -> dict:
    """Load SLURM config from local/slurm.yaml, merging onto cluster-agnostic defaults."""
    repo_root = Path(__file__).resolve().parent.parent
    local_cfg = repo_root / "local" / "slurm.yaml"
    defaults = OmegaConf.create(_SLURM_DEFAULTS)
    if local_cfg.exists():
        overrides = OmegaConf.load(local_cfg)
        defaults = OmegaConf.merge(defaults, overrides)
    return OmegaConf.to_container(defaults, resolve=True)


SLURM_DEFAULTS = _load_slurm_config()


# Example-specific configurations
EXAMPLE_CONFIGS = {
    "image_jepa": {
        "config": "examples/image_jepa/cfgs/default.yaml",
        "module": "examples.image_jepa.main",
        "metric": "val_acc",
    },
    "video_jepa": {
        "config": "examples/video_jepa/cfgs/default.yaml",
        "module": "examples.video_jepa.main",
        "metric": "AP_1",
    },
    "ac_video_jepa": {
        "config": "examples/ac_video_jepa/cfgs/train/two_rooms/vc.yaml",
        "module": "examples.ac_video_jepa.main",
        "metric": "success_rate",
    },
    "h_ac_video_jepa": {
        "config": "examples/h_ac_video_jepa/cfgs/train/two_rooms/vc.yaml",
        "module": "examples.h_ac_video_jepa.main",
        "metric": "success_rate",
    },
}

# =============================================================================
# Utility functions
# =============================================================================


def make_executor(
    folder: str,
    job_name: str,
    array_parallelism: int | None = None,
    gpus: int = 1,
    nodes: int = 1,
) -> submitit.AutoExecutor:
    """Create a submitit executor with standard SLURM parameters."""
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=20)

    params = {
        "name": job_name,
        "slurm_mem_per_gpu": SLURM_DEFAULTS["mem_per_gpu"],
        "cpus_per_task": SLURM_DEFAULTS["cpus_per_task"],
        "timeout_min": SLURM_DEFAULTS["timeout_min"],
        "nodes": nodes,
        "tasks_per_node": gpus,
        "gpus_per_node": gpus,
    }

    if SLURM_DEFAULTS["partition"] is not None:
        params["slurm_partition"] = SLURM_DEFAULTS["partition"]

    additional = {}
    if SLURM_DEFAULTS["qos"] is not None:
        additional["qos"] = SLURM_DEFAULTS["qos"]
    if SLURM_DEFAULTS["account"] is not None:
        additional["account"] = SLURM_DEFAULTS["account"]
    if additional:
        params["slurm_additional_parameters"] = additional

    if array_parallelism is not None:
        params["slurm_array_parallelism"] = array_parallelism

    executor.update_parameters(**params)
    return executor


def normalize_sweep_name(name: str) -> str:
    """Ensure sweep name has 'sweep_' prefix for consistency."""
    if name.startswith("sweep_"):
        return name
    return f"sweep_{name}"


def copy_code_folder(code_folder):
    """Copy the code folder to the experiment directory, ignoring unnecessary files."""
    # Patterns to always ignore (matched by name only)
    ignore_patterns = [
        "__pycache__",
        ".vscode",
        ".git",
        "core",
        "uv.lock",
        "Makefile",
        ".llms",
        "CLAUDE.md",
    ]
    # Paths to ignore (matched by name only, applies to any directory with this name)
    ignore_paths = [
        "traces",
        "docs",
        ".pytest_cache",
        "logs",
        ".venv",
        "eb_jepa.egg-info",
        "wandb",
        "assets",
    ]
    # Root-level directories to ignore (only ignored when at the source root)
    # This allows us to skip ./datasets (storage-intensive data) while keeping
    # ./eb_jepa/datasets (data code needed for experiments)
    root_only_ignore = [
        "eb_jepa_ICLR",
        "datasets",
        "checkpoints",
        "arxiv",
        "arxiv_eb2",
        "arxiv_hierarchy",
    ]
    source_root = os.path.abspath(".")

    def ignore_func(path, names):
        ignored = []
        for n in names:
            if n in ignore_patterns or n in ignore_paths:
                ignored.append(n)
            # Only ignore root-level directories specified in root_only_ignore
            elif n in root_only_ignore and os.path.abspath(path) == source_root:
                ignored.append(n)
        return ignored

    if not os.path.exists(code_folder):
        shutil.copytree(".", code_folder, ignore=ignore_func)


def setup_launch_environment(base_dir, logs_subdir: str | None = "slurm_logs"):
    """Setup directories and code folder for launching jobs."""
    base_dir = base_dir.absolute() if hasattr(base_dir, "absolute") else base_dir
    logs_dir = base_dir / logs_subdir if logs_subdir else base_dir
    code_folder = base_dir / "code"

    copy_code_folder(str(code_folder))
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Code folder: {code_folder}")
    os.chdir(code_folder)

    return logs_dir, code_folder


def generate_param_combinations(param_grid: dict):
    """Generate all parameter combinations from a grid."""
    param_names = list(param_grid.keys())
    param_values_list = list(param_grid.values())
    all_combinations = list(product(*param_values_list))
    return param_names, all_combinations


def print_submission_summary(jobs: list, logs_dir, extra_info: dict | None = None):
    """Print a compact summary of batch job submission."""
    job_ids = [job.job_id for job in jobs]
    batch_id = job_ids[0].split("_")[0] if "_" in job_ids[0] else job_ids[0]
    print(f"\n✓ Submitted {len(jobs)} jobs (batch {batch_id}_[0-{len(jobs)-1}])")
    print(f"  Logs: {logs_dir}")
    if extra_info:
        for key, value in extra_info.items():
            print(f"  {key}: {value}")


# =============================================================================
# Launch functions
# =============================================================================


def run_experiment(example_name: str, cfg, folder=None, gpus: int = 1):
    """Run the appropriate example based on example_name.

    Each SLURM task calls this function directly. When gpus > 1,
    SLURM spawns ntasks-per-node=gpus tasks, and each one calls
    this function. Distributed init happens inside the training
    script via setup_distributed().
    """
    print(f"Current working directory: {os.getcwd()}")
    print(f"EBJEPA_DSETS: {os.environ.get('EBJEPA_DSETS', 'not set')}")
    print(
        f"EBJEPA_DATA: {os.environ.get('EBJEPA_DATA', 'not set (using EBJEPA_DSETS)')}"
    )

    module = importlib.import_module(EXAMPLE_CONFIGS[example_name]["module"])
    return module.run(cfg=cfg, folder=folder)


def launch_job(example_name: str, fname: str, gpus: int = 1, nodes: int = 1, **kwargs):
    """Launch a single training job with the given config and overrides."""
    cfg = load_config(fname, kwargs)
    sweep_name = kwargs.get("sweep_name", get_default_run_name("sweep"))
    exp_name = get_exp_name(example_name, cfg)
    try:
        dataset_name = get_dataset_name(cfg)
    except ValueError:
        dataset_name = ""

    folder = get_unified_experiment_dir(
        example_name=example_name,
        sweep_name=sweep_name,
        exp_name=exp_name,
        seed=cfg.meta.seed,
        dataset_name=dataset_name,
    )

    logs_dir, _ = setup_launch_environment(folder, logs_subdir=None)

    executor = make_executor(
        folder=str(logs_dir),
        job_name=f"{example_name.upper()}",
        gpus=gpus,
        nodes=nodes,
    )
    job = executor.submit(run_experiment, example_name, cfg, folder, gpus)

    print(f"\n✓ Submitted job {job.job_id}")
    print(f"  Experiment folder: {folder}")

    return job


def _dot_notation_to_nested_params(param_grid: dict) -> dict:
    """Convert flat dot-notation param_grid to WandB nested parameters format.

    Example:
        {"model.cost.loss.detach_encoder": {"values": [True, False]}}
        becomes
        {"model": {"parameters": {"cost": {"parameters": {"loss": {"parameters": {
            "detach_encoder": {"values": [True, False]}}}}}}}}
    """
    result = {}
    for flat_key, value_spec in param_grid.items():
        parts = flat_key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"parameters": {}}
            current = current[part]["parameters"]
        current[parts[-1]] = value_spec
    return result


def create_wandb_sweep_config(param_grid: dict, metric: str, method: str = "grid"):
    """Create a wandb sweep configuration from a parameter grid.

    Converts flat dot-notation keys (e.g., 'model.cost.loss.detach_encoder') to
    WandB's nested parameters format for proper sweep UI correlation.
    """
    normalized_grid = {}
    for param_name, param_values in param_grid.items():
        if hasattr(param_values, "__iter__") and not isinstance(
            param_values, (str, dict)
        ):
            normalized_grid[param_name] = {"values": list(param_values)}
        elif isinstance(param_values, dict):
            normalized_grid[param_name] = param_values
        else:
            normalized_grid[param_name] = {"value": param_values}

    nested_params = _dot_notation_to_nested_params(normalized_grid)

    sweep_config = {
        "method": method,
        "metric": {"goal": "maximize", "name": metric},
        "parameters": nested_params,
    }

    return sweep_config


def launch_sweep(
    example_name: str,
    fname: str,
    param_grid: dict,
    array_parallelism: int = 256,
    use_wandb: bool = False,
    wandb_method: str = "grid",
    gpus: int = 1,
    nodes: int = 1,
    **base_overrides,
):
    """Launch a parameter sweep using submitit. Returns (sweep_id, jobs) if use_wandb else jobs."""
    param_names, all_combinations = generate_param_combinations(param_grid)

    if not all_combinations:
        print("No parameter combinations to sweep")
        return (None, []) if use_wandb else []

    sweep_name = base_overrides.get("sweep_name", get_default_run_name("sweep"))

    # Create wandb sweep if requested
    sweep_id = None
    if use_wandb:
        import wandb

        project_name = "eb_jepa"
        metric = EXAMPLE_CONFIGS[example_name]["metric"]
        sweep_config = create_wandb_sweep_config(param_grid, metric, wandb_method)
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        print(f"Created wandb sweep with ID: {sweep_id}")
        print(
            f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}"
        )

    # Setup environment (must happen before chdir)
    base_cfg = load_config(fname, {}, quiet=True)
    try:
        dataset_name = get_dataset_name(base_cfg)
    except ValueError:
        dataset_name = ""
    common_dir = get_checkpoints_dir() / example_name
    if dataset_name:
        common_dir = common_dir / dataset_name
    common_dir = common_dir / sweep_name
    logs_subdir = "wandb_sweep_slurm_logs" if use_wandb else "sweep_slurm_logs"
    logs_dir, _ = setup_launch_environment(common_dir, logs_subdir=logs_subdir)

    # Store checkpoints dir before chdir (for absolute paths in job configs)
    original_checkpoints_dir = get_checkpoints_dir().absolute()

    executor = make_executor(
        folder=str(logs_dir),
        job_name=f"{example_name.upper()}_{'wandb_' if use_wandb else ''}sweep",
        array_parallelism=array_parallelism,
        gpus=gpus,
        nodes=nodes,
    )

    print(f"\nPreparing {len(all_combinations)} tasks...")
    jobs = []
    with executor.batch():
        for values in all_combinations:
            param_overrides = dict(zip(param_names, values))
            final_overrides = {**base_overrides, **param_overrides}

            # Add wandb-specific overrides
            if use_wandb:
                final_overrides.update(
                    {
                        "logging.wandb_sweep": True,
                        "logging.wandb_sweep_id": sweep_id,
                        "logging.wandb_group": sweep_name,
                    }
                )

            cfg = load_config(fname, final_overrides, quiet=True)
            exp_name = get_exp_name(example_name, cfg, param_grid)
            folder = get_unified_experiment_dir(
                example_name=example_name,
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
                dataset_name=dataset_name,
                base_dir=original_checkpoints_dir,
            )

            job = executor.submit(run_experiment, example_name, cfg, folder, gpus)
            jobs.append(job)

    extra_info = {"Sweep ID": sweep_id} if use_wandb else None
    print_submission_summary(jobs, logs_dir, extra_info)

    return (sweep_id, jobs) if use_wandb else jobs


def launch_eval_sweep(
    example_name: str,
    sweep_dir: str,
    checkpoint_name: str = "latest.pth.tar",
    array_parallelism: int = 256,
    filter_pattern: str | None = None,
    **eval_overrides,
):
    """Launch eval-only jobs for all checkpoints in a sweep directory.

    Discovers subdirectories of sweep_dir containing checkpoint_name,
    loads each run's own config.yaml, sets eval_only_mode=True, and
    submits them as a SLURM job array.

    Args:
        example_name: Which example module to run (e.g. "ac_video_jepa").
        sweep_dir: Path to a sweep directory containing checkpoint subdirs.
        checkpoint_name: Checkpoint file to look for in each subdir.
        array_parallelism: Max parallel SLURM array jobs.
        filter_pattern: Optional regex to filter checkpoint folder names.
        **eval_overrides: Additional config overrides (e.g. eval.plan_cfg_path).
    """
    import re
    from pathlib import Path

    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory does not exist: {sweep_dir}")
        return []

    # Discover checkpoint folders
    checkpoint_folders = []
    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if filter_pattern and not re.search(filter_pattern, subdir.name):
            continue
        checkpoint_path = subdir / checkpoint_name
        config_path = subdir / "config.yaml"
        if checkpoint_path.exists() and config_path.exists():
            checkpoint_folders.append(subdir)

    if not checkpoint_folders:
        print(f"No checkpoint folders found in {sweep_dir} with {checkpoint_name}")
        return []

    print(f"Found {len(checkpoint_folders)} checkpoint folders to evaluate")

    # Setup environment for eval jobs
    eval_logs_dir = sweep_dir / "eval_slurm_logs"
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    # Copy code folder once for all eval jobs
    code_folder = sweep_dir / "eval_code"
    copy_code_folder(str(code_folder))
    print(f"Code folder: {code_folder}")
    os.chdir(code_folder)

    # Use lighter SLURM resources for eval
    executor = make_executor(
        folder=str(eval_logs_dir),
        job_name=f"{example_name.upper()}_eval",
        array_parallelism=array_parallelism,
    )
    # Override with eval-appropriate resources
    executor.update_parameters(
        slurm_mem_per_gpu="60G",
        # timeout_min=120,  # 2 hours
    )

    print(f"Preparing {len(checkpoint_folders)} eval tasks...")
    jobs = []
    with executor.batch():
        for folder in checkpoint_folders:
            # Load the checkpoint's own config.yaml
            config_path = folder / "config.yaml"
            cfg = load_config(str(config_path), {}, quiet=True)

            # Set eval-only overrides
            eval_mode_overrides = {
                "meta.eval_only_mode": True,
                "meta.load_model": True,
                "meta.model_folder": str(folder.absolute()),
                "meta.load_checkpoint": checkpoint_name,
                # Clear wandb sweep settings from training
                "logging.wandb_sweep": False,
                "logging.wandb_sweep_id": None,
            }

            # Merge with user-provided eval overrides (e.g., eval.plan_cfg_path)
            final_overrides = {**eval_mode_overrides, **eval_overrides}
            cfg = load_config(str(config_path), final_overrides, quiet=True)

            # Submit job with the same folder (will use existing folder, add eval results)
            job = executor.submit(run_experiment, example_name, cfg, folder.absolute())
            jobs.append(job)

    print_submission_summary(jobs, eval_logs_dir)
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified SLURM launcher for EB-JEPA examples"
    )
    parser.add_argument(
        "--example",
        type=str,
        required=True,
        choices=["image_jepa", "video_jepa", "ac_video_jepa", "h_ac_video_jepa"],
        help="Which example to run",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="Path to config file (defaults to example's default config)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Name for the sweep (default: sweep_YYYYMMDD_HHMM)",
    )
    parser.add_argument(
        "--array-parallelism",
        type=int,
        default=256,
        help="Number of jobs to run in parallel for the sweep",
    )
    parser.add_argument(
        "--use-wandb-sweep",
        action="store_true",
        help="Use wandb sweep for hyperparameter tracking",
    )
    parser.add_argument(
        "--sweep-method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="Wandb sweep method to use if use_wandb_sweep is true",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Enable full hyperparameter sweep (default: only sweep over 3 seeds)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs per node (default: 1, uses torchrun for >1)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1, read from config slurm.nodes if set)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Launch a single job (uses dev_YYYYMMDD_HHMM folder)",
    )
    parser.add_argument(
        "--eval-sweep",
        type=str,
        default=None,
        help="Path to sweep directory to evaluate all checkpoints in",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="latest.pth.tar",
        help="Checkpoint file to evaluate (default: latest.pth.tar)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional regex to filter checkpoint folder names",
    )

    # SLURM overrides (applied on top of local/slurm.yaml defaults)
    parser.add_argument("--slurm-partition", type=str, default=None)
    parser.add_argument("--slurm-qos", type=str, default=None)
    parser.add_argument("--slurm-account", type=str, default=None)
    parser.add_argument("--slurm-mem-per-gpu", type=str, default=None)

    # Common overrides
    parser.add_argument("--optim.lr", type=float)
    parser.add_argument("--meta.seed", type=int)

    # ac_video_jepa specific
    parser.add_argument("--model.regularizer.cov_coeff", type=float)
    parser.add_argument("--model.regularizer.std_coeff", type=float)
    parser.add_argument("--model.regularizer.sim_coeff_t", type=float)
    parser.add_argument("--model.regularizer.idm_coeff", type=float)

    # h_ac_video_jepa specific (per-level regularization)
    parser.add_argument("--model.level_1.regularizer.cov_coeff", type=float)
    parser.add_argument("--model.level_1.regularizer.std_coeff", type=float)
    parser.add_argument("--model.level_1.regularizer.sim_coeff_t", type=float)
    parser.add_argument("--model.level_1.regularizer.idm_coeff", type=float)
    parser.add_argument("--model.level_2.regularizer.cov_coeff", type=float)
    parser.add_argument("--model.level_2.regularizer.std_coeff", type=float)
    parser.add_argument("--model.level_2.regularizer.sim_coeff_t", type=float)
    parser.add_argument("--model.level_2.regularizer.idm_coeff", type=float)
    parser.add_argument("--model.level_3.regularizer.cov_coeff", type=float)
    parser.add_argument("--model.level_3.regularizer.std_coeff", type=float)
    parser.add_argument("--model.level_3.regularizer.sim_coeff_t", type=float)
    parser.add_argument("--model.level_3.regularizer.idm_coeff", type=float)

    # Use parse_known_args to allow dynamic overrides for any config key
    args, unknown = parser.parse_known_args()

    example_name = args.example
    example_config = EXAMPLE_CONFIGS[example_name]
    fname = args.fname or example_config["config"]

    # Load config to read sweep params from YAML (quiet mode to avoid duplicate logs)
    base_cfg = load_config(fname, {}, quiet=True)

    # Read sweep param_grid from config file
    # Fall back to default 3-seed sweep if not specified in config
    config_param_grid = (base_cfg.get("sweep") or {}).get("param_grid", {})
    if OmegaConf.is_config(config_param_grid):
        config_param_grid = OmegaConf.to_container(config_param_grid, resolve=True)

    default_seed_sweep = {"meta.seed": [1, 1000, 10000]}

    # Build overrides dict from known args
    excluded_keys = {
        "example",
        "fname",
        "sweep",
        "array_parallelism",
        "use_wandb_sweep",
        "sweep_method",
        "full_sweep",
        "single",
        "eval_sweep",
        "checkpoint_name",
        "filter",
        "gpus",
        "nodes",
        "slurm_partition",
        "slurm_qos",
        "slurm_account",
        "slurm_mem_per_gpu",
    }
    overrides = {
        k: v for k, v in vars(args).items() if v is not None and k not in excluded_keys
    }

    # Parse unknown args as additional config overrides (e.g., --data.batch_size 64)
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = unknown[i + 1]
                # Try to parse as JSON (handles numbers, bools, lists)
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    # Handle Python-style booleans (True/False)
                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                overrides[key] = value
                i += 2
            else:
                # Flag without value (e.g., --some_flag)
                overrides[key] = True
                i += 1
        else:
            i += 1

    # Apply CLI SLURM overrides
    for cli_key, slurm_key in [
        ("slurm_partition", "partition"),
        ("slurm_qos", "qos"),
        ("slurm_account", "account"),
        ("slurm_mem_per_gpu", "mem_per_gpu"),
    ]:
        val = getattr(args, cli_key)
        if val is not None:
            SLURM_DEFAULTS[slurm_key] = val

    # Read gpus/nodes from config if not overridden on CLI
    gpus = args.gpus
    if gpus == 1 and base_cfg.get("slurm", {}).get("gpus"):
        gpus = base_cfg.slurm.gpus
    nodes = args.nodes
    if nodes == 1 and base_cfg.get("slurm", {}).get("nodes"):
        nodes = base_cfg.slurm.nodes

    # Determine folder name based on mode
    if args.eval_sweep:
        # Eval sweep: evaluate all checkpoints in a directory
        print(f"Example: {example_name}")
        print(f"Eval sweep directory: {args.eval_sweep}")
        print(f"Checkpoint name: {args.checkpoint_name}")
        if args.filter:
            print(f"Filter pattern: {args.filter}")
        if overrides:
            print(f"Eval overrides: {overrides}")

        jobs = launch_eval_sweep(
            example_name=example_name,
            sweep_dir=args.eval_sweep,
            checkpoint_name=args.checkpoint_name,
            array_parallelism=args.array_parallelism,
            filter_pattern=args.filter,
            **overrides,
        )
    elif args.single:
        # Single job: use dev_ prefix
        prefix = base_cfg.logging.get("exp_tag") or "dev"
        sweep_name = get_default_run_name(prefix)
        param_grid = None  # No sweep, single job
    elif args.sweep:
        # Custom sweep name: normalize to have sweep_ prefix
        sweep_name = normalize_sweep_name(args.sweep)
        if args.full_sweep:
            param_grid = config_param_grid if config_param_grid else default_seed_sweep
        else:
            param_grid = default_seed_sweep
    else:
        # Default: 3-seed sweep with auto-generated name
        prefix = base_cfg.logging.get("exp_tag") or "sweep"
        sweep_name = get_default_run_name(prefix)
        if args.full_sweep:
            param_grid = config_param_grid if config_param_grid else default_seed_sweep
        else:
            param_grid = default_seed_sweep

    # Skip the rest for eval sweep mode
    if args.eval_sweep:
        import sys

        sys.exit(0)

    overrides["sweep_name"] = sweep_name
    overrides["logging.wandb_group"] = sweep_name

    print(f"Example: {example_name}")
    print(f"Config: {fname}")
    print(f"Sweep name: {sweep_name}")
    if param_grid:
        print(f"Param grid: {param_grid}")
    else:
        print("Mode: single job")
    if overrides:
        print(f"Overrides: {overrides}")

    if args.single:
        # Launch single job
        job = launch_job(example_name, fname, gpus=gpus, nodes=nodes, **overrides)
    elif args.use_wandb_sweep:
        sweep_id, jobs = launch_sweep(
            example_name,
            fname,
            param_grid,
            array_parallelism=args.array_parallelism,
            use_wandb=True,
            wandb_method=args.sweep_method,
            gpus=gpus,
            nodes=nodes,
            **overrides,
        )
    else:
        jobs = launch_sweep(
            example_name,
            fname,
            param_grid,
            array_parallelism=args.array_parallelism,
            gpus=gpus,
            nodes=nodes,
            **overrides,
        )
