# EB-JEPA Datasets

This directory contains dataset implementations for training action-conditioned JEPAs on various domains.

## Available Datasets

| Dataset | Module | Domain | Observations | Actions | State |
|---------|--------|--------|-------------|---------|-------|
| **Two Rooms** | `two_rooms_dset.py` | 2D navigation | 65×65 RGB | 2D velocity | 2D position |
| **DROID** | `droid_dset.py` | Real robot manipulation | Multi-view RGB video | 7-DoF delta poses | 7-DoF robot state |
| **PushT** | `pusht_dset.py` | 2D pushing | 96×96 RGB | 2D position | 2D position + angle |
| **PointMaze** | `point_maze_dset.py` | 2D maze navigation | 64×64 RGB | 2D velocity | 2D position |
| **RoboCasa** | `robocasa_dset.py` | Kitchen manipulation | Multi-view RGB | 7-DoF delta poses | 7-DoF robot state |

> **Note:** RoboCasa is partially implemented (dataset loader and env wrapper exist) but has not been tested end-to-end with training or evaluation.

Two Rooms generates data on-the-fly during training. All other datasets load from disk.

## Dataset Interface

All action-conditioned datasets follow a unified interface for compatibility with training and evaluation code.

### Base Class: `TrajDataset`

Located in `traj_dset.py`, this is the abstract base class for trajectory-based datasets.

**Required methods:**
```python
class TrajDataset(Dataset):
    def get_seq_length(self, idx: int) -> int:
        """Return the length of the idx-th trajectory."""
        raise NotImplementedError

    def __getitem__(self, idx: int):
        """Return a single trajectory sample."""
        raise NotImplementedError
```

**Required attributes:**
- `action_dim`: Dimension of action space
- `state_dim`: Dimension of state space
- `proprio_dim`: Dimension of proprioceptive observations
- `samples`: List of trajectory samples (optional, for debugging)

### Return Format

Datasets should return tuples with the following structure:

```python
(obs, actions, states, reward, extra)
```

Where:
- `obs`: Dictionary with keys:
  - `"visual"`: Visual observations `[T, C, H, W]`
  - `"proprio"`: Proprioceptive state `[T, D]` (optional)
- `actions`: Actions `[T, A]` or `[T-1, A]`
- `states`: States `[T, D]`
- `reward`: Reward signal (scalar or `[T]`)
- `extra`: Optional metadata (can be `None`)

### Batching with `traj_collate_fn`

The custom collate function in `utils.py` handles batching and dimension reordering:

```python
from eb_jepa.data.utils import traj_collate_fn

loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=traj_collate_fn,
    ...
)
```

**Batch format after collation:**
- `obs["visual"]`: `[B, C, T, H, W]` (channels first, suitable for CNN encoders)
- `obs["proprio"]`: `[B, T, D]`
- `actions`: `[B, A, T]` (actions transposed for predictor input)
- `states`: `[B, T, D]`
- `rewards`: `[B, T]`

### Trajectory Slicing with `TrajSlicerDataset`

For training with fixed-length sequences, wrap your dataset with `TrajSlicerDataset`:

```python
from eb_jepa.data.traj_dset import TrajSlicerDataset

sliced_dataset = TrajSlicerDataset(
    dataset=base_dataset,
    num_frames=10,         # Length of each slice
    frameskip=1,           # Sample every N frames
    action_skip=1,         # Sample actions every N frames
    process_actions="concat",  # How to combine actions
    generator=torch.Generator().manual_seed(42),
)
```

## Usage

### Quick Start

```python
from eb_jepa.data.utils import init_data

# Load dataset with config
train_loader, val_loader, config = init_data(
    env_name="two_rooms",  # or "droid", "pusht", "pointmaze", "robocasa"
    cfg_data={
        "batch_size": 64,
        "num_workers": 4,
    }
)

# Iterate through batches
for obs, actions, states, rewards, extra in train_loader:
    # obs["visual"]: [B, C, T, H, W]
    # obs["proprio"]: [B, T, D]
    # actions: [B, A, T]
    # states: [B, T, D]
    ...
```

### Dataset Configuration

Each dataset has a `data_config.yaml` file (in `cfgs/`) with default settings that can be overridden:

```python
train_loader, val_loader, config = init_data(
    env_name="droid",
    cfg_data={
        "data_path": "/path/to/dataset.csv",
        "batch_size": 16,
        "num_frames": 8,
        "normalize_action": True,
    }
)
```

## Adding a New Dataset

To add a new trajectory-based dataset:

1. **Inherit from `TrajDataset`**:
   ```python
   from eb_jepa.data.traj_dset import TrajDataset

   class MyDataset(TrajDataset):
       def __init__(self, ...):
           self.action_dim = ...
           self.state_dim = ...
           self.proprio_dim = ...

       def get_seq_length(self, idx):
           return len(self.trajectories[idx])

       def __getitem__(self, idx):
           obs = {"visual": ..., "proprio": ...}
           actions = ...
           states = ...
           return obs, actions, states, reward, None
   ```

2. **Create your dataset module** (e.g., `eb_jepa/data/my_dset.py`)

3. **Add to `init_data` in `utils.py`**:
   ```python
   elif env_name == "my_dataset":
       from eb_jepa.data.my_dset import MyDataset
       # ... initialization logic
   ```

4. **Use `traj_collate_fn` for dataloaders**

## File Organization

```
eb_jepa/data/
├── README.md             # This file
├── utils.py              # init_data, traj_collate_fn
├── traj_dset.py          # TrajDataset base class, TrajSlicerDataset
├── transforms.py         # Image transforms and augmentations
├── preprocessor.py       # Data preprocessing utilities
├── two_rooms_dset.py     # Two Rooms (on-the-fly generation)
├── droid_dset.py         # DROID real robot dataset
├── pusht_dset.py         # PushT pushing dataset
├── point_maze_dset.py    # PointMaze navigation dataset
├── robocasa_dset.py      # RoboCasa kitchen dataset
├── cfgs/                 # Per-dataset default configs
└── _video_transforms/    # Video-specific transforms
```

## Dependencies

- **Core**: torch, numpy, einops
- **Video loading**: decord (for DROID)
- **Data formats**: h5py (for HDF5 files), pandas (for CSV manifests)
- **Transforms**: scipy (for rotation utilities)

## Testing

Run dataset tests:
```bash
# Test basic functionality with mock data
python tests/test_droid_dataset.py

# Run full test suite with pytest
pytest tests/ -v
```

## Important Notes

### Training Loop Integration

The training loop unpacks 5 values from the dataloader:

```python
for idx, (x, a, loc, _, _) in loader:
    x = x.to(device)         # Visual observations [B, C, T, H, W]
    a = a[:, :, :-1].to(device)  # Actions [B, A, T-1]
    loc = loc.to(device)     # States/locations [B, T, D]
```

**For DROID and similar datasets**: The `obs` dictionary returned by `traj_collate_fn` needs to be unpacked:
- `x = obs["visual"]`
- `loc = obs["proprio"]` or `states`

### Normalization

Datasets may apply normalization to actions and states. Check the dataset's `normalize_action` parameter and stored statistics:
- `dataset.action_mean`, `dataset.action_std`
- `dataset.state_mean`, `dataset.state_std`

### Memory Considerations

- Video datasets can be memory-intensive. Use `num_workers > 0` for parallel loading.
- For large datasets, consider using `TrajSlicerDataset` to sample shorter clips.
- Use `droid_fraction < 1.0` for debugging with a subset of data.

## Examples

See training configs in `examples/ac_video_jepa/cfgs/` for example usage with different datasets.
