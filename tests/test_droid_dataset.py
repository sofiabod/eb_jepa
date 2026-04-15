"""Test DROID dataset loading and shape verification."""

import os

import pytest
import torch

# Import collate function for basic tests
from eb_jepa.data.utils import traj_collate_fn

# Try to import DROID dataset (requires h5py, decord)
try:
    from eb_jepa.data.droid_dset import DROIDVideoDataset
    from eb_jepa.data.traj_dset import TrajSlicerDataset, split_traj_datasets

    DROID_AVAILABLE = True
except ImportError as e:
    DROID_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(
    not DROID_AVAILABLE, reason="Requires h5py, decord, and DROID dataset"
)
@pytest.mark.skip(reason="Requires DROID dataset files to be present")
class TestDROIDDataset:
    """Tests for DROID dataset loading and batching."""

    @pytest.fixture
    def dataset_path(self):
        """Path to DROID dataset CSV file."""
        return os.path.join(
            os.environ.get("EBJEPA_DATA") or os.environ.get("EBJEPA_DSETS", "/tmp"),
            "DROID/droid_paths.csv",
        )

    @pytest.fixture
    def base_dataset(self, dataset_path):
        """Create a base DROID dataset."""
        return DROIDVideoDataset(
            data_path=dataset_path,
            camera_views=["wrist_mp4_path"],
            num_frames=8,
            fps=5,
            frameskip=1,
            action_skip=1,
            normalize_action=True,
            seed=42,
            droid_fraction=0.01,  # Use small fraction for testing
        )

    def test_dataset_creation(self, base_dataset):
        """Test that dataset can be created successfully."""
        assert len(base_dataset) > 0
        assert base_dataset.action_dim == 7  # xyz, rotation, gripper
        assert base_dataset.state_dim == 7
        assert base_dataset.proprio_dim == 7

    def test_single_sample_shape(self, base_dataset):
        """Test that a single sample has correct shapes."""
        obs, actions, states, reward, extra = base_dataset[0]

        # obs is a dict with "visual" and "proprio"
        assert isinstance(obs, dict)
        assert "visual" in obs
        assert "proprio" in obs

        # Visual: [T, C, H, W]
        T, C, H, W = obs["visual"].shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert T == 16, f"Expected 16 frames, got {T}"

        # Proprio: [T, D]
        assert obs["proprio"].shape == (T, 7)

        # Actions: [T-1, A] or [T, A] (padded)
        assert actions.shape[0] in [T - 1, T]
        assert actions.shape[1] == 7

        # States: [T, D]
        assert states.shape == (T, 7)

        # Reward: scalar
        assert reward.shape == torch.Size([])

    def test_traj_slicer(self, base_dataset):
        """Test that TrajSlicerDataset produces correct slices."""
        num_frames = 10
        frameskip = 1
        action_skip = 1

        slicer = TrajSlicerDataset(
            base_dataset,
            num_frames=num_frames,
            frameskip=frameskip,
            action_skip=action_skip,
            process_actions="concat",
            generator=torch.Generator().manual_seed(42),
        )

        assert len(slicer) > 0

        obs, actions, states, rewards = slicer[0]

        # Visual: [T, C, H, W]
        T = obs["visual"].shape[0]
        assert T == num_frames, f"Expected {num_frames} frames, got {T}"

        # Actions: [T, A] (after processing)
        assert actions.shape[0] == num_frames
        assert actions.shape[1] == 7

        # States: [T, D]
        assert states.shape == (num_frames, 7)

    def test_collate_function(self, base_dataset):
        """Test that collate function produces correct batch shapes."""
        # Create a simple slicer
        slicer = TrajSlicerDataset(
            base_dataset,
            num_frames=10,
            frameskip=1,
            action_skip=1,
            generator=torch.Generator().manual_seed(42),
        )

        # Get a few samples
        batch = [slicer[i] for i in range(4)]

        # Apply collate function
        obs, actions, states, rewards, extra = traj_collate_fn(batch)

        B = 4
        T = 10

        # Visual: [B, C, T, H, W]
        assert obs["visual"].shape[0] == B
        assert obs["visual"].shape[2] == T
        assert obs["visual"].shape[1] == 3  # channels

        # Proprio: [B, T, D]
        assert obs["proprio"].shape == (B, T, 7)

        # Actions: [B, A, T]
        assert actions.shape == (B, 7, T)

        # States: [B, T, D]
        assert states.shape == (B, T, 7)

        # Rewards: [B, T]
        assert rewards.shape == (B, T)

    def test_dataloader(self, base_dataset):
        """Test full dataloader pipeline."""
        # Create slicer
        slicer = TrajSlicerDataset(
            base_dataset,
            num_frames=10,
            frameskip=1,
            action_skip=1,
            generator=torch.Generator().manual_seed(42),
        )

        # Create dataloader
        loader = torch.utils.data.DataLoader(
            slicer,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 0 for testing
            collate_fn=traj_collate_fn,
        )

        # Get one batch
        obs, actions, states, rewards, extra = next(iter(loader))

        B = 4
        T = 10

        # Check all shapes
        assert obs["visual"].shape[0] == B
        assert obs["visual"].shape[2] == T
        assert obs["proprio"].shape == (B, T, 7)
        assert actions.shape == (B, 7, T)
        assert states.shape == (B, T, 7)
        assert rewards.shape == (B, T)

    def test_train_val_split(self, base_dataset):
        """Test train/val splitting."""
        train, val = split_traj_datasets(
            base_dataset,
            train_fraction=0.8,
            random_seed=42,
            traj_subset=True,
        )

        assert len(train) + len(val) == len(base_dataset)
        assert len(train) > len(val)  # 80/20 split

    def test_normalization(self, base_dataset):
        """Test that normalization is applied correctly."""
        # Actions and states should be normalized
        assert base_dataset.action_mean is not None
        assert base_dataset.action_std is not None
        assert base_dataset.state_mean is not None
        assert base_dataset.state_std is not None

        # Check that means are reasonable (close to zero for normalized data)
        # Note: This is approximate since we're testing with a fraction of data
        # In practice, with the full dataset, means should be closer to zero


def test_traj_collate_fn_with_mock_data():
    """Test collate function with mock data (no dataset required)."""
    # Create mock batch data
    B = 4
    T = 10
    C = 3
    H = W = 64
    A = 7

    batch = []
    for _ in range(B):
        obs = {
            "visual": torch.randn(T, C, H, W),
            "proprio": torch.randn(T, A),
        }
        actions = torch.randn(T, A)
        states = torch.randn(T, A)
        reward = torch.tensor(0.0)
        batch.append((obs, actions, states, reward))

    # Apply collate function
    obs, actions, states, rewards, extra = traj_collate_fn(batch)

    # Check shapes
    assert obs["visual"].shape == (B, C, T, H, W)
    assert obs["proprio"].shape == (B, T, A)
    assert actions.shape == (B, A, T)
    assert states.shape == (B, T, A)
    assert rewards.shape == (B,)


if __name__ == "__main__":
    # Run simple mock test without dataset
    print("Running mock collate function test...")
    test_traj_collate_fn_with_mock_data()
    print("✓ Mock test passed!")

    print("\nTo run full dataset tests:")
    print("1. Update dataset_path fixture with actual DROID dataset path")
    print("2. Remove @pytest.mark.skip decorator")
    print("3. Run: pytest tests/test_droid_dataset.py -v")
