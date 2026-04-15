"""Tests for DATA_STATS registry, config loading, and dataset imports.

Phase 1 validation: verifies that the foundation layer (DATA_STATS, YAML configs,
and import paths) works correctly after the jepa-wms import and restructuring.
"""

from pathlib import Path

import pytest
import yaml

from eb_jepa.data import DATA_STATS, get_data_dims, get_data_stats

EXPECTED_ENVS = ["two_rooms", "pusht", "pointmaze", "droid", "robocasa"]

DATASETS_DIR = Path(__file__).parent.parent / "eb_jepa" / "data"


# ---------------------------------------------------------------------------
# DATA_STATS registry
# ---------------------------------------------------------------------------


class TestDataStats:
    def test_all_envs_present(self):
        for env in EXPECTED_ENVS:
            assert env in DATA_STATS, f"{env} missing from DATA_STATS"

    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_required_fields(self, env):
        stats = DATA_STATS[env]
        for field in [
            "action_dim",
            "proprio_dim",
            "state_dim",
            "num_channels",
            "img_size",
        ]:
            assert field in stats, f"{env} missing field '{field}'"

    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_normalization_stats_present(self, env):
        stats = DATA_STATS[env]
        for field in ["action_mean", "action_std", "proprio_mean", "proprio_std"]:
            assert field in stats, f"{env} missing '{field}'"

    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_normalization_stats_length(self, env):
        stats = DATA_STATS[env]
        assert len(stats["action_mean"]) == stats["action_dim"]
        assert len(stats["action_std"]) == stats["action_dim"]
        assert len(stats["proprio_mean"]) == stats["proprio_dim"]
        assert len(stats["proprio_std"]) == stats["proprio_dim"]

    def test_two_rooms_values(self):
        s = DATA_STATS["two_rooms"]
        assert s["action_dim"] == 2
        assert s["proprio_dim"] == 2
        assert s["num_channels"] == 2
        assert s["img_size"] == 65

    def test_droid_values(self):
        s = DATA_STATS["droid"]
        assert s["action_dim"] == 7
        assert s["num_channels"] == 3
        assert s["img_size"] == 224


# ---------------------------------------------------------------------------
# get_data_dims / get_data_stats helpers
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_get_data_dims_two_rooms(self):
        a, p = get_data_dims("two_rooms")
        assert a == 2 and p == 2

    def test_get_data_dims_droid(self):
        a, p = get_data_dims("droid")
        assert a == 7 and p == 7

    def test_get_data_dims_unknown_raises(self):
        with pytest.raises(KeyError):
            get_data_dims("nonexistent_env")

    def test_get_data_stats_returns_copy(self):
        s1 = get_data_stats("droid")
        s1["action_dim"] = 999
        s2 = get_data_stats("droid")
        assert s2["action_dim"] == 7, "get_data_stats should return a copy"

    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_get_data_stats_all_envs(self, env):
        stats = get_data_stats(env)
        assert isinstance(stats, dict)
        assert "action_dim" in stats


# ---------------------------------------------------------------------------
# YAML config files in cfgs/
# ---------------------------------------------------------------------------


class TestConfigFiles:
    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_cfg_file_exists(self, env):
        cfg_path = DATASETS_DIR / "cfgs" / f"{env}.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"

    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_cfg_has_action_dim(self, env):
        cfg_path = DATASETS_DIR / "cfgs" / f"{env}.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "action_dim" in cfg, f"{env}.yaml missing 'action_dim'"

    @pytest.mark.parametrize("env", EXPECTED_ENVS)
    def test_cfg_action_dim_matches_registry(self, env):
        cfg_path = DATASETS_DIR / "cfgs" / f"{env}.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert (
            cfg["action_dim"] == DATA_STATS[env]["action_dim"]
        ), f"{env}: cfg action_dim={cfg['action_dim']} != registry {DATA_STATS[env]['action_dim']}"


# ---------------------------------------------------------------------------
# Config loading via utils.py
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_env_data_config_two_rooms(self):
        from eb_jepa.data.utils import load_env_data_config

        cfg = load_env_data_config("two_rooms")
        assert cfg["action_dim"] == 2
        assert cfg["img_size"] == 65

    def test_load_env_data_config_with_overrides(self):
        from eb_jepa.data.utils import load_env_data_config

        cfg = load_env_data_config("two_rooms", overrides={"batch_size": 999})
        assert cfg["batch_size"] == 999

    def test_load_env_data_config_unknown_raises(self):
        from eb_jepa.data.utils import load_env_data_config

        with pytest.raises(FileNotFoundError):
            load_env_data_config("nonexistent_env")


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all dataset modules can be imported without errors."""

    def test_import_traj_dset(self):
        from eb_jepa.data.traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset

    def test_import_transforms(self):
        from eb_jepa.data.transforms import VideoTransform, make_transforms

    def test_import_preprocessor(self):
        from eb_jepa.data.preprocessor import Preprocessor

    def test_import_two_rooms_wall_dataset(self):
        from eb_jepa.data.two_rooms_dset import WallDataset, WallDatasetConfig

    def test_import_droid(self):
        from eb_jepa.data.droid_dset import DROIDVideoDataset

    def test_import_utils(self):
        from eb_jepa.data.utils import init_data, load_env_data_config
