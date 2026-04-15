"""Tests for launch_sbatch.py sweep utilities.

Tests cover:
- Parameter combination generation
- Sweep name normalization
- Wandb sweep config generation
- Centralized SWEEP_PARAM_ABBREV dictionary
"""

import pytest

from eb_jepa.utils.config import (
    SWEEP_PARAM_ABBREV,
    _cfg_get,
)
from examples.launch_sbatch import (
    create_wandb_sweep_config,
    generate_param_combinations,
    normalize_sweep_name,
)


class TestGenerateParamCombinations:
    """Tests for generate_param_combinations function."""

    def test_single_param(self):
        param_grid = {"meta.seed": [1, 1000, 10000]}
        names, combinations = generate_param_combinations(param_grid)
        assert names == ["meta.seed"]
        assert combinations == [(1,), (1000,), (10000,)]

    def test_two_params(self):
        param_grid = {
            "model.cost.loss.detach_encoder": [True, False],
            "meta.seed": [1, 2],
        }
        names, combinations = generate_param_combinations(param_grid)
        assert names == ["model.cost.loss.detach_encoder", "meta.seed"]
        assert len(combinations) == 4
        assert (True, 1) in combinations
        assert (True, 2) in combinations
        assert (False, 1) in combinations
        assert (False, 2) in combinations

    def test_three_params(self):
        param_grid = {
            "model.cost.projector.mlp_spec": ["512-256-64", "512-256-32"],
            "model.cost.loss.detach_encoder": [True, False],
            "meta.seed": [1],
        }
        names, combinations = generate_param_combinations(param_grid)
        assert len(names) == 3
        assert len(combinations) == 4  # 2 * 2 * 1

    def test_empty_grid(self):
        param_grid = {}
        names, combinations = generate_param_combinations(param_grid)
        assert names == []
        assert combinations == [()]

    def test_preserves_order(self):
        param_grid = {
            "a": [1],
            "b": [2],
            "c": [3],
        }
        names, combinations = generate_param_combinations(param_grid)
        assert names == ["a", "b", "c"]
        assert combinations == [(1, 2, 3)]


class TestNormalizeSweepName:
    """Tests for normalize_sweep_name function."""

    def test_adds_prefix(self):
        assert normalize_sweep_name("my_experiment") == "sweep_my_experiment"

    def test_preserves_existing_prefix(self):
        assert normalize_sweep_name("sweep_my_experiment") == "sweep_my_experiment"

    def test_date_format(self):
        assert normalize_sweep_name("2026-02-09") == "sweep_2026-02-09"


class TestCreateWandbSweepConfig:
    """Tests for create_wandb_sweep_config function."""

    def test_basic_config(self):
        param_grid = {"meta.seed": [1, 2, 3]}
        config = create_wandb_sweep_config(param_grid, "success_rate")

        assert config["method"] == "grid"
        assert config["metric"]["goal"] == "maximize"
        assert config["metric"]["name"] == "success_rate"
        assert "parameters" in config

    def test_flat_dot_notation_params(self):
        """Test that dot-notation keys are converted to nested structure."""
        param_grid = {
            "model.cost.loss.detach_encoder": [True, False],
        }
        config = create_wandb_sweep_config(param_grid, "success_rate")

        # Check nested structure
        params = config["parameters"]
        assert "model" in params
        assert "parameters" in params["model"]
        assert "cost" in params["model"]["parameters"]
        assert "parameters" in params["model"]["parameters"]["cost"]
        assert "loss" in params["model"]["parameters"]["cost"]["parameters"]
        assert (
            "parameters" in params["model"]["parameters"]["cost"]["parameters"]["loss"]
        )
        assert (
            "detach_encoder"
            in params["model"]["parameters"]["cost"]["parameters"]["loss"]["parameters"]
        )
        assert params["model"]["parameters"]["cost"]["parameters"]["loss"][
            "parameters"
        ]["detach_encoder"] == {"values": [True, False]}

    def test_method_override(self):
        param_grid = {"meta.seed": [1, 2]}
        config = create_wandb_sweep_config(param_grid, "acc", method="random")
        assert config["method"] == "random"

    def test_list_values(self):
        param_grid = {"optim.lr": [0.001, 0.0001]}
        config = create_wandb_sweep_config(param_grid, "loss")

        # Nested structure format
        params = config["parameters"]
        assert "optim" in params
        assert "parameters" in params["optim"]
        lr_param = params["optim"]["parameters"]["lr"]
        assert lr_param == {"values": [0.001, 0.0001]}


class TestSweepParamAbbrev:
    """Tests for the centralized SWEEP_PARAM_ABBREV dictionary."""

    def test_required_keys_present(self):
        """Verify all commonly used sweep parameters have abbreviations."""
        required_keys = [
            "meta.seed",
            "model.regularizer.cov_coeff",
            "model.regularizer.std_coeff",
            "model.regularizer.sim_coeff_t",
            "model.regularizer.idm_coeff",
            "model.cost.projector.mlp_spec",
            "model.cost.loss.detach_encoder",
            "optim.lr",
        ]
        for key in required_keys:
            assert key in SWEEP_PARAM_ABBREV, f"Missing abbreviation for '{key}'"

    def test_abbreviations_are_short(self):
        """Verify abbreviations are reasonably short (max 9 characters)."""
        for key, abbrev in SWEEP_PARAM_ABBREV.items():
            assert (
                len(abbrev) <= 9
            ), f"Abbreviation '{abbrev}' for '{key}' is too long (max 9 chars)"

    def test_abbreviations_are_unique(self):
        """Verify all abbreviations are unique to avoid folder conflicts."""
        abbrevs = list(SWEEP_PARAM_ABBREV.values())
        assert len(abbrevs) == len(
            set(abbrevs)
        ), f"Duplicate abbreviations found: {abbrevs}"

    def test_abbreviations_are_filesystem_safe(self):
        """Verify abbreviations don't contain problematic characters."""
        unsafe_chars = ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|", "."]
        for key, abbrev in SWEEP_PARAM_ABBREV.items():
            for char in unsafe_chars:
                assert (
                    char not in abbrev
                ), f"Abbreviation '{abbrev}' for '{key}' contains unsafe char '{char}'"


class TestCfgGet:
    """Tests for _cfg_get helper function."""

    def test_single_level(self):
        cfg = {"seed": 42}
        assert _cfg_get(cfg, "seed") == 42

    def test_nested_dict(self):
        cfg = {"model": {"lr": 0.001}}
        assert _cfg_get(cfg, "model.lr") == 0.001

    def test_deeply_nested(self):
        cfg = {"model": {"regularizer": {"cov_coeff": 8.0}}}
        assert _cfg_get(cfg, "model.regularizer.cov_coeff") == 8.0

    def test_string_value(self):
        cfg = {"model": {"cost": {"projector": {"mlp_spec": "512-256-64"}}}}
        assert _cfg_get(cfg, "model.cost.projector.mlp_spec") == "512-256-64"

    def test_missing_key_returns_default(self):
        cfg = {"model": {"lr": 0.001}}
        assert _cfg_get(cfg, "model.missing_key", "fallback") == "fallback"

    def test_missing_nested_returns_default(self):
        cfg = {"model": {}}
        assert _cfg_get(cfg, "model.deep.nested", None) is None
