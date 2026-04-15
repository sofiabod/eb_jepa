"""Tests for get_exp_name sweep deduplication."""

import pytest
from omegaconf import OmegaConf

from eb_jepa.utils.config import (
    get_dataset_name,
    get_exp_name,
    get_unified_experiment_dir,
    resolve_experiment_folder,
)


def _make_h_ac_cfg(**overrides):
    """Minimal h_ac_video_jepa config with regularizer params."""
    base = {
        "data": {"env_name": "two_rooms"},
        "model": {
            "num_levels": 3,
            "level_1": {
                "encoder": {"architecture": "impala"},
                "predictor": {"type": "rnn"},
                "regularizer": {
                    "cov_coeff": 8,
                    "std_coeff": 16,
                    "sim_coeff_t": 12,
                    "idm_coeff": 1,
                },
            },
            "level_2": {
                "regularizer": {
                    "cov_coeff": 4,
                    "std_coeff": 8,
                    "sim_coeff_t": 6,
                    "idm_coeff": 0.5,
                }
            },
            "level_3": {
                "regularizer": {
                    "cov_coeff": 2,
                    "std_coeff": 4,
                    "sim_coeff_t": 3,
                    "idm_coeff": 0.25,
                }
            },
            "level_weights": [1.0, 0.5, 0.25],
        },
        "meta": {"seed": 1},
    }
    cfg = OmegaConf.create(base)
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_no_sweep_base_name():
    """Base name includes all level regularizer params, no sweep suffix."""
    cfg = _make_h_ac_cfg()
    name = get_exp_name("h_ac_video_jepa", cfg)
    assert name == (
        "two_rooms_h3lvl-imp-rnn-vc"
        "_lvl1cov8_lvl1std16_lvl1simt12_lvl1idm1"
        "_lvl2cov4_lvl2std8_lvl2simt6_lvl2idm0.5"
        "_lvl3cov2_lvl3std4_lvl3simt3_lvl3idm0.25"
    )


def test_sweep_no_redundancy():
    """Swept regularizer params must NOT appear twice in the name."""
    cfg = _make_h_ac_cfg()
    param_grid = {
        "model.level_1.regularizer.cov_coeff": [8, 12],
        "model.level_1.regularizer.std_coeff": [8, 16],
        "model.level_1.regularizer.sim_coeff_t": [8, 12],
        "model.level_1.regularizer.idm_coeff": [1, 2],
        "model.level_2.regularizer.cov_coeff": [4, 8],
        "model.level_weights": [[1.0, 0.5, 0.25], [1.0, 1.0, 1.0]],
        "meta.seed": [1, 1000, 10000],
    }
    name = get_exp_name("h_ac_video_jepa", cfg, param_grid)

    # level_weights IS appended (not in base), regularizer params are NOT repeated
    assert "lw[1.0, 0.5, 0.25]" in name
    assert name.count("lvl1cov") == 1
    assert name.count("lvl1std") == 1
    assert name.count("lvl1simt") == 1
    assert name.count("lvl1idm") == 1
    assert name.count("lvl2cov") == 1


def test_sweep_only_seed():
    """Sweeping only seed should produce no sweep suffix."""
    cfg = _make_h_ac_cfg()
    param_grid = {"meta.seed": [1, 42]}
    name = get_exp_name("h_ac_video_jepa", cfg, param_grid)
    assert name == get_exp_name("h_ac_video_jepa", cfg)


def test_sweep_novel_param_appended():
    """A swept param not in the base name IS appended."""
    cfg = _make_h_ac_cfg()
    param_grid = {"model.level_weights": [[1.0, 0.5, 0.25], [1.0, 1.0, 1.0]]}
    name = get_exp_name("h_ac_video_jepa", cfg, param_grid)
    assert name.endswith("_lw[1.0, 0.5, 0.25]")


def test_ac_video_jepa_no_redundancy():
    """ac_video_jepa: swept regularizer params should not duplicate."""
    cfg = OmegaConf.create(
        {
            "model": {
                "encoder": {"architecture": "impala"},
                "regularizer": {
                    "cov_coeff": 10,
                    "std_coeff": 5,
                    "sim_coeff_t": 3,
                    "idm_coeff": 1,
                },
            },
        }
    )
    param_grid = {
        "model.regularizer.cov_coeff": [10, 20],
        "model.regularizer.std_coeff": [5, 10],
    }
    name = get_exp_name("ac_video_jepa", cfg, param_grid)
    assert name.count("cov") == 1
    assert name.count("std") == 1


def _make_image_jepa_cfg(**overrides):
    """Minimal image_jepa config."""
    base = {
        "data": {"dataset": "cifar10", "batch_size": 256},
        "model": {
            "type": "resnet",
            "use_projector": True,
            "proj_hidden_dim": 2048,
            "proj_output_dim": 512,
        },
        "loss": {
            "type": "bcs",
            "lmbd": 0.1,
            "std_coeff": 1.0,
            "cov_coeff": 80.0,
        },
        "optim": {"epochs": 300},
        "meta": {"seed": 42},
    }
    cfg = OmegaConf.create(base)
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_image_jepa_bcs_no_redundancy():
    """image_jepa BCS: swept proj dims and lmbd should not duplicate."""
    cfg = _make_image_jepa_cfg()
    param_grid = {
        "model.proj_hidden_dim": [512, 2048],
        "model.proj_output_dim": [128, 512],
        "loss.lmbd": [0.01, 0.1, 1],
        "meta.seed": [1, 1000],
    }
    name = get_exp_name("image_jepa", cfg, param_grid)
    assert name.count("lmbd") == 1
    assert name.count("ph") == 1
    assert name.count("po") == 1


def test_image_jepa_vicreg_no_redundancy():
    """image_jepa VICReg: swept proj dims and std/cov should not duplicate."""
    cfg = _make_image_jepa_cfg(
        loss={"type": "vicreg", "std_coeff": 1.0, "cov_coeff": 80.0}
    )
    param_grid = {
        "model.proj_output_dim": [1024, 2048],
        "loss.std_coeff": [1.0, 10.0],
        "loss.cov_coeff": [1.0, 80.0],
    }
    name = get_exp_name("image_jepa", cfg, param_grid)
    assert name.count("std") == 1
    assert name.count("cov") == 1
    assert name.count("po") == 1


def test_image_jepa_novel_param_appended():
    """image_jepa: swept param NOT in base name IS appended."""
    cfg = _make_image_jepa_cfg()
    param_grid = {"optim.lr": [0.001, 0.01, 0.3]}
    name = get_exp_name("image_jepa", cfg, param_grid)
    assert "lr" in name


# -- Tests for get_dataset_name -------------------------------------------------


def test_get_dataset_name_env_name():
    cfg = OmegaConf.create({"data": {"env_name": "two_rooms"}})
    assert get_dataset_name(cfg) == "two_rooms"


def test_get_dataset_name_dataset():
    cfg = OmegaConf.create({"data": {"dataset": "cifar10"}})
    assert get_dataset_name(cfg) == "cifar10"


def test_get_dataset_name_env_name_priority():
    cfg = OmegaConf.create({"data": {"env_name": "two_rooms", "dataset": "cifar10"}})
    assert get_dataset_name(cfg) == "two_rooms"


def test_get_dataset_name_raises():
    cfg = OmegaConf.create({"data": {}})
    with pytest.raises(ValueError):
        get_dataset_name(cfg)


# -- Tests for resolve_experiment_folder -----------------------------------------


def test_resolve_experiment_folder_explicit(tmp_path):
    folder = tmp_path / "my_exp_seed42"
    cfg = OmegaConf.create({"meta": {"seed": 42}, "data": {"env_name": "two_rooms"}})
    result_folder, exp_name = resolve_experiment_folder("ac_video_jepa", cfg, folder)
    assert result_folder == folder
    assert exp_name == "my_exp"
    assert result_folder.exists()


def test_resolve_experiment_folder_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("EBJEPA_CKPTS", str(tmp_path))
    cfg = OmegaConf.create(
        {
            "meta": {"seed": 1},
            "data": {"env_name": "two_rooms"},
            "model": {
                "encoder": {"architecture": "impala"},
                "regularizer": {
                    "cov_coeff": 8,
                    "std_coeff": 16,
                    "sim_coeff_t": 12,
                    "idm_coeff": 1,
                },
            },
        }
    )
    result_folder, exp_name = resolve_experiment_folder("ac_video_jepa", cfg)
    assert "two_rooms" in str(result_folder)
    assert "ac_video_jepa" in str(result_folder)
    assert result_folder.exists()


# -- Tests for get_unified_experiment_dir with dataset_name ----------------------


def test_unified_dir_dataset_name(tmp_path):
    exp_dir = get_unified_experiment_dir(
        example_name="ac_video_jepa",
        sweep_name="sweep_test",
        exp_name="my_exp",
        seed=1,
        dataset_name="two_rooms",
        base_dir=tmp_path,
    )
    assert "two_rooms" in str(exp_dir)
    assert str(exp_dir).index("two_rooms") < str(exp_dir).index("sweep_test")


def test_unified_dir_no_dataset_name(tmp_path):
    exp_dir = get_unified_experiment_dir(
        example_name="ac_video_jepa",
        sweep_name="sweep_test",
        exp_name="my_exp",
        seed=1,
        base_dir=tmp_path,
    )
    assert "two_rooms" not in str(exp_dir)
