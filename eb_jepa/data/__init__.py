# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset dimension and normalization registry for EB-JEPA environments.

This module provides hardcoded action_dim, proprio_dim, and normalization
statistics (mean/std) for each environment. This enables model loading via
torchhub without requiring access to the actual datasets.

Note: The normalization statistics below are computed from the training datasets
and match the values used during model training.
"""

# Environment data dimensions and normalization statistics
# These are used by hubconf.py to load models without needing the datasets
# fmt: off
DATA_STATS = {
    # Toy environments
    "two_rooms": {
        "action_dim": 2,
        "proprio_dim": 2,
        "state_dim": 2,
        "num_channels": 2,
        "img_size": 65,
        "action_mean": [0.0, 0.0],
        "action_std": [1.0, 1.0],
        "proprio_mean": [31.5863, 32.0618],
        "proprio_std": [16.1025, 16.1353],
        "state_mean": [0.0026, 0.0989],
        "state_std": [0.0369, 0.2986],
    },
    # Simulation environments
    "pusht": {
        "action_dim": 2,
        "proprio_dim": 4,  # with_velocity=True: 2 pos + 2 vel
        "state_dim": 7,
        "num_channels": 3,
        "img_size": 224,
        "action_mean": [-0.008700000122189522, 0.006800000090152025],
        "action_std": [0.20190000534057617, 0.20020000636577606],
        "proprio_mean": [236.61549377441406, 264.5674133300781, -2.9303202629089355, 2.543079137802124],
        "proprio_std": [101.12020111083984, 87.01119995117188, 74.8455581665039, 74.14009094238281],
        "state_mean": [236.61549377441406, 264.5674133300781, 255.13070678710938, 266.3721008300781, 1.958400011062622, -2.9303202629089355, 2.543079137802124],
        "state_std": [101.12020111083984, 87.01119995117188, 52.70539855957031, 57.497100830078125, 1.7555999755859375, 74.8455581665039, 74.14009094238281],
    },
    "pointmaze": {
        "action_dim": 2,
        "proprio_dim": 4,
        "state_dim": 4,
        "num_channels": 3,
        "img_size": 224,
        "action_mean": [7.821338658686727e-05, 0.0006003659800626338],
        "action_std": [0.5769129991531372, 0.577587902545929],
        "proprio_mean": [1.8130563497543335, 1.9377127885818481, -0.0039275349117815495, -0.019777977839112282],
        "proprio_std": [1.0055010318756104, 0.9630089402198792, 1.6802769899368286, 1.9319307804107666],
        "state_mean": [1.8130563497543335, 1.9377127885818481, -0.0039275349117815495, -0.019777977839112282],
        "state_std": [1.0055010318756104, 0.9630089402198792, 1.6802769899368286, 1.9319307804107666],
    },
    # Real robot environments (normalize_action=False, so mean=0, std=1)
    "droid": {
        "action_dim": 7,  # 3 pos + 3 euler + 1 gripper
        "proprio_dim": 7,
        "state_dim": 7,
        "num_channels": 3,
        "img_size": 224,
        "action_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "action_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "proprio_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "proprio_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "state_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "robocasa": {
        "action_dim": 7,  # Same format as DROID
        "proprio_dim": 7,
        "state_dim": 7,
        "num_channels": 3,
        "img_size": 224,
        "action_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "action_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "proprio_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "proprio_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "state_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
}
# fmt: on


def get_data_dims(env_name: str) -> tuple[int, int]:
    """Get the action_dim and proprio_dim for a given environment.

    Args:
        env_name: Environment name (e.g., 'droid', 'pusht', 'two_rooms')

    Returns:
        Tuple of (action_dim, proprio_dim)

    Raises:
        KeyError: If the environment is not in the registry
    """
    env_lower = env_name.lower()
    if env_lower not in DATA_STATS:
        raise KeyError(
            f"Unknown environment: {env_name}. "
            f"Available environments: {list(DATA_STATS.keys())}"
        )
    stats = DATA_STATS[env_lower]
    return stats["action_dim"], stats["proprio_dim"]


def get_data_stats(env_name: str) -> dict:
    """Get all data statistics for a given environment.

    Args:
        env_name: Environment name (e.g., 'droid', 'pusht', 'two_rooms')

    Returns:
        Dictionary with action_dim

    Raises:
        KeyError: If the environment is not in the registry
    """
    env_lower = env_name.lower()
    if env_lower not in DATA_STATS:
        raise KeyError(
            f"Unknown environment: {env_name}. "
            f"Available environments: {list(DATA_STATS.keys())}"
        )
    return DATA_STATS[env_lower].copy()
