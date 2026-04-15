"""Utilities for loading and processing YAML configuration files."""

import os
import re

from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)

_ENV_FALLBACKS = {"EBJEPA_DATA": "EBJEPA_DSETS"}


def expand_env_vars(value, _path: str = ""):
    """Recursively expand ``${VAR_NAME}`` placeholders in YAML values.

    Walks dicts, lists, and strings. Logs successful expansions and warns
    (keeping the original placeholder) when an env var is not set.

    Args:
        value: The value to expand (str, dict, list, or passthrough).
        _path: Internal breadcrumb for log messages (e.g. ``"data_path"``).

    Returns:
        The value with environment variables expanded.
    """
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"

        def _replace(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None and var_name in _ENV_FALLBACKS:
                fallback_var = _ENV_FALLBACKS[var_name]
                env_value = os.environ.get(fallback_var)
                if env_value is not None:
                    logger.info(
                        f"'{var_name}' not set, falling back to "
                        f"'{fallback_var}'='{env_value}'"
                    )
            if env_value is None:
                logger.warning(
                    f"Environment variable '{var_name}' not found"
                    f"{' at ' + _path if _path else ''}. "
                    f"Keeping original placeholder: {match.group(0)}"
                )
                return match.group(0)
            logger.info(
                f"Expanded environment variable '{var_name}' to '{env_value}'"
                f"{' at ' + _path if _path else ''}"
            )
            return env_value

        return re.sub(pattern, _replace, value)
    elif isinstance(value, dict):
        return {
            k: expand_env_vars(v, f"{_path}.{k}" if _path else k)
            for k, v in value.items()
        }
    elif isinstance(value, list):
        return [expand_env_vars(item, f"{_path}[{i}]") for i, item in enumerate(value)]
    return value
