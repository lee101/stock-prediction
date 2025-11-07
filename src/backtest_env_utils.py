"""Environment variable parsing and configuration utilities for backtesting."""

import os
from typing import Iterable, Optional


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def read_env_flag(names: Iterable[str]) -> Optional[bool]:
    """Read boolean flag from one or more environment variables.

    Args:
        names: Iterable of environment variable names to check

    Returns:
        True if any variable is truthy, False if any is falsy, None if not set
    """
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    return None


def coerce_keepalive_seconds(env_name: str, *, default: float, logger=None) -> float:
    """Parse keepalive seconds from environment variable with validation.

    Args:
        env_name: Name of environment variable to read
        default: Default value if not set or invalid
        logger: Optional logger for warnings

    Returns:
        Parsed keepalive seconds or default value
    """
    value = os.getenv(env_name)
    if value is None or not value.strip():
        return float(default)
    try:
        seconds = float(value)
    except ValueError:
        if logger:
            logger.warning(f"Ignoring invalid {env_name}={value!r}; expected number of seconds.")
        return float(default)
    if seconds < 0.0:
        if logger:
            logger.warning(f"Ignoring negative {env_name}={value!r}; defaulting to {default:.1f}.")
        return float(default)
    return seconds


def cpu_fallback_enabled(env_name: str = "MARKETSIM_ALLOW_CPU_FALLBACK") -> bool:
    """Check if CPU fallback mode is enabled.

    Args:
        env_name: Name of environment variable to check

    Returns:
        True if CPU fallback is enabled
    """
    value = os.getenv(env_name)
    if value is None:
        return False
    return value.strip().lower() in _BOOL_TRUE


def in_test_mode() -> bool:
    """Check if running in test mode.

    Returns:
        True when unit-test machinery requests lightweight behavior
    """
    test_flag = os.getenv("TESTING")
    if test_flag is not None and test_flag.strip().lower() in _BOOL_TRUE:
        return True
    mock_flag = os.getenv("MARKETSIM_ALLOW_MOCK_ANALYTICS")
    if mock_flag is not None and mock_flag.strip().lower() in _BOOL_TRUE:
        return True
    return False
