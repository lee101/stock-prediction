"""Standardized environment variable parsing utilities.

This module consolidates environment variable parsing logic that was
duplicated across trade_stock_e2e.py, trade_stock_e2e_hourly.py, and
backtest_test3_inline.py.
"""

import os


# Standard truthy values for boolean environment variables
TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}


def parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.

    Args:
        name: Environment variable name
        default: Default value if not set or empty

    Returns:
        Boolean value based on environment variable or default

    Examples:
        >>> os.environ["ENABLE_FEATURE"] = "1"
        >>> parse_bool_env("ENABLE_FEATURE")
        True
        >>> parse_bool_env("MISSING_VAR", default=False)
        False
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    return normalized in TRUTHY_VALUES


def parse_int_env(
    name: str,
    default: int = 0,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Parse an integer environment variable with optional bounds.

    Args:
        name: Environment variable name
        default: Default value if not set, empty, or invalid
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Integer value, clamped to bounds if provided

    Examples:
        >>> os.environ["MAX_WORKERS"] = "8"
        >>> parse_int_env("MAX_WORKERS", default=4)
        8
        >>> parse_int_env("MAX_WORKERS", default=4, min_val=1, max_val=4)
        4
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        parsed = int(value.strip())
    except (ValueError, TypeError):
        return default

    if min_val is not None:
        parsed = max(parsed, min_val)
    if max_val is not None:
        parsed = min(parsed, max_val)

    return parsed


def parse_float_env(
    name: str,
    default: float = 0.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Parse a float environment variable with optional bounds.

    Args:
        name: Environment variable name
        default: Default value if not set, empty, or invalid
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Float value, clamped to bounds if provided

    Examples:
        >>> os.environ["THRESHOLD"] = "0.75"
        >>> parse_float_env("THRESHOLD", default=0.5)
        0.75
        >>> parse_float_env("THRESHOLD", default=0.5, min_val=0.0, max_val=1.0)
        0.75
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        parsed = float(value.strip())
    except (ValueError, TypeError):
        return default

    if min_val is not None:
        parsed = max(parsed, min_val)
    if max_val is not None:
        parsed = min(parsed, max_val)

    return parsed


def parse_enum_env(name: str, allowed: list[str], default: str) -> str:
    """Parse an environment variable that must be one of allowed values.

    Args:
        name: Environment variable name
        allowed: List of allowed values (case-insensitive)
        default: Default value if not set or not in allowed list

    Returns:
        Normalized value from allowed list, or default

    Examples:
        >>> os.environ["LOG_LEVEL"] = "INFO"
        >>> parse_enum_env("LOG_LEVEL", ["DEBUG", "INFO", "WARNING"], "INFO")
        'info'
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default.lower()

    normalized = value.strip().lower()
    allowed_lower = [val.lower() for val in allowed]

    if normalized in allowed_lower:
        return normalized

    return default.lower()


def parse_positive_int_env(name: str, default: int = 1) -> int:
    """Parse a positive integer environment variable.

    Args:
        name: Environment variable name
        default: Default value if not set, empty, invalid, or not positive

    Returns:
        Positive integer value (>= 1)

    Examples:
        >>> os.environ["MAX_RETRIES"] = "3"
        >>> parse_positive_int_env("MAX_RETRIES", default=1)
        3
        >>> os.environ["MAX_RETRIES"] = "0"
        >>> parse_positive_int_env("MAX_RETRIES", default=1)
        1
    """
    value = parse_int_env(name, default=default, min_val=1)
    return max(1, value)


def parse_positive_float_env(name: str, default: float = 1.0) -> float:
    """Parse a positive float environment variable.

    Args:
        name: Environment variable name
        default: Default value if not set, empty, invalid, or not positive

    Returns:
        Positive float value (> 0.0)

    Examples:
        >>> os.environ["TIMEOUT"] = "30.5"
        >>> parse_positive_float_env("TIMEOUT", default=10.0)
        30.5
    """
    value = parse_float_env(name, default=default, min_val=0.0)
    return max(0.0, value) if value == 0.0 and default > 0.0 else value
