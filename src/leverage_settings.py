"""
Centralised leverage configuration utilities.

Provides a single source of truth for leverage-related parameters such as the
annualised financing cost, effective trading days per year, and the maximum
gross exposure multiplier. Modules throughout the repository import this module
to guarantee consistent assumptions about leverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


DEFAULT_ANNUAL_LEVERAGE_COST = 0.0675  # 6.75% annualised financing rate
DEFAULT_TRADING_DAYS = 252
DEFAULT_MAX_GROSS_LEVERAGE = 2.0


def _parse_float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not (value == value):  # NaN check
        return default
    return value


def _parse_int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, value)


@dataclass(frozen=True)
class LeverageSettings:
    """Container for globally shared leverage parameters."""

    annual_cost: float = DEFAULT_ANNUAL_LEVERAGE_COST
    trading_days_per_year: int = DEFAULT_TRADING_DAYS
    max_gross_leverage: float = DEFAULT_MAX_GROSS_LEVERAGE

    @property
    def daily_cost(self) -> float:
        return self.annual_cost / self.trading_days_per_year


_OVERRIDE_SETTINGS: Optional[LeverageSettings] = None


def set_leverage_settings(settings: Optional[LeverageSettings]) -> None:
    """Override the global leverage parameters for the current process."""
    global _OVERRIDE_SETTINGS
    _OVERRIDE_SETTINGS = settings


def reset_leverage_settings() -> None:
    """Reset leverage settings to rely on environment/default values."""
    set_leverage_settings(None)


def get_leverage_settings() -> LeverageSettings:
    """
    Return the active leverage configuration.

    Order of precedence:
        1. Settings registered via :func:`set_leverage_settings`.
        2. Environment variables:
           - ``LEVERAGE_COST_ANNUAL`` for the annual financing rate.
           - ``LEVERAGE_TRADING_DAYS`` for the trading days per year.
           - ``GLOBAL_MAX_GROSS_LEVERAGE`` for the gross exposure cap.
        3. The defaults defined at module level.
    """
    if _OVERRIDE_SETTINGS is not None:
        return _OVERRIDE_SETTINGS

    annual = _parse_float_env("LEVERAGE_COST_ANNUAL", DEFAULT_ANNUAL_LEVERAGE_COST)
    trading_days = _parse_int_env("LEVERAGE_TRADING_DAYS", DEFAULT_TRADING_DAYS)
    max_leverage = _parse_float_env("GLOBAL_MAX_GROSS_LEVERAGE", DEFAULT_MAX_GROSS_LEVERAGE)
    max_leverage = max(1.0, max_leverage)
    return LeverageSettings(
        annual_cost=annual,
        trading_days_per_year=trading_days,
        max_gross_leverage=max_leverage,
    )


__all__ = [
    "LeverageSettings",
    "DEFAULT_ANNUAL_LEVERAGE_COST",
    "DEFAULT_TRADING_DAYS",
    "DEFAULT_MAX_GROSS_LEVERAGE",
    "get_leverage_settings",
    "set_leverage_settings",
    "reset_leverage_settings",
]
