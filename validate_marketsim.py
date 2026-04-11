"""Compatibility shim for the canonical marketsim validation entrypoint.

The maintained implementation lives in ``pufferlib_market.validate_marketsim``.
Keep this top-level module as a thin wrapper so older invocations like
``python validate_marketsim.py`` continue to work without drifting out of sync.
"""

from __future__ import annotations

from pufferlib_market.validate_marketsim import (  # noqa: F401
    DailyPPOTrader,
    FEE_TIERS,
    PPOTrader,
    SLIPPAGE_BPS,
    TradingSignal,
    compute_daily_feature_history,
    compute_hourly_feature_snapshot,
    load_daily_bars,
    load_hourly_bars,
    main,
)

__all__ = (
    "DailyPPOTrader",
    "FEE_TIERS",
    "PPOTrader",
    "SLIPPAGE_BPS",
    "TradingSignal",
    "compute_daily_feature_history",
    "compute_hourly_feature_snapshot",
    "load_daily_bars",
    "load_hourly_bars",
    "main",
)


if __name__ == "__main__":
    main()
