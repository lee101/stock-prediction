from __future__ import annotations

import pandas as pd

NO_FORECAST_FALLBACK_SCALE = 0.5
ATR_WINDOW = 14


def compute_position_scale(
    forecast_p10: float,
    forecast_p50: float,
    forecast_p90: float,
    current_price: float,
    atr_pct: float,
    base_mae_pct: float,
) -> float:
    """Return 0.0-1.0 scale factor for position sizing.

    Tight CI spread + high edge = larger position.
    Wide CI spread + low edge = smaller position.
    ATR normalizes across symbols.
    """
    if current_price <= 0 or base_mae_pct <= 0:
        return 0.0
    if forecast_p10 == 0 and forecast_p50 == 0 and forecast_p90 == 0:
        return 0.0

    ci_spread = (forecast_p90 - forecast_p10) / current_price
    edge = abs(forecast_p50 - current_price) / current_price
    normalized_edge = edge / base_mae_pct
    signal_quality = normalized_edge / (ci_spread / base_mae_pct + 1.0)
    scale = max(0.0, min(signal_quality, 1.0))

    if atr_pct > 2.0 * base_mae_pct:
        scale *= 0.5

    return scale


def compute_atr_pct(bars_df: pd.DataFrame, timestamp: pd.Timestamp, window: int = ATR_WINDOW) -> float:
    """ATR as fraction of price from bars up to timestamp."""
    prior = bars_df[bars_df["timestamp"] <= timestamp].tail(window)
    if len(prior) < 2:
        return 0.0
    tr = (prior["high"] - prior["low"]).abs()
    atr = float(tr.mean())
    mid = float(prior["close"].iloc[-1])
    if mid <= 0:
        return 0.0
    return atr / mid
