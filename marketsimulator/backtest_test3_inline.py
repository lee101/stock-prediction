from __future__ import annotations

import pandas as pd

from .state import get_state


def backtest_forecasts(symbol: str, num_simulations: int = 70) -> pd.DataFrame:
    state = get_state()
    series = state.prices.get(symbol)
    if series is None:
        raise ValueError(f"Unknown symbol {symbol} in simulation")

    frame = series.frame
    end_idx = min(series.cursor + num_simulations, len(frame))
    start_idx = max(0, end_idx - num_simulations)
    window = frame.iloc[start_idx:end_idx].copy()
    if window.empty:
        raise ValueError(f"No data for symbol {symbol}")

    window["close_return"] = window["Close"].pct_change().fillna(0.0)
    window["high_return"] = window["High"].pct_change().fillna(0.0)
    window["low_return"] = window["Low"].pct_change().fillna(0.0)

    predicted_close = window["Close"].shift(-1).fillna(window["Close"])
    predicted_high = window["High"].shift(-1).fillna(window["High"])
    predicted_low = window["Low"].shift(-1).fillna(window["Low"])

    simple = predicted_close.pct_change().fillna(0.0)
    all_signals = (predicted_close + predicted_high + predicted_low) / 3.0
    all_signals = all_signals.pct_change().fillna(0.0)
    takeprofit = (predicted_high - window["Close"]) / window["Close"]
    highlow = (predicted_high - predicted_low) / window["Close"]

    return pd.DataFrame(
        {
            "close": window["Close"],
            "predicted_close": predicted_close,
            "predicted_high": predicted_high,
            "predicted_low": predicted_low,
            "simple_strategy_return": simple,
            "all_signals_strategy_return": all_signals,
            "entry_takeprofit_return": takeprofit,
            "highlow_return": highlow,
        }
    )
