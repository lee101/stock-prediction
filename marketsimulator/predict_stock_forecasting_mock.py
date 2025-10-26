from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .state import get_state


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
_LOOKAHEAD_ENV_KEY = "MARKETSIM_FORECAST_LOOKAHEAD"


def _resolve_lookahead(default: int = 1) -> int:
    raw = os.getenv(_LOOKAHEAD_ENV_KEY)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, value)


def load_stock_data_from_csv(csv_file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def make_predictions(
    input_data_path: Optional[str] = None,
    pred_name: str = "",
    retrain: bool = False,
    alpaca_wrapper=None,
    symbols=None,
) -> pd.DataFrame:
    state = get_state()
    lookahead = _resolve_lookahead()
    records = []
    for symbol, series in state.prices.items():
        current_row = series.current_row
        # Determine the forecast horizon (inclusive of the target index).
        target_idx = min(series.cursor + lookahead, len(series.frame) - 1)
        future_slice = series.frame.iloc[series.cursor + 1 : target_idx + 1]
        if future_slice.empty:
            future_slice = series.frame.iloc[target_idx : target_idx + 1]
        close_pred = float(
            future_slice.get("Close", pd.Series(dtype=float)).iloc[-1]
            if "Close" in future_slice.columns and not future_slice.empty
            else series.frame.iloc[target_idx].get("Close", current_row.get("Close", 0.0))
        )
        if "High" in future_slice.columns and not future_slice.empty:
            high_pred = float(future_slice["High"].max())
        else:
            high_pred = float(series.frame.iloc[target_idx].get("High", close_pred))
        if "Low" in future_slice.columns and not future_slice.empty:
            low_pred = float(future_slice["Low"].min())
        else:
            low_pred = float(series.frame.iloc[target_idx].get("Low", close_pred))
        current_close = float(current_row.get("Close", close_pred))
        records.append(
            {
                "instrument": symbol,
                "close_predicted_price": close_pred,
                "high_predicted_price": high_pred,
                "low_predicted_price": low_pred,
                "entry_takeprofit_profit": (high_pred - close_pred) / close_pred if close_pred else 0.0,
                "maxdiffprofit_profit": (high_pred - low_pred) / close_pred if close_pred else 0.0,
                "takeprofit_profit": (high_pred - current_close) / close_pred if close_pred else 0.0,
                "generated_at": state.clock.current,
            }
        )
    predictions = pd.DataFrame(records)
    predictions.to_csv(RESULTS_DIR / "predictions-sim.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    return predictions
