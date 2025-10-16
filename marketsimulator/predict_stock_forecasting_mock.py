from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .state import get_state


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_stock_data_from_csv(csv_file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def make_predictions(input_data_path: Optional[str] = None, pred_name: str = "", retrain: bool = False, alpaca_wrapper=None) -> pd.DataFrame:
    state = get_state()
    records = []
    for symbol, series in state.prices.items():
        current_row = series.current_row
        target_idx = min(series.cursor + 1, len(series.frame) - 1)
        next_row = series.frame.iloc[target_idx]
        close_pred = float(next_row.get("Close", current_row.get("Close", 0.0)))
        high_pred = float(next_row.get("High", close_pred))
        low_pred = float(next_row.get("Low", close_pred))
        records.append(
            {
                "instrument": symbol,
                "close_predicted_price": close_pred,
                "high_predicted_price": high_pred,
                "low_predicted_price": low_pred,
                "entry_takeprofit_profit": (high_pred - close_pred) / close_pred if close_pred else 0.0,
                "maxdiffprofit_profit": (high_pred - low_pred) / close_pred if close_pred else 0.0,
                "takeprofit_profit": (high_pred - current_row.get("Close", close_pred)) / close_pred if close_pred else 0.0,
                "generated_at": state.clock.current,
            }
        )
    predictions = pd.DataFrame(records)
    predictions.to_csv(RESULTS_DIR / "predictions-sim.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    return predictions
