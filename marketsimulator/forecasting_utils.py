from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .logging_utils import logger
from .state import SimulationState


def export_price_history(
    state: SimulationState,
    destination: Path,
    padding: int = 0,
) -> None:
    """
    Write the simulated price history for all symbols to ``destination``.

    The exported CSVs mimic the structure expected by the real forecasting
    pipeline so we can reuse ``predict_stock_forecasting`` without major
    changes.  ``padding`` controls how many additional rows beyond the current
    cursor are included to provide enough context for validation windows.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extra = max(0, padding)

    for symbol, series in state.prices.items():
        frame = series.frame
        if frame.empty:
            logger.warning(f"[sim] No price data available for {symbol}; skipping export")
            continue

        end_idx = min(len(frame), series.cursor + 1 + extra)
        if end_idx <= 0:
            logger.warning(f"[sim] Unable to export data for {symbol}; invalid cursor {series.cursor}")
            continue

        export_frame = frame.iloc[:end_idx].copy()
        # Ensure timestamps are ISO formatted strings so downstream code can parse them.
        for column in export_frame.columns:
            if pd.api.types.is_datetime64_any_dtype(export_frame[column]):
                export_frame[column] = export_frame[column].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        export_path = destination / f"{symbol}.csv"
        export_frame.to_csv(export_path, index=False)
