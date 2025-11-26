"""Format market data as terse pct-change lines.

Format per day: close_pct,high_pct,low_pct
- close_pct: % change from previous close (5dp precision)
- high_pct: % above close (intraday high relative to close)
- low_pct: % below close (intraday low relative to close, negative)

Example line: +0.01234,+0.02100,-0.01500
Means: close up 1.234%, intraday high 2.1% above close, low 1.5% below close
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class PctLineData:
    """Container for formatted pct-change line data."""
    symbol: str
    last_close: float
    lines: str  # Terse multi-line string of pct changes
    num_days: int


def _format_pct(val: float) -> str:
    """Format percentage to 5dp with sign, very terse."""
    if val >= 0:
        return f"+{val:.5f}"
    return f"{val:.5f}"


def format_pctline_data(
    df: pd.DataFrame,
    symbol: str,
    max_days: int = 1000,
) -> PctLineData:
    """Convert OHLC DataFrame to terse pct-change lines.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        symbol: Stock symbol
        max_days: Maximum number of days to include (most recent)

    Returns:
        PctLineData with formatted lines
    """
    # Take most recent max_days
    df = df.tail(max_days + 1).copy()  # +1 to compute pct change

    if len(df) < 2:
        return PctLineData(symbol=symbol, last_close=0.0, lines="", num_days=0)

    # Compute pct changes
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # close_pct: % change from previous close
    close_pct = np.diff(close) / close[:-1]

    # high_pct: % that high was above the close of that day
    # (how much upside happened intraday)
    high_pct = (high[1:] - close[1:]) / close[1:]

    # low_pct: % that low was below the close of that day
    # (how much downside happened intraday, typically negative)
    low_pct = (low[1:] - close[1:]) / close[1:]

    # Build terse lines
    lines = []
    for i in range(len(close_pct)):
        line = f"{_format_pct(close_pct[i])},{_format_pct(high_pct[i])},{_format_pct(low_pct[i])}"
        lines.append(line)

    return PctLineData(
        symbol=symbol,
        last_close=float(close[-1]),
        lines="\n".join(lines),
        num_days=len(lines),
    )


def format_multi_symbol_data(
    dfs: dict[str, pd.DataFrame],
    max_days: int = 1000,
) -> dict[str, PctLineData]:
    """Format data for multiple symbols."""
    return {
        symbol: format_pctline_data(df, symbol, max_days)
        for symbol, df in dfs.items()
    }


def compute_daily_stats(df: pd.DataFrame) -> dict:
    """Compute summary stats for the data."""
    close = df["close"].values
    if len(close) < 2:
        return {}

    pct_changes = np.diff(close) / close[:-1]

    return {
        "mean_daily_return": float(np.mean(pct_changes)),
        "std_daily_return": float(np.std(pct_changes)),
        "max_daily_gain": float(np.max(pct_changes)),
        "max_daily_loss": float(np.min(pct_changes)),
        "positive_days_pct": float(np.mean(pct_changes > 0)),
    }
