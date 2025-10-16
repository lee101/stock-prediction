"""Helpers for preparing OHLC frames for prompts."""

from __future__ import annotations

import pandas as pd


def add_ohlc_percent_change(
    df: pd.DataFrame,
    *,
    price_columns: tuple[str, ...] = ("open", "high", "low", "close"),
    baseline_column: str = "close",
) -> pd.DataFrame:
    """Return copy with *_pct columns relative to previous close."""
    if baseline_column not in df.columns:
        raise ValueError(f"Baseline column '{baseline_column}' not found in dataframe")
    pct_df = df.sort_index().copy()
    baseline = pct_df[baseline_column].shift(1)
    for col in price_columns:
        if col not in pct_df.columns:
            continue
        change = (pct_df[col] - baseline) / baseline
        change = change.where(baseline.notna() & (baseline != 0), 0.0)
        pct_df[f"{col}_pct"] = change.fillna(0.0)
    return pct_df
