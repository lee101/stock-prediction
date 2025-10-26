from __future__ import annotations

import pandas as pd


def infer_freq(timestamps: pd.Series) -> pd.Timedelta:
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to infer frequency")
    diffs = timestamps.diff().dropna()
    return pd.Timedelta(diffs.mode().iloc[0])
