from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

PRICE_FEATURES: Tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_FEATURES: Tuple[str, ...] = ("volume", "amount")
TIME_FEATURES: Tuple[str, ...] = ("minute", "hour", "weekday", "day", "month")
ALL_FEATURES: Tuple[str, ...] = PRICE_FEATURES + VOLUME_FEATURES


def list_symbol_files(data_dir: Path | str) -> List[Tuple[str, Path]]:
    """Enumerate CSV files under ``data_dir`` and return (symbol, path) tuples."""
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Training data directory does not exist: {base}")

    csv_paths: List[Tuple[str, Path]] = []
    for path in sorted(base.glob("*.csv")):
        if not path.is_file():
            continue
        symbol = path.stem
        if not symbol.isupper():
            continue
        csv_paths.append((symbol.upper(), path))

    if not csv_paths:
        raise ValueError(f"No CSV files found under {base}")
    return csv_paths


def _ensure_amount_column(df: pd.DataFrame) -> pd.Series:
    volume = df.get("volume")
    if volume is None:
        return pd.Series(np.zeros(len(df)), name="amount", dtype=np.float32)

    price_mean = df[list(PRICE_FEATURES)].mean(axis=1)
    return (price_mean * volume.fillna(0.0)).astype(np.float32)


def load_symbol_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a symbol CSV and normalise columns for Kronos.

    Returns a dataframe sorted by timestamp with the expected Kronos columns.
    """
    df = pd.read_csv(path)
    if "timestamps" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "timestamps"})
        else:
            raise ValueError(f"{path} missing 'timestamp(s)' column")

    for col in PRICE_FEATURES:
        if col not in df.columns:
            raise ValueError(f"{path} missing required price column '{col}'")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=False)
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["amount"] = _ensure_amount_column(df)

    df["minute"] = df["timestamps"].dt.minute.astype(np.float32)
    df["hour"] = df["timestamps"].dt.hour.astype(np.float32)
    df["weekday"] = df["timestamps"].dt.weekday.astype(np.float32)
    df["day"] = df["timestamps"].dt.day.astype(np.float32)
    df["month"] = df["timestamps"].dt.month.astype(np.float32)

    feature_cols = list(ALL_FEATURES + TIME_FEATURES)
    return df[["timestamps", *feature_cols]]


def iter_symbol_dataframes(data_dir: Path | str) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Yield (symbol, dataframe) pairs for each CSV in the directory tree."""
    for symbol, path in list_symbol_files(data_dir):
        yield symbol, load_symbol_dataframe(path)
