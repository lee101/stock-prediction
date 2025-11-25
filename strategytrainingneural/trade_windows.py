from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import augment_metrics
from src.fixtures import all_crypto_symbols


SECONDS_PER_DAY = 86_400


def _normalize_paths(parquet_paths: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in parquet_paths:
        expanded = list(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in expanded:
            if path.exists():
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No parquet files found for patterns: {parquet_paths}")
    return paths


def _is_crypto(symbol: str) -> bool:
    return symbol.upper() in all_crypto_symbols


def _compute_roll_metric(series: pd.Series, window: int, func, default: float = 0.0) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=np.float32)
    return series.rolling(window=window, min_periods=1).apply(func, raw=True).fillna(default)


def _sortino_helper(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = values.mean()
    downside = np.sqrt(np.mean(np.square(np.minimum(values, 0.0))) + 1e-12)
    if downside == 0.0:
        return 0.0
    return float(np.clip(mean / downside, -5.0, 5.0))


def _sharpe_helper(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    std = values.std()
    if std == 0.0:
        return 0.0
    return float(np.clip(values.mean() / std, -5.0, 5.0))


def _annual_return_helper(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + values))
    if growth <= 0:
        return -1.0
    ann = growth ** (252.0 / max(len(values), 1)) - 1.0
    return float(np.clip(ann, -5.0, 5.0))


def _aggregate_group(
    df: pd.DataFrame,
    window_days: int,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Convert per-trade rows for a single symbol+strategy into daily metrics.
    """

    df = df.copy()
    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"], utc=True)
    df = df.sort_values("exit_timestamp")
    if df.empty:
        return pd.DataFrame(columns=["date"])
    df["date"] = df["exit_timestamp"].dt.floor("D")
    daily = (
        df.groupby("date")
        .agg(
            daily_pnl=("pnl", "sum"),
            daily_return=("pnl_pct", "sum"),
            trades=("pnl", "count"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["daily_return"] = daily["daily_return"].clip(-0.95, 0.95)
    daily["day_of_week"] = daily["date"].dt.day_name()
    daily["capital"] = initial_capital * (1.0 + daily["daily_return"]).cumprod()
    daily["rolling_sharpe"] = _compute_roll_metric(
        daily["daily_return"],
        window_days,
        _sharpe_helper,
    )
    daily["rolling_sortino"] = _compute_roll_metric(
        daily["daily_return"],
        window_days,
        _sortino_helper,
    )
    daily["rolling_ann_return"] = _compute_roll_metric(
        daily["daily_return"],
        window_days,
        _annual_return_helper,
    )
    daily["annualization_days"] = float(window_days)
    return daily


FORECAST_COLUMNS = [
    "forecast_move_pct",
    "forecast_volatility_pct",
    "predicted_close",
    "predicted_close_p10",
    "predicted_close_p90",
    "predicted_high",
    "predicted_low",
    "context_close",
]


def load_trade_window_metrics(
    parquet_paths: Sequence[str],
    *,
    symbols: Optional[Sequence[str]] = None,
    asset_class: str = "all",
    window_days: int = 10,
    min_trades: int = 5,
    forecast_cache_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a neural-training-friendly DataFrame out of short trade snapshots.

    Args:
        parquet_paths: glob patterns or explicit Parquet files from strategytraining/datasets.
        symbols: optional subset of symbols to keep (case-insensitive).
        asset_class: one of ``all``, ``stock``, ``crypto`` for filtering.
        window_days: length of the rolling window metrics (default 10 days).
        min_trades: drop symbol/strategy combos with fewer than this many trades.
    """

    allowed_symbols = {s.upper() for s in symbols} if symbols else None
    asset_class = asset_class.lower()
    paths = _normalize_paths(parquet_paths)
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        required = {"symbol", "strategy", "exit_timestamp", "pnl", "pnl_pct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        df = df[list(required)].copy()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        if allowed_symbols is not None:
            df = df[df["symbol"].isin(allowed_symbols)]
        if df.empty:
            continue
        df["is_crypto"] = df["symbol"].apply(_is_crypto)
        if asset_class == "stock":
            df = df[~df["is_crypto"]]
        elif asset_class == "crypto":
            df = df[df["is_crypto"]]
        df["strategy"] = df["strategy"].astype(str)
        frames.append(df)
    if not frames:
        raise ValueError("No trades matched the provided filters; cannot build dataset.")
    combined = pd.concat(frames, ignore_index=True)

    samples: List[pd.DataFrame] = []
    for (symbol, strategy), group in combined.groupby(["symbol", "strategy"]):
        if len(group) < min_trades:
            continue
        daily = _aggregate_group(group, window_days=window_days)
        if daily.empty:
            continue
        daily["strategy"] = f"{symbol}:{strategy}"
        daily["symbol"] = symbol
        daily["mode"] = "normal"
        daily["gate_config"] = "-"
        daily["day_class"] = "crypto" if _is_crypto(symbol) else "stock"
        samples.append(daily)

    if not samples:
        raise ValueError("No symbol/strategy cohorts produced enough data after filtering.")

    dataset = pd.concat(samples, ignore_index=True)
    dataset = dataset[
        [
            "strategy",
            "symbol",
            "date",
            "capital",
            "daily_return",
            "rolling_sharpe",
            "rolling_sortino",
            "rolling_ann_return",
            "daily_pnl",
            "mode",
            "day_class",
            "gate_config",
            "annualization_days",
            "day_of_week",
        ]
    ]
    dataset["date"] = pd.to_datetime(dataset["date"])
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        dataset = dataset[dataset["date"] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        dataset = dataset[dataset["date"] <= end_ts]

    if forecast_cache_dir:
        forecast_df = _load_forecast_frames(Path(forecast_cache_dir), dataset["symbol"].unique())
        if not forecast_df.empty:
            dataset = dataset.merge(forecast_df, on=["symbol", "date"], how="left")
        for column in FORECAST_COLUMNS:
            if column not in dataset.columns:
                dataset[column] = 0.0
            dataset[column] = dataset[column].fillna(0.0)

    dataset = dataset.replace([np.inf, -np.inf], 0.0)
    dataset = dataset.sort_values(["strategy", "date"]).reset_index(drop=True)
    dataset = augment_metrics(dataset)
    return dataset


def _load_forecast_frames(cache_dir: Path, symbols: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        safe = symbol.replace("/", "_").replace("\\", "_")
        path = cache_dir / f"{safe}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["symbol"] = symbol
        frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("D")
        frames.append(frame[["symbol", "date", *FORECAST_COLUMNS]])
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", *FORECAST_COLUMNS])
    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "date"])
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return merged


__all__ = ["load_trade_window_metrics", "FORECAST_COLUMNS"]
