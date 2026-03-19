from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pufferlib_market.export_data_daily import compute_daily_features, load_price_data


@dataclass(frozen=True)
class CoverageRow:
    symbol: str
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    num_rows: int


@dataclass(frozen=True)
class AlignedDailyData:
    index: pd.DatetimeIndex
    symbols: list[str]
    prices: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame]
    tradable: dict[str, np.ndarray]


@dataclass(frozen=True)
class LatestSnapshot:
    as_of: pd.Timestamp
    feature_matrix: np.ndarray
    prices: dict[str, float]
    tradable: dict[str, bool]


def summarize_symbol_coverage(
    symbols: list[str],
    *,
    data_root: str | Path = "trainingdata/train",
) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    data_root = Path(data_root)
    for sym in symbols:
        price_df = load_price_data(sym, data_root)
        rows.append(
            CoverageRow(
                symbol=sym.upper(),
                first_date=pd.Timestamp(price_df.index.min()),
                last_date=pd.Timestamp(price_df.index.max()),
                num_rows=int(len(price_df)),
            )
        )
    return rows


def align_daily_price_frames(
    symbols: list[str],
    *,
    data_root: str | Path = "trainingdata/train",
    start_date: str | None = None,
    end_date: str | None = None,
    min_days: int = 1,
) -> AlignedDailyData:
    symbols = [str(s).upper() for s in symbols]
    if not symbols:
        raise ValueError("No symbols provided")

    data_root = Path(data_root)
    original_prices = {sym: load_price_data(sym, data_root) for sym in symbols}

    starts = [df.index.min() for df in original_prices.values()]
    ends = [df.index.max() for df in original_prices.values()]
    start = max(starts)
    end = min(ends)
    if start_date is not None:
        start = max(start, pd.to_datetime(start_date, utc=True))
    if end_date is not None:
        end = min(end, pd.to_datetime(end_date, utc=True))
    if start >= end:
        raise ValueError(f"Invalid date window: start={start} end={end}")

    full_index = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    if len(full_index) < int(min_days):
        raise ValueError(f"Not enough aligned days: {len(full_index)} < {min_days}")

    aligned_prices: dict[str, pd.DataFrame] = {}
    aligned_features: dict[str, pd.DataFrame] = {}
    tradable: dict[str, np.ndarray] = {}

    for sym, raw in original_prices.items():
        mask = full_index.isin(raw.index).astype(np.uint8)
        aligned = raw.reindex(full_index, method="ffill")
        aligned["volume"] = aligned["volume"].where(mask.astype(bool), 0.0)
        aligned = aligned.bfill().fillna(0.0)
        aligned_prices[sym] = aligned
        aligned_features[sym] = compute_daily_features(aligned)
        tradable[sym] = mask.astype(bool, copy=False)

    return AlignedDailyData(
        index=full_index,
        symbols=symbols,
        prices=aligned_prices,
        features=aligned_features,
        tradable=tradable,
    )


def latest_snapshot(aligned: AlignedDailyData) -> LatestSnapshot:
    if len(aligned.index) < 1:
        raise ValueError("Aligned data is empty")
    as_of = pd.Timestamp(aligned.index[-1])
    feature_rows = []
    prices: dict[str, float] = {}
    tradable: dict[str, bool] = {}
    for sym in aligned.symbols:
        feature_rows.append(aligned.features[sym].iloc[-1].to_numpy(dtype=np.float32, copy=False))
        prices[sym] = float(aligned.prices[sym].iloc[-1]["close"])
        tradable[sym] = bool(aligned.tradable[sym][-1])
    return LatestSnapshot(
        as_of=as_of,
        feature_matrix=np.vstack(feature_rows).astype(np.float32, copy=False),
        prices=prices,
        tradable=tradable,
    )
