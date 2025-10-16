from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .state import PriceSeries


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "tototraining" / "trainingdata"


def _read_symbol_file(symbol: str, data_root: Path) -> Optional[pd.DataFrame]:
    candidates = [
        data_root / "train" / f"{symbol}.csv",
        data_root / "test" / f"{symbol}.csv",
    ]
    frames = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _synthetic_series(symbol: str, periods: int = 512) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    timestamp = pd.date_range("2024-01-01", periods=periods, freq="h")
    price = 100 + rng.standard_normal(periods).cumsum()
    high = price + np.abs(rng.normal(0, 0.5, periods))
    low = price - np.abs(rng.normal(0, 0.5, periods))
    open_price = price + rng.normal(0, 0.2, periods)
    volume = np.abs(rng.normal(1000, 100, periods)).astype(int)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": price,
            "Volume": volume,
        }
    )


def load_price_series(
    symbols: Iterable[str],
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Dict[str, PriceSeries]:
    series: Dict[str, PriceSeries] = {}
    data_root = data_root.resolve()
    for symbol in symbols:
        frame = _read_symbol_file(symbol, data_root)
        if frame is None:
            frame = _synthetic_series(symbol)
        series[symbol] = PriceSeries(symbol=symbol, frame=frame)
    return series
