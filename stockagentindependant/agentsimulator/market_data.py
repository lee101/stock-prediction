"""Utilities for assembling OHLC percent-change data (stateless agent)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger

from stock_data_utils import add_ohlc_percent_change

from ..constants import DEFAULT_SYMBOLS

DEFAULT_LOCAL_DATA_DIR = Path("trainingdata")
FALLBACK_DATA_DIRS = [
    Path("trainingdata/stockagent/marketdata"),
    Path("stockagentindependant_market_data"),
    Path("stockagent_market_data"),
    Path("trainingdata/marketdata"),
]


@dataclass
class MarketDataBundle:
    bars: Dict[str, pd.DataFrame]
    lookback_days: int
    as_of: datetime

    def get_symbol_bars(self, symbol: str) -> pd.DataFrame:
        return self.bars.get(symbol.upper(), pd.DataFrame()).copy()

    def trading_days(self) -> List[pd.Timestamp]:
        for df in self.bars.values():
            if not df.empty:
                return list(df.index)
        return []

    def to_payload(self, limit: Optional[int] = None) -> Dict[str, List[Dict[str, float]]]:
        payload: Dict[str, List[Dict[str, float]]] = {}
        for symbol, df in self.bars.items():
            frame = df.tail(limit) if limit else df
            frame_with_pct = add_ohlc_percent_change(frame)
            payload[symbol] = [
                {
                    "timestamp": row.name.isoformat(),
                    "open_pct": float(row["open_pct"]),
                    "high_pct": float(row["high_pct"]),
                    "low_pct": float(row["low_pct"]),
                    "close_pct": float(row["close_pct"]),
                    "close": float(row.get("close", 0.0)),
                }
                for _, row in frame_with_pct.iterrows()
            ]
        return payload


def fetch_latest_ohlc(
    symbols: Optional[Iterable[str]] = None,
    lookback_days: int = 60,
    as_of: Optional[datetime] = None,
    local_data_dir: Optional[Path] = DEFAULT_LOCAL_DATA_DIR,
    allow_remote_download: bool = False,
) -> MarketDataBundle:
    symbols = [str(symbol).upper() for symbol in (symbols or DEFAULT_SYMBOLS)]
    as_of = as_of or datetime.now(timezone.utc)
    start = as_of - timedelta(days=max(lookback_days * 2, 30))

    candidate_dirs: List[Path] = []
    if local_data_dir:
        candidate_dirs.append(Path(local_data_dir))
    candidate_dirs.extend(FALLBACK_DATA_DIRS)
    unique_dirs: List[Path] = []
    for path in candidate_dirs:
        path = Path(path)
        if path not in unique_dirs:
            unique_dirs.append(path)
    existing_dirs = [path for path in unique_dirs if path.exists()]
    for missing in [path for path in unique_dirs if not path.exists()]:
        logger.debug(f"Local market data dir {missing} not found.")
    if not existing_dirs:
        logger.warning("No local market data directories available; continuing without cached OHLC data.")

    bars: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = pd.DataFrame()
        for directory in existing_dirs:
            df = _load_local_symbol_data(symbol, directory)
            if not df.empty:
                break
        if df.empty and allow_remote_download:
            df = pd.DataFrame()  # this independent stack stays offline
        df = _ensure_datetime_index(df).tail(lookback_days)
        bars[symbol] = df

    return MarketDataBundle(bars=bars, lookback_days=lookback_days, as_of=as_of)


def _load_local_symbol_data(symbol: str, directory: Path) -> pd.DataFrame:
    normalized_symbol = symbol.replace("/", "-")
    patterns = [
        f"{normalized_symbol}*.parquet",
        f"{normalized_symbol}*.pq",
        f"{normalized_symbol}*.csv",
        f"{normalized_symbol}*.json",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(Path(directory).glob(pattern))
    if not candidates:
        return pd.DataFrame()
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        if latest.suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(latest)
        elif latest.suffix == ".json":
            df = pd.read_json(latest)
        else:
            df = pd.read_csv(latest)
    except Exception as exc:
        logger.warning(f"Failed to load {symbol} data from {latest}: {exc}")
        return pd.DataFrame()
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={"time": "timestamp", "date": "timestamp", "datetime": "timestamp"})
    return df


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "timestamp" not in df.columns:
        logger.warning("Received OHLC frame without timestamp column; skipping dataset")
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df
