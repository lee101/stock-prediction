"""Hourly data loading for market simulation."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from marketsimulator.hourly_utils import load_hourly_bars

from .config import DataConfigHourly

logger = logging.getLogger(__name__)

CRYPTO_SUFFIXES = ("USD", "BTC", "ETH", "USDT", "USDC")


def is_crypto_symbol(symbol: str) -> bool:
    return any(symbol.upper().endswith(suf) for suf in CRYPTO_SUFFIXES)


def load_symbol_data(
    symbol: str,
    data_root: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    frame = load_hourly_bars(symbol, data_root=data_root)
    if frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame.columns = frame.columns.str.lower()
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        logger.warning("Duplicate columns %s detected for %s; keeping first occurrence.", duplicates, symbol)
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()

    if "timestamp" not in frame.columns:
        logger.warning("Missing timestamp column for %s", symbol)
        return pd.DataFrame()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])

    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        logger.warning("Missing columns %s for %s", missing, symbol)
        return pd.DataFrame()

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    if start_date is not None:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        frame = frame[frame["timestamp"] >= start_dt]

    if end_date is not None:
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        frame = frame[frame["timestamp"] < end_dt]

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["symbol"] = symbol

    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")

    frame = frame.set_index("timestamp", drop=False)
    return frame[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


class HourlyDataLoader:
    """Loads and manages hourly price data for all symbols."""

    def __init__(self, config: DataConfigHourly) -> None:
        self.config = config
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def load_all_symbols(self) -> Dict[str, pd.DataFrame]:
        if self._data_cache:
            return self._data_cache

        for symbol in self.config.all_symbols:
            df = load_symbol_data(
                symbol,
                self.config.data_root,
                start_date=None,
                end_date=self.config.end_date,
            )
            if not df.empty:
                self._data_cache[symbol] = df
                logger.info("Loaded %d rows for %s", len(df), symbol)
            else:
                logger.warning("No hourly data loaded for %s", symbol)
        return self._data_cache

    def get_context_for_timestamp(
        self,
        symbol: str,
        target_ts: pd.Timestamp,
        context_hours: Optional[int] = None,
    ) -> pd.DataFrame:
        if symbol not in self._data_cache:
            self.load_all_symbols()
        if symbol not in self._data_cache:
            return pd.DataFrame()

        df = self._data_cache[symbol]
        context_len = context_hours or self.config.context_hours
        mask = df["timestamp"] < target_ts
        context = df.loc[mask].tail(context_len)
        return context.reset_index(drop=True).copy()

    def get_price_on_timestamp(
        self,
        symbol: str,
        target_ts: pd.Timestamp,
    ) -> Optional[Dict[str, float]]:
        if symbol not in self._data_cache:
            self.load_all_symbols()
        if symbol not in self._data_cache:
            return None
        df = self._data_cache[symbol]
        try:
            row = df.loc[target_ts]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        }

    def get_available_symbols_at(self, target_ts: pd.Timestamp) -> List[str]:
        if not self._data_cache:
            self.load_all_symbols()
        available: List[str] = []
        for symbol, df in self._data_cache.items():
            if target_ts in df.index:
                available.append(symbol)
        return available

    def get_tradable_symbols_at(self, target_ts: pd.Timestamp) -> List[str]:
        return self.get_available_symbols_at(target_ts)
