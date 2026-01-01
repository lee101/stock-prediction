"""Data loading for long-term daily market simulation."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from .config import DataConfigLong

logger = logging.getLogger(__name__)

# Crypto symbol suffixes
CRYPTO_SUFFIXES = ("USD", "BTC", "ETH", "USDT", "USDC")


def is_crypto_symbol(symbol: str) -> bool:
    """Check if symbol is a cryptocurrency."""
    return any(symbol.upper().endswith(suf) for suf in CRYPTO_SUFFIXES)


def get_trading_calendar(
    start_date: date,
    end_date: date,
    is_crypto: bool = False,
) -> List[date]:
    """Get list of trading days between start and end date.

    Args:
        start_date: Start of simulation period
        end_date: End of simulation period
        is_crypto: If True, include weekends (crypto trades 24/7)

    Returns:
        List of trading dates
    """
    trading_days = []
    current = start_date

    while current <= end_date:
        if is_crypto:
            # Crypto trades every day
            trading_days.append(current)
        else:
            # Stocks: exclude weekends (0=Monday, 6=Sunday)
            if current.weekday() < 5:
                trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def load_symbol_data(
    symbol: str,
    data_root: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Load daily OHLCV data for a single symbol.

    Args:
        symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
        data_root: Root directory for data files
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Try multiple file patterns
    symbol_normalized = symbol.replace("/", "-")
    patterns = [
        f"{symbol_normalized}-*.csv",
        f"{symbol_normalized}.csv",
        f"{symbol}.csv",
    ]

    csv_files: List[Path] = []
    for search_dir in [data_root, data_root / "train", data_root / "test"]:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            csv_files.extend(sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime))

    if not csv_files:
        logger.warning("No data files found for symbol %s in %s", symbol, data_root)
        return pd.DataFrame()

    # Use most recent file
    latest_file = csv_files[-1]
    logger.debug("Loading %s from %s", symbol, latest_file)

    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        logger.error("Failed to load %s: %s", latest_file, e)
        return pd.DataFrame()

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df["timestamp"] = df["date"]
        elif df.index.name and "date" in df.index.name.lower():
            df = df.reset_index()
            df["timestamp"] = df[df.columns[0]]
        else:
            # Try first column
            df["timestamp"] = df.iloc[:, 0]

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Ensure required columns
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning("Missing columns %s for %s", missing, symbol)
        return pd.DataFrame()

    # Add volume if missing
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Add symbol column
    df["symbol"] = symbol

    # Filter by date range
    if start_date is not None:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        df = df[df["timestamp"] >= start_dt]

    if end_date is not None:
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["timestamp"] < end_dt]

    # Sort and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Extract date column for easier grouping
    df["date"] = df["timestamp"].dt.date

    return df[["timestamp", "date", "symbol", "open", "high", "low", "close", "volume"]]


class DailyDataLoader:
    """Loads and manages daily price data for all symbols."""

    def __init__(self, config: DataConfigLong) -> None:
        self.config = config
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._combined_data: Optional[pd.DataFrame] = None

    def load_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Load data for all configured symbols.

        Returns:
            Dict mapping symbol -> DataFrame with daily OHLCV data
        """
        if self._data_cache:
            return self._data_cache

        for symbol in self.config.all_symbols:
            df = load_symbol_data(
                symbol,
                self.config.data_root,
                start_date=None,  # Load all available history
                end_date=self.config.end_date,
            )
            if not df.empty:
                self._data_cache[symbol] = df
                logger.info("Loaded %d rows for %s", len(df), symbol)
            else:
                logger.warning("No data loaded for %s", symbol)

        return self._data_cache

    def get_combined_data(self) -> pd.DataFrame:
        """Get combined data for all symbols.

        Returns:
            DataFrame with all symbols stacked, indexed by (timestamp, symbol)
        """
        if self._combined_data is not None:
            return self._combined_data

        data = self.load_all_symbols()
        if not data:
            return pd.DataFrame()

        self._combined_data = pd.concat(data.values(), ignore_index=True)
        return self._combined_data

    def get_context_for_date(
        self,
        symbol: str,
        target_date: date,
        context_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get historical context for forecasting on a specific date.

        Args:
            symbol: Trading symbol
            target_date: Date to forecast from (not included in context)
            context_days: Number of historical days to include

        Returns:
            DataFrame with context_days rows of historical data
        """
        if symbol not in self._data_cache:
            self.load_all_symbols()

        if symbol not in self._data_cache:
            return pd.DataFrame()

        df = self._data_cache[symbol]
        context_len = context_days or self.config.context_days

        # Filter to dates before target_date
        target_dt = pd.Timestamp(target_date, tz="UTC")
        mask = df["timestamp"] < target_dt
        context = df[mask].tail(context_len)

        return context.copy()

    def get_price_on_date(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[Dict[str, float]]:
        """Get OHLC prices for a symbol on a specific date.

        Args:
            symbol: Trading symbol
            target_date: Date to get prices for

        Returns:
            Dict with open, high, low, close prices, or None if not available
        """
        if symbol not in self._data_cache:
            self.load_all_symbols()

        if symbol not in self._data_cache:
            return None

        df = self._data_cache[symbol]
        mask = df["date"] == target_date

        if not mask.any():
            return None

        row = df[mask].iloc[0]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    def get_available_symbols_on_date(self, target_date: date) -> List[str]:
        """Get list of symbols that have data on a specific date.

        Args:
            target_date: Date to check

        Returns:
            List of symbols with data on that date
        """
        if not self._data_cache:
            self.load_all_symbols()

        available = []
        for symbol, df in self._data_cache.items():
            if (df["date"] == target_date).any():
                available.append(symbol)

        return available

    def get_tradable_symbols_on_date(
        self,
        target_date: date,
        require_previous_close: bool = True,
    ) -> List[str]:
        """Get symbols that are tradable on a specific date.

        Args:
            target_date: Trading date
            require_previous_close: If True, require at least one previous day of data

        Returns:
            List of tradable symbols
        """
        available = self.get_available_symbols_on_date(target_date)

        if not require_previous_close:
            return available

        # Filter to symbols with previous day data
        tradable = []
        prev_date = target_date - timedelta(days=1)

        # For stocks, go back to last trading day
        while prev_date.weekday() >= 5:  # Skip weekends
            prev_date -= timedelta(days=1)

        for symbol in available:
            df = self._data_cache[symbol]
            # Check if we have at least some historical data
            target_dt = pd.Timestamp(target_date, tz="UTC")
            has_history = (df["timestamp"] < target_dt).any()
            if has_history:
                tradable.append(symbol)

        return tradable


def download_latest_data(
    symbols: List[str],
    data_root: Path,
) -> Dict[str, pd.DataFrame]:
    """Download latest daily data for symbols from Alpaca.

    Args:
        symbols: List of trading symbols
        data_root: Directory to save data

    Returns:
        Dict mapping symbol -> DataFrame
    """
    try:
        from data_curate_daily import download_daily_stock_data
    except ImportError:
        logger.error("Could not import data_curate_daily")
        return {}

    data_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for symbol in symbols:
        try:
            df = download_daily_stock_data(
                path=str(data_root),
                symbols=[symbol],
                all_data_force=True,
            )
            if not df.empty:
                results[symbol] = df
                logger.info("Downloaded %d rows for %s", len(df), symbol)
        except Exception as e:
            logger.error("Failed to download %s: %s", symbol, e)

    return results
