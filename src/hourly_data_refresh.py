from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import pandas as pd
import requests
from alpaca.data import CryptoBarsRequest, StockBarsRequest, TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.hourly_data_utils import (
    HourlyDataIssue,
    HourlyDataStatus,
    HourlyDataValidator,
    resolve_hourly_symbol_path,
)
from src.symbol_utils import is_crypto_symbol
from src.stock_utils import remap_symbols

FetchFn = Callable[[str, datetime, datetime], pd.DataFrame]
BINANCE_ENDPOINT = "https://api.binance.com/api/v3/klines"
_PLACEHOLDER_TOKEN = "placeholder"

logger = logging.getLogger(__name__)


def _has_valid_alpaca_credentials() -> bool:
    return bool(
        ALP_KEY_ID_PROD
        and ALP_SECRET_KEY_PROD
        and _PLACEHOLDER_TOKEN not in ALP_KEY_ID_PROD
        and _PLACEHOLDER_TOKEN not in ALP_SECRET_KEY_PROD
    )


_STOCK_CLIENT: StockHistoricalDataClient | None = None
_CRYPTO_CLIENT: CryptoHistoricalDataClient | None = None


def _get_stock_client() -> StockHistoricalDataClient | None:
    global _STOCK_CLIENT
    if _STOCK_CLIENT is None and _has_valid_alpaca_credentials():
        _STOCK_CLIENT = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    return _STOCK_CLIENT


def _get_crypto_client() -> CryptoHistoricalDataClient | None:
    global _CRYPTO_CLIENT
    if _CRYPTO_CLIENT is None and _has_valid_alpaca_credentials():
        _CRYPTO_CLIENT = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    return _CRYPTO_CLIENT


def _normalize_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    bars = frame.copy()
    if isinstance(bars.index, pd.MultiIndex):
        # Drop the symbol level Alpaca attaches to the index
        bars = bars.reset_index(level="symbol", drop=True)
    index = bars.index
    if not isinstance(index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    else:
        if index.tzinfo is None:
            bars.index = index.tz_localize(timezone.utc)
        else:
            bars.index = index.tz_convert(timezone.utc)
    bars.index.name = "timestamp"
    bars = bars.sort_index()
    expected_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for column in expected_cols:
        if column not in bars.columns:
            bars[column] = 0.0
    bars["symbol"] = symbol.upper()
    selected = bars[expected_cols + ["symbol"]]
    return selected


def fetch_stock_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = _get_stock_client()
    if client is None:
        logger.warning("Stock client unavailable; cannot refresh %s", symbol)
        return pd.DataFrame()
    if start >= end:
        return pd.DataFrame()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    try:
        bars = client.get_stock_bars(request).df
    except Exception as exc:  # pragma: no cover - network issues
        logger.error("Failed to fetch stock bars for %s: %s", symbol, exc)
        return pd.DataFrame()
    return _normalize_bars(bars, symbol)


def fetch_crypto_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if not is_crypto_symbol(symbol):
        return pd.DataFrame()
    client = _get_crypto_client()
    if start >= end:
        return pd.DataFrame()
    remapped = remap_symbols(symbol)
    frame: pd.DataFrame | None = None
    if client is not None:
        request = CryptoBarsRequest(
            symbol_or_symbols=remapped,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start,
            end=end,
        )
        try:
            frame = client.get_crypto_bars(request).df
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("Alpaca crypto fetch failed for %s: %s", symbol, exc)
            frame = None
    normalized = _normalize_bars(frame, symbol) if frame is not None else pd.DataFrame()
    if not normalized.empty:
        return normalized
    return fetch_binance_bars(symbol, start, end)


def _binance_symbol(symbol: str) -> str | None:
    base = symbol.replace("/", "").upper()
    if base.endswith("USD"):
        base = base[:-3]
    # Binance pairs settle in USDT for most USD instruments
    candidate = f"{base}USDT"
    return candidate if candidate.isalpha() else None


def fetch_binance_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    pair = _binance_symbol(symbol)
    if not pair:
        return pd.DataFrame()
    limit = min(1000, max(10, math.ceil((end - start).total_seconds() / 3600.0) + 5))
    params = {
        "symbol": pair,
        "interval": "1h",
        "limit": limit,
        "startTime": max(0, int(start.timestamp() * 1000)),
        "endTime": max(0, int(end.timestamp() * 1000)),
    }
    try:
        response = requests.get(BINANCE_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network issues
        logger.error("Binance fallback failed for %s: %s", symbol, exc)
        return pd.DataFrame()
    rows = response.json()
    records = []
    for row in rows:
        open_time = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
        if open_time < start - timedelta(hours=1) or open_time > end + timedelta(hours=1):
            continue
        volume = float(row[5])
        quote_volume = float(row[7])
        vwap = quote_volume / volume if volume else float(row[4])
        records.append(
            {
                "timestamp": open_time,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": volume,
                "trade_count": int(row[8]),
                "vwap": vwap,
                "symbol": symbol.upper(),
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
    tz = frame.index.tz
    if tz is None:
        frame.index = frame.index.tz_localize(timezone.utc)
    else:
        frame.index = frame.index.tz_convert(timezone.utc)
    frame.index.name = "timestamp"
    return frame


class HourlyDataRefresher:
    """Backfill and maintain hourly CSVs for the trading loop."""

    def __init__(
        self,
        data_root: Path,
        validator: HourlyDataValidator,
        *,
        stock_fetcher: FetchFn | None = None,
        crypto_fetcher: FetchFn | None = None,
        backfill_hours: int = 48,
        overlap_hours: int = 2,
        crypto_max_staleness_hours: float = 1.5,
        sleep_seconds: float = 0.0,
        logger_override: logging.Logger | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.validator = validator
        self.stock_fetcher = stock_fetcher or fetch_stock_bars
        self.crypto_fetcher = crypto_fetcher or fetch_crypto_bars
        self.backfill_hours = max(1, backfill_hours)
        self.overlap_hours = max(0, overlap_hours)
        self.crypto_max_staleness_hours = max(0.1, crypto_max_staleness_hours)
        self.sleep_seconds = max(0.0, sleep_seconds)
        self._logger = logger_override or logger

    def refresh(self, symbols: Sequence[str]) -> Tuple[List[HourlyDataStatus], List[HourlyDataIssue]]:
        statuses, issues = self.validator.filter_ready(symbols)
        refresh_targets = self._symbols_requiring_refresh(statuses, issues)
        if not refresh_targets:
            return statuses, issues
        now = datetime.now(timezone.utc)
        self._logger.info(
            "Refreshing %d hourly symbols (targets=%s)",
            len(refresh_targets),
            ", ".join(refresh_targets),
        )
        for symbol in refresh_targets:
            try:
                refreshed = self._refresh_symbol(symbol, now)
            except Exception as exc:  # pragma: no cover - filesystem or parsing errors
                self._logger.exception("Failed to refresh %s: %s", symbol, exc)
                continue
            if refreshed:
                self._logger.info("Refreshed %s hourly data", symbol)
            else:
                self._logger.warning("No new hourly data fetched for %s", symbol)
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
        return self.validator.filter_ready(symbols)

    def _symbols_requiring_refresh(
        self,
        statuses: Sequence[HourlyDataStatus],
        issues: Sequence[HourlyDataIssue],
    ) -> List[str]:
        refresh = {issue.symbol for issue in issues if issue.reason in {"missing", "stale"}}
        for status in statuses:
            if is_crypto_symbol(status.symbol) and status.staleness_hours > self.crypto_max_staleness_hours:
                refresh.add(status.symbol)
        return sorted(refresh)

    def _refresh_symbol(self, symbol: str, now: datetime) -> bool:
        path = self._resolve_target_path(symbol)
        existing = self._load_existing_frame(path)
        last_timestamp = existing.index.max().to_pydatetime() if not existing.empty else None
        if last_timestamp is None:
            start = now - timedelta(hours=self.backfill_hours)
        else:
            start = last_timestamp - timedelta(hours=self.overlap_hours)
        start = max(start, now - timedelta(hours=self.backfill_hours))
        end = now + timedelta(minutes=1)
        fetcher = self.crypto_fetcher if is_crypto_symbol(symbol) else self.stock_fetcher
        new_frame = fetcher(symbol, start, end)
        if new_frame.empty:
            return False
        combined = new_frame if existing.empty else pd.concat([existing, new_frame])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.index.name = "timestamp"
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path)
        return True

    def _resolve_target_path(self, symbol: str) -> Path:
        resolved = resolve_hourly_symbol_path(symbol, self.data_root)
        if resolved is not None:
            return resolved
        folder = "crypto" if is_crypto_symbol(symbol) else "stocks"
        return self.data_root / folder / f"{symbol.upper()}.csv"

    def _load_existing_frame(self, path: Path) -> pd.DataFrame:
        """Load existing data using split cache: historical (pkl) + recent (csv)."""
        if not path.exists():
            return pd.DataFrame()

        # Split at Nov 1, 2025 - older data cached, recent data in CSV
        cache_cutoff = datetime(2025, 11, 1, tzinfo=timezone.utc)
        cache_path = path.parent / f"{path.stem}_hist.pkl"

        historical = pd.DataFrame()
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    historical = cache_data.get('data', pd.DataFrame())
                    cached_until = cache_data.get('cutoff', None)
                    if cached_until == cache_cutoff and not historical.empty:
                        self._logger.debug(f"{path.stem}: loaded {len(historical)} historical rows")
                    else:
                        historical = pd.DataFrame()
            except Exception:
                historical = pd.DataFrame()

        # Load recent CSV
        try:
            frame = pd.read_csv(path)
        except OSError:
            return historical if not historical.empty else pd.DataFrame()

        if frame.empty or "timestamp" not in frame.columns:
            return historical if not historical.empty else pd.DataFrame()

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            return historical if not historical.empty else pd.DataFrame()

        frame = frame.set_index("timestamp").sort_index()

        # Filter CSV to recent if cache exists
        if not historical.empty:
            frame = frame[frame.index >= cache_cutoff]

        # Combine
        if not historical.empty:
            combined = pd.concat([historical, frame])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            return combined

        # Create cache if enough historical data
        if len(frame) > 1000:
            hist_part = frame[frame.index < cache_cutoff]
            if len(hist_part) > 100:
                try:
                    import pickle
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'data': hist_part, 'cutoff': cache_cutoff}, f)
                    self._logger.info(f"{path.stem}: cached {len(hist_part)} historical rows")
                    # Trim CSV to recent only
                    recent = frame[frame.index >= cache_cutoff]
                    if not recent.empty:
                        recent.to_csv(path)
                except Exception:
                    pass

        return frame


__all__ = [
    "HourlyDataRefresher",
    "fetch_stock_bars",
    "fetch_crypto_bars",
    "fetch_binance_bars",
]
