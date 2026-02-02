from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import time

import pandas as pd
from alpaca.data import CryptoBarsRequest, CryptoHistoricalDataClient, TimeFrame, TimeFrameUnit

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.stock_utils import remap_symbols


@dataclass
class AppendResult:
    symbol: str
    appended: int
    total: int
    start: Optional[datetime]
    end: Optional[datetime]


@dataclass
class BackfillResult:
    symbol: str
    appended: int
    total: int
    requested_start: datetime
    requested_end: datetime
    start: Optional[datetime]
    end: Optional[datetime]
    status: str


def _merge_and_dedup(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly bars and drop duplicates by timestamp."""
    if existing.empty:
        combined = new
    else:
        # Ensure both indices are datetime with consistent timezone
        if not existing.empty and existing.index.dtype == 'object':
            existing.index = pd.to_datetime(existing.index, utc=True)
        if not new.empty and new.index.dtype == 'object':
            new.index = pd.to_datetime(new.index, utc=True)

        # Make both indices timezone-aware (UTC) for comparison
        if not existing.empty and existing.index.tz is None:
            existing.index = existing.index.tz_localize('UTC')
        if not new.empty and new.index.tz is None:
            new.index = new.index.tz_localize('UTC')

        combined = pd.concat([existing, new], ignore_index=False)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def append_recent_crypto_data(
    symbols: Iterable[str],
    *,
    days: int = 10,
    output_dir: Path = Path("trainingdatahourly") / "crypto",
    client: Optional[CryptoHistoricalDataClient] = None,
) -> List[AppendResult]:
    """Append last `days` of hourly bars for each symbol into csv cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, days))
    client = client or CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

    results: List[AppendResult] = []
    for sym in symbols:
        symbol = sym.upper()
        remapped = remap_symbols(symbol)
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=remapped,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=start,
                end=end,
            )
            bars = client.get_crypto_bars(req).df
        except Exception:
            results.append(AppendResult(symbol, 0, 0, None, None))
            continue
        if bars.empty:
            results.append(AppendResult(symbol, 0, 0, None, None))
            continue
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level="symbol", drop=True)
        bars = bars.sort_index()
        bars["symbol"] = symbol
        bars.index.name = "timestamp"

        out_file = output_dir / f"{symbol}.csv"
        try:
            existing = pd.read_csv(out_file, parse_dates=["timestamp"]).set_index("timestamp") if out_file.exists() else pd.DataFrame()
        except Exception:
            existing = pd.DataFrame()

        combined = _merge_and_dedup(existing, bars)
        combined.to_csv(out_file)
        results.append(
            AppendResult(
                symbol=symbol,
                appended=len(bars),
                total=len(combined),
                start=combined.index.min().to_pydatetime() if len(combined) else None,
                end=combined.index.max().to_pydatetime() if len(combined) else None,
            )
        )
    return results


def backfill_crypto_range(
    symbols: Iterable[str],
    *,
    start: datetime,
    end: datetime,
    output_dir: Path = Path("trainingdatahourly") / "crypto",
    client: Optional[CryptoHistoricalDataClient] = None,
    sleep_seconds: float = 0.0,
    dry_run: bool = False,
) -> List[BackfillResult]:
    """Backfill hourly crypto bars for a specific time range into csv cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start = _to_utc(start)
    end = _to_utc(end)
    if end <= start:
        raise ValueError("end must be after start for backfill range.")

    client = client or CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    results: List[BackfillResult] = []

    for sym in symbols:
        symbol = sym.upper()
        remapped = remap_symbols(symbol)
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=remapped,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=start,
                end=end,
            )
            bars = client.get_crypto_bars(req).df
        except Exception:
            results.append(
                BackfillResult(
                    symbol=symbol,
                    appended=0,
                    total=0,
                    requested_start=start,
                    requested_end=end,
                    start=None,
                    end=None,
                    status="error",
                )
            )
            continue

        if bars.empty:
            results.append(
                BackfillResult(
                    symbol=symbol,
                    appended=0,
                    total=0,
                    requested_start=start,
                    requested_end=end,
                    start=None,
                    end=None,
                    status="no_data",
                )
            )
            continue

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level="symbol", drop=True)
        bars = bars.sort_index()
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars[~bars.index.isna()]
        bars["symbol"] = symbol
        bars.index.name = "timestamp"

        out_file = output_dir / f"{symbol}.csv"
        try:
            existing = (
                pd.read_csv(out_file, parse_dates=["timestamp"]).set_index("timestamp")
                if out_file.exists()
                else pd.DataFrame()
            )
        except Exception:
            existing = pd.DataFrame()

        if not existing.empty:
            existing.index = pd.to_datetime(existing.index, utc=True, errors="coerce")
            existing = existing[~existing.index.isna()]

        combined = _merge_and_dedup(existing, bars)
        appended = max(0, len(combined) - len(existing)) if not existing.empty else len(bars)
        if not dry_run:
            combined.to_csv(out_file)

        results.append(
            BackfillResult(
                symbol=symbol,
                appended=int(appended),
                total=int(len(combined)),
                requested_start=start,
                requested_end=end,
                start=combined.index.min().to_pydatetime() if len(combined) else None,
                end=combined.index.max().to_pydatetime() if len(combined) else None,
                status="updated" if appended > 0 else "no_update",
            )
        )

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return results


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "append_recent_crypto_data",
    "backfill_crypto_range",
    "AppendResult",
    "BackfillResult",
    "_merge_and_dedup",
]
