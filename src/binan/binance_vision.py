from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional
from zipfile import ZipFile

import pandas as pd
import requests
from loguru import logger


_VISION_BASE_URL = "https://data.binance.vision/data"


def build_vision_daily_klines_zip_url(
    *,
    symbol: str,
    interval: str,
    day: date,
) -> str:
    """Build a Binance Vision daily klines zip URL for spot markets.

    Layout:
      .../spot/daily/klines/<SYMBOL>/<INTERVAL>/<SYMBOL>-<INTERVAL>-YYYY-MM-DD.zip
    """
    sym = str(symbol).strip().upper()
    itv = str(interval).strip()
    if not sym:
        raise ValueError("symbol is required")
    if not itv:
        raise ValueError("interval is required")

    stamp = day.strftime("%Y-%m-%d")
    return f"{_VISION_BASE_URL}/spot/daily/klines/{sym}/{itv}/{sym}-{itv}-{stamp}.zip"


def build_vision_monthly_klines_zip_url(
    *,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> str:
    sym = str(symbol).strip().upper()
    itv = str(interval).strip()
    if not sym:
        raise ValueError("symbol is required")
    if not itv:
        raise ValueError("interval is required")
    if not (1 <= int(month) <= 12):
        raise ValueError(f"month must be 1..12, received {month!r}")
    return (
        f"{_VISION_BASE_URL}/spot/monthly/klines/{sym}/{itv}/"
        f"{sym}-{itv}-{int(year):04d}-{int(month):02d}.zip"
    )


@dataclass(frozen=True)
class VisionDownload:
    url: str
    status_code: int
    content: bytes | None


def _download_zip(url: str, *, session: Optional[requests.Session] = None) -> VisionDownload:
    sess = session or requests.Session()
    try:
        resp = sess.get(url, stream=False)
    except Exception as exc:
        logger.warning("Binance Vision download failed: {} ({})", url, exc)
        return VisionDownload(url=url, status_code=-1, content=None)

    status = int(resp.status_code)
    if status == 200:
        return VisionDownload(url=url, status_code=status, content=resp.content)
    if status == 404:
        return VisionDownload(url=url, status_code=status, content=None)

    # Preserve detail for debugging, but do not throw unless the caller wants to.
    logger.warning("Binance Vision returned HTTP {} for {}", status, url)
    return VisionDownload(url=url, status_code=status, content=None)


_KLINES_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)


def _infer_epoch_unit(values: pd.Series) -> str:
    """Infer Binance Vision timestamp unit (s/ms/us/ns) from magnitude.

    Binance Vision has historically shipped kline timestamps in milliseconds, but
    some newer datasets appear to use microseconds. We infer the unit based on
    the numeric magnitude of the first non-null value.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    sample = numeric.dropna()
    if sample.empty:
        return "ms"
    try:
        first = float(sample.iloc[0])
    except Exception:
        return "ms"
    # Rough thresholds around current epoch (~1.7e9 seconds).
    if first >= 1e17:
        return "ns"
    if first >= 1e14:
        return "us"
    if first >= 1e11:
        return "ms"
    if first >= 1e8:
        return "s"
    return "ms"


def parse_vision_klines_zip_bytes(zip_bytes: bytes, *, symbol: str) -> pd.DataFrame:
    """Parse a Binance Vision kline zip payload into our canonical hourly schema."""
    if not zip_bytes:
        return pd.DataFrame()
    sym = str(symbol).strip().upper()
    if not sym:
        raise ValueError("symbol is required")

    with ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame()
        # Most zips contain a single CSV with the same stem as the zip.
        csv_name = sorted(names)[0]
        with zf.open(csv_name, "r") as fp:
            data = fp.read()

    frame = pd.read_csv(
        io.BytesIO(data),
        header=None,
        names=_KLINES_COLUMNS,
    )
    if frame.empty:
        return pd.DataFrame()

    # Convert to our canonical schema.
    unit = _infer_epoch_unit(frame["open_time"])
    ts = pd.to_datetime(pd.to_numeric(frame["open_time"], errors="coerce"), unit=unit, utc=True, errors="coerce")
    frame = frame.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return pd.DataFrame()
    frame = frame.set_index("timestamp")

    o = pd.to_numeric(frame["open"], errors="coerce").astype(float)
    h = pd.to_numeric(frame["high"], errors="coerce").astype(float)
    l = pd.to_numeric(frame["low"], errors="coerce").astype(float)
    c = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    v = pd.to_numeric(frame["volume"], errors="coerce").astype(float).fillna(0.0)
    qv = pd.to_numeric(frame["quote_asset_volume"], errors="coerce").astype(float).fillna(0.0)
    trades = pd.to_numeric(frame["number_of_trades"], errors="coerce").fillna(0).astype(int)

    # Use NaN (not pd.NA) to keep float dtypes and avoid pandas downcasting warnings.
    denom = v.where(v != 0.0)
    vwap = (qv / denom).fillna(c)

    out = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "trade_count": trades,
            "vwap": vwap,
            "symbol": sym,
        },
        index=pd.DatetimeIndex(frame.index, name="timestamp"),
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _month_starts(start_day: date, end_day: date) -> List[date]:
    """Return the first day-of-month dates covering [start_day, end_day]."""
    if end_day < start_day:
        return []
    cur = date(start_day.year, start_day.month, 1)
    end_month = date(end_day.year, end_day.month, 1)
    out: List[date] = []
    while cur <= end_month:
        out.append(cur)
        # Advance 1 month.
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _iter_days(start_day: date, end_day: date) -> Iterable[date]:
    if end_day < start_day:
        return []
    cur = start_day
    while cur <= end_day:
        yield cur
        cur = cur + timedelta(days=1)


def fetch_vision_hourly_klines(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1h",
    daily_lookback_days: int = 60,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch hourly spot klines via Binance Vision.

    Strategy:
    - Download monthly zips for the older portion of the window.
    - Download daily zips for the most-recent ``daily_lookback_days`` days (and always for the tail),
      because monthly zips may be missing for the current month.

    Returns an empty frame when no data is available (e.g., symbol not present on Vision).
    """
    sym = str(symbol).strip().upper()
    if not sym:
        raise ValueError("symbol is required")
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    if start >= end:
        return pd.DataFrame()

    start_day = start.date()
    end_day = end.date()
    daily_lookback_days = max(1, int(daily_lookback_days))
    daily_start_day = max(start_day, end_day - timedelta(days=daily_lookback_days))
    monthly_end_day = daily_start_day - timedelta(days=1)

    frames: List[pd.DataFrame] = []
    sess = session or requests.Session()

    # Monthly downloads for the older range.
    if monthly_end_day >= start_day:
        for month_start in _month_starts(start_day, monthly_end_day):
            url = build_vision_monthly_klines_zip_url(
                symbol=sym,
                interval=interval,
                year=month_start.year,
                month=month_start.month,
            )
            dl = _download_zip(url, session=sess)
            if dl.content is None:
                continue
            parsed = parse_vision_klines_zip_bytes(dl.content, symbol=sym)
            if not parsed.empty:
                frames.append(parsed)

    # Daily downloads for the recent range.
    for day in _iter_days(daily_start_day, end_day):
        url = build_vision_daily_klines_zip_url(symbol=sym, interval=interval, day=day)
        dl = _download_zip(url, session=sess)
        if dl.content is None:
            continue
        parsed = parse_vision_klines_zip_bytes(dl.content, symbol=sym)
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0, ignore_index=False)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Trim to the exact requested window.
    combined = combined[(combined.index >= pd.Timestamp(start)) & (combined.index <= pd.Timestamp(end))]
    return combined


__all__ = [
    "VisionDownload",
    "build_vision_monthly_klines_zip_url",
    "build_vision_daily_klines_zip_url",
    "parse_vision_klines_zip_bytes",
    "fetch_vision_hourly_klines",
]
