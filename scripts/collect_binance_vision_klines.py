#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.binan import binance_vision
from src.binance_symbol_utils import normalize_compact_symbol, unique_symbols


def _parse_utc(value: str) -> datetime:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp/date {value!r}. Expected ISO date/time (e.g. 2026-02-01).")
    return pd.Timestamp(ts).to_pydatetime()


def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = new
    elif new is None or new.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new], axis=0, ignore_index=True)
    if combined.empty:
        return combined
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return combined


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _fetch_symbol(
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    daily_lookback_days: int,
) -> pd.DataFrame:
    frame = binance_vision.fetch_vision_hourly_klines(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        daily_lookback_days=int(daily_lookback_days),
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download Binance Vision spot klines into local CSVs.")
    parser.add_argument("--symbols", nargs="+", default=("SOLFDUSD",), help="Symbols (e.g., SOLFDUSD).")
    parser.add_argument("--interval", default="5m", help="Binance interval string (e.g., 1m, 5m, 15m, 1h).")
    parser.add_argument(
        "--days",
        type=int,
        default=120,
        help="Lookback days from now when --start is not provided (default: 120).",
    )
    parser.add_argument("--start", default=None, help="UTC start date/time (ISO).")
    parser.add_argument("--end", default=None, help="UTC end date/time (ISO; default: now).")
    parser.add_argument(
        "--daily-lookback-days",
        type=int,
        default=99999,
        help="Force daily zip downloads for up to this many days from the end (default: 99999).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory (default: binance_spot_<interval>).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite instead of merging with existing CSV.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    interval = str(args.interval).strip()
    if not interval:
        raise ValueError("interval is required")

    symbols = unique_symbols(args.symbols or [])
    if not symbols:
        logger.error("No symbols provided.")
        return 2

    end = _parse_utc(args.end) if args.end else datetime.now(timezone.utc)
    if args.start:
        start = _parse_utc(args.start)
    else:
        start = end - timedelta(days=max(1, int(args.days)))
    if start >= end:
        raise ValueError(f"start must be before end (start={start.isoformat()} end={end.isoformat()})")

    out_root = Path(args.out_root) if args.out_root is not None else Path(f"binance_spot_{interval.lower()}")
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Binance Vision klines: interval={} window=[{}, {}) out_root={}", interval, start, end, out_root)

    for raw_symbol in symbols:
        symbol = normalize_compact_symbol(raw_symbol)
        out_path = out_root / f"{symbol}.csv"
        existing = pd.DataFrame()
        fetch_start = start
        if out_path.exists() and not bool(args.force) and not args.start:
            existing = _load_existing(out_path)
            if not existing.empty:
                # Re-fetch a small tail window to avoid gaps from partial days.
                last_ts = pd.Timestamp(existing["timestamp"].max()).to_pydatetime()
                fetch_start = max(start, last_ts - timedelta(days=2))
        logger.info("Fetching {} {} from {} to {} (existing_rows={})", symbol, interval, fetch_start, end, len(existing))

        fetched = _fetch_symbol(
            symbol=symbol,
            interval=interval,
            start=fetch_start,
            end=end,
            daily_lookback_days=int(args.daily_lookback_days),
        )
        if fetched.empty and existing.empty:
            logger.warning("No data fetched for {} (interval={}).", symbol, interval)
            continue

        if bool(args.force):
            merged = fetched
        else:
            merged = _merge(existing, fetched)

        if merged.empty:
            logger.warning("Merged frame empty for {} (interval={}).", symbol, interval)
            continue

        merged.to_csv(out_path, index=False)
        logger.info("Wrote {} rows for {} -> {}", len(merged), symbol, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

