from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from src.hourly_data_refresh import fetch_crypto_bars, fetch_stock_bars
from src.hourly_data_utils import resolve_hourly_symbol_path
from src.symbol_utils import is_crypto_symbol

FetchFn = Callable[[str, datetime, datetime], pd.DataFrame]


@dataclass(frozen=True)
class Gap:
    start: pd.Timestamp
    end: pd.Timestamp


def find_large_gaps(timestamps: Sequence[pd.Timestamp], *, min_gap: pd.Timedelta) -> List[Gap]:
    """Return gaps where the time delta between consecutive timestamps exceeds ``min_gap``.

    Notes:
    - This is a conservative detector designed to catch multi-day gaps in hourly datasets.
    - It does not attempt to distinguish expected market closures vs missing stock bars.
    """
    if not timestamps:
        return []
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce").dropna().sort_values()
    if ts.empty:
        return []
    diffs = ts.diff()
    gaps: List[Gap] = []
    for idx in range(1, len(ts)):
        delta = diffs.iloc[idx]
        if delta is not pd.NaT and delta > min_gap:
            gaps.append(Gap(start=ts.iloc[idx - 1], end=ts.iloc[idx]))
    return gaps


def _normalize_fetch_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if isinstance(work.index, pd.DatetimeIndex):
        idx = work.index
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        work = work.reset_index().rename(columns={"index": "timestamp"})
        work["timestamp"] = idx
    if "timestamp" not in work.columns:
        raise ValueError("Fetcher output must include a DatetimeIndex or 'timestamp' column.")
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    work.columns = [str(c).lower() for c in work.columns]
    work["symbol"] = symbol.upper()
    return work.reset_index(drop=True)


def fill_hourly_gaps_for_symbol(
    symbol: str,
    *,
    data_root: Path,
    min_gap: pd.Timedelta,
    scan_start: Optional[pd.Timestamp] = None,
    scan_end: Optional[pd.Timestamp] = None,
    overlap_hours: int = 4,
    stock_fetcher: FetchFn = fetch_stock_bars,
    crypto_fetcher: FetchFn = fetch_crypto_bars,
    sleep_seconds: float = 0.0,
) -> dict:
    """Fill multi-day gaps in an existing hourly CSV in-place.

    This is intended for "missing month" style gaps (e.g., due to a partial downloader run),
    not for enforcing a perfectly regular hourly index.
    """
    symbol = symbol.upper()
    data_root = Path(data_root)
    path = resolve_hourly_symbol_path(symbol, data_root)
    if path is None:
        raise FileNotFoundError(f"Hourly CSV for {symbol} not found under {data_root}")

    frame = pd.read_csv(path)
    if frame.empty or "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp column or is empty")
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if frame.empty:
        raise ValueError(f"{path} contained no valid timestamps for {symbol}")

    ts_full = frame["timestamp"]
    start_ts = pd.to_datetime(scan_start, utc=True, errors="coerce") if scan_start is not None else None
    end_ts = pd.to_datetime(scan_end, utc=True, errors="coerce") if scan_end is not None else None
    if end_ts is None:
        end_ts = ts_full.max()
    if start_ts is None:
        start_ts = ts_full.min()

    # Include one row before the scan window so we can detect a gap that crosses the boundary.
    before = frame[ts_full < start_ts].tail(1)
    inside = frame[(ts_full >= start_ts) & (ts_full <= end_ts)]
    scan_frame = pd.concat([before, inside], ignore_index=True)
    gaps = find_large_gaps(list(scan_frame["timestamp"]), min_gap=min_gap)

    if not gaps:
        return {"symbol": symbol, "status": "ok", "gaps_filled": 0, "added_rows": 0, "path": str(path)}

    fetcher = crypto_fetcher if is_crypto_symbol(symbol) else stock_fetcher
    overlap = pd.Timedelta(hours=max(0, int(overlap_hours)))
    added_rows = 0
    gap_count = 0

    for gap in gaps:
        fetch_start = (gap.start - overlap).to_pydatetime()
        fetch_end = (gap.end + overlap).to_pydatetime()
        logger.info("Filling gap for {}: {} -> {}", symbol, gap.start, gap.end)
        updates = fetcher(symbol, fetch_start, fetch_end)
        normalized = _normalize_fetch_frame(symbol, updates)
        if normalized.empty:
            logger.warning("No bars fetched for {} gap {} -> {}", symbol, gap.start, gap.end)
            continue
        before_len = len(frame)
        frame = (
            pd.concat([frame, normalized], ignore_index=True, sort=False)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        added_rows += max(0, len(frame) - before_len)
        gap_count += 1
        if sleep_seconds:
            time.sleep(float(sleep_seconds))

    if added_rows > 0:
        frame.to_csv(path, index=False)
        logger.info("{}: filled {} gaps (+{} rows) -> {}", symbol, gap_count, added_rows, path)
    else:
        logger.info("{}: detected {} gaps but fetched no new rows", symbol, len(gaps))

    return {"symbol": symbol, "status": "ok", "gaps_filled": gap_count, "added_rows": added_rows, "path": str(path)}


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill large gaps in hourly CSVs (stocks + crypto).")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols to repair.")
    parser.add_argument("--data-root", default="trainingdatahourly", help="Root directory containing hourly CSVs.")
    parser.add_argument(
        "--scan-days",
        type=int,
        default=180,
        help="Only scan the last N days of each file (default: 180).",
    )
    parser.add_argument(
        "--gap-days",
        type=float,
        default=5.0,
        help="Treat deltas larger than this as missing-data gaps (default: 5 days).",
    )
    parser.add_argument("--overlap-hours", type=int, default=4, help="Overlap on each side of fetched ranges.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    data_root = Path(args.data_root)
    now = pd.Timestamp(datetime.now(timezone.utc))
    scan_start = now - pd.Timedelta(days=max(1, int(args.scan_days)))
    min_gap = pd.Timedelta(days=float(args.gap_days))

    results = []
    for symbol in symbols:
        try:
            result = fill_hourly_gaps_for_symbol(
                symbol,
                data_root=data_root,
                min_gap=min_gap,
                scan_start=scan_start,
                scan_end=now,
                overlap_hours=args.overlap_hours,
                sleep_seconds=args.sleep_seconds,
            )
        except Exception as exc:
            logger.exception("Failed to fill gaps for {}: {}", symbol, exc)
            result = {"symbol": symbol, "status": "error", "error": str(exc)}
        results.append(result)

    logger.info("Gap fill complete: {}", results)


if __name__ == "__main__":
    main()
