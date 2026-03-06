from __future__ import annotations

import argparse
import logging
import shlex
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from alpaca.data import StockBarsRequest, TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "symbol",
]
DEFAULT_OUTPUT_ROOT = Path("trainingdata5min")
DEFAULT_STOCK_DIRNAME = "stocks"
DEFAULT_SUPERVISOR_CONFIG = Path("supervisor") / "unified-stock-trader.conf"
DEFAULT_FIVE_MIN_STOCK_SYMBOLS: tuple[str, ...] = ("NVDA", "PLTR", "GOOG", "DBX", "TRIP", "MTCH", "NYT")
DEFAULT_RECENT_LOOKBACK_MINUTES = 15
DEFAULT_OVERLAP_MINUTES = 10
DEFAULT_BOOTSTRAP_DAYS = 30
DEFAULT_REPAIR_DAYS = 7
DEFAULT_REPAIR_INTERVAL_HOURS = 24
DEFAULT_INTERVAL_SECONDS = 300
_PLACEHOLDER_TOKEN = "placeholder"
_STOCK_CLIENT: StockHistoricalDataClient | None = None

FetchFn = Callable[[str, datetime, datetime], pd.DataFrame]


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _ensure_utc(value: datetime | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def parse_symbol_tokens(values: Sequence[str] | None) -> list[str]:
    seen: set[str] = set()
    resolved: list[str] = []
    for raw in values or ():
        for token in str(raw).replace(",", " ").split():
            symbol = token.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            resolved.append(symbol)
    return resolved


def extract_stock_symbols_from_supervisor(config_path: Path) -> list[str]:
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("command="):
            continue
        try:
            tokens = shlex.split(line[len("command="):].strip())
        except ValueError:
            continue
        for idx, token in enumerate(tokens):
            if token == "--stock-symbols" and idx + 1 < len(tokens):
                return parse_symbol_tokens([tokens[idx + 1]])
            if token.startswith("--stock-symbols="):
                return parse_symbol_tokens([token.split("=", 1)[1]])
    return []


def resolve_stock_symbols(
    *,
    symbols: Sequence[str] | None,
    supervisor_config: Path = DEFAULT_SUPERVISOR_CONFIG,
    fallback: Sequence[str] = DEFAULT_FIVE_MIN_STOCK_SYMBOLS,
) -> list[str]:
    explicit = parse_symbol_tokens(symbols)
    if explicit:
        return explicit
    from_supervisor = extract_stock_symbols_from_supervisor(supervisor_config)
    if from_supervisor:
        return from_supervisor
    return parse_symbol_tokens(fallback)


def latest_complete_bar_open(
    *,
    now: datetime | None = None,
    bar_minutes: int = 5,
) -> datetime:
    current = pd.Timestamp(_ensure_utc(now or datetime.now(timezone.utc)))
    floored = current.floor(f"{int(bar_minutes)}min")
    return (floored - pd.Timedelta(minutes=int(bar_minutes))).to_pydatetime()


def _empty_bar_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_COLUMNS).set_index("timestamp")


def normalize_stock_bars(
    symbol: str,
    frame: pd.DataFrame,
    *,
    now: datetime | None = None,
    bar_minutes: int = 5,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_bar_frame()

    bars = frame.copy()
    if isinstance(bars.index, pd.MultiIndex) and "symbol" in bars.index.names:
        bars = bars.reset_index(level="symbol", drop=True)
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    elif bars.index.tz is None:
        bars.index = bars.index.tz_localize(timezone.utc)
    else:
        bars.index = bars.index.tz_convert(timezone.utc)

    bars = bars[~bars.index.isna()].sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"Missing required bar columns: {missing}")

    if "trade_count" not in bars.columns:
        bars["trade_count"] = 0
    if "vwap" not in bars.columns:
        bars["vwap"] = bars["close"]

    max_open = latest_complete_bar_open(now=now, bar_minutes=bar_minutes)
    bars = bars[bars.index <= max_open]
    if bars.empty:
        return _empty_bar_frame()

    bars.index.name = "timestamp"
    bars["symbol"] = symbol.upper()
    return bars[["open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]]


def load_existing_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_bar_frame()
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if frame.empty:
        return _empty_bar_frame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in SCHEMA_COLUMNS[1:]:
        if column not in frame.columns:
            frame[column] = 0 if column == "trade_count" else pd.NA
    return frame[[column for column in SCHEMA_COLUMNS[1:] if column in frame.columns]]


def merge_and_dedup_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    elif new.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new], axis=0, sort=False)
    if combined.empty:
        return _empty_bar_frame()
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.index.name = "timestamp"
    for column in ("trade_count", "vwap", "symbol"):
        if column not in combined.columns:
            combined[column] = 0 if column == "trade_count" else pd.NA
    return combined[["open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]]


def append_stock_bars(path: Path, new: pd.DataFrame) -> dict[str, int | str | None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_bars(path)
    combined = merge_and_dedup_bars(existing, new)
    appended = max(0, len(combined) - len(existing))
    combined.reset_index().to_csv(path, index=False)
    start = combined.index.min() if not combined.empty else None
    end = combined.index.max() if not combined.empty else None
    return {
        "appended": int(appended),
        "total": int(len(combined)),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
    }


def resolve_fetch_start(
    path: Path,
    *,
    now: datetime,
    recent_minutes: int = DEFAULT_RECENT_LOOKBACK_MINUTES,
    overlap_minutes: int = DEFAULT_OVERLAP_MINUTES,
    bootstrap_days: int = DEFAULT_BOOTSTRAP_DAYS,
) -> datetime:
    current = _ensure_utc(now)
    recent_start = current - timedelta(minutes=max(5, int(recent_minutes)))
    existing = load_existing_bars(path)
    if existing.empty:
        return current - timedelta(days=max(1, int(bootstrap_days)))
    latest = pd.Timestamp(existing.index.max()).to_pydatetime()
    overlap_start = latest - timedelta(minutes=max(0, int(overlap_minutes)))
    return min(recent_start, overlap_start)


def _has_valid_alpaca_credentials() -> bool:
    return bool(
        ALP_KEY_ID_PROD
        and ALP_SECRET_KEY_PROD
        and _PLACEHOLDER_TOKEN not in ALP_KEY_ID_PROD
        and _PLACEHOLDER_TOKEN not in ALP_SECRET_KEY_PROD
    )


def _get_stock_client() -> StockHistoricalDataClient | None:
    global _STOCK_CLIENT
    if _STOCK_CLIENT is None and _has_valid_alpaca_credentials():
        _STOCK_CLIENT = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    return _STOCK_CLIENT


def _coerce_feed(feed: str | DataFeed) -> DataFeed:
    if isinstance(feed, DataFeed):
        return feed
    token = str(feed).strip().upper()
    if not token:
        return DataFeed.IEX
    try:
        return getattr(DataFeed, token)
    except AttributeError as exc:
        raise ValueError(f"Unsupported stock data feed: {feed}") from exc


def fetch_stock_5min_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    feed: str | DataFeed = DataFeed.IEX,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    client = client or _get_stock_client()
    if client is None:
        logger.warning("Stock client unavailable; skipping %s", symbol)
        return _empty_bar_frame()

    start_dt = _ensure_utc(start)
    end_dt = _ensure_utc(end)
    if start_dt >= end_dt:
        return _empty_bar_frame()

    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_dt,
        end=end_dt,
        adjustment="raw",
        feed=_coerce_feed(feed),
    )
    try:
        frame = client.get_stock_bars(request).df
    except Exception as exc:  # pragma: no cover - network/runtime failures
        logger.error("Failed to fetch 5-minute stock bars for %s: %s", symbol, exc)
        return _empty_bar_frame()
    return normalize_stock_bars(symbol, frame, now=end_dt)


def collect_recent_cycle(
    *,
    symbols: Sequence[str],
    out_root: Path,
    now: datetime | None = None,
    recent_minutes: int = DEFAULT_RECENT_LOOKBACK_MINUTES,
    overlap_minutes: int = DEFAULT_OVERLAP_MINUTES,
    bootstrap_days: int = DEFAULT_BOOTSTRAP_DAYS,
    fetcher: FetchFn,
) -> list[dict[str, int | str | None]]:
    current = _ensure_utc(now or datetime.now(timezone.utc))
    out_root = Path(out_root)
    stock_dir = out_root / DEFAULT_STOCK_DIRNAME
    stock_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, int | str | None]] = []

    for symbol in resolve_stock_symbols(symbols=symbols):
        path = stock_dir / f"{symbol}.csv"
        start_dt = resolve_fetch_start(
            path,
            now=current,
            recent_minutes=recent_minutes,
            overlap_minutes=overlap_minutes,
            bootstrap_days=bootstrap_days,
        )
        bars = fetcher(symbol, start_dt, current)
        stats = append_stock_bars(path, bars)
        results.append(
            {
                "symbol": symbol,
                "start": start_dt.isoformat(),
                "end": current.isoformat(),
                **stats,
            }
        )
    return results


def repair_recent_history(
    *,
    symbols: Sequence[str],
    out_root: Path,
    days: int,
    now: datetime | None = None,
    fetcher: FetchFn,
) -> list[dict[str, int | str | None]]:
    if days <= 0:
        return []
    current = _ensure_utc(now or datetime.now(timezone.utc))
    start_dt = current - timedelta(days=max(1, int(days)))
    out_root = Path(out_root)
    stock_dir = out_root / DEFAULT_STOCK_DIRNAME
    stock_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, int | str | None]] = []
    for symbol in resolve_stock_symbols(symbols=symbols):
        path = stock_dir / f"{symbol}.csv"
        bars = fetcher(symbol, start_dt, current)
        stats = append_stock_bars(path, bars)
        results.append(
            {
                "symbol": symbol,
                "start": start_dt.isoformat(),
                "end": current.isoformat(),
                **stats,
            }
        )
    return results


def _log_cycle_results(name: str, results: Iterable[dict[str, int | str | None]]) -> None:
    rows = list(results)
    if not rows:
        logger.info("%s complete: no symbols", name)
        return
    appended = sum(int(row.get("appended") or 0) for row in rows)
    logger.info("%s complete: symbols=%d appended=%d", name, len(rows), appended)
    for row in rows:
        logger.info(
            "%s %s appended=%s total=%s range=[%s -> %s]",
            name,
            row.get("symbol"),
            row.get("appended"),
            row.get("total"),
            row.get("start"),
            row.get("end"),
        )


def run_collector_loop(
    *,
    symbols: Sequence[str] | None = None,
    out_root: Path = DEFAULT_OUTPUT_ROOT,
    supervisor_config: Path = DEFAULT_SUPERVISOR_CONFIG,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    recent_minutes: int = DEFAULT_RECENT_LOOKBACK_MINUTES,
    overlap_minutes: int = DEFAULT_OVERLAP_MINUTES,
    bootstrap_days: int = DEFAULT_BOOTSTRAP_DAYS,
    repair_days: int = DEFAULT_REPAIR_DAYS,
    repair_interval_hours: int = DEFAULT_REPAIR_INTERVAL_HOURS,
    feed: str | DataFeed = DataFeed.IEX,
    once: bool = False,
) -> None:
    resolved_symbols = resolve_stock_symbols(symbols=symbols, supervisor_config=supervisor_config)
    if not resolved_symbols:
        raise ValueError("No stock symbols resolved for 5-minute collector.")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    def _fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return fetch_stock_5min_bars(symbol, start, end, feed=feed)

    logger.info(
        "Starting 5-minute stock collector symbols=%s out_root=%s feed=%s",
        ",".join(resolved_symbols),
        out_root,
        _coerce_feed(feed).value,
    )

    last_repair: datetime | None = None
    while True:
        cycle_now = _ensure_utc(datetime.now(timezone.utc))
        try:
            cycle_results = collect_recent_cycle(
                symbols=resolved_symbols,
                out_root=out_root,
                now=cycle_now,
                recent_minutes=recent_minutes,
                overlap_minutes=overlap_minutes,
                bootstrap_days=bootstrap_days,
                fetcher=_fetch,
            )
            _log_cycle_results("recent", cycle_results)

            should_repair = repair_days > 0 and (
                last_repair is None
                or cycle_now - last_repair >= timedelta(hours=max(1, int(repair_interval_hours)))
            )
            if should_repair:
                repair_results = repair_recent_history(
                    symbols=resolved_symbols,
                    out_root=out_root,
                    days=repair_days,
                    now=cycle_now,
                    fetcher=_fetch,
                )
                _log_cycle_results("repair", repair_results)
                last_repair = cycle_now
        except Exception:  # pragma: no cover - defensive loop guard
            logger.exception("5-minute stock collector cycle failed")

        if once:
            return
        time.sleep(max(1, int(interval_seconds)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously collect 5-minute stock bars from Alpaca.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional stock symbols. Commas are supported.")
    parser.add_argument("--supervisor-config", type=Path, default=DEFAULT_SUPERVISOR_CONFIG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--lookback-minutes", type=int, default=DEFAULT_RECENT_LOOKBACK_MINUTES)
    parser.add_argument("--overlap-minutes", type=int, default=DEFAULT_OVERLAP_MINUTES)
    parser.add_argument("--bootstrap-days", type=int, default=DEFAULT_BOOTSTRAP_DAYS)
    parser.add_argument("--repair-days", type=int, default=DEFAULT_REPAIR_DAYS)
    parser.add_argument("--repair-interval-hours", type=int, default=DEFAULT_REPAIR_INTERVAL_HOURS)
    parser.add_argument("--feed", default=DataFeed.IEX.value)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_logging()
    args = build_arg_parser().parse_args(argv)
    run_collector_loop(
        symbols=args.symbols,
        out_root=args.out_root,
        supervisor_config=args.supervisor_config,
        interval_seconds=args.interval_seconds,
        recent_minutes=args.lookback_minutes,
        overlap_minutes=args.overlap_minutes,
        bootstrap_days=args.bootstrap_days,
        repair_days=args.repair_days,
        repair_interval_hours=args.repair_interval_hours,
        feed=args.feed,
        once=args.once,
    )
    return 0


__all__ = [
    "DEFAULT_BOOTSTRAP_DAYS",
    "DEFAULT_FIVE_MIN_STOCK_SYMBOLS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_OVERLAP_MINUTES",
    "DEFAULT_RECENT_LOOKBACK_MINUTES",
    "DEFAULT_REPAIR_DAYS",
    "DEFAULT_REPAIR_INTERVAL_HOURS",
    "DEFAULT_SUPERVISOR_CONFIG",
    "SCHEMA_COLUMNS",
    "append_stock_bars",
    "collect_recent_cycle",
    "extract_stock_symbols_from_supervisor",
    "latest_complete_bar_open",
    "load_existing_bars",
    "main",
    "merge_and_dedup_bars",
    "normalize_stock_bars",
    "parse_symbol_tokens",
    "repair_recent_history",
    "resolve_fetch_start",
    "resolve_stock_symbols",
    "run_collector_loop",
]
