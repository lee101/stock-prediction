from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from binance import Client

from src.binan import binance_wrapper, binance_vision

DEFAULT_HISTORY_YEARS = 10
DEFAULT_SLEEP_SECONDS = 0.25

DEFAULT_BINANCE_FDUSD_PAIRS = [
    "BTC/FDUSD",
    "ETH/FDUSD",
    "SOL/FDUSD",
    "BNB/FDUSD",
    "LINK/FDUSD",
    "ADA/FDUSD",
    "APT/FDUSD",
    "AVAX/FDUSD",
    "DOT/FDUSD",
    # MATIC markets have been deprecated on Binance.US (status BREAK); use POL instead.
    "POL/FDUSD",
    "ATOM/FDUSD",
    "LTC/FDUSD",
    "BCH/FDUSD",
    "UNI/FDUSD",
]

DEFAULT_BINANCE_U_PAIRS = [
    "BTC/U",
    "ETH/U",
    "SOL/U",
    "BNB/U",
    # Stablecoin conversion for manual USDT -> U moves.
    "U/USDT",
]

_STABLECOIN_QUOTES = ["FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USDP", "U", "USD"]
DEFAULT_FALLBACK_QUOTES = ["USDT", "U", "USDC"]


@dataclass
class DownloadResult:
    symbol: str
    status: str
    bars: int
    start: Optional[str] = None
    end: Optional[str] = None
    file: Optional[str] = None
    added_bars: Optional[int] = None
    error: Optional[str] = None
    resolved_symbol: Optional[str] = None


def _normalize_pair(pair: str) -> str:
    cleaned = pair.strip().upper().replace("-", "/").replace("_", "/")
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
    else:
        base = None
        quote = None
        for candidate in _STABLECOIN_QUOTES:
            if cleaned.endswith(candidate):
                base = cleaned[: -len(candidate)]
                quote = candidate
                break
        if base is None or quote is None:
            raise ValueError(f"Unable to parse pair '{pair}'. Use BASE/QUOTE format.")
    if not base or not quote:
        raise ValueError(f"Invalid pair '{pair}'.")
    symbol = f"{base}{quote}"
    if not symbol.isalnum():
        raise ValueError(f"Invalid pair '{pair}'. Expected alphanumeric asset symbols.")
    return symbol


def _split_pair(pair: str) -> Tuple[str, str]:
    normalized = pair.strip().upper().replace("-", "/").replace("_", "/")
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
    else:
        base = None
        quote = None
        for candidate in _STABLECOIN_QUOTES:
            if normalized.endswith(candidate):
                base = normalized[: -len(candidate)]
                quote = candidate
                break
        if base is None or quote is None:
            raise ValueError(f"Unable to parse pair '{pair}'. Use BASE/QUOTE format.")
    if not base or not quote:
        raise ValueError(f"Invalid pair '{pair}'.")
    return base, quote


def _symbol_exists(symbol: str, client: Client) -> bool:
    try:
        info = client.get_symbol_info(symbol)
    except Exception as exc:
        logger.warning(f"Failed to fetch Binance symbol info for {symbol}: {exc}")
        return False
    return isinstance(info, dict)


def resolve_pair_symbol(
    pair: str,
    client: Client,
    fallback_quotes: Optional[Sequence[str]] = None,
) -> Optional[str]:
    base, quote = _split_pair(pair)
    candidate = _normalize_pair(pair)
    if _symbol_exists(candidate, client):
        return candidate

    for fallback in fallback_quotes or []:
        fallback_symbol = f"{base}{fallback}".upper()
        if _symbol_exists(fallback_symbol, client):
            logger.warning(
                f"Pair {candidate} not available; falling back to {fallback_symbol}."
            )
            return fallback_symbol
    logger.warning(f"Pair {candidate} not available on this Binance endpoint.")
    return None


def _merge_and_dedup(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = new
    else:
        if existing.index.dtype == "object":
            existing.index = pd.to_datetime(existing.index, utc=True)
        if new.index.dtype == "object":
            new.index = pd.to_datetime(new.index, utc=True)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")
        if new.index.tz is None:
            new.index = new.index.tz_localize("UTC")
        combined = pd.concat([existing, new], ignore_index=False)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def fetch_binance_hourly_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    client: Optional[Client] = None,
) -> pd.DataFrame:
    client = client or binance_wrapper.get_client()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    start_ms = max(0, int(start.timestamp() * 1000))
    end_ms = max(0, int(end.timestamp() * 1000))

    try:
        rows = client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1HOUR,
            start_ms,
            end_ms,
        )
    except Exception as exc:
        logger.warning(f"Failed fetching Binance klines for {symbol} via REST API: {exc}")
        rows = []

    if not rows:
        # Fall back to Binance Vision (public dataset) when the live API is blocked
        # or the symbol is unavailable on the configured endpoint (e.g., FDUSD/U pairs
        # while using Binance.US).
        try:
            vision_frame = binance_vision.fetch_vision_hourly_klines(
                symbol=symbol,
                start=start,
                end=end,
                interval="1h",
            )
        except Exception as exc:
            logger.error(f"Failed fetching Binance Vision klines for {symbol}: {exc}")
            return pd.DataFrame()
        if not vision_frame.empty:
            logger.info(f"Fetched {len(vision_frame)} bars for {symbol} via Binance Vision.")
        return vision_frame

    records = []
    for row in rows:
        if not row:
            continue
        try:
            open_time = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
            close_time = datetime.fromtimestamp(row[6] / 1000, tz=timezone.utc) if len(row) > 6 else None
            if close_time is not None and close_time > end:
                # Binance may include the current in-progress candle when requesting up to "now".
                # Persisting it causes look-ahead leakage for training and evaluation.
                continue
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            volume = float(row[5])
            quote_volume = float(row[7]) if len(row) > 7 else 0.0
            trade_count = int(row[8]) if len(row) > 8 else 0
        except (TypeError, ValueError, IndexError):
            continue
        vwap = quote_volume / volume if volume else close_price
        records.append(
            {
                "timestamp": open_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "trade_count": trade_count,
                "vwap": vwap,
                "symbol": symbol,
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


def _resolve_window(
    start: Optional[datetime],
    end: Optional[datetime],
    history_years: int,
) -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    end_dt = end or now
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    start_dt = start or (end_dt - timedelta(days=int(history_years * 365.25)))
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if start_dt >= end_dt:
        raise ValueError("Start time must be before end time.")
    return start_dt, end_dt


def _drop_incomplete_current_hour(
    frame: Optional[pd.DataFrame],
    *,
    now: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """Drop the current-hour kline (open_time == current hour) if present.

    Binance may return the current open candle when requesting up to "now". We only
    want fully-closed hourly bars in persisted CSVs to avoid leakage.
    """

    if frame is None or frame.empty:
        return frame
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    current_hour = now.replace(minute=0, second=0, microsecond=0)
    current_hour_ts = pd.Timestamp(current_hour)

    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" not in frame.columns:
            return frame
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        keep = ts != current_hour_ts
        if keep.all():
            return frame
        return frame.loc[keep].copy()

    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    keep_mask = idx != current_hour_ts
    if keep_mask.all():
        return frame
    cleaned = frame.loc[keep_mask].copy()
    cleaned.index = pd.to_datetime(cleaned.index, utc=True, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()]
    return cleaned


def download_and_save_pair(
    pair: str,
    output_dir: Path,
    history_years: int = DEFAULT_HISTORY_YEARS,
    client: Optional[Client] = None,
    fallback_quotes: Optional[Sequence[str]] = None,
    skip_if_exists: bool = True,
) -> DownloadResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = client or binance_wrapper.get_client()

    try:
        requested_symbol = _normalize_pair(pair)
        resolved_symbol = resolve_pair_symbol(pair, client, fallback_quotes=fallback_quotes)
    except ValueError as exc:
        return DownloadResult(symbol=pair, status="invalid", bars=0, error=str(exc))

    if not resolved_symbol:
        # Symbol is not available on the configured Binance endpoint (e.g. Binance.US)
        # but may still be available in Binance Vision's public dataset.
        resolved_symbol = requested_symbol

    output_file = output_dir / f"{resolved_symbol}.csv"
    existing_df: Optional[pd.DataFrame] = None
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, parse_dates=["timestamp"]).set_index("timestamp")
        except Exception as exc:
            logger.warning(f"Error reading existing Binance file for {resolved_symbol}: {exc}")
            existing_df = None
    existing_df_before = existing_df
    existing_df = _drop_incomplete_current_hour(existing_df)
    if (
        existing_df_before is not None
        and existing_df is not None
        and not existing_df_before.empty
        and len(existing_df) < len(existing_df_before)
    ):
        # Persist cleanup even if we later decide the file is "up to date".
        existing_df.index.name = "timestamp"
        existing_df.to_csv(output_file)

    start_dt, end_dt = _resolve_window(None, None, history_years=history_years)
    update_mode = False

    if existing_df is not None and not existing_df.empty:
        existing_df.index = pd.to_datetime(existing_df.index, utc=True, errors="coerce")
        existing_df = existing_df[~existing_df.index.isna()]
        latest_ts = existing_df.index.max() if not existing_df.empty else None
        if latest_ts is not None:
            if skip_if_exists and latest_ts >= end_dt - timedelta(hours=2):
                return DownloadResult(
                    symbol=pair,
                    status="skipped",
                    bars=len(existing_df),
                    start=str(existing_df.index.min()),
                    end=str(existing_df.index.max()),
                    file=str(output_file),
                    resolved_symbol=resolved_symbol,
                )
            if skip_if_exists:
                update_mode = True
                start_dt = max(start_dt, latest_ts - timedelta(hours=2))

    logger.info(f"Downloading Binance data for {resolved_symbol} from {start_dt} to {end_dt}")
    frame = fetch_binance_hourly_bars(resolved_symbol, start=start_dt, end=end_dt, client=client)
    frame = _drop_incomplete_current_hour(frame, now=end_dt)

    if frame.empty and (existing_df is None or existing_df.empty) and fallback_quotes:
        base, _ = _split_pair(pair)
        for quote in fallback_quotes:
            fallback_symbol = f"{base}{quote}".upper()
            if fallback_symbol == resolved_symbol:
                continue
            if not _symbol_exists(fallback_symbol, client):
                continue
            logger.warning(
                f"No data for {resolved_symbol}; retrying with fallback {fallback_symbol}."
            )
            fallback_frame = fetch_binance_hourly_bars(
                fallback_symbol, start=start_dt, end=end_dt, client=client
            )
            fallback_frame = _drop_incomplete_current_hour(fallback_frame, now=end_dt)
            if not fallback_frame.empty:
                resolved_symbol = fallback_symbol
                output_file = output_dir / f"{resolved_symbol}.csv"
                frame = fallback_frame
                existing_df = None
                if output_file.exists():
                    try:
                        existing_df = pd.read_csv(
                            output_file, parse_dates=["timestamp"]
                        ).set_index("timestamp")
                    except Exception as exc:
                        logger.warning(
                            f"Error reading existing Binance file for {resolved_symbol}: {exc}"
                        )
                        existing_df = None
                existing_df_before = existing_df
                existing_df = _drop_incomplete_current_hour(existing_df)
                if (
                    existing_df_before is not None
                    and existing_df is not None
                    and not existing_df_before.empty
                    and len(existing_df) < len(existing_df_before)
                ):
                    existing_df.index.name = "timestamp"
                    existing_df.to_csv(output_file)
                update_mode = False
                if existing_df is not None and not existing_df.empty:
                    existing_df.index = pd.to_datetime(
                        existing_df.index, utc=True, errors="coerce"
                    )
                    existing_df = existing_df[~existing_df.index.isna()]
                    latest_ts = (
                        existing_df.index.max() if not existing_df.empty else None
                    )
                    if latest_ts is not None and skip_if_exists:
                        update_mode = True
                        start_dt = max(start_dt, latest_ts - timedelta(hours=2))
                        frame = fetch_binance_hourly_bars(
                            resolved_symbol, start=start_dt, end=end_dt, client=client
                        )
                        frame = _drop_incomplete_current_hour(frame, now=end_dt)
                break

    if frame.empty:
        if update_mode and existing_df is not None and not existing_df.empty:
            return DownloadResult(
                symbol=pair,
                status="no_update",
                bars=len(existing_df),
                start=str(existing_df.index.min()),
                end=str(existing_df.index.max()),
                file=str(output_file),
                resolved_symbol=resolved_symbol,
            )
        return DownloadResult(symbol=pair, status="no_data", bars=0, resolved_symbol=resolved_symbol)

    if update_mode and existing_df is not None and not existing_df.empty:
        combined = _merge_and_dedup(existing_df, frame)
        combined_clean = _drop_incomplete_current_hour(combined, now=end_dt)
        if combined_clean is not None:
            combined = combined_clean
        added = max(0, len(combined) - len(existing_df))
        combined.index.name = "timestamp"
        combined.to_csv(output_file)
        return DownloadResult(
            symbol=pair,
            status="updated",
            bars=len(combined),
            added_bars=added,
            start=str(combined.index.min()),
            end=str(combined.index.max()),
            file=str(output_file),
            resolved_symbol=resolved_symbol,
        )

    frame_clean = _drop_incomplete_current_hour(frame, now=end_dt)
    if frame_clean is not None:
        frame = frame_clean
    frame.index.name = "timestamp"
    frame.to_csv(output_file)
    return DownloadResult(
        symbol=pair,
        status="ok",
        bars=len(frame),
        start=str(frame.index.min()),
        end=str(frame.index.max()),
        file=str(output_file),
        resolved_symbol=resolved_symbol,
    )


def download_all_pairs(
    pairs: Optional[Iterable[str]] = None,
    output_dir: Path = Path("trainingdatahourlybinance"),
    history_years: int = DEFAULT_HISTORY_YEARS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    skip_if_exists: bool = True,
    fallback_quotes: Optional[Sequence[str]] = None,
) -> List[DownloadResult]:
    selected_pairs = list(pairs) if pairs else list(DEFAULT_BINANCE_FDUSD_PAIRS)
    client = binance_wrapper.get_client()
    fallback_quotes = list(fallback_quotes) if fallback_quotes is not None else list(DEFAULT_FALLBACK_QUOTES)

    results: List[DownloadResult] = []
    for idx, pair in enumerate(selected_pairs, 1):
        logger.info(f"Processing {idx}/{len(selected_pairs)}: {pair}")
        result = download_and_save_pair(
            pair,
            output_dir=output_dir,
            history_years=history_years,
            client=client,
            fallback_quotes=fallback_quotes,
            skip_if_exists=skip_if_exists,
        )
        results.append(result)
        if idx < len(selected_pairs) and sleep_seconds:
            time_sleep = max(0.0, float(sleep_seconds))
            if time_sleep:
                from time import sleep

                sleep(time_sleep)

    summary = pd.DataFrame([r.__dict__ for r in results])
    summary_path = output_dir / "download_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    return results


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp '{value}'.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download hourly Binance spot data")
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Pairs to download (default: curated FDUSD list).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trainingdatahourlybinance"),
        help="Directory for Binance hourly CSVs.",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=DEFAULT_HISTORY_YEARS,
        help=f"Years of history to download (default: {DEFAULT_HISTORY_YEARS}).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files are up to date.",
    )
    parser.add_argument(
        "--fallback-quote",
        action="append",
        default=None,
        help="Fallback quote asset(s) if the requested pair is unavailable (default: USDT, USDC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fallback_quotes = [q.upper() for q in args.fallback_quote] if args.fallback_quote else None

    try:
        download_all_pairs(
            pairs=args.pairs,
            output_dir=args.output_dir,
            history_years=args.history_years,
            sleep_seconds=args.sleep,
            skip_if_exists=not args.force,
            fallback_quotes=fallback_quotes,
        )
    except Exception as exc:
        logger.error(f"Binance download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
