"""Detect recent stock splits via yfinance and identify affected positions."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def check_recent_splits(
    symbols: list[str],
    lookback_days: int = 7,
) -> dict[str, float]:
    """Return {symbol: split_ratio} for symbols that split in the last lookback_days.

    split_ratio > 1 means forward split (e.g. 10.0 = 10:1 split).
    Returns empty dict if yfinance is unavailable or no splits found.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; split detection disabled")
        return {}

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    result = {}
    for sym in symbols:
        try:
            splits = yf.Ticker(sym).splits
            if len(splits) == 0:
                continue
            # Convert index to UTC if not already
            if splits.index.tz is None:
                splits.index = splits.index.tz_localize("UTC")
            else:
                splits.index = splits.index.tz_convert("UTC")
            recent = splits[splits.index >= cutoff]
            if len(recent) > 0:
                result[sym] = float(recent.iloc[-1])
                logger.warning(
                    f"Recent split detected: {sym} {recent.iloc[-1]:.0f}:1 "
                    f"on {recent.index[-1].date()}"
                )
        except Exception as e:
            logger.debug(f"Could not check splits for {sym}: {e}")
    return result


def get_split_affected_symbols(
    held_symbols: list[str],
    lookback_days: int = 7,
) -> list[str]:
    """Return symbols from held_symbols that have had a recent split.

    Call this before each trading decision. If any symbol is affected,
    close its position before the policy sees the distorted price data.
    """
    splits = check_recent_splits(held_symbols, lookback_days=lookback_days)
    return list(splits.keys())


def log_split_event(symbol: str, ratio: float, log_dir: str = "logs") -> None:
    """Append a split event to logs/split_events.log."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "split_events.log"
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    line = f"{timestamp}: {symbol} split {ratio:.0f}:1 — position closed\n"
    with open(log_path, "a") as f:
        f.write(line)
    logger.info(f"Split event logged: {line.strip()}")
