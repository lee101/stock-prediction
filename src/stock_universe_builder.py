from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.symbol_utils import is_crypto_symbol


_EXCLUDED_SYMBOLS = {"CORRELATION_MATRIX", "DATA_SUMMARY", "VOLATILITY_METRICS"}


@dataclass(frozen=True)
class StockUniverseCandidate:
    symbol: str
    rows: int
    last_timestamp: str
    last_close: float
    median_dollar_volume: float
    avg_dollar_volume: float
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def is_candidate_stock_symbol(symbol: str) -> bool:
    clean = str(symbol or "").strip().upper()
    if not clean:
        return False
    if clean in _EXCLUDED_SYMBOLS:
        return False
    if is_crypto_symbol(clean):
        return False
    if not clean.replace("-", "").replace("_", "").isalnum():
        return False
    return True


def summarize_daily_stock_csv(
    path: Path,
    *,
    lookback_rows: int = 60,
    min_history_rows: int = 252,
    min_last_close: float = 3.0,
    min_median_dollar_volume: float = 2_000_000.0,
    min_last_timestamp: str | None = None,
) -> StockUniverseCandidate | None:
    symbol = path.stem.upper()
    if not is_candidate_stock_symbol(symbol):
        return None

    frame = pd.read_csv(path)
    if frame.empty or len(frame) < int(min_history_rows):
        return None
    if "timestamp" not in frame.columns or "close" not in frame.columns or "volume" not in frame.columns:
        return None

    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    valid = pd.DataFrame({"timestamp": timestamp, "close": close, "volume": volume}).dropna()
    if valid.empty or len(valid) < int(min_history_rows):
        return None

    last_close = float(valid["close"].iloc[-1])
    if last_close < float(min_last_close):
        return None

    lookback = valid.tail(max(1, int(lookback_rows))).copy()
    lookback["dollar_volume"] = lookback["close"] * lookback["volume"]
    median_dollar_volume = float(lookback["dollar_volume"].median())
    if median_dollar_volume < float(min_median_dollar_volume):
        return None
    avg_dollar_volume = float(lookback["dollar_volume"].mean())
    last_timestamp = valid["timestamp"].iloc[-1].isoformat()
    if min_last_timestamp is not None and last_timestamp < str(min_last_timestamp):
        return None

    # Prefer names that are liquid first, with history as a weaker tiebreaker.
    score = median_dollar_volume + float(len(valid)) * 1_000.0
    return StockUniverseCandidate(
        symbol=symbol,
        rows=int(len(valid)),
        last_timestamp=last_timestamp,
        last_close=last_close,
        median_dollar_volume=median_dollar_volume,
        avg_dollar_volume=avg_dollar_volume,
        score=score,
    )


def rank_stock_universe(
    csv_paths: Iterable[Path],
    *,
    lookback_rows: int = 60,
    min_history_rows: int = 252,
    min_last_close: float = 3.0,
    min_median_dollar_volume: float = 2_000_000.0,
    min_last_timestamp: str | None = None,
    top_n: int | None = None,
) -> list[StockUniverseCandidate]:
    candidates: list[StockUniverseCandidate] = []
    for path in csv_paths:
        candidate = summarize_daily_stock_csv(
            Path(path),
            lookback_rows=lookback_rows,
            min_history_rows=min_history_rows,
            min_last_close=min_last_close,
            min_median_dollar_volume=min_median_dollar_volume,
            min_last_timestamp=min_last_timestamp,
        )
        if candidate is not None:
            candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda row: (-row.score, -row.median_dollar_volume, -row.rows, row.symbol),
    )
    if top_n is not None and top_n > 0:
        return ranked[:top_n]
    return ranked


__all__ = [
    "StockUniverseCandidate",
    "is_candidate_stock_symbol",
    "rank_stock_universe",
    "summarize_daily_stock_csv",
]
