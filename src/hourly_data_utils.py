from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _candidate_hourly_paths(symbol: str, data_root: Path) -> List[Path]:
    upper = symbol.upper()
    lower = symbol.lower()
    return [
        data_root / f"{upper}.csv",
        data_root / f"{lower}.csv",
        data_root / "stocks" / f"{upper}.csv",
        data_root / "stocks" / f"{lower}.csv",
        data_root / "crypto" / f"{upper}.csv",
        data_root / "crypto" / f"{lower}.csv",
    ]


def discover_hourly_symbols(data_root: Path) -> List[str]:
    """Return all symbols with hourly CSVs under ``data_root`` (stocks + crypto)."""
    root = Path(data_root)
    if not root.exists():
        return []
    candidates: List[str] = []
    seen = set()
    folders = [root, root / "stocks", root / "crypto"]
    for folder in folders:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            symbol = path.stem.upper()
            if symbol in seen:
                continue
            seen.add(symbol)
            candidates.append(symbol)
    return sorted(candidates)


class HourlyDataError(RuntimeError):
    """Base error for hourly data validation failures."""

    reason = "unknown"

    def __init__(self, message: str):
        super().__init__(message)


class HourlyDataMissing(HourlyDataError):
    reason = "missing"


class HourlyDataStale(HourlyDataError):
    reason = "stale"


@dataclass(frozen=True)
class HourlyDataStatus:
    symbol: str
    path: Path
    latest_timestamp: datetime
    latest_close: float
    staleness_hours: float


@dataclass(frozen=True)
class HourlyDataIssue:
    symbol: str
    reason: str
    detail: str


class HourlyDataValidator:
    """
    Validate that hourly CSV data exists and is fresh for the provided symbols.

    ``max_staleness_hours`` defines the allowable lag between the most recent bar and ``datetime.now(UTC)``.
    """

    def __init__(self, data_root: Path, *, max_staleness_hours: int = 6):
        self.data_root = Path(data_root)
        self.max_staleness_hours = max(1, int(max_staleness_hours))

    def filter_ready(self, symbols: Iterable[str]) -> Tuple[List[HourlyDataStatus], List[HourlyDataIssue]]:
        ready: List[HourlyDataStatus] = []
        issues: List[HourlyDataIssue] = []
        for symbol in symbols:
            normalized = symbol.upper()
            try:
                status = self._build_status(normalized)
            except HourlyDataError as exc:
                issues.append(HourlyDataIssue(symbol=normalized, reason=exc.reason, detail=str(exc)))
                continue
            ready.append(status)
        return ready, issues

    def _build_status(self, symbol: str) -> HourlyDataStatus:
        path = self._resolve_symbol_path(symbol)
        if path is None:
            raise HourlyDataMissing(f"No hourly CSV found for {symbol} under {self.data_root}")
        try:
            frame = pd.read_csv(path)
        except OSError as exc:  # pragma: no cover - filesystem errors
            raise HourlyDataMissing(f"Failed to read {path}: {exc}") from exc

        if "timestamp" not in frame.columns or frame.empty:
            raise HourlyDataMissing(f"{path} has no timestamp column or is empty")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            raise HourlyDataMissing(f"{path} contained no valid timestamps for {symbol}")

        frame = frame.sort_values("timestamp")
        latest_row = frame.iloc[-1]
        latest_timestamp = latest_row["timestamp"].to_pydatetime()
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
        else:
            latest_timestamp = latest_timestamp.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        staleness_hours = max(0.0, (now - latest_timestamp).total_seconds() / 3600.0)
        if staleness_hours > self.max_staleness_hours:
            raise HourlyDataStale(
                f"{symbol} hourly data is {staleness_hours:.2f}h old "
                f"(threshold {self.max_staleness_hours}h)"
            )

        close_value = float(latest_row.get("Close") or latest_row.get("close") or 0.0)
        return HourlyDataStatus(
            symbol=symbol,
            path=path,
            latest_timestamp=latest_timestamp,
            latest_close=close_value,
            staleness_hours=staleness_hours,
        )

    def _resolve_symbol_path(self, symbol: str) -> Path | None:
        return resolve_hourly_symbol_path(symbol, self.data_root)


def resolve_hourly_symbol_path(symbol: str, data_root: Path) -> Optional[Path]:
    """Return the first CSV path matching ``symbol`` under ``data_root`` if present."""
    root = Path(data_root)
    if not root.exists():
        return None
    for candidate in _candidate_hourly_paths(symbol, root):
        if candidate.exists():
            return candidate
    return None


def summarize_statuses(statuses: Sequence[HourlyDataStatus]) -> str:
    """Human readable summary for logging."""
    if not statuses:
        return "no hourly data available"
    hours = [status.staleness_hours for status in statuses]
    return (
        f"{len(statuses)} symbols | "
        f"median lag={median(hours):.2f}h | "
        f"max lag={max(hours):.2f}h"
    )


__all__ = [
    "discover_hourly_symbols",
    "HourlyDataValidator",
    "HourlyDataStatus",
    "HourlyDataIssue",
    "summarize_statuses",
    "resolve_hourly_symbol_path",
]
