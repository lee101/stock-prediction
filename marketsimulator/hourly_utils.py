"""Helpers for loading and querying hourly bar data for simulator fills."""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_HOURLY_ROOT = Path(__file__).resolve().parents[1] / "trainingdatahourly"
ENV_HOURLY_ROOT = "MARKETSIM_HOURLY_DATA_ROOT"
logger = logging.getLogger(__name__)


def _resolve_hourly_root(data_root: Path) -> Path:
    env_value = os.getenv(ENV_HOURLY_ROOT, "").strip()
    if not env_value:
        return data_root
    candidate = Path(env_value).expanduser()
    if candidate.exists():
        return candidate
    logger.warning(
        "MARKETSIM_HOURLY_DATA_ROOT=%s does not exist; falling back to %s",
        candidate,
        data_root,
    )
    return data_root


def _candidate_hourly_paths(symbol: str, data_root: Path) -> Iterable[Path]:
    upper = symbol.upper()
    lower = symbol.lower()
    yield data_root / f"{upper}.csv"
    yield data_root / f"{lower}.csv"
    yield data_root / "crypto" / f"{upper}.csv"
    yield data_root / "crypto" / f"{lower}.csv"
    yield data_root / "stocks" / f"{upper}.csv"
    yield data_root / "stocks" / f"{lower}.csv"


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key == "timestamp":
            rename_map[column] = "timestamp"
        elif key == "open":
            rename_map[column] = "Open"
        elif key == "high":
            rename_map[column] = "High"
        elif key == "low":
            rename_map[column] = "Low"
        elif key == "close":
            rename_map[column] = "Close"
        elif key == "volume":
            rename_map[column] = "Volume"
    if rename_map:
        frame = frame.rename(columns=rename_map)
    return frame


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise ValueError("hourly data requires a timestamp column")
    frame = _normalise_columns(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _resolve_hourly_path(symbol: str, data_root: Path) -> Optional[Path]:
    for candidate in _candidate_hourly_paths(symbol, data_root):
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=64)
def load_hourly_bars(symbol: str, data_root: Path = DEFAULT_HOURLY_ROOT) -> pd.DataFrame:
    """
    Load hourly bars for ``symbol`` from ``data_root``.

    Returns an empty DataFrame when no data is available.
    """
    if data_root == DEFAULT_HOURLY_ROOT:
        resolved_root = _resolve_hourly_root(data_root)
        if resolved_root != data_root:
            return load_hourly_bars(symbol, data_root=resolved_root)
    if not data_root.exists():
        return pd.DataFrame()
    path = _resolve_hourly_path(symbol, data_root)
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    try:
        return _prepare_frame(frame)
    except ValueError:
        return pd.DataFrame()


def get_intraday_slice(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_root: Path = DEFAULT_HOURLY_ROOT,
) -> pd.DataFrame:
    """
    Return hourly bars between ``start`` (inclusive) and ``end`` (inclusive).

    ``start`` and ``end`` must be timezone-aware ``Timestamp`` objects.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware timestamps")
    frame = load_hourly_bars(symbol, data_root)
    if frame.empty:
        return frame
    mask = (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
    return frame.loc[mask].copy()


def get_intraday_for_date(
    symbol: str,
    session_date: pd.Timestamp,
    data_root: Path = DEFAULT_HOURLY_ROOT,
) -> pd.DataFrame:
    """
    Convenience wrapper returning hourly bars for the UTC date of ``session_date``.
    """
    if session_date.tzinfo is None:
        raise ValueError("session_date must be timezone-aware")
    session_utc = session_date.tz_convert("UTC")
    start = session_utc.normalize()
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return get_intraday_slice(symbol, start, end, data_root=data_root)
