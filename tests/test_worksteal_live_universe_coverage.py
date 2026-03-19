"""Guards that the live work-steal universe is locally replayable."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SCRIPT = REPO_ROOT / "deployments" / "binance-worksteal-daily" / "launch.sh"
MIN_HOURLY_ROWS = 24 * 60
MIN_DAILY_ROWS = 60
MAX_END_LAG = pd.Timedelta(days=14)


def _extract_live_symbols() -> list[str]:
    text = LAUNCH_SCRIPT.read_text()
    match = re.search(r"--symbols(?P<body>.*?)\"\$@\"", text, re.S)
    if match is None:
        raise AssertionError(f"Unable to extract --symbols block from {LAUNCH_SCRIPT}.")
    symbols = [token for token in match.group("body").replace("\\", " ").split() if token]
    if not symbols:
        raise AssertionError(f"No live symbols found in {LAUNCH_SCRIPT}.")
    return symbols


def test_live_launch_symbols_have_local_daily_and_hourly_coverage():
    symbols = _extract_live_symbols()
    daily_root = REPO_ROOT / "trainingdatadailybinance"
    hourly_root = REPO_ROOT / "trainingdatahourly" / "crypto"

    missing_daily = [symbol for symbol in symbols if not (daily_root / f"{symbol}.csv").exists()]
    missing_hourly = [symbol for symbol in symbols if not (hourly_root / f"{symbol}.csv").exists()]

    assert not missing_daily, f"Missing daily replay coverage for live symbols: {missing_daily}"
    assert not missing_hourly, f"Missing hourly replay coverage for live symbols: {missing_hourly}"

    hourly_ends: dict[str, pd.Timestamp] = {}
    daily_ends: dict[str, pd.Timestamp] = {}
    sparse_hourly: list[str] = []
    sparse_daily: list[str] = []

    for symbol in symbols:
        hourly = pd.read_csv(hourly_root / f"{symbol}.csv", usecols=["timestamp"])
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
        hourly = hourly.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(hourly) < MIN_HOURLY_ROWS:
            sparse_hourly.append(f"{symbol}({len(hourly)})")
        hourly_ends[symbol] = pd.Timestamp(hourly["timestamp"].iloc[-1])

        daily = pd.read_csv(daily_root / f"{symbol}.csv", usecols=["timestamp"])
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True, errors="coerce")
        daily = daily.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(daily) < MIN_DAILY_ROWS:
            sparse_daily.append(f"{symbol}({len(daily)})")
        daily_ends[symbol] = pd.Timestamp(daily["timestamp"].iloc[-1])

    reference_hourly_end = max(hourly_ends.values())
    reference_daily_end = max(daily_ends.values())
    stale_hourly = [
        symbol
        for symbol, end in hourly_ends.items()
        if reference_hourly_end - end > MAX_END_LAG
    ]
    stale_daily = [
        symbol
        for symbol, end in daily_ends.items()
        if reference_daily_end - end > MAX_END_LAG
    ]

    assert not sparse_hourly, f"Insufficient hourly replay depth for live symbols: {sparse_hourly}"
    assert not sparse_daily, f"Insufficient daily replay depth for live symbols: {sparse_daily}"
    assert not stale_hourly, (
        f"Hourly replay coverage lags the live universe by more than {MAX_END_LAG}: {stale_hourly}"
    )
    assert not stale_daily, (
        f"Daily replay coverage lags the live universe by more than {MAX_END_LAG}: {stale_daily}"
    )
