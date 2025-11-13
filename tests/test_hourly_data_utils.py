from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.hourly_data_utils import HourlyDataStatus, HourlyDataValidator, discover_hourly_symbols


def _write_hourly_csv(path: Path, timestamps):
    frame = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "Open": [1.0 + idx for idx in range(len(timestamps))],
            "High": [1.5 + idx for idx in range(len(timestamps))],
            "Low": [0.5 + idx for idx in range(len(timestamps))],
            "Close": [1.2 + idx for idx in range(len(timestamps))],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_discover_hourly_symbols_prefers_uppercase(tmp_path: Path):
    stocks_dir = tmp_path / "stocks"
    crypto_dir = tmp_path / "crypto"
    _write_hourly_csv(stocks_dir / "aapl.csv", [datetime.now(timezone.utc)])
    _write_hourly_csv(crypto_dir / "BTCUSD.csv", [datetime.now(timezone.utc)])
    _write_hourly_csv(tmp_path / "ethusd.csv", [datetime.now(timezone.utc)])

    discovered = discover_hourly_symbols(tmp_path)
    assert discovered == ["AAPL", "BTCUSD", "ETHUSD"]


def test_hourly_data_validator_flags_missing_and_stale(tmp_path: Path):
    fresh_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    stale_ts = datetime.now(timezone.utc) - timedelta(hours=5)

    _write_hourly_csv(tmp_path / "stocks" / "AAPL.csv", [fresh_ts])
    _write_hourly_csv(tmp_path / "crypto" / "BTCUSD.csv", [stale_ts])

    validator = HourlyDataValidator(tmp_path, max_staleness_hours=2)
    statuses, issues = validator.filter_ready(["AAPL", "BTCUSD", "MSFT"])

    assert [status.symbol for status in statuses] == ["AAPL"]
    assert len(issues) == 2
    reasons = {issue.symbol: issue.reason for issue in issues}
    assert reasons == {"BTCUSD": "stale", "MSFT": "missing"}


def test_hourly_data_validator_reports_close_value(tmp_path: Path):
    ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    _write_hourly_csv(tmp_path / "ETHUSD.csv", [ts])

    validator = HourlyDataValidator(tmp_path, max_staleness_hours=3)
    statuses, issues = validator.filter_ready(["ETHUSD"])

    assert issues == []
    assert len(statuses) == 1
    status = statuses[0]
    assert isinstance(status, HourlyDataStatus)
    assert status.symbol == "ETHUSD"
    assert abs(status.latest_close - 1.2) < 1e-6
