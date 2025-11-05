import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

import trade_stock_e2e as trade_module


@pytest.fixture
def reset_forecast_cache(monkeypatch):
    monkeypatch.setattr(trade_module, "_LATEST_FORECAST_CACHE", {}, raising=False)
    monkeypatch.setattr(trade_module, "_LATEST_FORECAST_PATH", None, raising=False)
    return None


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        (float("nan"), None),
        (7, 7.0),
        (3.25, 3.25),
        ("  4.5  ", 4.5),
        ("invalid", None),
    ],
)
def test_coerce_optional_float_handles_common_inputs(raw, expected):
    assert trade_module.coerce_optional_float(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("[1, 2.5, None]", [1.0, 2.5]),
        ("[]", None),
        ("", None),
        ("not-a-list", None),
    ],
)
def test_parse_float_list_filters_invalid_entries(raw, expected):
    assert trade_module.parse_float_list(raw) == expected


def test_load_latest_forecast_snapshot_prefers_newer_file(tmp_path, monkeypatch, reset_forecast_cache):
    monkeypatch.setattr(trade_module, "_results_dir", lambda: tmp_path)

    older_file = tmp_path / "predictions-20240101.csv"
    newer_file = tmp_path / "predictions-20250101.csv"

    pd.DataFrame(
        {
            "instrument": ["AAPL"],
            "maxdiffprofit_profit": [1.0],
            "entry_takeprofit_profit": [0.5],
        }
    ).to_csv(older_file, index=False)

    old_ts = datetime.now() - timedelta(days=1)
    os.utime(older_file, (old_ts.timestamp(), old_ts.timestamp()))

    pd.DataFrame(
        {
            "instrument": ["MSFT"],
            "maxdiffprofit_profit": [2.5],
            "entry_takeprofit_profit": [0.75],
            "entry_takeprofit_profit_values": ["[0.05, None, 0.1]"],
            "takeprofit_low_price": ["301.4"],
        }
    ).to_csv(newer_file, index=False)

    snapshot = trade_module._load_latest_forecast_snapshot()

    assert "MSFT" in snapshot and "AAPL" not in snapshot
    msft_entry = snapshot["MSFT"]
    assert msft_entry["entry_takeprofit_profit"] == 0.75
    assert msft_entry["takeprofit_low_price"] == 301.4
    assert msft_entry["entry_takeprofit_profit_values"] == [0.05, 0.1]

    pd.DataFrame(
        {
            "instrument": ["MSFT"],
            "entry_takeprofit_profit": [0.12],
        }
    ).to_csv(newer_file, index=False)

    cached = trade_module._load_latest_forecast_snapshot()
    assert cached is snapshot


def test_load_latest_forecast_snapshot_handles_missing_directory(tmp_path, monkeypatch, reset_forecast_cache):
    missing = tmp_path / "nope"
    monkeypatch.setattr(trade_module, "_results_dir", lambda: missing)

    snapshot = trade_module._load_latest_forecast_snapshot()
    assert snapshot == {}
    assert trade_module._LATEST_FORECAST_PATH is None


def test_load_latest_forecast_snapshot_handles_corrupt_file(tmp_path, monkeypatch, reset_forecast_cache):
    monkeypatch.setattr(trade_module, "_results_dir", lambda: tmp_path)

    corrupt_file = tmp_path / "predictions-20250202.csv"
    corrupt_file.write_text("instrument,maxdiffprofit_profit\naapl,1\n\"broken")

    snapshot = trade_module._load_latest_forecast_snapshot()
    assert snapshot == {}
    assert trade_module._LATEST_FORECAST_PATH == corrupt_file


def test_find_latest_prediction_file_prefers_recent(tmp_path, monkeypatch, reset_forecast_cache):
    monkeypatch.setattr(trade_module, "_results_dir", lambda: tmp_path)

    older = tmp_path / "predictions-1.csv"
    newer = tmp_path / "predictions-2.csv"
    older.write_text("instrument\nAAPL\n")
    newer.write_text("instrument\nMSFT\n")

    past = datetime.now() - timedelta(days=2)
    os.utime(older, (past.timestamp(), past.timestamp()))

    result = trade_module._find_latest_prediction_file()
    assert result == newer
