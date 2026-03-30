"""Tests for binance_worksteal/fetch_data.py"""
import csv
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from binance_worksteal.fetch_data import (
    CSV_COLUMNS,
    MIN_DAYS,
    fetch_and_save,
    get_top_symbols,
    kline_to_row,
    read_existing_csv,
    validate_symbol,
    write_csv,
)

SAMPLE_KLINE = [
    1640995200000,  # 2022-01-01 00:00:00 UTC
    "46216.93",
    "47954.63",
    "46208.37",
    "47722.65",
    "19604.46325",
    1641081599999,  # close time
    "924123456.78",  # quote volume
    714899,  # trade count
    "10000.0",
    "470000000.0",
    "0",
]

SAMPLE_KLINE_2 = [
    1641081600000,  # 2022-01-02 00:00:00 UTC
    "47722.66",
    "47990.0",
    "46654.0",
    "47286.18",
    "18340.4604",
    1641167999999,
    "866789012.34",
    709624,
    "9000.0",
    "425000000.0",
    "0",
]


def test_kline_to_row_basic():
    row = kline_to_row(SAMPLE_KLINE, "BTCUSDT")
    assert row["symbol"] == "BTCUSDT"
    assert row["timestamp"] == "2022-01-01 00:00:00+00:00"
    assert row["open"] == 46216.93
    assert row["high"] == 47954.63
    assert row["low"] == 46208.37
    assert row["close"] == 47722.65
    assert row["volume"] == 19604.46325
    assert row["trade_count"] == 714899
    assert isinstance(row["vwap"], float)
    assert row["vwap"] > 0


def test_kline_to_row_vwap_calculation():
    row = kline_to_row(SAMPLE_KLINE, "BTCUSDT")
    expected_vwap = 924123456.78 / 19604.46325
    assert abs(row["vwap"] - expected_vwap) < 0.01


def test_kline_to_row_zero_volume():
    kline = list(SAMPLE_KLINE)
    kline[5] = "0"
    row = kline_to_row(kline, "TESTUSDT")
    assert row["vwap"] == pytest.approx((46216.93 + 47954.63 + 46208.37 + 47722.65) / 4, rel=1e-4)


def test_csv_columns_match_existing():
    btc_path = Path(__file__).resolve().parent.parent / "trainingdata" / "train" / "BTCUSDT.csv"
    if not btc_path.exists():
        pytest.skip("BTCUSDT.csv not available")
    with open(btc_path) as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == CSV_COLUMNS


def test_write_and_read_csv():
    rows = [kline_to_row(SAMPLE_KLINE, "TESTUSDT"), kline_to_row(SAMPLE_KLINE_2, "TESTUSDT")]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "TESTUSDT.csv"
        write_csv(path, rows)
        read_rows, last_ts = read_existing_csv(path)
        assert len(read_rows) == 2
        assert last_ts is not None
        assert last_ts.year == 2022
        assert last_ts.month == 1
        assert last_ts.day == 2
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_COLUMNS


def test_read_nonexistent_csv():
    rows, last_ts = read_existing_csv(Path("/nonexistent/path.csv"))
    assert rows == []
    assert last_ts is None


def test_write_csv_creates_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "dir" / "TEST.csv"
        write_csv(path, [kline_to_row(SAMPLE_KLINE, "TESTUSDT")])
        assert path.exists()


def test_incremental_update_deduplicates():
    """Existing rows should not be duplicated when fetching overlapping data."""
    row1 = kline_to_row(SAMPLE_KLINE, "TESTUSDT")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "TESTUSDT.csv"
        write_csv(path, [row1])

        with patch("binance_worksteal.fetch_data.fetch_all_klines") as mock_fetch:
            mock_fetch.return_value = [SAMPLE_KLINE, SAMPLE_KLINE_2]
            result = fetch_and_save("TESTUSDT", data_dir=Path(tmp))

        read_rows, _ = read_existing_csv(result)
        dates = [r["timestamp"][:10] for r in read_rows]
        assert len(dates) == len(set(dates)), "Duplicate dates found"
        assert len(read_rows) == 2


def test_fetch_and_save_new_symbol():
    """Fetching a new symbol should create CSV from scratch."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("binance_worksteal.fetch_data.fetch_all_klines") as mock_fetch:
            mock_fetch.return_value = [SAMPLE_KLINE, SAMPLE_KLINE_2]
            result = fetch_and_save("NEWUSDT", data_dir=Path(tmp))

        assert result.exists()
        rows, _ = read_existing_csv(result)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "NEWUSDT"


def test_fetch_and_save_auto_append_usdt():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("binance_worksteal.fetch_data.fetch_all_klines") as mock_fetch:
            mock_fetch.return_value = [SAMPLE_KLINE]
            result = fetch_and_save("BTC", data_dir=Path(tmp))
        assert result.name == "BTCUSDT.csv"


def test_fetch_and_save_no_data_warning(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        with patch("binance_worksteal.fetch_data.fetch_all_klines") as mock_fetch:
            mock_fetch.return_value = []
            fetch_and_save("DEADUSDT", data_dir=Path(tmp))
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "no data" in captured.out


def test_fetch_and_save_min_days_warning(capsys):
    klines = [list(SAMPLE_KLINE) for _ in range(5)]
    for i, k in enumerate(klines):
        k[0] = SAMPLE_KLINE[0] + i * 86400000
    with tempfile.TemporaryDirectory() as tmp:
        with patch("binance_worksteal.fetch_data.fetch_all_klines") as mock_fetch:
            mock_fetch.return_value = klines
            fetch_and_save("SMALLUSDT", data_dir=Path(tmp))
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert str(MIN_DAYS) in captured.out


@patch("binance_worksteal.fetch_data.requests.get")
def test_get_top_symbols(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = [
        {"symbol": "BTCUSDT", "quoteVolume": "1000000000"},
        {"symbol": "ETHUSDT", "quoteVolume": "500000000"},
        {"symbol": "FDUSDUSDT", "quoteVolume": "999999999"},  # stablecoin
        {"symbol": "SOLUSDT", "quoteVolume": "200000000"},
        {"symbol": "BTCEUR", "quoteVolume": "100000000"},  # not USDT
    ]
    mock_get.return_value = mock_resp
    result = get_top_symbols(n=3)
    assert "BTCUSDT" in result
    assert "ETHUSDT" in result
    assert "SOLUSDT" in result
    assert "FDUSDUSDT" not in result
    assert "BTCEUR" not in result


@patch("binance_worksteal.fetch_data.requests.get")
def test_validate_symbol_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    assert validate_symbol("BTCUSDT") is True


@patch("binance_worksteal.fetch_data.requests.get")
def test_validate_symbol_failure(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_get.return_value = mock_resp
    assert validate_symbol("FAKECOIN") is False


def test_date_parsing_formats():
    """Ensure we parse the timestamp format used in existing CSVs."""
    ts_str = "2022-01-01 00:00:00+00:00"
    dt = datetime.fromisoformat(ts_str)
    assert dt.year == 2022
    assert dt.tzinfo is not None


def test_csv_format_matches_existing():
    """Generated CSV should match format of existing training data."""
    row = kline_to_row(SAMPLE_KLINE, "BTCUSDT")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "TEST.csv"
        write_csv(path, [row])
        with open(path) as f:
            content = f.read()
        lines = content.strip().split("\n")
        header = lines[0]
        assert header == ",".join(CSV_COLUMNS)
        data_line = lines[1]
        fields = data_line.split(",")
        assert len(fields) == len(CSV_COLUMNS)
        assert "+00:00" in fields[0]  # timezone-aware timestamp
        assert fields[-1] == "BTCUSDT"  # symbol column
