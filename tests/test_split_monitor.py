"""Tests for src/split_monitor — use unittest.mock to avoid real yfinance calls."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta, timezone


def make_mock_splits(splits_dict: dict) -> pd.Series:
    """Build a timezone-aware pd.Series for mocking yfinance splits."""
    if not splits_dict:
        return pd.Series([], dtype=float)
    index = pd.DatetimeIndex(list(splits_dict.keys()), tz="UTC")
    return pd.Series(list(splits_dict.values()), index=index)


def test_check_recent_splits_finds_split():
    from src.split_monitor import check_recent_splits
    recent_date = datetime.now(tz=timezone.utc) - timedelta(days=2)
    mock_splits = make_mock_splits({recent_date: 10.0})

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.splits = mock_splits
        mock_ticker_cls.return_value = mock_ticker

        result = check_recent_splits(["NVDA"], lookback_days=7)

    assert "NVDA" in result
    assert result["NVDA"] == 10.0


def test_check_recent_splits_ignores_old_split():
    from src.split_monitor import check_recent_splits
    old_date = datetime.now(tz=timezone.utc) - timedelta(days=30)
    mock_splits = make_mock_splits({old_date: 4.0})

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.splits = mock_splits
        mock_ticker_cls.return_value = mock_ticker

        result = check_recent_splits(["AAPL"], lookback_days=7)

    assert "AAPL" not in result


def test_check_recent_splits_handles_no_yfinance():
    from src.split_monitor import check_recent_splits
    with patch.dict("sys.modules", {"yfinance": None}):
        # Should return empty dict gracefully
        result = check_recent_splits(["AAPL"])
    assert result == {}


def test_get_split_affected_symbols_filters_held():
    from src.split_monitor import get_split_affected_symbols
    recent_date = datetime.now(tz=timezone.utc) - timedelta(days=1)

    with patch("src.split_monitor.check_recent_splits") as mock_check:
        mock_check.return_value = {"NVDA": 10.0}
        affected = get_split_affected_symbols(["AAPL", "NVDA", "MSFT"])

    assert affected == ["NVDA"]
    assert "AAPL" not in affected


def test_log_split_event_creates_file(tmp_path):
    from src.split_monitor import log_split_event
    log_split_event("NVDA", 10.0, log_dir=str(tmp_path))
    log_file = tmp_path / "split_events.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "NVDA" in content
    assert "10:1" in content


def test_check_recent_splits_empty_list():
    from src.split_monitor import check_recent_splits
    result = check_recent_splits([])
    assert result == {}


def test_check_recent_splits_multiple_symbols():
    from src.split_monitor import check_recent_splits
    recent_date = datetime.now(tz=timezone.utc) - timedelta(days=3)
    old_date = datetime.now(tz=timezone.utc) - timedelta(days=30)

    mock_nvda_splits = make_mock_splits({recent_date: 10.0})
    mock_aapl_splits = make_mock_splits({old_date: 4.0})
    mock_msft_splits = make_mock_splits({})

    def ticker_side_effect(sym):
        m = MagicMock()
        if sym == "NVDA":
            m.splits = mock_nvda_splits
        elif sym == "AAPL":
            m.splits = mock_aapl_splits
        else:
            m.splits = mock_msft_splits
        return m

    with patch("yfinance.Ticker", side_effect=ticker_side_effect):
        result = check_recent_splits(["NVDA", "AAPL", "MSFT"], lookback_days=7)

    assert "NVDA" in result
    assert result["NVDA"] == 10.0
    assert "AAPL" not in result
    assert "MSFT" not in result


def test_check_recent_splits_handles_ticker_error():
    from src.split_monitor import check_recent_splits

    def ticker_side_effect(sym):
        raise RuntimeError("network error")

    with patch("yfinance.Ticker", side_effect=ticker_side_effect):
        # Should return empty dict without raising
        result = check_recent_splits(["NVDA"])

    assert result == {}


def test_log_split_event_appends(tmp_path):
    from src.split_monitor import log_split_event
    log_split_event("NVDA", 10.0, log_dir=str(tmp_path))
    log_split_event("AAPL", 4.0, log_dir=str(tmp_path))
    log_file = tmp_path / "split_events.log"
    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    assert "NVDA" in lines[0]
    assert "10:1" in lines[0]
    assert "AAPL" in lines[1]
    assert "4:1" in lines[1]


def test_get_split_affected_symbols_empty_positions():
    from src.split_monitor import get_split_affected_symbols
    result = get_split_affected_symbols([])
    assert result == []


def test_check_recent_splits_naive_index_localized():
    """Ensure splits with naive datetime index are handled correctly."""
    from src.split_monitor import check_recent_splits
    recent_date = datetime.now(tz=timezone.utc) - timedelta(days=2)
    # Create a naive (no tz) index
    naive_index = pd.DatetimeIndex([recent_date.replace(tzinfo=None)])
    mock_splits = pd.Series([10.0], index=naive_index)

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.splits = mock_splits
        mock_ticker_cls.return_value = mock_ticker

        result = check_recent_splits(["TSLA"], lookback_days=7)

    assert "TSLA" in result
    assert result["TSLA"] == 10.0
