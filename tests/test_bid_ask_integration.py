"""Integration tests for bid/ask price fetching in data_curate_daily.

This test module verifies that:
1. Bid/ask prices are always populated after download_exchange_latest_data
2. Synthetic values are used when ADD_LATEST is False (default)
3. Synthetic values are used as fallback when API returns invalid data
4. Real values are used when ADD_LATEST is True and API returns valid data
"""
from __future__ import annotations

import unittest.mock as mock
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global bid/ask dictionaries before each test."""
    import data_curate_daily
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}
    yield
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}


@pytest.fixture
def mock_stock_data():
    """Create mock historical stock data."""
    now = datetime.now(timezone.utc)
    dates = pd.date_range(end=now, periods=10, freq='D', tz=timezone.utc)
    return pd.DataFrame({
        'open': [100.0] * 10,
        'high': [102.0] * 10,
        'low': [98.0] * 10,
        'close': [101.0] * 10,
    }, index=dates)


@pytest.fixture
def mock_client():
    """Create a mock Alpaca client."""
    return mock.MagicMock()


def test_bid_ask_populated_when_add_latest_false(mock_client, mock_stock_data, monkeypatch):
    """Test that synthetic bid/ask values are populated when ADD_LATEST is False."""
    # Import modules after patching
    import data_curate_daily
    from data_curate_daily import download_exchange_latest_data, get_bid, get_ask

    # Patch ADD_LATEST directly in the data_curate_daily module namespace
    import sys
    sys.modules['data_curate_daily'].ADD_LATEST = False

    # Mock download_stock_data_between_times to return our test data
    monkeypatch.setattr(
        data_curate_daily,
        'download_stock_data_between_times',
        lambda api, end, start, symbol: mock_stock_data
    )

    symbol = 'ETHUSD'

    # Call download_exchange_latest_data
    result = download_exchange_latest_data(mock_client, symbol)

    # Verify bid/ask are populated with synthetic values
    bid = get_bid(symbol)
    ask = get_ask(symbol)

    assert bid is not None, "Bid should not be None"
    assert ask is not None, "Ask should not be None"
    assert bid > 0, "Bid should be positive"
    assert ask > 0, "Ask should be positive"

    # Verify synthetic spread is 0 (both equal to last close)
    last_close = mock_stock_data.iloc[-1]['close']
    assert bid == last_close, f"Expected bid to equal last_close {last_close}, got {bid}"
    assert ask == last_close, f"Expected ask to equal last_close {last_close}, got {ask}"
    assert bid == ask, "Bid and ask should be equal (0 spread)"


def test_bid_ask_populated_when_api_returns_none(mock_client, mock_stock_data, monkeypatch):
    """Test that synthetic values are used when API returns None for bid/ask."""
    # Import modules and patch
    import data_curate_daily
    from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
    import sys
    sys.modules['data_curate_daily'].ADD_LATEST = True

    # Mock download_stock_data_between_times
    monkeypatch.setattr(
        data_curate_daily,
        'download_stock_data_between_times',
        lambda api, end, start, symbol: mock_stock_data
    )

    # Mock latest_data to raise an exception (simulating API failure)
    def mock_latest_data_exception(symbol):
        raise Exception("API error")

    import alpaca_wrapper
    monkeypatch.setattr(alpaca_wrapper, 'latest_data', mock_latest_data_exception)

    symbol = 'ETHUSD'

    # Call download_exchange_latest_data
    result = download_exchange_latest_data(mock_client, symbol)

    # Verify bid/ask are still populated with synthetic values
    bid = get_bid(symbol)
    ask = get_ask(symbol)

    assert bid is not None, "Bid should not be None even when API fails"
    assert ask is not None, "Ask should not be None even when API fails"
    assert bid > 0, "Bid should be positive"
    assert ask > 0, "Ask should be positive"


def test_bid_ask_populated_when_api_returns_zero(mock_client, mock_stock_data, monkeypatch):
    """Test that synthetic values are used when API returns zero for bid/ask."""
    # Import modules and patch
    import data_curate_daily
    from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
    import sys
    sys.modules['data_curate_daily'].ADD_LATEST = True

    # Mock download_stock_data_between_times
    monkeypatch.setattr(
        data_curate_daily,
        'download_stock_data_between_times',
        lambda api, end, start, symbol: mock_stock_data
    )

    # Mock latest_data to return zero bid/ask
    def mock_latest_data_zero(symbol):
        return SimpleNamespace(
            ask_price=0.0,
            bid_price=0.0
        )

    import alpaca_wrapper
    monkeypatch.setattr(alpaca_wrapper, 'latest_data', mock_latest_data_zero)

    symbol = 'ETHUSD'

    # Call download_exchange_latest_data
    result = download_exchange_latest_data(mock_client, symbol)

    # Verify bid/ask are populated with synthetic values
    bid = get_bid(symbol)
    ask = get_ask(symbol)

    assert bid is not None, "Bid should not be None when API returns zero"
    assert ask is not None, "Ask should not be None when API returns zero"
    assert bid > 0, "Bid should be positive"
    assert ask > 0, "Ask should be positive"


def test_bid_ask_use_real_values_when_available(mock_client, mock_stock_data, monkeypatch):
    """Test that real bid/ask values are used when API returns valid data."""
    # Import modules and patch
    import data_curate_daily
    from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
    import sys
    sys.modules['data_curate_daily'].ADD_LATEST = True

    # Mock download_stock_data_between_times
    monkeypatch.setattr(
        data_curate_daily,
        'download_stock_data_between_times',
        lambda api, end, start, symbol: mock_stock_data
    )

    # Mock latest_data to return valid bid/ask
    real_bid = 3900.0
    real_ask = 3910.0

    def mock_latest_data_valid(symbol):
        return SimpleNamespace(
            ask_price=real_ask,
            bid_price=real_bid
        )

    import alpaca_wrapper
    monkeypatch.setattr(alpaca_wrapper, 'latest_data', mock_latest_data_valid)

    symbol = 'ETHUSD'

    # Call download_exchange_latest_data
    result = download_exchange_latest_data(mock_client, symbol)

    # Verify bid/ask match the real values from API
    bid = get_bid(symbol)
    ask = get_ask(symbol)

    assert bid == real_bid, f"Expected bid {real_bid}, got {bid}"
    assert ask == real_ask, f"Expected ask {real_ask}, got {ask}"


def test_bid_ask_retries_on_api_failure(mock_client, mock_stock_data, monkeypatch):
    """Test that the system retries when API initially fails but succeeds later."""
    # Import modules and patch
    import data_curate_daily
    from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
    import sys
    sys.modules['data_curate_daily'].ADD_LATEST = True

    # Mock download_stock_data_between_times
    monkeypatch.setattr(
        data_curate_daily,
        'download_stock_data_between_times',
        lambda api, end, start, symbol: mock_stock_data
    )

    # Mock latest_data to fail twice then succeed
    call_count = 0
    real_bid = 3900.0
    real_ask = 3910.0

    def mock_latest_data_retry(symbol):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception(f"API error on attempt {call_count}")
        return SimpleNamespace(
            ask_price=real_ask,
            bid_price=real_bid
        )

    import alpaca_wrapper
    monkeypatch.setattr(alpaca_wrapper, 'latest_data', mock_latest_data_retry)

    symbol = 'ETHUSD'

    # Call download_exchange_latest_data
    result = download_exchange_latest_data(mock_client, symbol)

    # Verify it retried and got the real values
    bid = get_bid(symbol)
    ask = get_ask(symbol)

    assert call_count == 3, f"Expected 3 API calls (2 failures + 1 success), got {call_count}"
    assert bid == real_bid, f"Expected bid {real_bid}, got {bid}"
    assert ask == real_ask, f"Expected ask {real_ask}, got {ask}"


def test_get_bid_returns_none_for_unknown_symbol():
    """Test that get_bid returns None for a symbol that hasn't been fetched."""
    from data_curate_daily import get_bid
    # Don't call download_exchange_latest_data
    bid = get_bid('UNKNOWN_SYMBOL')
    assert bid is None, "get_bid should return None for unknown symbol"


def test_get_ask_returns_none_for_unknown_symbol():
    """Test that get_ask returns None for a symbol that hasn't been fetched."""
    from data_curate_daily import get_ask
    # Don't call download_exchange_latest_data
    ask = get_ask('UNKNOWN_SYMBOL')
    assert ask is None, "get_ask should return None for unknown symbol"


def test_multiple_symbols_independent(mock_client, mock_stock_data, monkeypatch):
    """Test that bid/ask for multiple symbols are independent."""
    # Import modules and patch
    import data_curate_daily
    from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
    import sys
    sys.modules['data_curate_daily'].ADD_LATEST = False

    # Mock download_stock_data_between_times
    monkeypatch.setattr(
        data_curate_daily,
        'download_stock_data_between_times',
        lambda api, end, start, symbol: mock_stock_data
    )

    # Fetch data for two different symbols
    symbol1 = 'ETHUSD'
    symbol2 = 'BTCUSD'

    download_exchange_latest_data(mock_client, symbol1)
    download_exchange_latest_data(mock_client, symbol2)

    # Verify both have bid/ask
    bid1 = get_bid(symbol1)
    ask1 = get_ask(symbol1)
    bid2 = get_bid(symbol2)
    ask2 = get_ask(symbol2)

    assert bid1 is not None
    assert ask1 is not None
    assert bid2 is not None
    assert ask2 is not None

    # Verify they're the same (since same close price in mock data)
    assert bid1 == bid2
    assert ask1 == ask2
