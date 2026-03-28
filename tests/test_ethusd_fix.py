#!/usr/bin/env python3
"""Regression tests for ETHUSD bid/ask handling."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from data_curate_daily import download_exchange_latest_data, get_ask, get_bid, get_spread


@pytest.fixture(autouse=True)
def reset_global_state():
    import data_curate_daily

    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}
    yield
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}


@pytest.fixture
def mock_price_frame() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    dates = pd.date_range(end=now, periods=5, freq="D", tz=timezone.utc)
    return pd.DataFrame(
        {
            "open": [3800.0, 3810.0, 3820.0, 3830.0, 3840.0],
            "high": [3810.0, 3820.0, 3830.0, 3840.0, 3850.0],
            "low": [3790.0, 3800.0, 3810.0, 3820.0, 3830.0],
            "close": [3805.0, 3815.0, 3825.0, 3835.0, 3845.0],
        },
        index=dates,
    )


def test_ethusd_with_add_latest_true(monkeypatch, mock_price_frame: pd.DataFrame) -> None:
    import alpaca_wrapper
    import data_curate_daily

    data_curate_daily.ADD_LATEST = True
    monkeypatch.setattr(
        data_curate_daily,
        "download_stock_data_between_times",
        lambda api, end, start, symbol: mock_price_frame.copy(),
    )
    monkeypatch.setattr(
        alpaca_wrapper,
        "latest_data",
        lambda symbol: SimpleNamespace(bid_price=3900.0, ask_price=3910.0),
    )

    result = download_exchange_latest_data(object(), "ETHUSD")

    assert not result.empty
    assert result.iloc[-1]["close"] == pytest.approx(3905.0)
    assert get_bid("ETHUSD") == pytest.approx(3900.0)
    assert get_ask("ETHUSD") == pytest.approx(3910.0)


def test_ethusd_with_add_latest_false(monkeypatch, mock_price_frame: pd.DataFrame) -> None:
    import data_curate_daily

    data_curate_daily.ADD_LATEST = False
    monkeypatch.setattr(
        data_curate_daily,
        "download_stock_data_between_times",
        lambda api, end, start, symbol: mock_price_frame.copy(),
    )

    result = download_exchange_latest_data(object(), "ETHUSD")
    last_close = float(mock_price_frame.iloc[-1]["close"])

    assert not result.empty
    assert float(result.iloc[-1]["close"]) == pytest.approx(last_close)
    assert get_bid("ETHUSD") == pytest.approx(last_close)
    assert get_ask("ETHUSD") == pytest.approx(last_close)
    assert get_spread("ETHUSD") == pytest.approx(0.0)


def test_multiple_symbols(monkeypatch, mock_price_frame: pd.DataFrame) -> None:
    import alpaca_wrapper
    import data_curate_daily

    data_curate_daily.ADD_LATEST = True

    def _mock_frame(api, end, start, symbol):
        frame = mock_price_frame.copy()
        offset = {"ETHUSD": 0.0, "BTCUSD": 10_000.0, "LTCUSD": -3_500.0}[symbol]
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col] + offset
        return frame

    latest_quotes = {
        "ETHUSD": SimpleNamespace(bid_price=3900.0, ask_price=3910.0),
        "BTCUSD": SimpleNamespace(bid_price=62_000.0, ask_price=62_020.0),
        "LTCUSD": SimpleNamespace(bid_price=125.0, ask_price=126.0),
    }

    monkeypatch.setattr(data_curate_daily, "download_stock_data_between_times", _mock_frame)
    monkeypatch.setattr(alpaca_wrapper, "latest_data", lambda symbol: latest_quotes[symbol])

    for symbol, quote in latest_quotes.items():
        result = download_exchange_latest_data(object(), symbol)
        assert not result.empty
        assert get_bid(symbol) == pytest.approx(float(quote.bid_price))
        assert get_ask(symbol) == pytest.approx(float(quote.ask_price))

