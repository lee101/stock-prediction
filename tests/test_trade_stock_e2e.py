from contextlib import ExitStack, contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from trade_stock_e2e import (
    analyze_symbols,
    get_market_hours,
    manage_market_close,
    manage_positions,
)


def make_position(symbol, side, qty=1, current_price=100):
    """Create a lightweight alpaca position mock for testing."""
    position = MagicMock()
    position.symbol = symbol
    position.side = side
    position.qty = str(qty)
    position.current_price = str(current_price)
    return position


@contextmanager
def stub_trading_env(
    positions=None,
    *,
    qty=5,
    bid=99.0,
    ask=101.0,
    trading_day_now=False,
):
    """Patch trading-related helpers so tests never touch real APIs."""
    if positions is None:
        positions = []

    with ExitStack() as stack:
        mocks = {}
        mocks["get_all_positions"] = stack.enter_context(
            patch("trade_stock_e2e.alpaca_wrapper.get_all_positions", return_value=positions)
        )
        mocks["filter_positions"] = stack.enter_context(
            patch("trade_stock_e2e.filter_to_realistic_positions", return_value=positions)
        )
        mocks["client_cls"] = stack.enter_context(
            patch("trade_stock_e2e.StockHistoricalDataClient")
        )
        mocks["download_latest"] = stack.enter_context(
            patch("trade_stock_e2e.download_exchange_latest_data")
        )
        mocks["get_bid"] = stack.enter_context(
            patch("trade_stock_e2e.get_bid", return_value=bid)
        )
        mocks["get_ask"] = stack.enter_context(
            patch("trade_stock_e2e.get_ask", return_value=ask)
        )
        mocks["get_qty"] = stack.enter_context(
            patch("trade_stock_e2e.get_qty", return_value=qty)
        )
        mocks["ramp"] = stack.enter_context(
            patch("trade_stock_e2e.ramp_into_position")
        )
        mocks["spawn_tp"] = stack.enter_context(
            patch("trade_stock_e2e.spawn_close_position_at_takeprofit")
        )
        mocks["open_order"] = stack.enter_context(
            patch("trade_stock_e2e.alpaca_wrapper.open_order_at_price_or_all")
        )
        mocks["trading_day_now"] = stack.enter_context(
            patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=trading_day_now)
        )
        yield mocks


@pytest.fixture
def test_data():
    return {
        "symbols": ["AAPL", "MSFT"],
        "mock_picks": {
            "AAPL": {
                "sharpe": 1.5,
                "avg_return": 0.03,
                "side": "buy",
                "strategy": "simple",
                "predicted_movement": 0.02,
                "predictions": pd.DataFrame(),
            }
        },
    }


@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols(mock_backtest, test_data):
    mock_df = pd.DataFrame(
        {
            "simple_strategy_return": [0.02],
            "all_signals_strategy_return": [0.01],
            "entry_takeprofit_return": [0.005],
            "highlow_return": [0.004],
            "predicted_close": [105],
            "predicted_high": [106],
            "predicted_low": [104],
            "close": [100],
        }
    )
    mock_backtest.return_value = mock_df

    results = analyze_symbols(test_data["symbols"])

    assert isinstance(results, dict)
    assert len(results) > 0
    first_symbol = list(results.keys())[0]
    assert "avg_return" in results[first_symbol]
    assert "side" in results[first_symbol]
    assert "predicted_movement" in results[first_symbol]


def test_get_market_hours():
    market_open, market_close = get_market_hours()
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)

    assert market_open.hour == 9
    assert market_open.minute == 30
    assert market_close.hour == 16
    assert market_close.minute == 0


@patch("trade_stock_e2e.analyze_next_day_positions")
@patch("trade_stock_e2e.alpaca_wrapper.get_all_positions")
@patch("trade_stock_e2e.logger")
def test_manage_market_close(mock_logger, mock_get_positions, mock_analyze, test_data):
    mock_position = MagicMock()
    mock_position.symbol = "MSFT"
    mock_position.side = "buy"
    mock_get_positions.return_value = [mock_position]
    mock_analyze.return_value = test_data["mock_picks"]

    result = manage_market_close(test_data["symbols"], {}, test_data["mock_picks"])
    assert isinstance(result, dict)
    mock_logger.info.assert_called()


def test_manage_positions_only_closes_on_opposite_forecast():
    """Ensure we only issue exits when the forecast flips direction."""
    positions = [
        make_position("AAPL", "buy"),
        make_position("MSFT", "buy"),
        make_position("GOOG", "buy"),
        make_position("TSLA", "sell"),
    ]

    all_analyzed_results = {
        "MSFT": {
            "side": "buy",
            "sharpe": 1.5,
            "avg_return": 0.05,
            "predicted_movement": 0.02,
            "predictions": pd.DataFrame(),
            "strategy": "simple",
        },
        "GOOG": {
            "side": "sell",
            "sharpe": 1.2,
            "avg_return": 0.01,
            "predicted_movement": -0.02,
            "predictions": pd.DataFrame(),
            "strategy": "simple",
        },
        "TSLA": {
            "side": "sell",
            "sharpe": 1.1,
            "avg_return": 0.02,
            "predicted_movement": -0.01,
            "predictions": pd.DataFrame(),
            "strategy": "simple",
        },
    }

    current_picks = {k: v for k, v in all_analyzed_results.items() if v["sharpe"] > 0}

    with stub_trading_env(positions=positions) as mocks, patch(
        "trade_stock_e2e.backout_near_market"
    ) as mock_backout:
        manage_positions(current_picks, {}, all_analyzed_results)

    mock_backout.assert_called_once_with("GOOG")
    assert mocks["ramp"].call_count >= 1  # new entries can still be scheduled


@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_strategy_selection(mock_backtest):
    """Test that analyze_symbols correctly selects and applies strategies."""
    test_cases = [
        {
            "simple_strategy_return": [0.06],
            "all_signals_strategy_return": [0.03],
            "entry_takeprofit_return": [0.01],
            "highlow_return": [0.02],
            "close": [100],
            "predicted_close": [105],
            "predicted_high": [106],
            "predicted_low": [104],
            "expected_strategy": "simple",
        },
        {
            "simple_strategy_return": [0.02],
            "all_signals_strategy_return": [0.06],
            "entry_takeprofit_return": [0.03],
            "highlow_return": [0.01],
            "close": [100],
            "predicted_close": [105],
            "predicted_high": [106],
            "predicted_low": [104],
            "expected_strategy": "all_signals",
        },
        {
            "simple_strategy_return": [0.02],
            "all_signals_strategy_return": [0.05],
            "entry_takeprofit_return": [0.01],
            "highlow_return": [0.015],
            "close": [100],
            "predicted_close": [105],
            "predicted_high": [99],
            "predicted_low": [104],
            "expected_strategy": None,
        },
    ]

    symbols = ["TEST1", "TEST2", "TEST3"]

    for symbol, test_case in zip(symbols, test_cases):
        mock_backtest.return_value = pd.DataFrame(test_case)

        results = analyze_symbols([symbol])

        if test_case["expected_strategy"] is None:
            assert symbol not in results
            continue

        result = results[symbol]
        assert result["strategy"] == test_case["expected_strategy"]

        if test_case["expected_strategy"] == "simple":
            expected_side = (
                "buy" if test_case["predicted_close"] > test_case["close"] else "sell"
            )
            assert result["side"] == expected_side
        elif test_case["expected_strategy"] == "all_signals":
            pc = test_case["predicted_close"][0]
            c = test_case["close"][0]
            ph = test_case["predicted_high"][0]
            pl = test_case["predicted_low"][0]
            movements = [pc - c, ph - c, pl - c]
            if all(x > 0 for x in movements):
                assert result["side"] == "buy"
            elif all(x < 0 for x in movements):
                assert result["side"] == "sell"

        assert "avg_return" in result
        assert "predicted_movement" in result
        assert "predictions" in result


def test_manage_positions_enters_new_simple_position_without_real_trades():
    current_picks = {
        "AAPL": {
            "side": "buy",
            "avg_return": 0.07,
            "predicted_movement": 0.03,
            "strategy": "simple",
            "predicted_high": 120.0,
            "predicted_low": 115.0,
            "predictions": pd.DataFrame(),
        }
    }

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_called_once_with("AAPL", "buy", target_qty=5)
    mocks["get_qty"].assert_called()
    mocks["spawn_tp"].assert_not_called()
    mocks["open_order"].assert_not_called()


def test_manage_positions_highlow_strategy_uses_limit_orders():
    current_picks = {
        "AAPL": {
            "side": "buy",
            "avg_return": 0.12,
            "predicted_movement": 0.06,
            "strategy": "highlow",
            "predicted_high": 125.0,
            "predicted_low": 100.0,
            "predictions": pd.DataFrame(
                [{"predicted_low": 100.0, "predicted_high": 125.0}]
            ),
        }
    }

    with stub_trading_env(positions=[], qty=3, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_called_once_with("AAPL", "buy", target_qty=3)
    mocks["spawn_tp"].assert_called_once_with("AAPL", 125.0)
    assert mocks["open_order"].call_count == 1
    call_args, call_kwargs = mocks["open_order"].call_args
    assert call_args == ("AAPL",)
    assert call_kwargs == {"qty": 3, "side": "buy", "price": 100.0}
