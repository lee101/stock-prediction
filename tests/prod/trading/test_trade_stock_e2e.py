from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz
import sys
import types

if "backtest_test3_inline" not in sys.modules:
    _backtest_stub = types.ModuleType("backtest_test3_inline")

    def _stub_backtest_forecasts(*args, **kwargs):
        raise RuntimeError("backtest_forecasts stub should be patched in tests")

    def _stub_release_model_resources():
        return None

    _backtest_stub.backtest_forecasts = _stub_backtest_forecasts
    _backtest_stub.release_model_resources = _stub_release_model_resources
    sys.modules["backtest_test3_inline"] = _backtest_stub

import trade_stock_e2e as trade_module
from trade_stock_e2e import (
    analyze_symbols,
    build_portfolio,
    get_market_hours,
    manage_market_close,
    manage_positions,
    reset_symbol_entry_counters,
    is_tradeable,
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
        mocks["spawn_open_maxdiff"] = stack.enter_context(
            patch("trade_stock_e2e.spawn_open_position_at_maxdiff_takeprofit")
        )
        mocks["spawn_close_maxdiff"] = stack.enter_context(
            patch("trade_stock_e2e.spawn_close_position_at_maxdiff_takeprofit")
        )
        mocks["spawn_tp"] = stack.enter_context(
            patch("trade_stock_e2e.spawn_close_position_at_takeprofit")
        )
        mocks["open_order"] = stack.enter_context(
            patch("trade_stock_e2e.alpaca_wrapper.open_order_at_price_or_all")
        )
        stack.enter_context(
            patch("trade_stock_e2e.PROBE_SYMBOLS", set())
        )
        stack.enter_context(
            patch.object(
                trade_module.alpaca_wrapper,
                "equity",
                250000.0,
            )
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


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols(mock_backtest, mock_snapshot, mock_trading_day_now, test_data):
    mock_df = pd.DataFrame(
        {
            "simple_strategy_return": [0.02],
            "simple_strategy_avg_daily_return": [0.02],
            "simple_strategy_annual_return": [0.02 * 252],
            "ci_guard_return": [0.018],
            "ci_guard_avg_daily_return": [0.018],
            "ci_guard_annual_return": [0.018 * 252],
            "ci_guard_sharpe": [1.1],
            "all_signals_strategy_return": [0.01],
            "all_signals_strategy_avg_daily_return": [0.01],
            "all_signals_strategy_annual_return": [0.01 * 252],
            "entry_takeprofit_return": [0.005],
            "entry_takeprofit_avg_daily_return": [0.005],
            "entry_takeprofit_annual_return": [0.005 * 252],
            "highlow_return": [0.004],
            "highlow_avg_daily_return": [0.004],
            "highlow_annual_return": [0.004 * 252],
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
    assert "annual_return" in results[first_symbol]
    assert "side" in results[first_symbol]
    assert "predicted_movement" in results[first_symbol]
    assert results[first_symbol]["ci_guard_return"] == pytest.approx(0.018)
    assert results[first_symbol]["ci_guard_sharpe"] == pytest.approx(1.1)
    expected_penalty = trade_module.resolve_spread_cap(first_symbol) / 10000.0
    expected_primary = results[first_symbol]["avg_return"]
    assert results[first_symbol]["composite_score"] == pytest.approx(
        expected_primary - expected_penalty, rel=1e-4
    )


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_falls_back_to_maxdiff_when_all_signals_conflict(
    mock_backtest, mock_snapshot, mock_trading_day_now
):
    rows = []
    for _ in range(70):
        rows.append(
            {
                "simple_strategy_return": 0.0,
                "simple_strategy_avg_daily_return": 0.0,
                "simple_strategy_annual_return": 0.0,
                "simple_strategy_sharpe": 0.3,
                "simple_strategy_turnover": 0.5,
                "simple_strategy_max_drawdown": -0.02,
                "ci_guard_return": 0.0,
                "ci_guard_avg_daily_return": 0.0,
                "ci_guard_annual_return": 0.0,
                "ci_guard_sharpe": 0.0,
                "ci_guard_turnover": 0.5,
                "ci_guard_max_drawdown": -0.02,
                "all_signals_strategy_return": 0.05,
                "all_signals_strategy_avg_daily_return": 0.05,
                "all_signals_strategy_annual_return": 0.05 * 365,
                "all_signals_strategy_sharpe": 1.2,
                "all_signals_strategy_turnover": 0.6,
                "all_signals_strategy_max_drawdown": -0.03,
                "entry_takeprofit_return": 0.01,
                "entry_takeprofit_avg_daily_return": 0.01,
                "entry_takeprofit_annual_return": 0.01 * 365,
                "entry_takeprofit_sharpe": 0.6,
                "entry_takeprofit_turnover": 0.7,
                "entry_takeprofit_max_drawdown": -0.04,
                "highlow_return": 0.015,
                "highlow_avg_daily_return": 0.015,
                "highlow_annual_return": 0.015 * 365,
                "highlow_sharpe": 0.8,
                "highlow_turnover": 0.9,
                "highlow_max_drawdown": -0.05,
                "maxdiff_return": 0.03,
                "maxdiff_avg_daily_return": 0.03,
                "maxdiff_annual_return": 0.03 * 365,
                "maxdiff_sharpe": 1.0,
                "maxdiff_turnover": 1.0,
                "maxdiff_max_drawdown": -0.04,
                "close": 10.0,
                "predicted_close": 10.8,
                "predicted_high": 11.0,
                "predicted_low": 9.6,
            }
        )
    mock_backtest.return_value = pd.DataFrame(rows)

    with patch("trade_stock_e2e.ALLOW_HIGHLOW_ENTRY", True), patch("trade_stock_e2e.ALLOW_MAXDIFF_ENTRY", True):
        results = analyze_symbols(["UNIUSD"])
    assert "UNIUSD" in results
    assert results["UNIUSD"]["strategy"] == "maxdiff"
    assert results["UNIUSD"]["maxdiff_entry_allowed"] is True
    ineligible = results["UNIUSD"]["strategy_entry_ineligible"]
    assert ineligible.get("all_signals") == "mixed_directional_signals"
    notes = results["UNIUSD"].get("strategy_selection_notes") or []
    assert any("mixed_directional_signals" in note for note in notes)
    sequence = results["UNIUSD"].get("strategy_sequence") or []
    assert sequence and sequence[0] == "all_signals"


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
@patch("trade_stock_e2e._log_detail")
def test_analyze_symbols_marks_crypto_sell_ineligible(
    mock_log, mock_backtest, mock_snapshot, mock_trading_day_now
):
    row = {
        "simple_strategy_return": 0.04,
        "simple_strategy_avg_daily_return": 0.04,
        "simple_strategy_annual_return": 0.04 * 365,
        "simple_strategy_sharpe": 0.9,
        "simple_strategy_turnover": 0.5,
        "simple_strategy_max_drawdown": -0.03,
        "all_signals_strategy_return": 0.03,
        "all_signals_strategy_avg_daily_return": 0.03,
        "all_signals_strategy_annual_return": 0.03 * 365,
        "all_signals_strategy_sharpe": 0.8,
        "all_signals_strategy_turnover": 0.6,
        "all_signals_strategy_max_drawdown": -0.04,
        "entry_takeprofit_return": 0.02,
        "entry_takeprofit_avg_daily_return": 0.02,
        "entry_takeprofit_annual_return": 0.02 * 365,
        "entry_takeprofit_sharpe": 0.7,
        "entry_takeprofit_turnover": 0.7,
        "entry_takeprofit_max_drawdown": -0.05,
        "highlow_return": 0.01,
        "highlow_avg_daily_return": 0.01,
        "highlow_annual_return": 0.01 * 365,
        "highlow_sharpe": 0.6,
        "highlow_turnover": 0.6,
        "highlow_max_drawdown": -0.05,
        "maxdiff_return": 0.015,
        "maxdiff_avg_daily_return": 0.015,
        "maxdiff_annual_return": 0.015 * 365,
        "maxdiff_sharpe": 0.7,
        "maxdiff_turnover": 0.8,
        "maxdiff_max_drawdown": -0.05,
        "close": 10.0,
        "predicted_close": 9.6,
        "predicted_high": 9.7,
        "predicted_low": 9.3,
    }
    mock_backtest.return_value = pd.DataFrame([row] * 70)

    with patch.object(trade_module, "ALLOW_HIGHLOW_ENTRY", True), patch.object(
        trade_module, "ALLOW_MAXDIFF_ENTRY", True
    ):
        results = analyze_symbols(["UNIUSD"])

    assert results == {}
    logged_messages = " ".join(call.args[0] for call in mock_log.call_args_list)
    assert "crypto_sell_disabled" in logged_messages


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=False)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_skips_equities_when_market_closed(mock_backtest, mock_snapshot, mock_trading_day_now):
    mock_df = pd.DataFrame(
        {
            "simple_strategy_return": [0.02],
            "simple_strategy_avg_daily_return": [0.02],
            "simple_strategy_annual_return": [0.02 * 252],
            "all_signals_strategy_return": [0.01],
            "all_signals_strategy_avg_daily_return": [0.01],
            "all_signals_strategy_annual_return": [0.01 * 252],
            "entry_takeprofit_return": [0.005],
            "entry_takeprofit_avg_daily_return": [0.005],
            "entry_takeprofit_annual_return": [0.005 * 252],
            "highlow_return": [0.004],
            "highlow_avg_daily_return": [0.004],
            "highlow_annual_return": [0.004 * 252],
            "close": [100.0],
            "predicted_close": [102.0],
            "predicted_high": [103.0],
            "predicted_low": [99.0],
        }
    )
    mock_backtest.return_value = mock_df

    with patch.dict(os.environ, {"MARKETSIM_SKIP_CLOSED_EQUITY": "1"}, clear=False):
        results = analyze_symbols(["AAPL", "BTCUSD"])

    assert "AAPL" not in results
    assert "BTCUSD" in results
    assert mock_backtest.call_count == 1
    assert mock_backtest.call_args[0][0] == "BTCUSD"


@patch("trade_stock_e2e.fetch_bid_ask", return_value=(100.0, 101.0))
@patch("trade_stock_e2e.is_tradeable", return_value=(True, "ok"))
@patch("trade_stock_e2e.pass_edge_threshold", return_value=(True, "ok"))
@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=False)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_respects_skip_override(
    mock_backtest,
    mock_snapshot,
    mock_trading_day_now,
    mock_edge,
    mock_tradeable,
    mock_bid_ask,
    monkeypatch,
):
    monkeypatch.setenv("MARKETSIM_SKIP_CLOSED_EQUITY", "0")
    mock_df = pd.DataFrame(
        {
            "simple_strategy_return": [0.01],
            "simple_strategy_avg_daily_return": [0.01],
            "simple_strategy_annual_return": [0.01 * 252],
            "all_signals_strategy_return": [0.009],
            "all_signals_strategy_avg_daily_return": [0.009],
            "all_signals_strategy_annual_return": [0.009 * 252],
            "entry_takeprofit_return": [0.008],
            "entry_takeprofit_avg_daily_return": [0.008],
            "entry_takeprofit_annual_return": [0.008 * 252],
            "highlow_return": [0.007],
            "highlow_avg_daily_return": [0.007],
            "highlow_annual_return": [0.007 * 252],
            "ci_guard_return": [0.015],
            "ci_guard_avg_daily_return": [0.015],
            "ci_guard_annual_return": [0.015 * 252],
            "ci_guard_sharpe": [0.8],
            "close": [100.0],
            "predicted_close": [101.5],
            "predicted_high": [102.0],
            "predicted_low": [99.5],
        }
    )
    mock_backtest.return_value = mock_df

    results = analyze_symbols(["AAPL"])

    assert "AAPL" in results
    assert results["AAPL"]["ci_guard_return"] == pytest.approx(0.015)


@patch("trade_stock_e2e.fetch_bid_ask", return_value=(100.0, 101.0))
@patch("trade_stock_e2e.is_tradeable", return_value=(True, "ok"))
@patch("trade_stock_e2e.pass_edge_threshold", return_value=(True, "ok"))
@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_ci_guard_shapes_price_skill(
    mock_backtest,
    mock_snapshot,
    mock_trading_day_now,
    mock_edge,
    mock_tradeable,
    mock_bid_ask,
):
    mock_df = pd.DataFrame(
        {
            "simple_strategy_return": [-0.01],
            "simple_strategy_avg_daily_return": [-0.01],
            "simple_strategy_annual_return": [-0.01 * 252],
            "simple_strategy_sharpe": [-0.2],
            "all_signals_strategy_return": [-0.02],
            "all_signals_strategy_avg_daily_return": [-0.02],
            "all_signals_strategy_annual_return": [-0.02 * 252],
            "entry_takeprofit_return": [0.0],
            "entry_takeprofit_avg_daily_return": [0.0],
            "entry_takeprofit_annual_return": [0.0],
            "highlow_return": [0.0],
            "highlow_avg_daily_return": [0.0],
            "highlow_annual_return": [0.0],
            "ci_guard_return": [0.02],
            "ci_guard_avg_daily_return": [0.02],
            "ci_guard_annual_return": [0.02 * 252],
            "ci_guard_sharpe": [1.4],
            "maxdiff_return": [0.0],
            "close": [100.0],
            "predicted_close": [102.0],
            "predicted_high": [103.0],
            "predicted_low": [99.0],
        }
    )
    mock_backtest.return_value = mock_df

    results = analyze_symbols(["AAPL"])

    assert "AAPL" in results
    row = results["AAPL"]
    # With Kronos contribution zero, price_skill should be driven by CI Guard stats.
    expected_price_skill = 0.02 + 0.25 * 1.4
    assert row["price_skill"] == pytest.approx(expected_price_skill)
    assert row["strategy"] == "ci_guard"


@patch("trade_stock_e2e.fetch_bid_ask", return_value=(100.0, 101.0))
@patch("trade_stock_e2e.is_tradeable", return_value=(True, "ok"))
@patch("trade_stock_e2e.pass_edge_threshold", return_value=(True, "ok"))
@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_blocks_on_negative_recent_sum(
    mock_backtest,
    mock_snapshot,
    mock_trading_day_now,
    mock_edge,
    mock_tradeable,
    mock_bid_ask,
):
    mock_df = pd.DataFrame(
        {
            "simple_strategy_return": [-0.02, -0.015, 0.04],
            "simple_strategy_avg_daily_return": [-0.02, -0.015, 0.04],
            "simple_strategy_annual_return": [-0.02 * 252, -0.015 * 252, 0.04 * 252],
            "all_signals_strategy_return": [-0.03, -0.02, -0.01],
            "all_signals_strategy_avg_daily_return": [-0.03, -0.02, -0.01],
            "all_signals_strategy_annual_return": [-0.03 * 252, -0.02 * 252, -0.01 * 252],
            "ci_guard_return": [-0.04, -0.03, -0.02],
            "ci_guard_avg_daily_return": [-0.04, -0.03, -0.02],
            "ci_guard_annual_return": [-0.04 * 252, -0.03 * 252, -0.02 * 252],
            "entry_takeprofit_return": [-0.045, -0.04, -0.03],
            "entry_takeprofit_avg_daily_return": [-0.045, -0.04, -0.03],
            "entry_takeprofit_annual_return": [-0.045 * 252, -0.04 * 252, -0.03 * 252],
            "highlow_return": [-0.05, -0.045, -0.035],
            "highlow_avg_daily_return": [-0.05, -0.045, -0.035],
            "highlow_annual_return": [-0.05 * 252, -0.045 * 252, -0.035 * 252],
            "maxdiff_return": [-0.055, -0.05, -0.04],
            "maxdiff_avg_daily_return": [-0.055, -0.05, -0.04],
            "maxdiff_annual_return": [-0.055 * 252, -0.05 * 252, -0.04 * 252],
            "close": [100.0, 100.0, 100.0],
            "predicted_close": [101.5, 100.8, 102.0],
            "predicted_high": [102.0, 101.0, 103.0],
            "predicted_low": [99.0, 98.5, 100.0],
        }
    )
    mock_backtest.return_value = mock_df

    results = analyze_symbols(["AAPL"])

    assert "AAPL" in results
    row = results["AAPL"]
    assert row["trade_blocked"] is True
    assert row["recent_return_sum"] == pytest.approx(-0.035)
    assert "Recent simple returns sum" in (row.get("block_reason") or "")


def test_is_tradeable_relaxes_spread_gate():
    ok, reason = is_tradeable(
        "AAPL",
        bid=100.0,
        ask=101.5,
        avg_dollar_vol=6_000_000,
        atr_pct=15.0,
    )
    assert ok is True
    assert "Spread" in reason
    assert "gates relaxed" in reason


def test_get_market_hours():
    market_open, market_close = get_market_hours()
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)

    assert market_open.hour == 9
    assert market_open.minute == 30
    expected_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    expected_close -= timedelta(minutes=trade_module.MARKET_CLOSE_SHIFT_MINUTES)
    if expected_close <= market_open:
        expected_close = market_open + timedelta(minutes=1)
    assert market_close.hour == expected_close.hour
    assert market_close.minute == expected_close.minute


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


def test_manage_market_close_closes_on_negative_strategy(monkeypatch):
    position = make_position("AAPL", "buy")

    monkeypatch.setattr(
        trade_module.alpaca_wrapper,
        "get_all_positions",
        lambda: [position],
    )
    monkeypatch.setattr(trade_module, "filter_to_realistic_positions", lambda positions: positions)
    monkeypatch.setattr(trade_module, "build_portfolio", lambda *args, **kwargs: {})

    close_calls = []
    outcome_calls = []
    monkeypatch.setattr(trade_module, "backout_near_market", lambda symbol: close_calls.append(symbol))
    monkeypatch.setattr(
        trade_module,
        "_record_trade_outcome",
        lambda pos, reason: outcome_calls.append((pos.symbol, reason)),
    )

    monkeypatch.setattr(
        trade_module,
        "_get_active_trade",
        lambda symbol, side: {"mode": "normal", "entry_strategy": "simple"},
    )

    all_results = {
        "AAPL": {
            "side": "buy",
            "strategy": "simple",
            "strategy_returns": {"simple": -0.012},
            "avg_return": -0.012,
            "predicted_movement": 0.001,
            "probe_expired": False,
        }
    }
    previous_picks = {
        "AAPL": {
            "strategy": "simple",
            "trade_mode": "normal",
        }
    }

    manage_market_close(["AAPL"], previous_picks, all_results)

    assert close_calls == ["AAPL"]
    assert outcome_calls == [("AAPL", "simple_strategy_loss")]


def test_manage_market_close_skips_probe_when_negative(monkeypatch):
    position = make_position("AAPL", "buy")

    monkeypatch.setattr(trade_module.alpaca_wrapper, "get_all_positions", lambda: [position])
    monkeypatch.setattr(trade_module, "filter_to_realistic_positions", lambda positions: positions)
    monkeypatch.setattr(trade_module, "build_portfolio", lambda *args, **kwargs: {})
    close_calls = []
    monkeypatch.setattr(trade_module, "backout_near_market", lambda symbol: close_calls.append(symbol))
    monkeypatch.setattr(trade_module, "_record_trade_outcome", lambda pos, reason: None)

    monkeypatch.setattr(
        trade_module,
        "_get_active_trade",
        lambda symbol, side: {"mode": "probe", "entry_strategy": "simple"},
    )

    all_results = {
        "AAPL": {
            "side": "buy",
            "strategy": "simple",
            "strategy_returns": {"simple": -0.05},
            "avg_return": -0.05,
            "predicted_movement": 0.002,
            "probe_expired": False,
        }
    }
    previous_picks = {
        "AAPL": {
            "strategy": "simple",
            "trade_mode": "probe",
        }
    }

    manage_market_close(["AAPL"], previous_picks, all_results)

    assert close_calls == []


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


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_strategy_selection(mock_backtest, mock_snapshot, mock_trading_day_now):
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
            "expected_strategy": "simple",
        },
        {
            "simple_strategy_return": [-0.01],
            "all_signals_strategy_return": [-0.015],
            "entry_takeprofit_return": [-0.02],
            "highlow_return": [-0.03],
            "close": [100],
            "predicted_close": [99],
            "predicted_high": [101],
            "predicted_low": [95],
            "expected_strategy": None,
        },
    ]

    for case in test_cases:
        for prefix in ("simple_strategy", "all_signals_strategy", "entry_takeprofit", "highlow"):
            return_key = f"{prefix}_return"
            if return_key in case and case[return_key]:
                value = case[return_key][0]
                case.setdefault(f"{prefix}_avg_daily_return", [value])
                case.setdefault(f"{prefix}_annual_return", [value * 252])

    symbols = ["TEST1", "TEST2", "TEST3", "TEST4"]

    for symbol, test_case in zip(symbols, test_cases):
        mock_backtest.return_value = pd.DataFrame(test_case)

        results = analyze_symbols([symbol])

        if test_case["expected_strategy"] is None:
            assert symbol not in results
            continue

        result = results[symbol]
        assert result["strategy"] == test_case["expected_strategy"]

        if test_case["expected_strategy"] == "simple":
            expected_side = "buy" if test_case["predicted_close"] > test_case["close"] else "sell"
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


@pytest.mark.parametrize("limit_map", ["AAPL:2", "AAPL@simple:2"])
def test_manage_positions_respects_max_entries_per_run(monkeypatch, limit_map):
    monkeypatch.setenv("MARKETSIM_SYMBOL_MAX_ENTRIES_MAP", limit_map)
    reset_symbol_entry_counters()

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
        manage_positions(current_picks, {}, current_picks)
        manage_positions(current_picks, {}, current_picks)

    assert mocks["ramp"].call_count == 2


def test_reset_symbol_entry_counters_allows_additional_runs(monkeypatch):
    monkeypatch.setenv("MARKETSIM_SYMBOL_MAX_ENTRIES_MAP", "AAPL:1")
    reset_symbol_entry_counters()

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

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks_first:
        manage_positions(current_picks, {}, current_picks)
        manage_positions(current_picks, {}, current_picks)

    assert mocks_first["ramp"].call_count == 1

    reset_symbol_entry_counters()

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks_second:
        manage_positions(current_picks, {}, current_picks)
        manage_positions(current_picks, {}, current_picks)

    assert mocks_second["ramp"].call_count == 1


@patch("trade_stock_e2e._symbol_force_probe", return_value=True)
def test_manage_positions_force_probe_override(mock_force_probe):
    current_picks = {
        "AAPL": {
            "side": "sell",
            "avg_return": 0.07,
            "predicted_movement": -0.03,
            "strategy": "ci_guard",
            "predicted_high": 120.0,
            "predicted_low": 115.0,
            "predictions": pd.DataFrame(),
            "trade_mode": "normal",
        }
    }

    with ExitStack() as stack:
        mock_probe_active = stack.enter_context(
            patch("trade_stock_e2e._mark_probe_active")
        )
        mocks = stack.enter_context(stub_trading_env(positions=[], qty=5, trading_day_now=True))
        manage_positions(current_picks, {}, current_picks)

    mock_force_probe.assert_called()
    mock_probe_active.assert_called_once()
    mocks["ramp"].assert_called_once()


def test_manage_positions_min_strategy_return_gating(monkeypatch):
    monkeypatch.setenv("MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP", "AAPL:-0.02")
    current_picks = {
        "AAPL": {
            "side": "sell",
            "avg_return": -0.01,
            "predicted_movement": -0.05,
            "strategy": "ci_guard",
            "strategy_returns": {"ci_guard": -0.01},
            "predicted_high": 120.0,
            "predicted_low": 115.0,
            "predictions": pd.DataFrame(),
            "trade_mode": "probe",
        }
    }

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_not_called()


@patch("trade_stock_e2e._load_trend_summary", return_value={"AAPL": {"pnl": -6000.0}})
def test_manage_positions_trend_pnl_gating(mock_summary, monkeypatch):
    monkeypatch.setenv("MARKETSIM_TREND_PNL_SUSPEND_MAP", "AAPL:-5000")
    current_picks = {
        "AAPL": {
            "side": "sell",
            "avg_return": -0.03,
            "predicted_movement": -0.09,
            "strategy": "ci_guard",
            "strategy_returns": {"ci_guard": -0.04},
            "predicted_high": 120.0,
            "predicted_low": 110.0,
            "predictions": pd.DataFrame(),
            "trade_mode": "probe",
        }
    }

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_not_called()
    mock_summary.assert_called()


@patch("trade_stock_e2e._load_trend_summary", return_value={"AAPL": {"pnl": -2000.0}})
def test_manage_positions_trend_pnl_resume(mock_summary, monkeypatch):
    monkeypatch.setenv("MARKETSIM_TREND_PNL_SUSPEND_MAP", "AAPL:-5000")
    monkeypatch.setenv("MARKETSIM_TREND_PNL_RESUME_MAP", "AAPL:-3000")
    current_picks = {
        "AAPL": {
            "side": "sell",
            "avg_return": -0.03,
            "predicted_movement": -0.09,
            "strategy": "ci_guard",
            "strategy_returns": {"ci_guard": -0.04},
            "predicted_high": 120.0,
            "predicted_low": 110.0,
            "predictions": pd.DataFrame(),
            "trade_mode": "probe",
        }
    }

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_called_once()
    mock_summary.assert_called()


@pytest.mark.parametrize("strategy_name", ["highlow", "maxdiff"])
def test_manage_positions_highlow_strategy_uses_limit_orders(strategy_name):
    current_picks = {
        "AAPL": {
            "side": "buy",
            "avg_return": 0.12,
            "predicted_movement": 0.06,
            "strategy": strategy_name,
            "predicted_high": 125.0,
            "predicted_low": 100.0,
            "maxdiffprofit_low_price": 98.5,
            "maxdiffprofit_high_price": 132.0,
            "predictions": pd.DataFrame(
                [{"predicted_low": 100.0, "predicted_high": 125.0}]
            ),
        }
    }

    with stub_trading_env(positions=[], qty=3, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_not_called()
    mocks["open_order"].assert_not_called()
    mocks["spawn_open_maxdiff"].assert_called_once()
    args, _ = mocks["spawn_open_maxdiff"].call_args
    assert args[0] == "AAPL"
    assert args[1] == "buy"
    assert args[2] == pytest.approx(98.5)
    assert args[3] == pytest.approx(3.0)
    mocks["spawn_close_maxdiff"].assert_called_once_with("AAPL", "buy", 132.0)
    mocks["spawn_tp"].assert_not_called()


@pytest.mark.parametrize("strategy_name", ["highlow", "maxdiff"])
def test_manage_positions_highlow_short_uses_maxdiff_prices(strategy_name):
    current_picks = {
        "UNIUSD": {
            "side": "sell",
            "avg_return": 0.08,
            "predicted_movement": -0.04,
            "strategy": strategy_name,
            "predicted_high": 6.8,
            "predicted_low": 6.1,
            "maxdiffprofit_high_price": 6.9,
            "maxdiffprofit_low_price": 6.05,
            "predictions": pd.DataFrame([{"predicted_high": 6.8, "predicted_low": 6.1}]),
        }
    }

    with stub_trading_env(positions=[], qty=2, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_not_called()
    mocks["open_order"].assert_not_called()
    mocks["spawn_open_maxdiff"].assert_called_once()
    args, _ = mocks["spawn_open_maxdiff"].call_args
    assert args[0] == "UNIUSD"
    assert args[1] == "sell"
    assert args[2] == pytest.approx(6.9)
    assert args[3] == pytest.approx(2.0)
    mocks["spawn_close_maxdiff"].assert_called_once_with("UNIUSD", "sell", 6.05)
    mocks["spawn_tp"].assert_not_called()


def test_build_portfolio_core_prefers_profitable_strategies():
    results = {
        "AAA": {
            "avg_return": 0.03,
            "unprofit_shutdown_return": 0.02,
            "simple_return": 0.01,
            "composite_score": 0.5,
            "trade_blocked": False,
        },
        "BBB": {
            "avg_return": -0.01,
            "unprofit_shutdown_return": -0.02,
            "simple_return": 0.02,
            "composite_score": 0.6,
            "trade_blocked": False,
        },
    }

    picks = build_portfolio(results, min_positions=1, max_positions=2)

    assert "AAA" in picks
    assert picks["AAA"]["avg_return"] > 0
    assert "BBB" not in picks  # fails core profitability screen


def test_build_portfolio_expands_to_meet_minimum():
    results = {
        "AAA": {
            "avg_return": 0.03,
            "unprofit_shutdown_return": 0.02,
            "simple_return": 0.02,
            "composite_score": 0.4,
            "trade_blocked": False,
        },
        "BBB": {
            "avg_return": 0.0,
            "unprofit_shutdown_return": -0.01,
            "simple_return": 0.01,
            "composite_score": 0.3,
            "trade_blocked": False,
        },
        "CCC": {
            "avg_return": -0.02,
            "unprofit_shutdown_return": 0.0,
            "simple_return": 0.0,
            "composite_score": 0.2,
            "trade_blocked": True,
        },
    }

    picks = build_portfolio(results, min_positions=2, max_positions=3)

    assert len(picks) == 2
    assert {"AAA", "BBB"} == set(picks.keys())


def test_build_portfolio_default_max_positions_allows_ten():
    assert trade_module.DEFAULT_MAX_PORTFOLIO == 10
    results = {
        f"SYM{i}": {
            "avg_return": 0.05 - i * 0.001,
            "unprofit_shutdown_return": 0.03,
            "simple_return": 0.02,
            "composite_score": 1.0 - i * 0.05,
            "trade_blocked": False,
        }
        for i in range(12)
    }

    picks = build_portfolio(results)

    assert len(picks) == trade_module.DEFAULT_MAX_PORTFOLIO


def test_build_portfolio_includes_probe_candidate():
    results = {
        "CORE": {
            "avg_return": 0.05,
            "unprofit_shutdown_return": 0.04,
            "simple_return": 0.02,
            "composite_score": 0.6,
            "trade_blocked": False,
        },
        "WEAK": {
            "avg_return": 0.01,
            "unprofit_shutdown_return": 0.0,
            "simple_return": 0.01,
            "composite_score": 0.2,
            "trade_blocked": False,
        },
        "PROBE": {
            "avg_return": -0.01,
            "unprofit_shutdown_return": -0.02,
            "simple_return": 0.0,
            "composite_score": 0.1,
            "trade_blocked": False,
            "trade_mode": "probe",
        },
    }

    picks = build_portfolio(results, min_positions=1, max_positions=2)

    assert "CORE" in picks
    assert "PROBE" in picks
    assert "WEAK" not in picks  # replaced to respect probe inclusion
