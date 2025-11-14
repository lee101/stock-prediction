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
from src.risk_state import ProbeState


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
    assert "maxdiff" in sequence


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
@patch("trade_stock_e2e._log_detail")
def test_analyze_symbols_allows_maxdiff_when_highlow_disabled(
    mock_log, mock_backtest, mock_snapshot, mock_trading_day_now
):
    row = {
        "simple_strategy_return": 0.01,
        "simple_strategy_avg_daily_return": 0.01,
        "simple_strategy_annual_return": 0.01 * 365,
        "simple_strategy_sharpe": 0.6,
        "simple_strategy_turnover": 0.4,
        "simple_strategy_max_drawdown": -0.03,
        "all_signals_strategy_return": 0.02,
        "all_signals_strategy_avg_daily_return": 0.02,
        "all_signals_strategy_annual_return": 0.02 * 365,
        "all_signals_strategy_sharpe": 1.0,
        "all_signals_strategy_turnover": 0.7,
        "all_signals_strategy_max_drawdown": -0.04,
        "entry_takeprofit_return": 0.005,
        "entry_takeprofit_avg_daily_return": 0.005,
        "entry_takeprofit_annual_return": 0.005 * 365,
        "entry_takeprofit_sharpe": 0.55,
        "entry_takeprofit_turnover": 0.8,
        "entry_takeprofit_max_drawdown": -0.05,
        "highlow_return": 0.006,
        "highlow_avg_daily_return": 0.006,
        "highlow_annual_return": 0.006 * 365,
        "highlow_sharpe": 0.65,
        "highlow_turnover": 0.9,
        "highlow_max_drawdown": -0.05,
        "maxdiff_return": 0.03,
        "maxdiff_avg_daily_return": 0.03,
        "maxdiff_annual_return": 0.03 * 365,
        "maxdiff_sharpe": 1.2,
        "maxdiff_turnover": 0.9,
        "maxdiff_max_drawdown": -0.05,
        "close": 10.0,
        "predicted_close": 10.8,
        "predicted_high": 11.0,
        "predicted_low": 9.6,
    }
    mock_backtest.return_value = pd.DataFrame([row] * 70)

    with patch.object(trade_module, "ALLOW_HIGHLOW_ENTRY", False):
        results = analyze_symbols(["UNIUSD"])

    assert "UNIUSD" in results
    assert results["UNIUSD"]["strategy"] == "maxdiff"
    ineligible = results["UNIUSD"]["strategy_entry_ineligible"]
    assert ineligible.get("highlow") == "disabled_by_config"
    sequence = results["UNIUSD"].get("strategy_sequence") or []
    assert sequence and sequence[0] == "maxdiff"
    assert "all_signals" in sequence


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
@patch("trade_stock_e2e._log_detail")
def test_analyze_symbols_prefers_maxdiff_for_crypto_when_primary_side_buy(
    mock_log, mock_backtest, mock_snapshot, mock_trading_day_now
):
    row = {
        "simple_strategy_return": 0.01,
        "simple_strategy_avg_daily_return": 0.01,
        "simple_strategy_annual_return": 0.01 * 365,
        "simple_strategy_sharpe": 0.6,
        "simple_strategy_turnover": 0.5,
        "simple_strategy_max_drawdown": -0.04,
        "all_signals_strategy_return": -0.005,
        "all_signals_strategy_avg_daily_return": -0.005,
        "all_signals_strategy_annual_return": -0.005 * 365,
        "all_signals_strategy_sharpe": 0.2,
        "all_signals_strategy_turnover": 0.4,
        "all_signals_strategy_max_drawdown": -0.06,
        "entry_takeprofit_return": 0.0,
        "entry_takeprofit_avg_daily_return": 0.0,
        "entry_takeprofit_annual_return": 0.0,
        "entry_takeprofit_sharpe": 0.0,
        "entry_takeprofit_turnover": 0.5,
        "entry_takeprofit_max_drawdown": -0.05,
        "highlow_return": 0.015,
        "highlow_avg_daily_return": 0.015,
        "highlow_annual_return": 0.015 * 365,
        "highlow_sharpe": 0.7,
        "highlow_turnover": 0.7,
        "highlow_max_drawdown": -0.05,
        "maxdiff_return": 0.04,
        "maxdiff_avg_daily_return": 0.04,
        "maxdiff_annual_return": 0.04 * 365,
        "maxdiff_sharpe": 1.4,
        "maxdiff_turnover": 0.9,
        "maxdiff_max_drawdown": -0.03,
        "maxdiffprofit_high_price": 103.0,
        "maxdiffprofit_low_price": 96.5,
        "maxdiffprofit_profit": 0.04,
        "maxdiffprofit_profit_high_multiplier": 0.02,
        "maxdiffprofit_profit_low_multiplier": -0.01,
        "maxdiff_primary_side": "buy",
        "maxdiff_trade_bias": 0.6,
        "maxdiff_trades_positive": 5,
        "maxdiff_trades_negative": 0,
        "maxdiff_trades_total": 5,
        "close": 100.0,
        "predicted_close": 98.5,
        "predicted_high": 103.5,
        "predicted_low": 96.0,
    }
    mock_backtest.return_value = pd.DataFrame([row] * 70)

    with patch.object(trade_module, "ALLOW_MAXDIFF_ENTRY", True), patch.object(
        trade_module, "crypto_symbols", ["BTCUSD"]
    ):
        results = analyze_symbols(["BTCUSD"])

    assert "BTCUSD" in results
    outcome = results["BTCUSD"]
    assert outcome["strategy"] == "maxdiff"
    assert outcome["side"] == "buy"
    assert outcome["maxdiff_entry_allowed"] is True


def test_collect_forced_probe_reasons_uses_pnl_sum(monkeypatch):
    probe_state = ProbeState(force_probe=False, reason=None, probe_date=None, state={})
    data = {"side": "buy"}

    monkeypatch.setattr(trade_module, "_recent_trade_pnls", lambda *_, **__: [-2.0, 1.0])
    reasons = trade_module._collect_forced_probe_reasons("AAPL", data, probe_state)
    assert any("recent_pnl_sum" in reason for reason in reasons)

    monkeypatch.setattr(trade_module, "_recent_trade_pnls", lambda *_, **__: [1.5, -0.5])
    data = {"side": "buy"}
    reasons = trade_module._collect_forced_probe_reasons("AAPL", data, probe_state)
    assert not any("recent_pnl_sum" in reason for reason in reasons)


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
@patch("trade_stock_e2e._log_detail")
def test_positive_forecast_overrides_entry_gate_and_records_candidate_map(
    mock_log, mock_backtest, mock_snapshot, mock_trading_day_now
):
    row = {
        "simple_strategy_return": 0.01,
        "simple_strategy_avg_daily_return": 0.01,
        "simple_strategy_annual_return": 0.01 * 365,
        "simple_strategy_sharpe": 0.7,
        "simple_strategy_turnover": 0.6,
        "simple_strategy_max_drawdown": -0.03,
        "all_signals_strategy_return": 0.008,
        "all_signals_strategy_avg_daily_return": 0.008,
        "all_signals_strategy_annual_return": 0.008 * 365,
        "all_signals_strategy_sharpe": 0.65,
        "all_signals_strategy_turnover": 0.7,
        "all_signals_strategy_max_drawdown": -0.04,
        "entry_takeprofit_return": 0.004,
        "entry_takeprofit_avg_daily_return": 0.004,
        "entry_takeprofit_annual_return": 0.004 * 365,
        "entry_takeprofit_sharpe": 0.55,
        "entry_takeprofit_turnover": 0.8,
        "entry_takeprofit_max_drawdown": -0.05,
        "highlow_return": 0.006,
        "highlow_avg_daily_return": 0.006,
        "highlow_annual_return": 0.006 * 365,
        "highlow_sharpe": 0.7,
        "highlow_turnover": 0.8,
        "highlow_max_drawdown": -0.05,
        "maxdiff_return": 0.005,
        "maxdiff_avg_daily_return": 0.0002,
        "maxdiff_annual_return": 0.005 * 365,
        "maxdiff_sharpe": 1.1,
        "maxdiff_turnover": 0.6,
        "maxdiff_max_drawdown": -0.04,
        "maxdiffalwayson_return": 0.007,
        "maxdiffalwayson_avg_daily_return": 0.007,
        "maxdiffalwayson_annual_return": 0.007 * 365,
        "maxdiffalwayson_sharpe": 0.9,
        "maxdiffalwayson_turnover": 0.7,
        "maxdiffalwayson_max_drawdown": -0.05,
        "pctdiff_return": 0.004,
        "pctdiff_avg_daily_return": 0.004,
        "pctdiff_annual_return": 0.004 * 365,
        "pctdiff_sharpe": 0.75,
        "pctdiff_turnover": 1.1,
        "pctdiff_max_drawdown": -0.05,
        "close": 50.0,
        "predicted_close": 50.6,
        "predicted_high": 51.2,
        "predicted_low": 49.8,
        "maxdiffprofit_high_price": 51.4,
        "maxdiffprofit_low_price": 49.2,
        "maxdiff_primary_side": "buy",
        "maxdiff_trade_bias": 0.4,
        "pctdiff_entry_low_price": 49.7,
        "pctdiff_takeprofit_high_price": 51.1,
        "pctdiff_entry_high_price": 50.9,
        "pctdiff_takeprofit_low_price": 49.2,
        "pctdiff_trade_bias": 0.1,
        "simple_forecasted_pnl": 0.003,
        "all_signals_forecasted_pnl": 0.002,
        "entry_takeprofit_forecasted_pnl": 0.001,
        "highlow_forecasted_pnl": 0.0015,
        "maxdiff_forecasted_pnl": 0.012,
        "maxdiffalwayson_forecasted_pnl": 0.004,
        "pctdiff_forecasted_pnl": 0.0025,
    }

    mock_backtest.return_value = pd.DataFrame([row] * 70)

    with patch.object(trade_module, "ALLOW_MAXDIFF_ENTRY", True):
        results = analyze_symbols(["UNIUSD"])

    assert "UNIUSD" in results
    entry = results["UNIUSD"]
    assert entry["strategy"] == "maxdiff"
    assert entry["strategy_candidate_forecasted_pnl"]["maxdiff"] == pytest.approx(0.012)
    reason = entry["strategy_entry_ineligible"].get("maxdiff")
    assert reason and reason.startswith("edge")
    notes = entry.get("strategy_selection_notes", [])
    assert any("allowed_by_forecast" in note for note in notes)


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_prefers_pctdiff_when_percent_edge(
    mock_backtest, mock_snapshot, mock_trading_day_now
):
    rows = []
    for _ in range(70):
        rows.append(
            {
                "simple_strategy_return": 0.0,
                "simple_strategy_avg_daily_return": 0.0,
                "simple_strategy_annual_return": 0.0,
                "simple_strategy_sharpe": 0.1,
                "simple_strategy_turnover": 0.2,
                "simple_strategy_max_drawdown": -0.02,
                "all_signals_strategy_return": 0.0,
                "all_signals_strategy_avg_daily_return": 0.0,
                "all_signals_strategy_annual_return": 0.0,
                "entry_takeprofit_return": 0.0,
                "entry_takeprofit_avg_daily_return": 0.0,
                "entry_takeprofit_annual_return": 0.0,
                "highlow_return": 0.0,
                "highlow_avg_daily_return": 0.0,
                "highlow_annual_return": 0.0,
                "maxdiff_return": -0.01,
                "maxdiff_avg_daily_return": -0.01,
                "maxdiff_annual_return": -0.01 * 365,
                "maxdiff_sharpe": -0.2,
                "maxdiff_turnover": 0.4,
                "maxdiff_max_drawdown": -0.05,
                "pctdiff_return": 0.02,
                "pctdiff_avg_daily_return": 0.02,
                "pctdiff_annual_return": 0.02 * 365,
                "pctdiff_sharpe": 1.5,
                "pctdiff_turnover": 0.5,
                "pctdiff_profit": 0.02,
                "pctdiff_profit_values": [0.02],
                "pctdiff_entry_low_price": 94.5,
                "pctdiff_entry_high_price": 105.5,
                "pctdiff_takeprofit_high_price": 96.39,
                "pctdiff_takeprofit_low_price": 103.9,
                "pctdiff_entry_low_multiplier": -0.01,
                "pctdiff_entry_high_multiplier": 0.01,
                "pctdiff_long_pct": 0.02,
                "pctdiff_short_pct": 0.015,
                "pctdiff_primary_side": "buy",
                "pctdiff_trade_bias": 0.4,
                "pctdiff_trades_positive": 8,
                "pctdiff_trades_negative": 2,
                "pctdiff_trades_total": 10,
                "predicted_close": 100.0,
                "predicted_high": 106.0,
                "predicted_low": 94.0,
                "close": 100.0,
            }
        )

    mock_backtest.return_value = pd.DataFrame(rows)

    results = analyze_symbols(["AAPL"])

    assert results["AAPL"]["strategy"] == "pctdiff"
    assert results["AAPL"]["pctdiff_entry_low_price"] == pytest.approx(94.5)
    assert results["AAPL"]["pctdiff_takeprofit_high_price"] == pytest.approx(96.39)


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


@patch("trade_stock_e2e.is_nyse_trading_day_now", return_value=True)
@patch("trade_stock_e2e._load_latest_forecast_snapshot", return_value={})
@patch("trade_stock_e2e.backtest_forecasts")
def test_analyze_symbols_selects_maxdiffalwayson_when_maxdiff_blocked(
    mock_backtest, mock_snapshot, mock_trading_day_now
):
    rows = []
    for _ in range(70):
        rows.append(
            {
                "simple_strategy_return": -0.001,
                "simple_strategy_avg_daily_return": -0.001,
                "simple_strategy_annual_return": -0.001 * 365,
                "simple_strategy_sharpe": 0.2,
                "simple_strategy_turnover": 0.4,
                "simple_strategy_max_drawdown": -0.03,
                "all_signals_strategy_return": -0.002,
                "all_signals_strategy_avg_daily_return": -0.002,
                "all_signals_strategy_annual_return": -0.002 * 365,
                "all_signals_strategy_sharpe": -0.2,
                "all_signals_strategy_turnover": 0.5,
                "all_signals_strategy_max_drawdown": -0.04,
                "entry_takeprofit_return": -0.0015,
                "entry_takeprofit_avg_daily_return": -0.0015,
                "entry_takeprofit_annual_return": -0.0015 * 365,
                "entry_takeprofit_sharpe": -0.1,
                "entry_takeprofit_turnover": 0.45,
                "entry_takeprofit_max_drawdown": -0.035,
                "highlow_return": -0.001,
                "highlow_avg_daily_return": -0.001,
                "highlow_annual_return": -0.001 * 365,
                "highlow_sharpe": -0.05,
                "highlow_turnover": 0.55,
                "highlow_max_drawdown": -0.03,
                "maxdiff_return": 0.002,
                "maxdiff_avg_daily_return": 0.002,
                "maxdiff_annual_return": 0.002 * 365,
                "maxdiff_sharpe": 0.4,
                "maxdiff_turnover": 0.02,
                "maxdiffprofit_high_price": 4120.0,
                "maxdiffprofit_low_price": 3520.0,
                "maxdiffprofit_profit_high_multiplier": 0.01,
                "maxdiffprofit_profit_low_multiplier": -0.015,
                "maxdiffprofit_profit": 0.002,
                "maxdiffprofit_profit_values": [0.002],
                "maxdiff_primary_side": "sell",
                "maxdiff_trade_bias": -0.25,
                "maxdiff_trades_positive": 0,
                "maxdiff_trades_negative": 6,
                "maxdiff_trades_total": 6,
                "maxdiffalwayson_return": 0.005,
                "maxdiffalwayson_avg_daily_return": 0.005,
                "maxdiffalwayson_annual_return": 0.005 * 365,
                "maxdiffalwayson_sharpe": 1.3,
                "maxdiffalwayson_turnover": 0.018,
                "maxdiffalwayson_profit": 0.005,
                "maxdiffalwayson_profit_values": [0.005],
                "maxdiffalwayson_high_multiplier": 0.02,
                "maxdiffalwayson_low_multiplier": -0.02,
                "maxdiffalwayson_high_price": 4165.0,
                "maxdiffalwayson_low_price": 3535.0,
                "maxdiffalwayson_buy_contribution": 0.003,
                "maxdiffalwayson_sell_contribution": 0.002,
                "maxdiffalwayson_filled_buy_trades": 7,
                "maxdiffalwayson_filled_sell_trades": 5,
                "maxdiffalwayson_trades_total": 12,
                "maxdiffalwayson_trade_bias": 0.2,
                "buy_hold_return": 0.0,
                "buy_hold_avg_daily_return": 0.0,
                "buy_hold_annual_return": 0.0,
                "buy_hold_sharpe": 0.0,
                "buy_hold_finalday": 0.0,
                "unprofit_shutdown_return": 0.0,
                "unprofit_shutdown_avg_daily_return": 0.0,
                "unprofit_shutdown_annual_return": 0.0,
                "unprofit_shutdown_sharpe": 0.0,
                "unprofit_shutdown_finalday": 0.0,
                "predicted_close": 4100.0,
                "predicted_high": 4200.0,
                "predicted_low": 3600.0,
                "close": 3800.0,
                "toto_expected_move_pct": 0.003,
                "kronos_expected_move_pct": -0.001,
                "realized_volatility_pct": 1.2,
                "dollar_vol_20d": 1.2e6,
                "atr_pct_14": 0.015,
                "walk_forward_oos_sharpe": 0.05,
                "walk_forward_turnover": 0.8,
                "walk_forward_highlow_sharpe": -0.05,
                "walk_forward_takeprofit_sharpe": -0.04,
                "walk_forward_maxdiff_sharpe": 0.35,
                "walk_forward_maxdiffalwayson_sharpe": 0.9,
            }
        )

    mock_backtest.return_value = pd.DataFrame(rows)

    with patch.dict(os.environ, {"MARKETSIM_DISABLE_GATES": "1"}, clear=False):
        results = analyze_symbols(["ETHUSD"])

    assert "ETHUSD" in results
    selected = results["ETHUSD"]
    assert selected["strategy"] == "maxdiffalwayson"
    assert selected["strategy_entry_ineligible"].get("maxdiff") is not None
    assert selected["maxdiffalwayson_return"] == pytest.approx(0.005)
    assert selected["maxdiffalwayson_high_price"] == pytest.approx(4165.0)

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
            "close": [100.0],
            "predicted_close": [101.5],
            "predicted_high": [102.0],
            "predicted_low": [99.5],
        }
    )
    mock_backtest.return_value = mock_df

    results = analyze_symbols(["AAPL"])

    assert "AAPL" in results


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

    def record_backout(symbol, **kwargs):
        close_calls.append((symbol, kwargs))

    monkeypatch.setattr(trade_module, "backout_near_market", record_backout)
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

    assert close_calls, "Expected backout_near_market to be invoked"
    symbol, kwargs = close_calls[0]
    assert symbol == "AAPL"
    assert kwargs == {
        "start_offset_minutes": trade_module.BACKOUT_START_OFFSET_MINUTES,
        "sleep_seconds": trade_module.BACKOUT_SLEEP_SECONDS,
        "market_close_buffer_minutes": trade_module.BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
        "market_close_force_minutes": trade_module.BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
    }
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

    mock_backout.assert_called_once_with(
        "GOOG",
        start_offset_minutes=trade_module.BACKOUT_START_OFFSET_MINUTES,
        sleep_seconds=trade_module.BACKOUT_SLEEP_SECONDS,
        market_close_buffer_minutes=trade_module.BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
        market_close_force_minutes=trade_module.BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
    )
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
            "expected_strategy": "all_signals",  # Changed: inverted high/low predictions, simple rejected
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


@patch("trade_stock_e2e._resolve_model_passes")
@patch("trade_stock_e2e._analyze_symbols_impl")
def test_analyze_symbols_merges_best_strategy_across_models(mock_impl, mock_resolve_passes):
    symbol = "BTCUSD"

    base_row = {
        "strategy": "maxdiff",
        "strategy_candidate_forecasted_pnl": {"maxdiff": 0.08, "maxdiffalwayson": 0.05},
        "composite_score": 0.10,
        "close_prediction_source": "chronos2",
        "forecast_model": "chronos2",
        "avg_return": 0.08,
    }
    secondary_row = {
        "strategy": "maxdiff",
        "strategy_candidate_forecasted_pnl": {"maxdiff": 0.07, "maxdiffalwayson": 0.09},
        "composite_score": 0.09,
        "close_prediction_source": "toto",
        "forecast_model": "toto",
        "avg_return": 0.07,
    }

    rerun_row = {**secondary_row, "strategy": "maxdiffalwayson"}

    def impl_side_effect(symbols, *, model_overrides=None, strategy_priorities=None):
        target = symbols[0]
        override = (model_overrides or {}).get(target)
        priority = (strategy_priorities or {}).get(target)
        if override == "toto" and priority == ["maxdiffalwayson"]:
            return {target: rerun_row}
        if override == "toto":
            return {target: secondary_row}
        return {target: base_row}

    mock_impl.side_effect = impl_side_effect
    mock_resolve_passes.return_value = [None, "toto"]

    results = analyze_symbols([symbol])

    assert mock_impl.call_count == 3  # base, secondary, rerun with priority
    assert symbol in results
    assert results[symbol]["strategy"] == "maxdiffalwayson"
    assert results[symbol]["forecast_model"] == "toto"


@patch("trade_stock_e2e._resolve_model_passes", return_value=[None])
@patch("trade_stock_e2e._analyze_symbols_impl", return_value={"ETHUSD": {"strategy": "simple", "composite_score": 0.1}})
def test_analyze_symbols_skips_secondary_when_not_needed(mock_impl, mock_resolve):
    results = analyze_symbols(["ETHUSD"])
    assert results["ETHUSD"]["strategy"] == "simple"
    mock_impl.assert_called_once()

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

    mocks["ramp"].assert_called_once()
    args, kwargs = mocks["ramp"].call_args
    assert args == ("AAPL", "buy")
    assert kwargs["target_qty"] == pytest.approx(5)
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
            "strategy": "maxdiff",
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
            "strategy": "maxdiff",
            "strategy_returns": {"maxdiff": -0.01},
            "predicted_high": 120.0,
            "predicted_low": 115.0,
            "predictions": pd.DataFrame(),
            "trade_mode": "probe",
        }
    }

    with stub_trading_env(positions=[], qty=5, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_not_called()


@patch("src.trade_stock_env_utils._load_trend_summary", return_value={"AAPL": {"pnl": -6000.0}})
def test_manage_positions_trend_pnl_gating(mock_summary, monkeypatch):
    monkeypatch.setenv("MARKETSIM_TREND_PNL_SUSPEND_MAP", "AAPL:-5000")
    current_picks = {
        "AAPL": {
            "side": "sell",
            "avg_return": -0.03,
            "predicted_movement": -0.09,
            "strategy": "maxdiff",
            "strategy_returns": {"maxdiff": -0.04},
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


@patch("src.trade_stock_env_utils._load_trend_summary", return_value={"AAPL": {"pnl": -2000.0}})
def test_manage_positions_trend_pnl_resume(mock_summary, monkeypatch):
    monkeypatch.setenv("MARKETSIM_TREND_PNL_SUSPEND_MAP", "AAPL:-5000")
    monkeypatch.setenv("MARKETSIM_TREND_PNL_RESUME_MAP", "AAPL:-3000")
    current_picks = {
        "AAPL": {
            "side": "sell",
            "avg_return": -0.03,
            "predicted_movement": -0.09,
            "strategy": "maxdiff",
            "strategy_returns": {"maxdiff": -0.04},
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


@pytest.mark.parametrize("strategy_name", ["highlow", "maxdiff", "pctdiff"])
def test_manage_positions_highlow_strategy_uses_limit_orders(strategy_name):
    current_picks = {
        "AAPL": {
            "side": "buy",
            "avg_return": 0.12,
            "predicted_movement": 0.06,
            "strategy": strategy_name,
            "predicted_high": 125.0,
            "predicted_low": 100.0,
            **(
                {
                    "pctdiff_entry_low_price": 99.0,
                    "pctdiff_takeprofit_high_price": 130.5,
                }
                if strategy_name == "pctdiff"
                else {
                    "maxdiffprofit_low_price": 98.5,
                    "maxdiffprofit_high_price": 132.0,
                }
            ),
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
    args, kwargs = mocks["spawn_open_maxdiff"].call_args
    assert args[0] == "AAPL"
    assert args[1] == "buy"
    expected_entry = (
        current_picks["AAPL"].get("pctdiff_entry_low_price")
        if strategy_name == "pctdiff"
        else current_picks["AAPL"].get("maxdiffprofit_low_price")
    )
    assert args[2] == pytest.approx(expected_entry)
    assert kwargs.get("poll_seconds") == trade_module.MAXDIFF_ENTRY_WATCHER_POLL_SECONDS
    assert kwargs.get("force_immediate") is False
    assert kwargs.get("priority_rank") is None
    mocks["spawn_close_maxdiff"].assert_called_once()
    close_args, close_kwargs = mocks["spawn_close_maxdiff"].call_args
    expected_exit = (
        current_picks["AAPL"].get("pctdiff_takeprofit_high_price")
        if strategy_name == "pctdiff"
        else current_picks["AAPL"].get("maxdiffprofit_high_price")
    )
    assert close_args == ("AAPL", "buy", expected_exit)
    assert close_kwargs.get("poll_seconds") == trade_module.MAXDIFF_EXIT_WATCHER_POLL_SECONDS
    assert close_kwargs.get("price_tolerance") == pytest.approx(
        trade_module.MAXDIFF_EXIT_WATCHER_PRICE_TOLERANCE
    )
    mocks["spawn_tp"].assert_not_called()


@pytest.mark.parametrize("strategy_name", ["highlow", "maxdiff", "pctdiff"])
def test_manage_positions_highlow_short_uses_maxdiff_prices(strategy_name):
    current_picks = {
        "UNIUSD": {
            "side": "sell",
            "avg_return": 0.08,
            "predicted_movement": -0.04,
            "strategy": strategy_name,
            "predicted_high": 6.8,
            "predicted_low": 6.1,
            **(
                {
                    "pctdiff_entry_high_price": 6.95,
                    "pctdiff_takeprofit_low_price": 6.02,
                }
                if strategy_name == "pctdiff"
                else {
                    "maxdiffprofit_high_price": 6.9,
                    "maxdiffprofit_low_price": 6.05,
                }
            ),
            "predictions": pd.DataFrame([{"predicted_high": 6.8, "predicted_low": 6.1}]),
        }
    }

    with stub_trading_env(positions=[], qty=2, trading_day_now=True) as mocks:
        manage_positions(current_picks, {}, current_picks)

    mocks["ramp"].assert_not_called()
    mocks["open_order"].assert_not_called()
    mocks["spawn_open_maxdiff"].assert_called_once()
    args, kwargs = mocks["spawn_open_maxdiff"].call_args
    assert args[0] == "UNIUSD"
    assert args[1] == "sell"
    expected_entry = (
        current_picks["UNIUSD"].get("pctdiff_entry_high_price")
        if strategy_name == "pctdiff"
        else current_picks["UNIUSD"].get("maxdiffprofit_high_price")
    )
    assert args[2] == pytest.approx(expected_entry)
    assert kwargs.get("poll_seconds") == trade_module.MAXDIFF_ENTRY_WATCHER_POLL_SECONDS
    assert kwargs.get("force_immediate") is False
    assert kwargs.get("priority_rank") is None
    mocks["spawn_close_maxdiff"].assert_called_once()
    close_args, close_kwargs = mocks["spawn_close_maxdiff"].call_args
    expected_exit = (
        current_picks["UNIUSD"].get("pctdiff_takeprofit_low_price")
        if strategy_name == "pctdiff"
        else current_picks["UNIUSD"].get("maxdiffprofit_low_price")
    )
    assert close_args == ("UNIUSD", "sell", expected_exit)
    assert close_kwargs.get("poll_seconds") == trade_module.MAXDIFF_EXIT_WATCHER_POLL_SECONDS
    assert close_kwargs.get("price_tolerance") == pytest.approx(
        trade_module.MAXDIFF_EXIT_WATCHER_PRICE_TOLERANCE
    )
    mocks["spawn_tp"].assert_not_called()


def test_manage_positions_prioritises_maxdiffalwayson_force_immediate():
    current_picks = {
        "AAPL": {
            "side": "buy",
            "avg_return": 0.05,
            "predicted_movement": 0.04,
            "strategy": "maxdiffalwayson",
            "predicted_high": 210.0,
            "predicted_low": 180.0,
            "maxdiffalwayson_low_price": 182.0,
            "maxdiffalwayson_high_price": 208.0,
            "maxdiffprofit_low_price": 181.0,
            "maxdiffprofit_high_price": 207.0,
            "predictions": pd.DataFrame([
                {"predicted_low": 180.0, "predicted_high": 210.0}
            ]),
        },
        "MSFT": {
            "side": "sell",
            "avg_return": 0.04,
            "predicted_movement": -0.03,
            "strategy": "maxdiffalwayson",
            "predicted_high": 350.0,
            "predicted_low": 320.0,
            "maxdiffalwayson_low_price": 322.0,
            "maxdiffalwayson_high_price": 348.0,
            "maxdiffprofit_low_price": 321.0,
            "maxdiffprofit_high_price": 347.0,
            "predictions": pd.DataFrame([
                {"predicted_low": 320.0, "predicted_high": 350.0}
            ]),
        },
        "GOOG": {
            "side": "buy",
            "avg_return": 0.02,
            "predicted_movement": 0.02,
            "strategy": "maxdiffalwayson",
            "predicted_high": 150.0,
            "predicted_low": 130.0,
            "maxdiffalwayson_low_price": 132.0,
            "maxdiffalwayson_high_price": 148.0,
            "maxdiffprofit_low_price": 131.0,
            "maxdiffprofit_high_price": 147.0,
            "predictions": pd.DataFrame([
                {"predicted_low": 130.0, "predicted_high": 150.0}
            ]),
        },
    }

    with patch.object(trade_module, "MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT", 2):
        with stub_trading_env(positions=[], qty=4, trading_day_now=True) as mocks:
            manage_positions(current_picks, {}, current_picks)

    spawn_calls = mocks["spawn_open_maxdiff"].call_args_list
    assert len(spawn_calls) >= 3

    priority_map = {}
    force_map = {}
    for args, kwargs in spawn_calls:
        symbol = args[0]
        priority_map.setdefault(symbol, set()).add(kwargs.get("priority_rank"))
        force_map.setdefault(symbol, set()).add(kwargs.get("force_immediate"))

    assert priority_map["AAPL"] == {1}
    assert force_map["AAPL"] == {True}
    assert priority_map["MSFT"] == {2}
    assert force_map["MSFT"] == {True}
    assert priority_map["GOOG"] == {3}
    assert force_map["GOOG"] == {False}


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
