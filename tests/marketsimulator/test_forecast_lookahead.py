import os

import pandas as pd
import pytest

from marketsimulator import backtest_test3_inline
from marketsimulator.environment import activate_simulation
from marketsimulator.predict_stock_forecasting_mock import make_predictions


@pytest.fixture
def simulation_env(monkeypatch):
    monkeypatch.setenv("MARKETSIM_ALLOW_MOCK_ANALYTICS", "1")
    monkeypatch.setenv("MARKETSIM_SKIP_REAL_IMPORT", "1")
    with activate_simulation(symbols=["AAPL"], initial_cash=100_000.0, use_mock_analytics=True) as controller:
        yield controller


def _slice_window(series, count):
    frame = series.frame.iloc[:count].copy()
    if isinstance(frame["timestamp"].iloc[0], str):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def test_make_predictions_respects_lookahead(simulation_env, monkeypatch):
    controller = simulation_env
    state = controller.state
    series = state.prices["AAPL"]
    series.cursor = 0
    lookahead = 3
    monkeypatch.setenv("MARKETSIM_FORECAST_LOOKAHEAD", str(lookahead))

    predictions = make_predictions(symbols=["AAPL"])
    assert not predictions.empty
    row = predictions.loc[predictions["instrument"] == "AAPL"].iloc[0]

    target_idx = min(series.cursor + lookahead, len(series.frame) - 1)
    future_slice = series.frame.iloc[series.cursor + 1 : target_idx + 1]
    if future_slice.empty:
        future_slice = series.frame.iloc[target_idx : target_idx + 1]
    expected_close = float(future_slice["Close"].iloc[-1])
    expected_high = float(future_slice["High"].max())
    expected_low = float(future_slice["Low"].min())

    assert pytest.approx(row["close_predicted_price"], rel=1e-9) == expected_close
    assert pytest.approx(row["high_predicted_price"], rel=1e-9) == expected_high
    assert pytest.approx(row["low_predicted_price"], rel=1e-9) == expected_low


def test_fallback_backtest_lookahead_alignment(simulation_env, monkeypatch):
    controller = simulation_env
    state = controller.state
    series = state.prices["AAPL"]
    series.cursor = 0
    lookahead = 4
    monkeypatch.setenv("MARKETSIM_FORECAST_LOOKAHEAD", str(lookahead))

    sims = 12
    window = _slice_window(series, sims)
    result = backtest_test3_inline.backtest_forecasts("AAPL", num_simulations=sims)
    assert not result.empty

    oldest_row = result.iloc[-1]
    expected_close = float(window["Close"].iloc[min(len(window) - 1, lookahead)])
    future_high_slice = window["High"].iloc[1 : lookahead + 1]
    future_low_slice = window["Low"].iloc[1 : lookahead + 1]
    if future_high_slice.empty:
        future_high = float(window["High"].iloc[min(len(window) - 1, lookahead)])
    else:
        future_high = float(future_high_slice.max())
    if future_low_slice.empty:
        future_low = float(window["Low"].iloc[min(len(window) - 1, lookahead)])
    else:
        future_low = float(future_low_slice.min())

    assert pytest.approx(oldest_row["predicted_close"], rel=1e-9) == expected_close
    assert pytest.approx(oldest_row["predicted_high"], rel=1e-9) == future_high
    assert pytest.approx(oldest_row["predicted_low"], rel=1e-9) == future_low
