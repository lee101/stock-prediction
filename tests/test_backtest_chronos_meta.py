"""Tests for Chronos-2 PnL forecast meta-selector."""
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binanceleveragesui.backtest_chronos_meta import (
    resample_trace_hourly,
    forecast_equity_growth,
    chronos_select_model,
)


def _make_5m_trace(hours=4, start_equity=1000.0, growth_per_hour=10.0):
    timestamps = pd.date_range("2026-01-01", periods=hours * 12, freq="5min", tz="UTC")
    equities = [start_equity + (i / 12) * growth_per_hour for i in range(len(timestamps))]
    return pd.DataFrame({"timestamp": timestamps, "equity": equities})


class TestResampleTraceHourly:
    def test_basic_resampling(self):
        trace = _make_5m_trace(hours=3, start_equity=1000, growth_per_hour=10)
        hourly = resample_trace_hourly(trace)
        assert len(hourly) == 3
        assert hourly.iloc[0] == pytest.approx(1000 + 10 * (11 / 12), rel=1e-3)

    def test_empty_trace(self):
        trace = pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC"), "equity": []})
        hourly = resample_trace_hourly(trace)
        assert len(hourly) == 0

    def test_single_hour(self):
        trace = _make_5m_trace(hours=1, start_equity=500, growth_per_hour=5)
        hourly = resample_trace_hourly(trace)
        assert len(hourly) == 1


class TestForecastEquityGrowth:
    def _mock_pipeline(self, forecast_final: float):
        pipeline = MagicMock()
        tensor = torch.zeros(1, 21, 6)
        tensor[0, 10, -1] = forecast_final  # median quantile at last step
        pipeline.predict.return_value = [tensor]
        return pipeline

    def test_positive_growth(self):
        pipeline = self._mock_pipeline(110.0)
        eq = np.array([100.0] * 48)
        growth = forecast_equity_growth(pipeline, eq, prediction_length=6)
        assert growth == pytest.approx(0.1, rel=1e-6)

    def test_negative_growth(self):
        pipeline = self._mock_pipeline(90.0)
        eq = np.array([100.0] * 48)
        growth = forecast_equity_growth(pipeline, eq, prediction_length=6)
        assert growth == pytest.approx(-0.1, rel=1e-6)

    def test_short_array_returns_zero(self):
        pipeline = self._mock_pipeline(110.0)
        growth = forecast_equity_growth(pipeline, np.array([100.0, 101.0]), prediction_length=6)
        assert growth == 0.0

    def test_zero_equity_returns_zero(self):
        pipeline = self._mock_pipeline(110.0)
        eq = np.array([0.0] * 10)
        growth = forecast_equity_growth(pipeline, eq, prediction_length=6)
        assert growth == 0.0


class TestChronosSelectModel:
    def _make_equities(self, doge_vals, aave_vals):
        ts = pd.date_range("2026-01-01", periods=len(doge_vals), freq="1h", tz="UTC")
        return {
            "doge": pd.Series(doge_vals, index=ts),
            "aave": pd.Series(aave_vals, index=ts),
        }

    def _mock_pipeline_growth(self, growths: dict):
        """Mock pipeline that returns specified growth for each call."""
        call_count = [0]
        model_order = list(growths.keys())

        def predict_fn(inputs, prediction_length=6):
            name = model_order[call_count[0] % len(model_order)]
            call_count[0] += 1
            current_val = float(inputs[0][-1])
            forecast_val = current_val * (1.0 + growths[name])
            tensor = torch.zeros(1, 21, prediction_length)
            tensor[0, 10, -1] = forecast_val
            return [tensor]

        pipeline = MagicMock()
        pipeline.predict.side_effect = predict_fn
        return pipeline

    def test_picks_higher_growth(self):
        equities = self._make_equities(
            [1000 + i for i in range(48)],
            [1000 + i * 0.5 for i in range(48)],
        )
        pipeline = self._mock_pipeline_growth({"doge": 0.05, "aave": 0.02})
        current_ts = equities["doge"].index[-1]
        chosen, forecasts = chronos_select_model(pipeline, equities, current_ts)
        assert chosen == "doge"
        assert forecasts["doge"] > forecasts["aave"]

    def test_cash_when_both_negative(self):
        equities = self._make_equities(
            [1000 - i for i in range(48)],
            [1000 - i * 0.5 for i in range(48)],
        )
        pipeline = self._mock_pipeline_growth({"doge": -0.05, "aave": -0.02})
        current_ts = equities["doge"].index[-1]
        chosen, forecasts = chronos_select_model(
            pipeline, equities, current_ts, growth_threshold=0.0
        )
        assert chosen == ""

    def test_cold_start_short_history(self):
        ts = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
        equities = {
            "doge": pd.Series([100, 101], index=ts),
            "aave": pd.Series([100, 99], index=ts),
        }
        pipeline = MagicMock()
        chosen, forecasts = chronos_select_model(pipeline, equities, ts[-1])
        assert forecasts["doge"] == 0.0
        assert forecasts["aave"] == 0.0
        pipeline.predict.assert_not_called()
