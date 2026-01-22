from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from bagsfm.config import BagsConfig, DataConfig, ForecastConfig, TokenConfig
from bagsfm.data_collector import DataCollector, OHLCBar
from bagsfm.forecaster import TokenForecaster


def _make_bars(count: int, start: datetime, step_minutes: int = 10) -> list[OHLCBar]:
    bars = []
    price = 1e-7
    for i in range(count):
        timestamp = start + timedelta(minutes=step_minutes * i)
        price *= 1.001
        bars.append(
            OHLCBar(
                timestamp=timestamp,
                token_mint="TEST",
                token_symbol="TST",
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=0.0,
                num_ticks=1,
            )
        )
    return bars


def test_forecast_from_bars_simple_fallback(tmp_path, monkeypatch):
    data_config = DataConfig(data_dir=tmp_path)
    collector = DataCollector(BagsConfig(), data_config)
    forecast_config = ForecastConfig(prediction_length=3, context_length=32)
    forecaster = TokenForecaster(collector, forecast_config)

    monkeypatch.setattr(TokenForecaster, "_ensure_initialized", lambda self: None)
    forecaster._wrapper = None

    token = TokenConfig(symbol="TST", mint="TEST", decimals=9)
    bars = _make_bars(40, datetime(2026, 1, 1))

    forecast = forecaster.forecast_from_bars(token, bars)
    assert forecast is not None
    assert len(forecast.predicted_prices) == 3
    assert forecast.predicted_return > -1.0


def test_forecast_from_bars_insufficient_data(tmp_path, monkeypatch):
    data_config = DataConfig(data_dir=tmp_path)
    collector = DataCollector(BagsConfig(), data_config)
    forecast_config = ForecastConfig(prediction_length=2, context_length=32)
    forecaster = TokenForecaster(collector, forecast_config)

    monkeypatch.setattr(TokenForecaster, "_ensure_initialized", lambda self: None)
    forecaster._wrapper = None

    token = TokenConfig(symbol="TST", mint="TEST", decimals=9)
    bars = _make_bars(5, datetime(2026, 1, 1))

    assert forecaster.forecast_from_bars(token, bars) is None
