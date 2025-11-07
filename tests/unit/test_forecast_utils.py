"""Tests for forecast_utils module."""

from unittest.mock import patch

from src.forecast_utils import extract_forecasted_pnl, load_latest_forecast_snapshot


class TestExtractForecastedPnl:
    def test_extracts_maxdiff_forecasted_pnl(self):
        forecast = {"maxdiff_forecasted_pnl": 123.45}
        assert extract_forecasted_pnl(forecast) == 123.45

    def test_extracts_maxdiffalwayson_forecasted_pnl(self):
        forecast = {"maxdiffalwayson_forecasted_pnl": 67.89}
        assert extract_forecasted_pnl(forecast) == 67.89

    def test_extracts_highlow_forecasted_pnl(self):
        forecast = {"highlow_forecasted_pnl": -23.45}
        assert extract_forecasted_pnl(forecast) == -23.45

    def test_extracts_avg_return(self):
        forecast = {"avg_return": 0.042}
        assert extract_forecasted_pnl(forecast) == 0.042

    def test_priority_order(self):
        forecast = {
            "avg_return": 10.0,
            "highlow_forecasted_pnl": 20.0,
            "maxdiffalwayson_forecasted_pnl": 30.0,
            "maxdiff_forecasted_pnl": 40.0,
        }
        assert extract_forecasted_pnl(forecast) == 40.0

    def test_skips_invalid_values(self):
        forecast = {
            "maxdiff_forecasted_pnl": "invalid",
            "maxdiffalwayson_forecasted_pnl": None,
            "highlow_forecasted_pnl": 99.99,
        }
        assert extract_forecasted_pnl(forecast) == 99.99

    def test_returns_default_when_no_valid_field(self):
        forecast = {"unrelated_field": 123}
        assert extract_forecasted_pnl(forecast, default=0.0) == 0.0
        assert extract_forecasted_pnl(forecast, default=-1.0) == -1.0

    def test_handles_empty_forecast(self):
        assert extract_forecasted_pnl({}) == 0.0

    def test_handles_string_numbers(self):
        forecast = {"maxdiff_forecasted_pnl": "123.45"}
        assert extract_forecasted_pnl(forecast) == 123.45

    def test_handles_negative_pnl(self):
        forecast = {"maxdiff_forecasted_pnl": -456.78}
        assert extract_forecasted_pnl(forecast) == -456.78


class TestLoadLatestForecastSnapshot:
    @patch("src.forecast_utils.load_latest_forecast_snapshot")
    def test_loads_forecast_snapshot(self, mock_load):
        mock_load.return_value = {
            "AAPL": {"maxdiff_forecasted_pnl": 100.0},
            "BTCUSD": {"avg_return": 0.05},
        }

        result = load_latest_forecast_snapshot()
        assert "AAPL" in result
        assert "BTCUSD" in result
        mock_load.assert_called_once()
