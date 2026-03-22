"""Tests for binance_worksteal/cute_forecast.py and batch_forecast.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.cute_forecast import (
    _check_cute_available,
    _extract_quantiles,
    _find_lora_path,
    _naive_fallback,
    _pad_to_horizon,
    _prepare_ohlc_contexts,
    forecast_batch,
    forecast_symbol,
    get_pipeline,
    _pipeline_cache,
    LORA_MAP,
    LORA_ROOT,
)


def _make_bars(n=100, base_price=50000.0) -> pd.DataFrame:
    np.random.seed(42)
    ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = base_price + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - np.abs(np.random.randn(n) * 50),
        "high": close + np.abs(np.random.randn(n) * 100),
        "low": close - np.abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.random.rand(n) * 1e6,
    })


class TestHelpers:
    def test_pad_to_horizon_short(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _pad_to_horizon(arr, 5)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 3.0, 3.0])

    def test_pad_to_horizon_exact(self):
        arr = np.array([1.0, 2.0])
        result = _pad_to_horizon(arr, 2)
        assert result.shape == (2,)

    def test_pad_to_horizon_long(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _pad_to_horizon(arr, 2)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_pad_to_horizon_empty(self):
        arr = np.array([])
        result = _pad_to_horizon(arr, 3)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_extract_quantiles(self):
        q = torch.zeros(1, 10, 3)
        q[..., 0] = 1.0
        q[..., 1] = 2.0
        q[..., 2] = 3.0
        result = _extract_quantiles(q, 10)
        assert set(result.keys()) == {"p10", "p50", "p90"}
        np.testing.assert_array_equal(result["p10"], np.full(10, 1.0))

    def test_naive_fallback(self):
        result = _naive_fallback(100.0, 5)
        assert result["p50"].shape == (5,)
        assert np.all(result["p10"] < result["p50"])
        assert np.all(result["p50"] < result["p90"])


class TestPrepareOHLCContexts:
    def test_basic(self):
        df = _make_bars(200)
        ctx = _prepare_ohlc_contexts(df, max_len=128)
        assert set(ctx.keys()) == {"close", "high", "low"}
        for k, t in ctx.items():
            assert isinstance(t, torch.Tensor)
            assert t.shape == (128,)
            assert t.dtype == torch.float32

    def test_short_df(self):
        df = _make_bars(10)
        ctx = _prepare_ohlc_contexts(df, max_len=512)
        assert ctx["close"].shape == (10,)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        ctx = _prepare_ohlc_contexts(df, max_len=512)
        assert "high" in ctx
        assert "low" in ctx
        np.testing.assert_array_equal(ctx["high"].numpy(), ctx["close"].numpy())


class TestFindLoraPath:
    def test_known_symbol(self):
        path = _find_lora_path("BTCUSDT")
        if LORA_ROOT.exists():
            # may or may not exist on CI
            if path is not None:
                assert (path / "config.json").exists()
        else:
            assert path is None

    def test_unknown_symbol(self):
        path = _find_lora_path("FAKECOINUSDT")
        assert path is None


class FakePipeline:
    """Mock pipeline that returns correctly shaped quantile predictions."""
    model_prediction_length = 64

    def predict_quantiles(self, context, prediction_length=None, quantile_levels=None):
        if isinstance(context, list):
            batch_size = len(context)
        elif isinstance(context, torch.Tensor):
            batch_size = context.shape[0] if context.ndim == 2 else 1
        else:
            batch_size = 1

        pred_len = prediction_length or 24
        n_q = len(quantile_levels) if quantile_levels else 3

        quantiles = []
        means = []
        for i in range(batch_size):
            base = 50000.0 + i * 1000
            q = torch.zeros(1, pred_len, n_q)
            q[..., 0] = base * 0.98  # p10
            q[..., 1] = base         # p50
            q[..., 2] = base * 1.02  # p90
            quantiles.append(q)
            means.append(torch.full((1, pred_len), base))
        return quantiles, means


class TestForecastSymbol:
    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_output_structure(self, mock_get):
        mock_get.return_value = FakePipeline()
        df = _make_bars(100)
        result = forecast_symbol("BTCUSDT", df, horizon=24)

        assert set(result.keys()) == {"close", "high", "low"}
        for col in ("close", "high", "low"):
            assert set(result[col].keys()) == {"p10", "p50", "p90"}
            for q_key in ("p10", "p50", "p90"):
                arr = result[col][q_key]
                assert isinstance(arr, np.ndarray)
                assert arr.shape == (24,)
                assert np.all(np.isfinite(arr))

    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_p10_lt_p50_lt_p90(self, mock_get):
        mock_get.return_value = FakePipeline()
        df = _make_bars(100)
        result = forecast_symbol("BTCUSDT", df, horizon=24)
        for col in result:
            assert np.all(result[col]["p10"] <= result[col]["p50"])
            assert np.all(result[col]["p50"] <= result[col]["p90"])

    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_horizon_padding(self, mock_get):
        """If model returns fewer steps than horizon, should pad."""
        pipe = FakePipeline()
        pipe.model_prediction_length = 10  # less than requested 24
        mock_get.return_value = pipe
        df = _make_bars(100)
        result = forecast_symbol("BTCUSDT", df, horizon=24)
        assert result["close"]["p50"].shape == (24,)

    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_fallback_on_error(self, mock_get):
        pipe = MagicMock()
        pipe.model_prediction_length = 64
        pipe.predict_quantiles.side_effect = RuntimeError("GPU OOM")
        mock_get.return_value = pipe
        df = _make_bars(100)
        result = forecast_symbol("BTCUSDT", df, horizon=24)
        # should return naive fallback without raising
        assert result["close"]["p50"].shape == (24,)
        assert np.all(np.isfinite(result["close"]["p50"]))


class TestForecastBatch:
    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_batch_multiple_symbols(self, mock_get):
        mock_get.return_value = FakePipeline()
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        bars = {s: _make_bars(100, base_price=50000 - i * 10000) for i, s in enumerate(symbols)}
        results = forecast_batch(symbols, bars, horizon=24, batch_size=2)
        assert set(results.keys()) == set(symbols)
        for sym in symbols:
            assert "close" in results[sym]
            assert results[sym]["close"]["p50"].shape == (24,)

    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_batch_skips_missing(self, mock_get):
        mock_get.return_value = FakePipeline()
        bars = {"BTCUSDT": _make_bars(100)}
        results = forecast_batch(["BTCUSDT", "MISSING"], bars, horizon=24)
        assert "BTCUSDT" in results
        assert "MISSING" not in results

    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_batch_error_fallback(self, mock_get):
        pipe = MagicMock()
        pipe.model_prediction_length = 64
        pipe.predict_quantiles.side_effect = RuntimeError("kaboom")
        mock_get.return_value = pipe
        bars = {"BTCUSDT": _make_bars(100)}
        results = forecast_batch(["BTCUSDT"], bars, horizon=24)
        assert "BTCUSDT" in results
        assert results["BTCUSDT"]["close"]["p50"].shape == (24,)


class TestGetPipeline:
    def test_caching(self):
        _pipeline_cache.clear()
        fake = FakePipeline()
        with patch("binance_worksteal.cute_forecast._check_cute_available", return_value=False), \
             patch("binance_worksteal.cute_forecast._load_original_pipeline", return_value=fake):
            p1 = get_pipeline("test-model", "cpu", use_cute=False)
            p2 = get_pipeline("test-model", "cpu", use_cute=False)
            assert p1 is p2
        _pipeline_cache.clear()


class TestCuteAvailability:
    def test_check_returns_bool(self):
        import binance_worksteal.cute_forecast as mod
        old = mod._cute_available
        mod._cute_available = None
        result = _check_cute_available()
        assert isinstance(result, bool)
        mod._cute_available = old

    @patch("binance_worksteal.cute_forecast.get_pipeline")
    def test_fallback_when_cute_unavailable(self, mock_get):
        """Ensure forecast_symbol works when cute is not available."""
        mock_get.return_value = FakePipeline()
        df = _make_bars(50)
        result = forecast_symbol("ETHUSDT", df, horizon=12, use_cute=False)
        assert result["close"]["p50"].shape == (12,)


class TestBatchForecastCLI:
    def test_forecast_to_cache_rows(self):
        from binance_worksteal.batch_forecast import forecast_to_cache_rows
        df = _make_bars(50)
        forecast = {
            "close": {"p10": np.ones(24), "p50": np.ones(24) * 2, "p90": np.ones(24) * 3},
            "high": {"p10": np.ones(24), "p50": np.ones(24) * 4, "p90": np.ones(24) * 5},
            "low": {"p10": np.ones(24), "p50": np.ones(24) * 0.5, "p90": np.ones(24) * 1.5},
        }
        rows = forecast_to_cache_rows("BTCUSDT", df, forecast, horizon=24)
        assert len(rows) == 24
        expected_cols = [
            "timestamp", "symbol", "issued_at", "target_timestamp",
            "horizon_hours", "predicted_close_p50", "predicted_close_p10",
            "predicted_close_p90", "predicted_high_p50", "predicted_low_p50",
        ]
        for col in expected_cols:
            assert col in rows.columns, f"Missing column: {col}"
        assert rows["symbol"].iloc[0] == "BTCUSDT"
        assert rows["predicted_close_p50"].iloc[0] == 2.0

    def test_load_bars(self, tmp_path):
        from binance_worksteal.batch_forecast import load_bars
        df = _make_bars(20)
        csv_path = tmp_path / "TESTUSDT.csv"
        df.to_csv(csv_path, index=False)
        loaded = load_bars(tmp_path, "TESTUSDT")
        assert len(loaded) == 20
        assert "close" in loaded.columns

    def test_load_bars_missing(self, tmp_path):
        from binance_worksteal.batch_forecast import load_bars
        loaded = load_bars(tmp_path, "NONEXIST")
        assert loaded.empty
