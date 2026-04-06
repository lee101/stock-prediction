"""Tests for RL signal feature computation edge cases."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rl-trading-agent-binance"))

from rl_signal import (
    FEATURES_PER_SYM,
    PortfolioSnapshot,
    RLSignal,
    RLSignalGenerator,
    _build_action_names,
    _forecast_confidence,
    _forecast_delta,
    compute_symbol_features,
)


def _make_price_df(n=100, base_price=100.0, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = base_price * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    opn = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.uniform(1000, 10000, n)
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _make_forecast_df(idx, close, horizon=1):
    return pd.DataFrame({
        "predicted_close_p50": close * 1.001,
        "predicted_high_p50": close * 1.005,
        "predicted_low_p50": close * 0.995,
        "predicted_close_p90": close * 1.01,
        "predicted_close_p10": close * 0.99,
    }, index=idx)


class TestComputeSymbolFeatures:
    def test_output_shape_and_dtype(self):
        df = _make_price_df(100)
        fc1 = _make_forecast_df(df.index, df["close"])
        fc24 = _make_forecast_df(df.index, df["close"], 24)
        result = compute_symbol_features(df, fc1, fc24)
        assert result.shape == (100, FEATURES_PER_SYM)
        assert result.dtype == np.float32

    def test_no_nans_or_infs(self):
        df = _make_price_df(200)
        fc1 = _make_forecast_df(df.index, df["close"])
        fc24 = _make_forecast_df(df.index, df["close"], 24)
        result = compute_symbol_features(df, fc1, fc24)
        assert np.all(np.isfinite(result))

    def test_empty_forecasts(self):
        df = _make_price_df(50)
        result = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        assert result.shape == (50, FEATURES_PER_SYM)
        assert np.all(np.isfinite(result))
        assert result[:, 0].sum() == 0.0  # chronos deltas should be 0

    def test_single_bar(self):
        df = _make_price_df(1)
        result = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        assert result.shape == (1, FEATURES_PER_SYM)
        assert np.all(np.isfinite(result))

    def test_constant_price(self):
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1000.0
        }, index=idx)
        result = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        assert np.all(np.isfinite(result))
        assert result[-1, 8] == 0.0  # return_1h should be 0
        assert result[-1, 10] == 0.0  # volatility should be 0

    def test_extreme_price_spike(self):
        df = _make_price_df(100)
        df.iloc[50, df.columns.get_loc("close")] = 1e8
        df.iloc[50, df.columns.get_loc("high")] = 1e8
        result = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        assert np.all(np.isfinite(result))
        # return clipping should cap extreme moves
        assert abs(result[50, 8]) <= 0.5  # return_1h clipped

    def test_near_zero_price(self):
        n = 50
        idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": 1e-9, "high": 1e-9, "low": 1e-9,
            "close": 1e-9, "volume": 1000.0
        }, index=idx)
        result = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        assert np.all(np.isfinite(result))

    def test_features_clipped_correctly(self):
        df = _make_price_df(200)
        result = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        # return_1h clipped to [-0.5, 0.5]
        assert np.all(result[:, 8] >= -0.5)
        assert np.all(result[:, 8] <= 0.5)
        # return_24h clipped to [-1.0, 1.0]
        assert np.all(result[:, 9] >= -1.0)
        assert np.all(result[:, 9] <= 1.0)
        # volatility clipped to [0, 1]
        assert np.all(result[:, 10] >= 0.0)
        assert np.all(result[:, 10] <= 1.0)
        # drawdown clipped to [-1, 0]
        assert np.all(result[:, 15] >= -1.0)
        assert np.all(result[:, 15] <= 0.0)


class TestForecastDelta:
    def test_missing_column(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        fc = pd.DataFrame({"other_col": [1, 2, 3, 4, 5]}, index=idx)
        close = pd.Series(100.0, index=idx)
        result = _forecast_delta(fc, idx, "nonexistent", close)
        assert len(result) == 5
        assert (result == 0.0).all()

    def test_empty_forecast(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        close = pd.Series(100.0, index=idx)
        result = _forecast_delta(pd.DataFrame(), idx, "predicted_close_p50", close)
        assert (result == 0.0).all()


class TestForecastConfidence:
    def test_narrow_spread_high_confidence(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        fc = pd.DataFrame({
            "predicted_close_p90": [101.0] * 5,
            "predicted_close_p10": [99.0] * 5,
        }, index=idx)
        close = pd.Series(100.0, index=idx)
        result = _forecast_confidence(fc, idx, close)
        assert np.all(result > 0.9)

    def test_wide_spread_low_confidence(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        fc = pd.DataFrame({
            "predicted_close_p90": [200.0] * 5,
            "predicted_close_p10": [50.0] * 5,
        }, index=idx)
        close = pd.Series(100.0, index=idx)
        result = _forecast_confidence(fc, idx, close)
        assert np.all(result < 0.5)

    def test_empty_forecast_returns_half(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        close = pd.Series(100.0, index=idx)
        result = _forecast_confidence(pd.DataFrame(), idx, close)
        assert np.allclose(result, 0.5)


class TestEndToEndSignal:
    def _make_gen(self, num_symbols=2):
        gen = object.__new__(RLSignalGenerator)
        if num_symbols == 2:
            gen.symbols = ("BTCUSD", "ETHUSD")
        else:
            gen.symbols = ("BTCUSD",)
        gen.num_symbols = num_symbols
        gen.features_per_sym = FEATURES_PER_SYM
        gen.action_allocation_bins = 1
        gen.action_level_bins = 1
        gen.per_symbol_actions = 1
        gen.disable_shorts = False
        gen.action_names = _build_action_names(gen.symbols)
        gen.obs_size = gen.num_symbols * FEATURES_PER_SYM + 5 + gen.num_symbols
        gen.num_actions = 1 + 2 * gen.num_symbols
        gen._episode_step = 0
        gen.max_steps = 720
        gen.forecast_cache_root = None
        gen.device = "cpu"
        return gen

    def _make_policy(self, logits):
        import torch

        class _P:
            def __call__(self, obs):
                return torch.tensor([logits], dtype=torch.float32), torch.tensor([0.0])
        return _P()

    def test_flat_signal_with_zero_logits(self):
        gen = self._make_gen(1)
        gen.policy = self._make_policy([10.0, -5.0, -5.0])
        portfolio = PortfolioSnapshot(cash_usd=10000.0)
        sig = gen.get_signal(
            portfolio=portfolio,
            klines_map={"BTCUSD": None},
            tradable_symbols=["BTCUSD"],
            spot_only=True,
        )
        assert sig.direction == "flat"
        assert sig.allocation_pct == 0.0

    def test_long_signal(self):
        gen = self._make_gen(1)
        gen.policy = self._make_policy([-5.0, 10.0, -5.0])
        portfolio = PortfolioSnapshot(cash_usd=10000.0)
        sig = gen.get_signal(
            portfolio=portfolio,
            klines_map={"BTCUSD": None},
            tradable_symbols=["BTCUSD"],
            spot_only=True,
        )
        assert sig.direction == "long"
        assert sig.target_symbol == "BTCUSD"
        assert sig.confidence > 0.5

    def test_short_masked_to_flat(self):
        gen = self._make_gen(1)
        gen.policy = self._make_policy([-5.0, -10.0, 10.0])
        portfolio = PortfolioSnapshot(cash_usd=10000.0)
        sig = gen.get_signal(
            portfolio=portfolio,
            klines_map={"BTCUSD": None},
            tradable_symbols=["BTCUSD"],
            spot_only=True,
        )
        assert sig.direction == "flat"

    def test_episode_step_increments(self):
        gen = self._make_gen(1)
        gen.policy = self._make_policy([1.0, -1.0, -1.0])
        portfolio = PortfolioSnapshot(cash_usd=10000.0)
        assert gen._episode_step == 0
        gen.get_signal(portfolio=portfolio, klines_map={"BTCUSD": None},
                       tradable_symbols=["BTCUSD"], spot_only=True)
        assert gen._episode_step == 1
        gen.get_signal(portfolio=portfolio, klines_map={"BTCUSD": None},
                       tradable_symbols=["BTCUSD"], spot_only=True)
        assert gen._episode_step == 2

    def test_reset_episode_clears_step(self):
        gen = self._make_gen(1)
        gen._episode_step = 50
        gen.reset_episode()
        assert gen._episode_step == 0

    def test_degenerate_portfolio_zero_cash(self):
        gen = self._make_gen(1)
        gen.policy = self._make_policy([1.0, 0.0, -1.0])
        portfolio = PortfolioSnapshot(cash_usd=0.0, position_value_usd=10000.0,
                                     position_symbol="BTCUSD", hold_hours=5)
        sig = gen.get_signal(portfolio=portfolio, klines_map={"BTCUSD": None},
                             tradable_symbols=["BTCUSD"], spot_only=True)
        assert isinstance(sig, RLSignal)

    def test_all_symbols_masked(self):
        gen = self._make_gen(2)
        gen.policy = self._make_policy([0.0, 5.0, 5.0, -1.0, -1.0])
        portfolio = PortfolioSnapshot(cash_usd=10000.0)
        sig = gen.get_signal(portfolio=portfolio,
                             klines_map={"BTCUSD": None, "ETHUSD": None},
                             tradable_symbols=[], spot_only=True)
        assert sig.direction == "flat"

    def test_portfolio_with_short_position(self):
        gen = self._make_gen(1)
        gen.policy = self._make_policy([1.0, -1.0, -1.0])
        portfolio = PortfolioSnapshot(
            cash_usd=5000.0, position_value_usd=5000.0,
            position_symbol="BTCUSD", is_short=True,
            unrealized_pnl_usd=-100.0, hold_hours=3
        )
        sig = gen.get_signal(portfolio=portfolio, klines_map={"BTCUSD": None},
                             tradable_symbols=["BTCUSD"], spot_only=False)
        assert isinstance(sig, RLSignal)


class TestPortfolioSnapshot:
    def test_defaults(self):
        p = PortfolioSnapshot()
        assert p.cash_usd == 0.0
        assert p.position_symbol is None
        assert p.is_short is False
        assert p.hold_hours == 0

    def test_custom_values(self):
        p = PortfolioSnapshot(
            cash_usd=5000.0, total_value_usd=10000.0,
            position_symbol="ETHUSD", position_value_usd=5000.0,
            unrealized_pnl_usd=200.0, hold_hours=3, is_short=False
        )
        assert p.position_symbol == "ETHUSD"
        assert p.hold_hours == 3
