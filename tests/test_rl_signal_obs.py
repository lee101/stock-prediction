"""Tests for rl_signal.py obs construction and portfolio model support."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl_trading_agent_binance"))

from rl_signal import (
    FEATURES_PER_SYM,
    INITIAL_CASH,
    DEFAULT_MAX_STEPS,
    _infer_num_symbols,
    _OBS_SIZE_TO_SYMBOLS,
    PortfolioSnapshot,
    RLSignalGenerator,
    compute_symbol_features,
)


class TestInferNumSymbols:
    def test_legacy_4sym(self):
        assert _infer_num_symbols(73) == 4

    def test_crypto15(self):
        assert _infer_num_symbols(260) == 15

    def test_mixed23(self):
        assert _infer_num_symbols(396) == 23

    def test_custom_features_per_sym(self):
        # F=20: obs = S*(20+1)+5 = S*21+5
        obs_size = 4 * 21 + 5  # 89
        assert _infer_num_symbols(obs_size, features_per_sym=20) == 4

    def test_all_known_obs_sizes(self):
        for obs_size, symbols in _OBS_SIZE_TO_SYMBOLS.items():
            n = _infer_num_symbols(obs_size)
            assert n == len(symbols), f"obs_size={obs_size}: got {n}, expected {len(symbols)}"


class TestBuildObs:
    """Test obs construction via a mock-like approach without loading a real checkpoint."""

    def _make_generator_stub(self, num_symbols=4, features_per_sym=16, max_steps=720):
        """Create a minimal RLSignalGenerator-like object for testing _build_obs."""
        gen = object.__new__(RLSignalGenerator)
        gen.num_symbols = num_symbols
        gen.features_per_sym = features_per_sym
        gen.max_steps = max_steps
        gen.obs_size = num_symbols * features_per_sym + 5 + num_symbols
        gen.symbols = tuple(f"SYM{i}" for i in range(num_symbols))
        gen._episode_step = 0
        return gen

    def test_obs_shape(self):
        gen = self._make_generator_stub(num_symbols=4)
        mf = np.zeros((4, 16), dtype=np.float32)
        port = PortfolioSnapshot(cash_usd=10000.0)
        obs = gen._build_obs(mf, port)
        assert obs.shape == (73,)

    def test_obs_shape_23sym(self):
        gen = self._make_generator_stub(num_symbols=23)
        mf = np.zeros((23, 16), dtype=np.float32)
        port = PortfolioSnapshot(cash_usd=10000.0)
        obs = gen._build_obs(mf, port)
        assert obs.shape == (396,)

    def test_cash_normalization(self):
        gen = self._make_generator_stub()
        mf = np.zeros((4, 16), dtype=np.float32)
        port = PortfolioSnapshot(cash_usd=5000.0)
        obs = gen._build_obs(mf, port)
        base = 4 * 16
        assert abs(obs[base + 0] - 0.5) < 1e-6

    def test_hold_hours_uses_max_steps(self):
        gen = self._make_generator_stub(max_steps=168)
        mf = np.zeros((4, 16), dtype=np.float32)
        port = PortfolioSnapshot(cash_usd=10000.0, hold_hours=84)
        obs = gen._build_obs(mf, port)
        base = 4 * 16
        assert abs(obs[base + 3] - 0.5) < 1e-6  # 84/168 = 0.5

    def test_episode_progress_increments(self):
        gen = self._make_generator_stub(max_steps=100)
        mf = np.zeros((4, 16), dtype=np.float32)
        port = PortfolioSnapshot(cash_usd=10000.0)
        base = 4 * 16

        gen._episode_step = 0
        obs0 = gen._build_obs(mf, port)
        assert abs(obs0[base + 4] - 0.0) < 1e-6

        gen._episode_step = 50
        obs50 = gen._build_obs(mf, port)
        assert abs(obs50[base + 4] - 0.5) < 1e-6

    def test_position_onehot_long(self):
        gen = self._make_generator_stub()
        mf = np.zeros((4, 16), dtype=np.float32)
        port = PortfolioSnapshot(
            cash_usd=5000.0,
            position_symbol="SYM2",
            position_value_usd=5000.0,
            is_short=False,
        )
        obs = gen._build_obs(mf, port)
        base = 4 * 16
        assert obs[base + 5 + 2] == 1.0

    def test_position_onehot_short(self):
        gen = self._make_generator_stub()
        mf = np.zeros((4, 16), dtype=np.float32)
        port = PortfolioSnapshot(
            cash_usd=15000.0,
            position_symbol="SYM1",
            position_value_usd=5000.0,
            is_short=True,
        )
        obs = gen._build_obs(mf, port)
        base = 4 * 16
        assert obs[base + 5 + 1] == -1.0
        assert obs[base + 1] < 0  # pos_val negative for short

    def test_reset_episode(self):
        gen = self._make_generator_stub()
        gen._episode_step = 42
        gen.reset_episode()
        assert gen._episode_step == 0


class TestComputeSymbolFeatures:
    def test_output_shape(self):
        import pandas as pd
        idx = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.uniform(90, 110, 100),
            "high": np.random.uniform(100, 120, 100),
            "low": np.random.uniform(80, 100, 100),
            "close": np.random.uniform(90, 110, 100),
        }, index=idx)
        feat = compute_symbol_features(df, pd.DataFrame(), pd.DataFrame())
        assert feat.shape == (100, 16)
        assert feat.dtype == np.float32
        assert np.all(np.isfinite(feat))
