"""Tests for chronos2_calibrated_backtest.py — run_backtest statistics."""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos2_linear_calibration import CalibrationParams
from chronos2_calibrated_backtest import run_backtest


def _make_preds(
    N: int = 200,
    upward_bias: float = 0.001,
    noise: float = 0.005,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    prev = np.full(N, 100.0)
    noise_arr = rng.randn(N) * noise
    q50  = prev * (1 + upward_bias + noise_arr * 0.5)
    actual = prev * (1 + upward_bias + noise_arr)
    q10  = q50 * 0.995
    q90  = q50 * 1.005
    syms = [f"SYM{i % 5}" for i in range(N)]
    return q10, q50, q90, actual, prev, syms


class TestRunBacktest:
    def test_returns_dict_with_expected_keys(self):
        q10, q50, q90, actual, prev, syms = _make_preds()
        params = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001)
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params)
        for key in ("n_windows", "n_trades", "sharpe_annualized", "hold_rate",
                    "win_rate", "bnh_sharpe_annualized", "pnl_30d_est_pct"):
            assert key in stats, f"Missing key: {key}"

    def test_sharpe_positive_for_good_predictor(self):
        """Model with consistent upward bias → positive Sharpe."""
        q10, q50, q90, actual, prev, syms = _make_preds(N=500, upward_bias=0.003)
        params = CalibrationParams(
            buy_threshold=-0.001, sell_threshold=-0.001, signal_weight=1.0
        )
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params, fee_bps=1.0)
        assert stats["sharpe_annualized"] > 0.0

    def test_n_windows_matches_input(self):
        q10, q50, q90, actual, prev, syms = _make_preds(N=123)
        params = CalibrationParams()
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params)
        assert stats["n_windows"] == 123

    def test_hold_rate_plus_long_rate_plus_short_rate_equals_one(self):
        q10, q50, q90, actual, prev, syms = _make_preds(N=300)
        params = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001)
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params)
        total = stats["hold_rate"] + stats["long_rate"] + stats["short_rate"]
        assert abs(total - 1.0) < 1e-6

    def test_per_symbol_breakdown_present_when_symbols_provided(self):
        q10, q50, q90, actual, prev, syms = _make_preds(N=200)
        params = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001)
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params)
        assert "per_symbol" in stats
        assert len(stats["per_symbol"]) > 0

    def test_confidence_filter_reduces_trades(self):
        """Wide confidence threshold → fewer trades."""
        q10, q50, q90, actual, prev, syms = _make_preds(N=300)
        params_no_filter = CalibrationParams(
            buy_threshold=-0.002, sell_threshold=-0.002, confidence_threshold=0.0
        )
        params_filtered = CalibrationParams(
            buy_threshold=-0.002, sell_threshold=-0.002,
            confidence_threshold=0.001  # very tight → filter most trades
        )
        stats_no = run_backtest(q10, q50, q90, actual, prev, syms, params_no_filter)
        stats_fi = run_backtest(q10, q50, q90, actual, prev, syms, params_filtered)
        assert stats_fi["n_trades"] <= stats_no["n_trades"]

    def test_calibration_stored_in_stats(self):
        q10, q50, q90, actual, prev, syms = _make_preds()
        params = CalibrationParams(signal_weight=2.0, buy_threshold=0.0005)
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params)
        assert stats["calibration"]["signal_weight"] == 2.0
        assert stats["calibration"]["buy_threshold"] == 0.0005

    def test_no_trades_when_threshold_very_high(self):
        q10, q50, q90, actual, prev, syms = _make_preds(N=200)
        params = CalibrationParams(buy_threshold=10.0, sell_threshold=10.0)
        stats = run_backtest(q10, q50, q90, actual, prev, syms, params)
        assert stats["n_trades"] == 0
        assert stats["hold_rate"] == pytest.approx(1.0)
