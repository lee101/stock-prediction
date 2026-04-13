"""Tests for chronos2_linear_calibration.py — CalibrationParams and fit_calibration."""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos2_linear_calibration import (
    CalibrationParams,
    compute_sharpe,
    fit_calibration,
)


# ---------------------------------------------------------------------------
# CalibrationParams.apply
# ---------------------------------------------------------------------------

class TestCalibrationParamsApply:
    def test_buy_signal(self):
        p = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001, allow_short=False)
        assert p.apply(0.002) == "buy"   # strong positive signal

    def test_hold_signal(self):
        p = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001, allow_short=False)
        assert p.apply(0.0005) == "hold"  # below buy, above -sell

    def test_exit_signal(self):
        p = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001, allow_short=False)
        assert p.apply(-0.002) == "exit"  # below -sell_threshold

    def test_short_signal_with_allow_short(self):
        p = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001, allow_short=True)
        assert p.apply(-0.002) == "sell"

    def test_no_short_without_allow_short(self):
        p = CalibrationParams(buy_threshold=0.001, sell_threshold=0.001, allow_short=False)
        result = p.apply(-0.002)
        assert result in ("exit", "hold")
        assert result != "sell"

    def test_signal_weight_scales(self):
        """signal_weight=2.0 means we need 2x predicted_return to trigger buy."""
        p = CalibrationParams(signal_weight=2.0, buy_threshold=0.001, sell_threshold=0.001)
        # predicted_return = 0.0006: signal = 0.0012 > 0.001 → buy
        assert p.apply(0.0006) == "buy"
        # predicted_return = 0.0004: signal = 0.0008 < 0.001 → hold
        assert p.apply(0.0004) == "hold"

    def test_negative_buy_threshold_is_very_aggressive(self):
        """Negative buy_threshold means almost always buy."""
        p = CalibrationParams(buy_threshold=-0.0008, sell_threshold=-0.0008, allow_short=False)
        assert p.apply(0.0) == "buy"      # neutral signal still buys
        assert p.apply(-0.0007) == "buy"  # slight down still buys
        # exit when signal < -sell_threshold = +8bps
        # -0.001 < +0.0008 → exit
        assert p.apply(-0.001) == "exit"


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    def test_pure_buy_positive_market(self):
        """If every trade wins, Sharpe should be high."""
        N = 500
        signals = np.full(N, 0.002)   # all above buy_thresh=0.001
        actual  = np.full(N, 0.003)   # all gain 30bps
        sharpe  = compute_sharpe(signals, actual, buy_thresh=0.001, sell_thresh=0.001,
                                  allow_short=False, fee_bps=10.0)
        assert sharpe > 1.0

    def test_no_trades_returns_penalty(self):
        """If buy_thresh is very high, no trades → penalty."""
        N = 200
        signals = np.zeros(N)
        actual  = np.zeros(N)
        sharpe  = compute_sharpe(signals, actual, buy_thresh=1.0, sell_thresh=1.0,
                                  allow_short=False, fee_bps=10.0)
        assert sharpe == -999.0

    def test_fee_hurts_break_even_trades(self):
        """Exactly break-even returns minus fee should produce negative Sharpe."""
        N = 500
        signals = np.full(N, 0.002)
        actual  = np.full(N, 0.0)   # 0 return, but pay 10bps fee each trade
        sharpe  = compute_sharpe(signals, actual, buy_thresh=0.001, sell_thresh=0.001,
                                  allow_short=False, fee_bps=10.0)
        assert sharpe < 0.0

    def test_shorting_helps_on_pure_down_market(self):
        """When allow_short=True and market always falls, short should dominate."""
        N = 500
        signals = np.full(N, -0.002)   # all below -sell_thresh
        actual  = np.full(N, -0.003)   # market always falls 30bps
        # Long only: no buys triggered → penalty
        sharpe_long = compute_sharpe(signals, actual, buy_thresh=0.001, sell_thresh=0.001,
                                      allow_short=False, fee_bps=1.0)
        # Short allowed: shorts triggered → profits from decline
        sharpe_short = compute_sharpe(signals, actual, buy_thresh=0.001, sell_thresh=0.001,
                                       allow_short=True, fee_bps=1.0)
        assert sharpe_short > sharpe_long


# ---------------------------------------------------------------------------
# fit_calibration
# ---------------------------------------------------------------------------

def _make_synthetic_data(N: int = 1000, upward_bias: float = 0.001, seed: int = 42):
    """Generate synthetic data where q50 is a noisy predictor of actual close."""
    rng = np.random.RandomState(seed)
    prev_close = np.full(N, 100.0)
    noise      = rng.randn(N) * 0.005
    # q50 predicts actual with some noise, modest upward bias
    q50    = prev_close * (1 + upward_bias + noise * 0.5)
    actual = prev_close * (1 + upward_bias + noise)
    q10    = q50 * 0.995
    q90    = q50 * 1.005
    return q10, q50, q90, actual, prev_close


class TestFitCalibration:
    def test_returns_calibration_params(self):
        q10, q50, q90, actual, prev = _make_synthetic_data()
        params = fit_calibration(q10, q50, q90, actual, prev, max_shift_bps=8.0)
        assert isinstance(params, CalibrationParams)
        assert params.n_cal_windows == len(actual)

    def test_sharpe_non_trivial(self):
        """Calibrated strategy should have positive Sharpe for a biased predictor."""
        q10, q50, q90, actual, prev = _make_synthetic_data(upward_bias=0.002)
        params = fit_calibration(q10, q50, q90, actual, prev, max_shift_bps=8.0)
        assert params.cal_sharpe > 0.0

    def test_thresholds_within_search_range(self):
        """buy_threshold and sell_threshold should stay within ±max_shift_bps."""
        q10, q50, q90, actual, prev = _make_synthetic_data()
        max_bps = 8.0
        max_frac = max_bps / 10_000.0
        params = fit_calibration(q10, q50, q90, actual, prev, max_shift_bps=max_bps,
                                  search_signal_weight=False)
        assert params.buy_threshold >= -max_frac - 1e-9
        assert params.buy_threshold <=  max_frac + 1e-9
        assert params.sell_threshold >= -max_frac - 1e-9
        assert params.sell_threshold <=  max_frac + 1e-9

    def test_signal_weight_search_finds_plausible_weight(self):
        """Signal weight should be one of the pre-defined candidates."""
        q10, q50, q90, actual, prev = _make_synthetic_data(N=2000)
        params = fit_calibration(q10, q50, q90, actual, prev,
                                  search_signal_weight=True)
        # weight must come from [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        expected_weights = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
        assert params.signal_weight in expected_weights

    def test_no_signal_weight_search_fixed_at_one(self):
        q10, q50, q90, actual, prev = _make_synthetic_data()
        params = fit_calibration(q10, q50, q90, actual, prev,
                                  search_signal_weight=False)
        assert params.signal_weight == 1.0

    def test_short_allowed_constraint_enforced(self):
        """When allow_short=True, sell_threshold must be >= buy_threshold - min_gap."""
        q10, q50, q90, actual, prev = _make_synthetic_data(N=2000, seed=7)
        min_gap_bps = 4.0
        params = fit_calibration(q10, q50, q90, actual, prev,
                                  allow_short=True, min_gap_bps=min_gap_bps,
                                  search_signal_weight=False)
        min_gap = min_gap_bps / 10_000.0
        assert params.sell_threshold >= params.buy_threshold - min_gap - 1e-9

    def test_improves_on_biased_predictor_with_weight_search(self):
        """Searching signal_weight should not degrade vs fixed weight for reasonable data."""
        q10, q50, q90, actual, prev = _make_synthetic_data(N=3000, upward_bias=0.001)
        params_fixed  = fit_calibration(q10, q50, q90, actual, prev,
                                         search_signal_weight=False, max_shift_bps=8.0)
        params_search = fit_calibration(q10, q50, q90, actual, prev,
                                         search_signal_weight=True,  max_shift_bps=8.0)
        # Weight search should find ≥ fixed (or equal) in-sample Sharpe
        assert params_search.cal_sharpe >= params_fixed.cal_sharpe - 0.01  # tiny tolerance

    def test_model_id_stored(self):
        q10, q50, q90, actual, prev = _make_synthetic_data()
        mid = "test/model/path"
        params = fit_calibration(q10, q50, q90, actual, prev, model_id=mid)
        assert params.model_id == mid

    def test_from_dict_roundtrip(self):
        p = CalibrationParams(signal_weight=1.5, buy_threshold=0.0005,
                              sell_threshold=0.0003, allow_short=True,
                              model_id="foo", n_cal_windows=100, cal_sharpe=0.5)
        d = p.to_dict()
        p2 = CalibrationParams.from_dict(d)
        assert p2.signal_weight == p.signal_weight
        assert p2.buy_threshold == p.buy_threshold
        assert p2.model_id == p.model_id
