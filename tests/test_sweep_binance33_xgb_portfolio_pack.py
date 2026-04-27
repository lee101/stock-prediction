from __future__ import annotations

import numpy as np
import pytest

from pufferlib_market.hourly_replay import MktdData, P_CLOSE
from scripts.sweep_binance33_xgb import Experiment
from scripts.sweep_binance33_xgb_portfolio_pack import (
    PackConfig,
    _simulate_pack_window,
    _target_weights,
)


def _fake_data() -> MktdData:
    features = np.zeros((6, 4, 16), dtype=np.float32)
    features[:, :, 4] = np.array([0.10, 0.20, 0.40, 0.80], dtype=np.float32)
    prices = np.ones((6, 4, 5), dtype=np.float32) * 100.0
    prices[:, :, P_CLOSE] = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [101.0, 102.0, 98.0, 97.0],
            [102.0, 103.0, 97.0, 96.0],
            [103.0, 104.0, 96.0, 95.0],
            [104.0, 105.0, 95.0, 94.0],
            [105.0, 106.0, 94.0, 93.0],
        ],
        dtype=np.float32,
    )
    return MktdData(
        version=2,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD"],
        features=features,
        prices=prices,
        tradable=np.ones((6, 4), dtype=np.uint8),
    )


def _short_exp() -> Experiment:
    return Experiment(
        name="short",
        horizon=5,
        label="raw",
        mode="short_bottom",
        rebalance_days=1,
        max_depth=2,
        eta=0.1,
        subsample=1.0,
        colsample=1.0,
        min_child_weight=1.0,
        reg_lambda=1.0,
        min_abs_score=0.0,
        btc_gate=-99.0,
    )


def test_target_weights_pack_top_n_shorts_with_gross_cap() -> None:
    data = _fake_data()
    scores = np.array([[0.01, -0.03, -0.05, -0.02]], dtype=np.float64).repeat(6, axis=0)
    cfg = PackConfig(
        top_n=2,
        allocation_mode="equal",
        min_abs_score=0.0,
        score_temp=0.01,
        target_vol=0.0,
        max_weight=0.5,
        max_gross=1.0,
    )

    weights = _target_weights(data, scores, _short_exp(), cfg, t=0)

    assert weights.tolist() == pytest.approx([0.0, -0.5, -0.5, 0.0])
    assert np.abs(weights).sum() == pytest.approx(1.0)


def test_target_weights_inverse_vol_prefers_lower_volatility_short() -> None:
    data = _fake_data()
    scores = np.array([[0.01, -0.03, -0.05, -0.02]], dtype=np.float64).repeat(6, axis=0)
    cfg = PackConfig(
        top_n=2,
        allocation_mode="inv_vol_score",
        min_abs_score=0.0,
        score_temp=0.01,
        target_vol=0.0,
        max_weight=1.0,
        max_gross=1.0,
    )

    weights = _target_weights(data, scores, _short_exp(), cfg, t=0)

    assert weights[1] < weights[2]
    assert np.abs(weights).sum() == pytest.approx(1.0)


def test_simulate_pack_window_charges_turnover_and_closes_final_weights() -> None:
    data = _fake_data()
    scores = np.array([[0.01, -0.03, -0.05, -0.02]], dtype=np.float64).repeat(6, axis=0)
    cfg = PackConfig(
        top_n=2,
        allocation_mode="equal",
        min_abs_score=0.0,
        score_temp=0.01,
        target_vol=0.0,
        max_weight=0.5,
        max_gross=1.0,
    )

    no_cost = _simulate_pack_window(
        data,
        scores,
        _short_exp(),
        cfg,
        eval_days=4,
        fee_rate=0.0,
        slippage_bps=0.0,
        fill_buffer_bps=0.0,
        decision_lag=0,
    )
    with_cost = _simulate_pack_window(
        data,
        scores,
        _short_exp(),
        cfg,
        eval_days=4,
        fee_rate=0.001,
        slippage_bps=20.0,
        fill_buffer_bps=5.0,
        decision_lag=0,
    )

    assert no_cost["total_return"] > 0.0
    assert with_cost["total_return"] < no_cost["total_return"]
    assert with_cost["trades"] >= 4
