from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, P_HIGH, P_LOW
from scripts.search_binance33_linear_rules import LinearRule
from scripts.sweep_binance33_hedged_cvar_pack import FEATURES, HedgedConfig, _build_cache, _target_weights


def _tiny_data() -> MktdData:
    features = np.zeros((6, 4, 16), dtype=np.float32)
    # return_5d feature drives the linear score in this test.
    features[:, :, 1] = np.asarray([2.0, 1.5, -2.0, -1.5], dtype=np.float32)
    features[:, :, 2] = 0.1
    features[:, :, 4] = 0.02
    prices = np.ones((6, 4, 5), dtype=np.float32) * 100.0
    prices[:, :, P_CLOSE] = 100.0
    prices[:, :, P_LOW] = 99.0
    prices[:, :, P_HIGH] = 101.0
    return MktdData(
        version=2,
        symbols=["L1USD", "L2USD", "S1USD", "S2USD"],
        features=features,
        prices=prices,
        tradable=np.ones((6, 4), dtype=np.uint8),
    )


def _rule() -> LinearRule:
    coeffs = np.zeros(10, dtype=np.float64)
    coeffs[1] = 1.0
    return LinearRule(
        name="test",
        coeffs=coeffs,
        rebalance_days=1,
        min_abs_score=0.0,
        btc_min_return_20d=-99.0,
    )


def test_hedged_target_selects_longs_and_shorts() -> None:
    data = _tiny_data()
    rule = _rule()
    cache = _build_cache(data, rule, exclude_symbols=set())
    cfg = HedgedConfig(
        n_longs=1,
        n_shorts=1,
        long_signal="rule",
        allocation_mode="equal",
        max_gross=2.0,
        max_weight=1.0,
        long_fraction=0.5,
        max_net=0.0,
        lookback_days=3,
        cvar_alpha=0.9,
        edge_power=1.0,
        risk_power=1.0,
        min_abs_score=0.0,
        rebalance_days=1,
        rebalance_style="full",
    )

    weights = _target_weights(data, cache, rule, None, cfg, t=3)

    assert weights[0] > 0.0
    assert weights[2] < 0.0
    assert np.isclose(float(np.abs(weights).sum()), 2.0)
    assert abs(float(weights.sum())) < 1e-12


def test_hedged_target_respects_net_cap() -> None:
    data = _tiny_data()
    rule = _rule()
    cache = _build_cache(data, rule, exclude_symbols=set())
    cfg = HedgedConfig(
        n_longs=1,
        n_shorts=1,
        long_signal="rule",
        allocation_mode="equal",
        max_gross=2.0,
        max_weight=1.5,
        long_fraction=0.75,
        max_net=0.25,
        lookback_days=3,
        cvar_alpha=0.9,
        edge_power=1.0,
        risk_power=1.0,
        min_abs_score=0.0,
        rebalance_days=1,
        rebalance_style="full",
    )

    weights = _target_weights(data, cache, rule, None, cfg, t=3)

    assert abs(float(weights.sum())) <= 0.25 + 1e-12


def test_hedged_target_respects_zero_net_cap() -> None:
    data = _tiny_data()
    rule = _rule()
    cache = _build_cache(data, rule, exclude_symbols=set())
    cfg = HedgedConfig(
        n_longs=1,
        n_shorts=1,
        long_signal="rule",
        allocation_mode="equal",
        max_gross=2.0,
        max_weight=1.5,
        long_fraction=0.75,
        max_net=0.0,
        lookback_days=3,
        cvar_alpha=0.9,
        edge_power=1.0,
        risk_power=1.0,
        min_abs_score=0.0,
        rebalance_days=1,
        rebalance_style="full",
    )

    weights = _target_weights(data, cache, rule, None, cfg, t=3)

    assert abs(float(weights.sum())) <= 1e-12


def test_hedged_target_can_use_separate_long_signal() -> None:
    data = _tiny_data()
    data.features[:, :, FEATURES["trend_60d"]] = np.asarray([0.1, 2.0, -1.0, -2.0], dtype=np.float32)
    rule = _rule()
    cache = _build_cache(data, rule, exclude_symbols=set())
    cfg = HedgedConfig(
        n_longs=1,
        n_shorts=1,
        long_signal="trend60",
        allocation_mode="equal",
        max_gross=2.0,
        max_weight=1.0,
        long_fraction=0.5,
        max_net=0.0,
        lookback_days=3,
        cvar_alpha=0.9,
        edge_power=1.0,
        risk_power=1.0,
        min_abs_score=0.0,
        rebalance_days=1,
        rebalance_style="full",
    )

    weights = _target_weights(data, cache, rule, None, cfg, t=3)

    assert weights[1] > 0.0
    assert weights[2] < 0.0
