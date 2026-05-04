from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData
from scripts.search_binance33_linear_rules import FEATURE_NAMES, LinearRule, _make_policy
from scripts.sweep_binance33_rules import FEATURES


def test_linear_short_policy_selects_highest_recent_winner() -> None:
    features = np.zeros((3, 2, 16), dtype=np.float32)
    prices = np.ones((3, 2, 5), dtype=np.float32) * 100.0
    tradable = np.ones((3, 2), dtype=np.uint8)
    features[:, 0, FEATURES["return_5d"]] = 0.05
    features[:, 1, FEATURES["return_5d"]] = 0.25

    coeffs = np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    coeffs[FEATURE_NAMES.index("return_5d")] = -1.0
    rule = LinearRule(
        name="test",
        coeffs=coeffs,
        rebalance_days=1,
        min_abs_score=0.0,
        btc_min_return_20d=-99.0,
    )
    data = MktdData(version=2, symbols=["AAA", "BBB"], features=features, prices=prices, tradable=tradable)

    policy = _make_policy(data, rule, decision_lag=0)

    assert policy(np.zeros((1,), dtype=np.float32)) == 1 + data.num_symbols + 1
