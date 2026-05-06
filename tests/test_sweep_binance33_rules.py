from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData
from scripts.sweep_binance33_rules import FEATURES, RuleConfig, ScoreSpec, _make_policy


def test_short_bottom_policy_ignores_untradable_lowest_score_symbol() -> None:
    features = np.zeros((3, 2, 16), dtype=np.float32)
    prices = np.ones((3, 2, 5), dtype=np.float32)
    prices[:, :, :4] = 100.0
    tradable = np.ones((3, 2), dtype=np.uint8)
    tradable[:, 0] = 0
    features[:, 0, FEATURES["return_5d"]] = -0.50
    features[:, 1, FEATURES["return_5d"]] = -0.10

    data = MktdData(version=2, symbols=["BAD", "GOOD"], features=features, prices=prices, tradable=tradable)
    config = RuleConfig(
        score=ScoreSpec("mom5", "return_5d"),
        mode="short_bottom",
        rebalance_days=1,
        min_abs_score=0.0,
        btc_min_return_20d=-99.0,
    )

    policy = _make_policy(data, config, decision_lag=0)

    assert policy(np.zeros((1,), dtype=np.float32)) == 1 + data.num_symbols + 1
