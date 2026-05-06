from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData
from scripts.search_binance33_regime_gates import GateSpec, _gate_allows
from scripts.sweep_binance33_rules import FEATURES


def test_gate_allows_supports_conjunction() -> None:
    features = np.zeros((2, 1, 16), dtype=np.float32)
    features[:, 0, FEATURES["return_20d"]] = 0.04
    prices = np.ones((2, 1, 5), dtype=np.float32)
    data = MktdData(version=2, symbols=["BTCUSD"], features=features, prices=prices, tradable=None)
    left = GateSpec(
        name="btc_ret20",
        fn=lambda frame, t: float(frame.features[t, 0, FEATURES["return_20d"]]),
        op="gt",
        threshold=0.0,
    )
    right = GateSpec(
        name="btc_ret20",
        fn=lambda frame, t: float(frame.features[t, 0, FEATURES["return_20d"]]),
        op="lt",
        threshold=0.1,
    )

    assert _gate_allows((left, right), data, 1) is True
