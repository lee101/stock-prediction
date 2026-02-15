from __future__ import annotations

import numpy as np
import pytest

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, simulate_daily_policy


def test_simulate_daily_policy_tracks_max_drawdown_over_time() -> None:
    # Price goes up, then down, then up: max drawdown should be from 110 -> 90 = 18.1818%.
    close = np.asarray([100.0, 110.0, 90.0, 120.0], dtype=np.float32)
    T = close.shape[0]

    features = np.zeros((T, 1, 16), dtype=np.float32)
    prices = np.zeros((T, 1, 5), dtype=np.float32)
    for t, c in enumerate(close):
        prices[t, 0, :] = np.asarray([c, c, c, c, 0.0], dtype=np.float32)
    tradable = np.ones((T, 1), dtype=np.uint8)

    data = MktdData(version=2, symbols=["AAA"], features=features, prices=prices, tradable=tradable)

    def always_long(_obs: np.ndarray) -> int:
        return 1  # long symbol 0

    result = simulate_daily_policy(
        data,
        always_long,
        max_steps=3,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        initial_cash=10_000.0,
    )

    assert result.total_return == 0.2
    assert result.max_drawdown == pytest.approx((110.0 - 90.0) / 110.0, rel=1e-6, abs=1e-6)
