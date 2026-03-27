from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.hourly_replay import (
    HourlyMarket,
    InitialPositionSpec,
    MktdData,
    replay_hourly_frozen_daily_actions,
    simulate_daily_policy,
    simulate_hourly_policy,
)


def _build_daily_data() -> MktdData:
    features = np.zeros((3, 1, 16), dtype=np.float32)
    prices = np.zeros((3, 1, 5), dtype=np.float32)
    prices[:, 0, :] = np.asarray(
        [
            [100.0, 100.0, 100.0, 100.0, 0.0],
            [110.0, 110.0, 110.0, 110.0, 0.0],
            [121.0, 121.0, 121.0, 121.0, 0.0],
        ],
        dtype=np.float32,
    )
    return MktdData(
        version=2,
        symbols=["AAA"],
        features=features,
        prices=prices,
        tradable=np.ones((3, 1), dtype=np.uint8),
    )


def _build_hourly_market() -> HourlyMarket:
    market_index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T23:00:00Z", freq="h", tz="UTC")
    close = np.full((len(market_index),), 100.0, dtype=np.float64)
    close[-1] = 121.0
    return HourlyMarket(
        index=market_index,
        close={"AAA": close},
        tradable={"AAA": np.ones((len(market_index),), dtype=bool)},
    )


def test_simulate_daily_policy_supports_initial_long_position() -> None:
    data = _build_daily_data()
    result = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.25),
    )

    assert result.total_return != 0.0
    assert result.num_trades >= 1


def test_replay_hourly_frozen_daily_actions_supports_initial_short_position() -> None:
    data = _build_daily_data()
    market = _build_hourly_market()
    result = replay_hourly_frozen_daily_actions(
        data=data,
        actions=np.asarray([0, 0], dtype=np.int32),
        market=market,
        start_date="2026-01-01",
        end_date="2026-01-03",
        max_steps=2,
        initial_position=InitialPositionSpec(symbol="AAA", side="short", allocation_pct=0.25),
    )

    assert result.num_trades >= 1
    assert result.num_orders >= 1


def test_simulate_hourly_policy_supports_initial_long_position() -> None:
    data = _build_daily_data()
    market = _build_hourly_market()
    result = simulate_hourly_policy(
        data=data,
        policy_fn=lambda obs: 0,
        market=market,
        start_date="2026-01-01",
        end_date="2026-01-03",
        max_steps_days=2,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.25),
    )

    assert result.num_trades >= 1
    assert result.total_return != 0.0


def test_initial_position_rejects_unknown_symbol() -> None:
    data = _build_daily_data()
    with pytest.raises(ValueError, match="not found in symbols"):
        simulate_daily_policy(
            data,
            lambda obs: 0,
            max_steps=2,
            initial_position=InitialPositionSpec(symbol="BBB", side="long", allocation_pct=0.25),
        )
