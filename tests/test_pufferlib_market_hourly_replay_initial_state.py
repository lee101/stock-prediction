from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.hourly_replay import (
    HourlyMarket,
    HourlyReplayResult,
    InitialPositionSpec,
    MktdData,
    load_hourly_market,
    replay_hourly_frozen_daily_actions,
    simulate_daily_policy,
    simulate_hourly_policy,
)


def _make_daily_data() -> MktdData:
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


def _make_hourly_market() -> HourlyMarket:
    index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T23:00:00Z", freq="h", tz="UTC")
    close = np.full((len(index),), 100.0, dtype=np.float64)
    return HourlyMarket(
        index=index,
        close={"AAA": close},
        tradable={"AAA": np.ones((len(index),), dtype=bool)},
    )


def test_simulate_daily_policy_scales_obs_to_initial_cash() -> None:
    data = _make_daily_data()
    observed: list[np.ndarray] = []

    def policy_fn(obs: np.ndarray) -> int:
        observed.append(obs.copy())
        return 0

    result = simulate_daily_policy(
        data,
        policy_fn,
        max_steps=1,
        initial_cash=2_000.0,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )

    base = data.num_symbols * data.features.shape[2]
    assert observed
    assert observed[0][base + 0] == pytest.approx(1.0)
    assert result.total_return == pytest.approx(0.0)


def test_simulate_hourly_policy_scales_obs_to_initial_cash() -> None:
    data = _make_daily_data()
    market = _make_hourly_market()
    observed: list[np.ndarray] = []

    def policy_fn(obs: np.ndarray) -> int:
        if not observed:
            observed.append(obs.copy())
        return 0

    simulate_hourly_policy(
        data=data,
        policy_fn=policy_fn,
        market=market,
        start_date="2026-01-01",
        end_date="2026-01-03",
        max_steps_days=1,
        initial_cash=2_000.0,
        fee_rate=0.0,
    )

    base = data.num_symbols * data.features.shape[2]
    assert observed
    assert observed[0][base + 0] == pytest.approx(1.0)


def test_simulate_daily_policy_initial_long_position_carries_pnl() -> None:
    data = _make_daily_data()

    result = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        initial_cash=10_000.0,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.5),
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )

    assert result.total_return > 0.0
    assert result.num_trades == 1


def test_replay_hourly_initial_position_uses_first_positive_tradable_price() -> None:
    data = _make_daily_data()
    index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T23:00:00Z", freq="h", tz="UTC")
    close = np.zeros((len(index),), dtype=np.float64)
    tradable = np.zeros((len(index),), dtype=bool)
    close[10:] = 100.0
    tradable[10:] = True
    market = HourlyMarket(
        index=index,
        close={"AAA": close},
        tradable={"AAA": tradable},
    )

    result = replay_hourly_frozen_daily_actions(
        data=data,
        actions=np.asarray([0], dtype=np.int32),
        market=market,
        start_date="2026-01-01",
        end_date="2026-01-03",
        max_steps=1,
        fee_rate=0.0,
        initial_cash=10_000.0,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.5),
    )

    assert isinstance(result, HourlyReplayResult)
    assert np.isfinite(result.total_return)



def test_replay_hourly_initial_position_falls_back_to_daily_price_when_hourly_mark_missing() -> None:
    data = _make_daily_data()
    index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T23:00:00Z", freq="h", tz="UTC")
    market = HourlyMarket(
        index=index,
        close={"AAA": np.zeros((len(index),), dtype=np.float64)},
        tradable={"AAA": np.zeros((len(index),), dtype=bool)},
    )

    result = replay_hourly_frozen_daily_actions(
        data=data,
        actions=np.asarray([0], dtype=np.int32),
        market=market,
        start_date="2026-01-01",
        end_date="2026-01-03",
        max_steps=1,
        fee_rate=0.0,
        initial_cash=10_000.0,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.5),
    )

    assert isinstance(result, HourlyReplayResult)
    assert result.total_return == pytest.approx(0.0)
    assert result.num_orders == 1
    assert result.num_trades == 1



def test_simulate_hourly_policy_missing_hourly_bars_uses_daily_fallback_for_price_and_tradability() -> None:
    data = _make_daily_data()
    index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T23:00:00Z", freq="h", tz="UTC")
    market = HourlyMarket(
        index=index,
        close={"AAA": np.zeros((len(index),), dtype=np.float64)},
        tradable={"AAA": np.zeros((len(index),), dtype=bool)},
    )

    result = simulate_hourly_policy(
        data=data,
        policy_fn=lambda obs: 0,
        market=market,
        start_date="2026-01-01",
        end_date="2026-01-03",
        max_steps_days=1,
        fee_rate=0.0,
        initial_cash=10_000.0,
        initial_position=InitialPositionSpec(symbol="AAA", side="long", allocation_pct=0.5),
    )

    assert result.total_return == pytest.approx(0.0)
    assert result.num_orders == 1
    assert result.num_trades == 1



def test_load_hourly_market_aligns_half_hour_bars_to_hour_grid(tmp_path: Path) -> None:
    root = tmp_path / "stocks"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T13:30:00Z",
                "2026-01-01T14:30:00Z",
            ],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000.0, 1_200.0],
        }
    ).to_csv(root / "AAA.csv", index=False)

    market = load_hourly_market(
        ["AAA"],
        tmp_path,
        start="2026-01-01T13:00:00Z",
        end="2026-01-01T15:00:00Z",
    )

    tradable_idx = np.flatnonzero(market.tradable["AAA"])
    assert tradable_idx.tolist() == [0, 1]
    assert market.index[tradable_idx[0]] == pd.Timestamp("2026-01-01T13:00:00Z")
    assert market.index[tradable_idx[1]] == pd.Timestamp("2026-01-01T14:00:00Z")
    assert market.close["AAA"][tradable_idx[0]] == pytest.approx(100.5)
    assert market.close["AAA"][tradable_idx[1]] == pytest.approx(101.5)
