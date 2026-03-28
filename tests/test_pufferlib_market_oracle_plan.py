from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData
from pufferlib_market.oracle_plan import (
    OraclePortfolioState,
    build_legacy_oracle_trace,
    compare_policy_actions_to_oracle,
    legacy_action_label,
    rollout_policy_actions,
    score_all_legacy_actions,
)


def _make_data(*, closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> MktdData:
    highs = highs or closes
    lows = lows or closes
    prices = np.zeros((len(closes), 1, 5), dtype=np.float32)
    for idx, close in enumerate(closes):
        prices[idx, 0, 0] = close
        prices[idx, 0, 1] = highs[idx]
        prices[idx, 0, 2] = lows[idx]
        prices[idx, 0, 3] = close
        prices[idx, 0, 4] = 1.0
    features = np.zeros((len(closes), 1, 16), dtype=np.float32)
    tradable = np.ones((len(closes), 1), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=["BTCUSD"],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def test_legacy_action_label_formats_expected_names() -> None:
    assert legacy_action_label(0, ["BTCUSD", "ETHUSD"]) == "flat"
    assert legacy_action_label(1, ["BTCUSD", "ETHUSD"]) == "long:BTCUSD"
    assert legacy_action_label(4, ["BTCUSD", "ETHUSD"]) == "short:ETHUSD"


def test_score_all_legacy_actions_prefers_long_when_next_close_rises() -> None:
    data = _make_data(closes=[100.0, 110.0, 115.0])

    scores = score_all_legacy_actions(
        data,
        start_step=0,
        state=OraclePortfolioState(),
        lookahead_steps=1,
        fee_rate=0.0,
        fill_buffer_bps_values=[0.0],
        slippage_bps_values=[0.0],
    )

    assert scores[0].action_label == "long:BTCUSD"
    assert scores[-1].action_label == "short:BTCUSD"


def test_score_all_legacy_actions_prefers_short_when_next_close_falls() -> None:
    data = _make_data(closes=[100.0, 90.0, 85.0])

    scores = score_all_legacy_actions(
        data,
        start_step=0,
        state=OraclePortfolioState(),
        lookahead_steps=1,
        fee_rate=0.0,
        fill_buffer_bps_values=[0.0],
        slippage_bps_values=[0.0],
    )

    assert scores[0].action_label == "short:BTCUSD"
    assert scores[-1].action_label == "long:BTCUSD"


def test_build_legacy_oracle_trace_marks_near_best_actions() -> None:
    data = _make_data(closes=[100.0, 100.0, 100.0])

    steps = build_legacy_oracle_trace(
        data,
        max_steps=2,
        lookahead_steps=1,
        fee_rate=0.001,
        fill_buffer_bps_values=[0.0],
        slippage_bps_values=[0.0],
        near_best_score_gap=0.0,
    )

    assert len(steps) == 2
    assert steps[0].best_action_label == "flat"
    assert steps[0].near_best_action_labels == ("flat",)


def test_rollout_policy_actions_and_oracle_comparison_match() -> None:
    data = _make_data(closes=[100.0, 110.0, 121.0, 130.0])
    oracle_steps = build_legacy_oracle_trace(
        data,
        max_steps=3,
        lookahead_steps=1,
        fee_rate=0.0,
        fill_buffer_bps_values=[0.0],
        slippage_bps_values=[0.0],
        near_best_score_gap=0.0,
    )

    def _always_long(_obs: np.ndarray) -> int:
        return 1

    policy_actions = rollout_policy_actions(
        data,
        policy_fn=_always_long,
        max_steps=3,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        slippage_bps=0.0,
    )
    summary = compare_policy_actions_to_oracle(oracle_steps, policy_actions)

    assert policy_actions == [1, 1, 1]
    assert summary.exact_match_rate == 1.0
    assert summary.near_best_match_rate == 1.0
    assert summary.mean_regret == 0.0
