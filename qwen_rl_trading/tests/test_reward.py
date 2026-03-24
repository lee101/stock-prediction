"""Tests for qwen_rl_trading.reward."""
import json

import numpy as np
import pandas as pd
import pytest

from qwen_rl_trading.data_prompt import MarketSnapshot
from qwen_rl_trading.reward import (
    DEFAULT_REWARD_TYPE,
    PENALTY_FLAT,
    PENALTY_INVALID,
    REWARD_SIM_CONFIG,
    GRPORewardFn,
    compute_reward,
    compute_sortino,
    score_reward_metrics,
)


def _make_snapshot(n_hours=24, symbols=("BTCUSD",)):
    rows = []
    base_ts = pd.Timestamp("2025-06-01", tz="UTC")
    for sym in symbols:
        base_price = 67000 if sym == "BTCUSD" else 3500
        for h in range(n_hours):
            rows.append({
                "timestamp": base_ts + pd.Timedelta(hours=h + 1),
                "symbol": sym,
                "open": base_price + h * 5,
                "high": base_price + h * 5 + 50,
                "low": base_price + h * 5 - 30,
                "close": base_price + h * 5 + 10,
                "volume": 500.0,
            })
    forward_bars = pd.DataFrame(rows)
    return MarketSnapshot(
        window_id="test123",
        timestamp=base_ts,
        symbols=list(symbols),
        features={s: {"price": 67000 if s == "BTCUSD" else 3500} for s in symbols},
        chronos_forecasts={},
        forward_bars=forward_bars,
    )


def _valid_plan_json(entry_price=67050):
    return json.dumps({
        "plans": [{
            "symbol": "BTCUSD",
            "action": "LONG",
            "allocation_pct": 50,
            "entry_price": entry_price,
            "stop_loss": entry_price * 0.99,
            "take_profit": entry_price * 1.02,
            "max_hold_hours": 6,
            "confidence": 0.7,
            "reasoning": "test",
        }],
        "cash_reserve_pct": 20,
        "market_regime": "trending_up",
    })


class TestComputeSortino:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.005, 0.01, -0.001])
        s = compute_sortino(returns)
        assert s > 0

    def test_all_negative(self):
        returns = np.array([-0.01, -0.02, -0.005])
        s = compute_sortino(returns)
        assert s < 0

    def test_empty(self):
        assert compute_sortino(np.array([])) == 0.0
        assert compute_sortino(np.array([0.01])) == 0.0

    def test_all_positive(self):
        returns = np.array([0.01, 0.02, 0.015])
        s = compute_sortino(returns)
        assert s > 0  # no downside = large positive


class TestComputeReward:
    def test_invalid_json_penalty(self):
        snap = _make_snapshot()
        assert compute_reward("not valid json", snap) == PENALTY_INVALID

    def test_all_flat_penalty(self):
        snap = _make_snapshot()
        plan = json.dumps({
            "plans": [{"symbol": "BTCUSD", "action": "FLAT", "allocation_pct": 0,
                        "entry_price": 67000, "stop_loss": 66000, "take_profit": 68000}],
            "cash_reserve_pct": 100, "market_regime": "ranging",
        })
        assert compute_reward(plan, snap) == PENALTY_FLAT

    def test_valid_plan_not_penalty(self):
        snap = _make_snapshot()
        reward = compute_reward(_valid_plan_json(), snap)
        assert reward != PENALTY_INVALID
        assert reward != PENALTY_FLAT

    def test_uses_decision_lag(self):
        assert REWARD_SIM_CONFIG.decision_lag_bars == 2

    def test_uses_10bps_fee(self):
        assert REWARD_SIM_CONFIG.maker_fee == 0.001

    def test_uses_binary_fills(self):
        # fill_buffer_bps > 0 means price must penetrate limit
        assert REWARD_SIM_CONFIG.fill_buffer_bps == 5.0

    def test_max_hold_6h(self):
        assert REWARD_SIM_CONFIG.max_hold_hours == 6

    def test_reward_type_variants_change_score(self):
        base_reward = score_reward_metrics(
            sortino=2.0,
            total_return=0.05,
            max_drawdown=-0.20,
            pnl_smoothness=0.01,
            reward_type=DEFAULT_REWARD_TYPE,
        )
        drawdown_reward = score_reward_metrics(
            sortino=2.0,
            total_return=0.05,
            max_drawdown=-0.20,
            pnl_smoothness=0.01,
            reward_type="sortino_drawdown",
        )
        smoothness_reward = score_reward_metrics(
            sortino=2.0,
            total_return=0.05,
            max_drawdown=-0.20,
            pnl_smoothness=0.01,
            reward_type="sortino_smoothness",
        )

        assert drawdown_reward < base_reward
        assert smoothness_reward < base_reward
        assert drawdown_reward != smoothness_reward

    def test_unknown_reward_type_raises(self):
        with pytest.raises(ValueError, match="unsupported reward_type"):
            score_reward_metrics(
                sortino=1.0,
                total_return=0.01,
                max_drawdown=-0.05,
                pnl_smoothness=0.001,
                reward_type="does_not_exist",
            )


class TestGRPORewardFn:
    def test_callable(self):
        snap = _make_snapshot()
        fn = GRPORewardFn({"test123": snap})
        rewards = fn(
            completions=[_valid_plan_json()],
            prompts=["[window:test123] Market data..."],
        )
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_missing_snapshot_penalty(self):
        fn = GRPORewardFn({})
        rewards = fn(completions=["{}"], prompts=["no window id here"])
        assert rewards[0] == PENALTY_INVALID

    def test_window_id_extraction(self):
        snap = _make_snapshot()
        fn = GRPORewardFn({"abc123": snap})
        result = fn._find_snapshot("[window:abc123] some data")
        assert result is snap

    def test_window_id_extraction_from_chat_prompt(self):
        snap = _make_snapshot()
        fn = GRPORewardFn({"abc123": snap})
        prompt = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "[window:abc123] some data"},
        ]
        result = fn._find_snapshot(prompt)
        assert result is snap

    def test_batch_multiple(self):
        snap = _make_snapshot()
        fn = GRPORewardFn({"test123": snap})
        rewards = fn(
            completions=[_valid_plan_json(), "invalid", _valid_plan_json()],
            prompts=["[window:test123] d1", "[window:test123] d2", "[window:missing] d3"],
        )
        assert len(rewards) == 3
        assert rewards[1] == PENALTY_INVALID  # invalid json
        assert rewards[2] == PENALTY_INVALID  # missing snapshot

    def test_conversational_completion_payload(self):
        snap = _make_snapshot()
        snap.window_id = "abc123"
        fn = GRPORewardFn({"abc123": snap})
        rewards = fn(
            completions=[[{"role": "assistant", "content": _valid_plan_json()}]],
            prompts=["[window:abc123] market data"],
        )
        assert len(rewards) == 1
        assert rewards[0] != PENALTY_INVALID
