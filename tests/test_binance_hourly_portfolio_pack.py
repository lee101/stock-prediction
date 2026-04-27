from __future__ import annotations

import argparse

import pandas as pd

from scripts.sweep_binance_hourly_portfolio_pack import (
    PackConfig,
    build_actions_and_bars,
    compute_pack_selection_score,
    iter_pack_configs,
    sample_pack_configs,
)


def _pack_config(**overrides) -> PackConfig:
    values = dict(
        risk_penalty=0.5,
        cvar_weight=0.0,
        entry_gap_bps=50.0,
        entry_alpha=0.5,
        exit_alpha=0.8,
        edge_threshold=0.003,
        edge_to_full_size=0.02,
        max_positions=2,
        max_pending_entries=4,
        entry_ttl_hours=3,
        max_hold_hours=24,
        max_leverage=1.0,
        entry_selection_mode="edge_rank",
        entry_allocator_mode="concentrated",
        entry_allocator_edge_power=2.0,
    )
    values.update(overrides)
    return PackConfig(**values)


def test_build_actions_and_bars_creates_long_watcher_levels():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    scored = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "BTCUSDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.04,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.02,
                "cvar_loss_72h": 0.002,
            }
        ]
    )

    bars, actions = build_actions_and_bars(
        scored,
        cfg=_pack_config(),
        label_horizon=24,
        min_take_profit_bps=35.0,
        max_entry_gap_bps=120.0,
        max_exit_gap_bps=250.0,
        fee_rate=0.001,
        top_candidates_per_hour=10,
    )

    action = actions.iloc[0]
    assert action["buy_amount"] > 0.0
    assert action["buy_price"] < 100.0
    assert action["sell_price"] > 100.0
    assert "predicted_high_p50_h24" in bars.columns


def test_build_actions_and_bars_applies_top_candidate_gate():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    scored = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAAUSDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.04,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.02,
                "cvar_loss_72h": 0.002,
            },
            {
                "timestamp": ts,
                "symbol": "BBBUSDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "reference_close": 100.0,
                "pred_high_ret_xgb": 0.02,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.01,
                "cvar_loss_72h": 0.002,
            },
        ]
    )

    _, actions = build_actions_and_bars(
        scored,
        cfg=_pack_config(),
        label_horizon=24,
        min_take_profit_bps=35.0,
        max_entry_gap_bps=120.0,
        max_exit_gap_bps=250.0,
        fee_rate=0.001,
        top_candidates_per_hour=1,
    )

    active = actions[actions["buy_amount"] > 0.0]
    assert active["symbol"].tolist() == ["AAAUSDT"]


def test_sample_pack_configs_spreads_across_full_grid_deterministically():
    args = argparse.Namespace(
        risk_penalties="0.2,0.5",
        cvar_weights="0.0,0.5",
        entry_gap_bps_grid="25,50,75",
        entry_alpha_grid="0.5",
        exit_alpha_grid="0.8",
        edge_threshold_grid="0.003,0.006",
        edge_to_full_size_grid="0.02",
        max_positions_grid="5,8",
        max_pending_entries_grid="12,24",
        entry_ttl_hours_grid="3,6",
        max_hold_hours_grid="24",
        max_leverage_grid="1.0",
        entry_selection_modes="edge_rank,first_trigger",
        entry_allocator_modes="legacy,concentrated",
        entry_allocator_edge_power_grid="2.0",
    )
    configs = iter_pack_configs(args)
    sampled_a = sample_pack_configs(configs, limit=20, seed=1337)
    sampled_b = sample_pack_configs(configs, limit=20, seed=1337)

    assert sampled_a == sampled_b
    assert len(sampled_a) == 20
    assert {cfg.entry_gap_bps for cfg in sampled_a} == {25.0, 50.0, 75.0}
    assert {cfg.risk_penalty for cfg in sampled_a} == {0.2, 0.5}


def test_selection_score_penalizes_idle_configs():
    base = {
        "monthly_return_pct": 0.0,
        "sortino": 0.0,
        "pnl_smoothness_score": 1.0,
        "goodness_score": 0.5,
        "max_drawdown_pct": 0.0,
    }
    idle = dict(base, num_sells=0)
    active = dict(base, num_sells=10)

    assert compute_pack_selection_score(active) > compute_pack_selection_score(idle)
