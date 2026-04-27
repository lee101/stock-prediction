from __future__ import annotations

import argparse

import pandas as pd

from scripts.sweep_binance_hourly_portfolio_pack import (
    PackConfig,
    _filter_liquid_frames,
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
        min_close_ret=-0.2,
        close_edge_weight=0.0,
        min_upside_downside_ratio=0.0,
        min_recent_ret_24h=-1.0,
        min_recent_ret_72h=-1.0,
        max_recent_vol_72h=0.0,
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


def test_build_actions_and_bars_can_gate_weak_close_consensus():
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
                "pred_high_ret_xgb": 0.08,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": -0.01,
                "cvar_loss_72h": 0.001,
            }
        ]
    )

    _, actions = build_actions_and_bars(
        scored,
        cfg=_pack_config(min_close_ret=0.0),
        label_horizon=24,
        min_take_profit_bps=35.0,
        max_entry_gap_bps=120.0,
        max_exit_gap_bps=250.0,
        fee_rate=0.001,
        top_candidates_per_hour=10,
    )

    action = actions.iloc[0]
    assert action["xgb_edge"] > 0.0
    assert action["buy_amount"] == 0.0


def test_build_actions_and_bars_can_gate_weak_recent_momentum():
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
                "pred_high_ret_xgb": 0.08,
                "pred_low_ret_xgb": -0.005,
                "pred_close_ret_xgb": 0.02,
                "cvar_loss_72h": 0.001,
                "ret_24h": -0.03,
                "ret_72h": 0.04,
                "vol_72h": 0.01,
            }
        ]
    )

    _, actions = build_actions_and_bars(
        scored,
        cfg=_pack_config(min_recent_ret_24h=0.0),
        label_horizon=24,
        min_take_profit_bps=35.0,
        max_entry_gap_bps=120.0,
        max_exit_gap_bps=250.0,
        fee_rate=0.001,
        top_candidates_per_hour=10,
    )

    action = actions.iloc[0]
    assert action["xgb_edge"] > 0.0
    assert action["recent_ret_24h"] == -0.03
    assert action["buy_amount"] == 0.0


def test_sample_pack_configs_spreads_across_full_grid_deterministically():
    args = argparse.Namespace(
        risk_penalties="0.2,0.5",
        cvar_weights="0.0,0.5",
        entry_gap_bps_grid="25,50,75",
        entry_alpha_grid="0.5",
        exit_alpha_grid="0.8",
        edge_threshold_grid="0.003,0.006",
        edge_to_full_size_grid="0.02",
        min_close_ret_grid="-0.2",
        close_edge_weight_grid="0.0",
        min_upside_downside_ratio_grid="0.0",
        min_recent_ret_24h_grid="-1.0",
        min_recent_ret_72h_grid="-1.0",
        max_recent_vol_72h_grid="0.0",
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


def test_sample_pack_configs_keeps_randomized_order():
    args = argparse.Namespace(
        risk_penalties="0.2,0.5",
        cvar_weights="0.0",
        entry_gap_bps_grid="25,50,75",
        entry_alpha_grid="0.5",
        exit_alpha_grid="0.8",
        edge_threshold_grid="0.003,0.006",
        edge_to_full_size_grid="0.02",
        min_close_ret_grid="0.0",
        close_edge_weight_grid="0.0",
        min_upside_downside_ratio_grid="0.0",
        min_recent_ret_24h_grid="-1.0",
        min_recent_ret_72h_grid="-1.0",
        max_recent_vol_72h_grid="0.0",
        max_positions_grid="5,8",
        max_pending_entries_grid="12",
        entry_ttl_hours_grid="3",
        max_hold_hours_grid="24",
        max_leverage_grid="1.0",
        entry_selection_modes="edge_rank",
        entry_allocator_modes="concentrated",
        entry_allocator_edge_power_grid="2.0",
    )
    configs = iter_pack_configs(args)
    sampled = sample_pack_configs(configs, limit=10, seed=20260427)
    sampled_indices = [configs.index(cfg) for cfg in sampled]

    assert sampled_indices != sorted(sampled_indices)


def test_filter_liquid_frames_keeps_top_dollar_volume_symbols():
    ts = pd.date_range("2026-03-01T00:00:00Z", periods=4, freq="h")
    frames = {
        "LOWUSDT": pd.DataFrame({"timestamp": ts, "close": [1.0] * 4, "volume": [10.0] * 4}),
        "MIDUSDT": pd.DataFrame({"timestamp": ts, "close": [10.0] * 4, "volume": [100.0] * 4}),
        "HIGHUSDT": pd.DataFrame({"timestamp": ts, "close": [100.0] * 4, "volume": [1000.0] * 4}),
    }

    filtered, metrics = _filter_liquid_frames(
        frames,
        end=pd.Timestamp("2026-03-01T03:00:00Z"),
        lookback_days=1,
        min_median_dollar_volume=0.0,
        max_symbols=2,
    )

    assert list(filtered) == ["HIGHUSDT", "MIDUSDT"]
    assert metrics.iloc[0]["symbol"] == "HIGHUSDT"


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
