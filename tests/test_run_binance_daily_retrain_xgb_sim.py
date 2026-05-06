from __future__ import annotations

import pandas as pd
from scripts.run_binance_daily_retrain_xgb_sim import controlled_clamp_config, daily_eval_windows, parse_args


def test_daily_eval_windows_split_partial_utc_days():
    windows = daily_eval_windows(
        pd.Timestamp("2026-05-04T07:00:00Z"),
        pd.Timestamp("2026-05-06T01:00:00Z"),
    )

    assert windows == [
        (pd.Timestamp("2026-05-04T07:00:00Z"), pd.Timestamp("2026-05-04T23:00:00Z")),
        (pd.Timestamp("2026-05-05T00:00:00Z"), pd.Timestamp("2026-05-05T23:00:00Z")),
        (pd.Timestamp("2026-05-06T00:00:00Z"), pd.Timestamp("2026-05-06T01:00:00Z")),
    ]


def test_controlled_clamp_config_matches_safe_research_candidate():
    cfg = controlled_clamp_config()

    assert cfg.risk_penalty == 0.35
    assert cfg.cvar_weight == 0.5
    assert cfg.max_positions == 2
    assert cfg.max_leverage == 2.0
    assert cfg.entry_allocator_mode == "concentrated"


def test_daily_retrain_defaults_use_binary_fill_research_controls():
    args = parse_args(["--out", "analysis/test.csv"])

    assert args.side_mode == "short"
    assert args.decision_lag == 2
    assert args.entry_allocator_min_active_positions == 2
    assert args.entry_allocator_min_second_position_fraction == 0.2
    assert args.entry_corr_window_bars == 168
    assert args.entry_corr_max_signed == 0.8
