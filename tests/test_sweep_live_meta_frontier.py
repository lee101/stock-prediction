from __future__ import annotations

from pathlib import Path

from binanceleveragesui.sweep_live_meta_frontier import (
    _build_backtest_cmd,
    _build_seeded_scenarios,
    _candidate_rank,
    _config_name,
    _dedupe_by_signature,
    _group_by_signature,
    _parse_float_list,
    _parse_int_list,
    _select_phase2c_rows,
    _select_refine_seed_rows,
    _summarize_windows,
    _validated_rank,
)


def test_parse_lists_drop_duplicates_and_blanks() -> None:
    assert _parse_int_list("1,2,2, 3 ,,") == [1, 2, 3]
    assert _parse_float_list("0.1,0.2,0.1,,0.3") == [0.1, 0.2, 0.3]


def test_build_seeded_scenarios_supports_custom_pair() -> None:
    scenarios = _build_seeded_scenarios(
        [
            {"name": "btc"},
            {"name": "sui"},
        ],
        [0.001, 30.0],
    )

    assert scenarios == (
        ("flat_1d", {"days": 1, "initial_model": "", "initial_inv": 0.0}),
        ("flat_7d", {"days": 7, "initial_model": "", "initial_inv": 0.0}),
        ("btc_long_7d", {"days": 7, "initial_model": "btc", "initial_inv": 0.001}),
        ("btc_short_7d", {"days": 7, "initial_model": "btc", "initial_inv": -0.001}),
        ("sui_long_7d", {"days": 7, "initial_model": "sui", "initial_inv": 30.0}),
        ("sui_short_7d", {"days": 7, "initial_model": "sui", "initial_inv": -30.0}),
    )


def test_build_seeded_scenarios_skips_zero_inventory() -> None:
    scenarios = _build_seeded_scenarios([{"name": "btc"}, {"name": "eth"}], [0.0, -0.0])
    assert scenarios == (
        ("flat_1d", {"days": 1, "initial_model": "", "initial_inv": 0.0}),
        ("flat_7d", {"days": 7, "initial_model": "", "initial_inv": 0.0}),
    )


def test_build_backtest_cmd_uses_generic_model_flags() -> None:
    cmd = _build_backtest_cmd(
        model_specs=[
            {
                "name": "btc",
                "symbol": "BTCUSDT",
                "data_symbol": "BTCUSD",
                "base_asset": "BTC",
                "maker_fee": 0.001,
                "checkpoint": Path("/tmp/btc.pt"),
            },
            {
                "name": "sui",
                "symbol": "SUIUSDT",
                "data_symbol": "SUIUSDT",
                "base_asset": "SUI",
                "maker_fee": 0.0015,
                "checkpoint": Path("/tmp/sui.pt"),
            },
        ],
        output_path=Path("/tmp/out.json"),
        data_root=Path("/tmp/data"),
        forecast_cache=Path("/tmp/cache"),
        selection_mode="winner_cash",
        selection_metric="omega",
        lookback=3,
        short_max_leverage=0.08,
        switch_margin=0.005,
        min_score_gap=0.002,
        profit_gate_mode="live_like",
        days=7,
        initial_model="btc",
        initial_inv=-0.125,
        initial_entry_ts="2026-03-08T00:00:00+00:00",
    )

    assert "--model-a-name" in cmd and "btc" in cmd
    assert "--model-b-name" in cmd and "sui" in cmd
    assert "--model-a-checkpoint" in cmd and "/tmp/btc.pt" in cmd
    assert "--model-b-checkpoint" in cmd and "/tmp/sui.pt" in cmd
    assert "--doge-checkpoint" not in cmd
    assert "--aave-checkpoint" not in cmd
    assert "--initial-model" in cmd and "--initial-inv" in cmd and "--initial-entry-ts" in cmd


def test_config_name_stable_and_compact() -> None:
    assert (
        _config_name(
            {
                "selection_mode": "winner_cash",
                "selection_metric": "calmar",
                "lookback": 1,
                "short_max_leverage": 0.16,
                "switch_margin": 0.002,
                "min_score_gap": 0.005,
            }
        )
        == "winner_cash_calmar_lb1_s160_sm002_gap005"
    )


def test_dedupe_by_signature_keeps_best_row() -> None:
    rows = [
        {
            "name": "better",
            "config": {},
            "windows": {
                "7d": {
                    "return_pct": 0.2,
                    "max_drawdown_pct": -1.0,
                    "trade_count": 10,
                    "switch_count": 3,
                }
            },
        },
        {
            "name": "worse_same_path",
            "config": {},
            "windows": {
                "7d": {
                    "return_pct": 0.2,
                    "max_drawdown_pct": -1.0,
                    "trade_count": 10,
                    "switch_count": 3,
                }
            },
        },
        {
            "name": "different",
            "config": {},
            "windows": {
                "7d": {
                    "return_pct": 0.1,
                    "max_drawdown_pct": -0.5,
                    "trade_count": 4,
                    "switch_count": 1,
                }
            },
        },
    ]

    deduped = _dedupe_by_signature(rows, "7d")
    assert [row["name"] for row in deduped] == ["better", "different"]


def test_group_by_signature_preserves_membership() -> None:
    rows = [
        {
            "name": "a",
            "config": {},
            "windows": {"7d": {"return_pct": 0.2, "max_drawdown_pct": -1.0, "trade_count": 10, "switch_count": 3}},
        },
        {
            "name": "b",
            "config": {},
            "windows": {"7d": {"return_pct": 0.2, "max_drawdown_pct": -1.0, "trade_count": 10, "switch_count": 3}},
        },
        {
            "name": "c",
            "config": {},
            "windows": {"7d": {"return_pct": 0.1, "max_drawdown_pct": -0.5, "trade_count": 4, "switch_count": 1}},
        },
    ]
    grouped = _group_by_signature(rows, "7d")
    assert [member["name"] for member in grouped[0][1]] == ["a", "b"]
    assert [member["name"] for member in grouped[1][1]] == ["c"]


def test_candidate_rank_prefers_positive_recent_and_drawdown_gate() -> None:
    better = {
        "windows": {
            "1d": {"return_pct": 0.05},
            "7d": {"return_pct": 0.10},
            "30d": {"return_pct": 2.0, "max_drawdown_pct": -10.0},
        }
    }
    worse = {
        "windows": {
            "1d": {"return_pct": -0.01},
            "7d": {"return_pct": 0.50},
            "30d": {"return_pct": 5.0, "max_drawdown_pct": -5.0},
        }
    }
    assert _candidate_rank(better) > _candidate_rank(worse)


def test_candidate_rank_sortino_objective_prefers_stronger_realized_sortino() -> None:
    higher_sortino = {
        "windows": {
            "1d": {"return_pct": 0.05, "sortino_ratio": 0.2},
            "7d": {"return_pct": 0.08, "sortino_ratio": 0.9},
            "30d": {"return_pct": 2.0, "max_drawdown_pct": -10.0, "sortino_ratio": 0.5},
        }
    }
    higher_return = {
        "windows": {
            "1d": {"return_pct": 0.06, "sortino_ratio": 0.2},
            "7d": {"return_pct": 0.12, "sortino_ratio": 0.3},
            "30d": {"return_pct": 2.5, "max_drawdown_pct": -10.0, "sortino_ratio": 0.4},
        }
    }
    assert _candidate_rank(higher_sortino, objective="sortino") > _candidate_rank(higher_return, objective="sortino")


def test_validated_rank_penalizes_seeded_drift() -> None:
    stable = {
        "windows": {
            "1d": {"return_pct": 0.05},
            "7d": {"return_pct": 0.08, "max_drawdown_pct": -1.0},
            "30d": {"return_pct": 1.0, "max_drawdown_pct": -10.0},
        },
        "seeded": {
            "doge_long_7d": {"return_pct": 0.001},
            "doge_short_7d": {"return_pct": -0.001},
        },
    }
    unstable = {
        "windows": {
            "1d": {"return_pct": 0.05},
            "7d": {"return_pct": 0.08, "max_drawdown_pct": -1.0},
            "30d": {"return_pct": 1.0, "max_drawdown_pct": -10.0},
        },
        "seeded": {
            "doge_long_7d": {"return_pct": 0.5},
            "doge_short_7d": {"return_pct": -0.4},
        },
    }
    assert _validated_rank(stable) > _validated_rank(unstable)


def test_validated_rank_sortino_objective_prefers_stronger_realized_sortino() -> None:
    better_sortino = {
        "windows": {
            "1d": {"return_pct": 0.05},
            "7d": {"return_pct": 0.09, "max_drawdown_pct": -1.0, "sortino_ratio": 0.8},
            "30d": {"return_pct": 10.0, "max_drawdown_pct": -10.0, "sortino_ratio": 0.4},
        },
        "seeded": {
            "doge_long_7d": {"return_pct": 0.001},
            "doge_short_7d": {"return_pct": -0.001},
        },
    }
    weaker_sortino = {
        "windows": {
            "1d": {"return_pct": 0.07},
            "7d": {"return_pct": 0.11, "max_drawdown_pct": -1.0, "sortino_ratio": 0.2},
            "30d": {"return_pct": 11.0, "max_drawdown_pct": -10.0, "sortino_ratio": 0.1},
        },
        "seeded": {
            "doge_long_7d": {"return_pct": 0.001},
            "doge_short_7d": {"return_pct": -0.001},
        },
    }
    assert _validated_rank(better_sortino, objective="sortino") > _validated_rank(weaker_sortino, objective="sortino")


def test_select_refine_seed_rows_keeps_long_window_challenger() -> None:
    rows = [
        {
            "name": "winner_cash_calmar_lb1_s160_sm000_gap000",
            "config": {
                "selection_mode": "winner_cash",
                "selection_metric": "calmar",
                "lookback": 1,
            },
            "windows": {
                "1d": {"return_pct": 0.08},
                "7d": {"return_pct": 0.154, "max_drawdown_pct": -2.0, "trade_count": 8, "switch_count": 3},
                "30d": {"return_pct": 13.7, "max_drawdown_pct": -15.0, "trade_count": 29, "switch_count": 11},
            },
        },
        {
            "name": "winner_cash_calmar_lb2_s160_sm000_gap000",
            "config": {
                "selection_mode": "winner_cash",
                "selection_metric": "calmar",
                "lookback": 2,
            },
            "windows": {
                "1d": {"return_pct": 0.08},
                "7d": {"return_pct": 0.153, "max_drawdown_pct": -2.0, "trade_count": 8, "switch_count": 3},
                "30d": {"return_pct": 13.6, "max_drawdown_pct": -15.0, "trade_count": 29, "switch_count": 11},
            },
        },
        {
            "name": "winner_cash_omega_lb1_s160_sm000_gap000",
            "config": {
                "selection_mode": "winner_cash",
                "selection_metric": "omega",
                "lookback": 1,
            },
            "windows": {
                "1d": {"return_pct": 0.08},
                "7d": {"return_pct": 0.146, "max_drawdown_pct": -2.0, "trade_count": 6, "switch_count": 2},
                "30d": {"return_pct": 14.2, "max_drawdown_pct": -15.0, "trade_count": 47, "switch_count": 11},
            },
        },
        {
            "name": "winner_cash_p10_lb1_s160_sm000_gap000",
            "config": {
                "selection_mode": "winner_cash",
                "selection_metric": "p10",
                "lookback": 1,
            },
            "windows": {
                "1d": {"return_pct": 0.0},
                "7d": {"return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 0, "switch_count": 0},
                "30d": {"return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 0, "switch_count": 0},
            },
        },
    ]

    selected = _select_refine_seed_rows(rows, top_unique=1)
    assert [row["name"] for row in selected] == [
        "winner_cash_calmar_lb1_s160_sm000_gap000",
        "winner_cash_omega_lb1_s160_sm000_gap000",
    ]


def test_validated_rank_prefers_stronger_30d_when_gates_match() -> None:
    better_long = {
        "windows": {
            "1d": {"return_pct": 0.05},
            "7d": {"return_pct": 0.09, "max_drawdown_pct": -1.0},
            "30d": {"return_pct": 12.0, "max_drawdown_pct": -10.0},
        },
        "seeded": {
            "doge_long_7d": {"return_pct": 0.001},
            "doge_short_7d": {"return_pct": -0.001},
        },
    }
    weaker_long = {
        "windows": {
            "1d": {"return_pct": 0.05},
            "7d": {"return_pct": 0.10, "max_drawdown_pct": -1.0},
            "30d": {"return_pct": 2.0, "max_drawdown_pct": -10.0},
        },
        "seeded": {
            "doge_long_7d": {"return_pct": 0.001},
            "doge_short_7d": {"return_pct": -0.001},
        },
    }
    assert _validated_rank(better_long) > _validated_rank(weaker_long)


def test_summarize_windows_accepts_precomputed_summaries() -> None:
    reports = {
        "1d": {
            "meta": {
                "return_pct": 0.1,
                "max_drawdown_pct": -0.5,
                "trade_count": 2,
                "switch_count": 1,
                "final_equity": 10010.0,
            }
        },
        "7d": {
            "return_pct": 0.2,
            "max_drawdown_pct": -1.0,
            "trade_count": 5,
            "switch_count": 2,
            "final_equity": 10020.0,
        },
    }

    assert _summarize_windows(reports) == {
        "1d": {
            "return_pct": 0.1,
            "max_drawdown_pct": -0.5,
            "trade_count": 2,
            "switch_count": 1,
            "final_equity": 10010.0,
        },
        "7d": {
            "return_pct": 0.2,
            "max_drawdown_pct": -1.0,
            "trade_count": 5,
            "switch_count": 2,
            "final_equity": 10020.0,
        },
    }


def test_summarize_windows_preserves_ratio_metrics_when_present() -> None:
    reports = {
        "7d": {
            "meta": {
                "return_pct": 0.2,
                "max_drawdown_pct": -1.0,
                "trade_count": 5,
                "switch_count": 2,
                "final_equity": 10020.0,
                "sortino_ratio": 0.7,
                "sharpe_ratio": 0.6,
            }
        },
    }

    assert _summarize_windows(reports) == {
        "7d": {
            "return_pct": 0.2,
            "max_drawdown_pct": -1.0,
            "trade_count": 5,
            "switch_count": 2,
            "final_equity": 10020.0,
            "sortino_ratio": 0.7,
            "sharpe_ratio": 0.6,
        }
    }


def test_select_phase2c_rows_works_with_7d_only_inputs() -> None:
    rows = [
        {
            "name": "winner_cash_calmar_lb1",
            "config": {
                "selection_mode": "winner_cash",
                "selection_metric": "calmar",
                "lookback": 1,
            },
            "windows": {
                "7d": {
                    "return_pct": 0.15,
                    "max_drawdown_pct": -2.0,
                    "trade_count": 10,
                    "switch_count": 4,
                    "final_equity": 10150.0,
                }
            },
        },
        {
            "name": "winner_cash_omega_lb1",
            "config": {
                "selection_mode": "winner_cash",
                "selection_metric": "omega",
                "lookback": 1,
            },
            "windows": {
                "7d": {
                    "return_pct": 0.14,
                    "max_drawdown_pct": -1.9,
                    "trade_count": 8,
                    "switch_count": 3,
                    "final_equity": 10140.0,
                }
            },
        },
    ]

    assert [row["name"] for row in _select_phase2c_rows(rows, top_finalists=2)] == [
        "winner_cash_calmar_lb1",
        "winner_cash_omega_lb1",
    ]
