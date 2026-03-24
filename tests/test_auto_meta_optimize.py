from __future__ import annotations

import argparse
import json
from pathlib import Path

from unified_hourly_experiment.auto_meta_optimize import (
    build_entry_allocator_mode_summary,
    build_deploy_command,
    eligible_summary,
    iter_entry_allocator_grid,
    parse_float_list,
    parse_int_list,
    rank_key,
    run_once,
)


def test_parse_float_list() -> None:
    assert parse_float_list("0.1, 0.2,0.3") == [0.1, 0.2, 0.3]


def test_parse_int_list() -> None:
    assert parse_int_list("0, 1,2") == [0, 1, 2]


def test_rank_key_prefers_higher_sortino_then_return() -> None:
    a = {
        "min_sortino": 0.4,
        "mean_sortino": 1.0,
        "min_return_pct": 0.2,
        "mean_return_pct": 0.7,
        "mean_max_drawdown_pct": 0.4,
    }
    b = {
        "min_sortino": 0.1,
        "mean_sortino": 2.0,
        "min_return_pct": 0.5,
        "mean_return_pct": 1.2,
        "mean_max_drawdown_pct": 0.2,
    }
    assert rank_key(a) > rank_key(b)


def test_eligible_summary_uses_min_num_buys() -> None:
    row = {"min_num_buys": 2}
    assert eligible_summary(row, min_num_buys=2)
    assert eligible_summary(row, min_num_buys=1)
    assert not eligible_summary(row, min_num_buys=3)


def test_build_entry_allocator_mode_summary_groups_and_ranks_modes() -> None:
    rows = [
        {
            "entry_allocator_mode": "legacy",
            "metric": "sortino",
            "lookback_days": 5,
            "selection_mode": "winner",
            "switch_margin": 0.0,
            "min_score_gap": 0.0,
            "recency_halflife_days": 0.0,
            "sit_out_threshold": 0.2,
            "trade_amount_scale": 100.0,
            "min_buy_amount": 0.0,
            "entry_intensity_power": 1.0,
            "entry_min_intensity_fraction": 0.0,
            "long_intensity_multiplier": 1.0,
            "short_intensity_multiplier": 1.0,
            "entry_allocator_edge_power": 2.0,
            "entry_allocator_max_single_position_fraction": 0.6,
            "entry_allocator_reserve_fraction": 0.1,
            "min_sortino": 0.5,
            "mean_sortino": 0.7,
            "min_return_pct": 1.0,
            "mean_return_pct": 2.0,
            "min_num_buys": 2,
            "output": "/tmp/legacy.json",
        },
        {
            "entry_allocator_mode": "concentrated",
            "metric": "sortino",
            "lookback_days": 7,
            "selection_mode": "winner",
            "switch_margin": 0.0,
            "min_score_gap": 0.0,
            "recency_halflife_days": 1.0,
            "sit_out_threshold": 0.2,
            "trade_amount_scale": 120.0,
            "min_buy_amount": 0.0,
            "entry_intensity_power": 1.3,
            "entry_min_intensity_fraction": 0.0,
            "long_intensity_multiplier": 1.0,
            "short_intensity_multiplier": 1.0,
            "entry_allocator_edge_power": 3.0,
            "entry_allocator_max_single_position_fraction": 0.8,
            "entry_allocator_reserve_fraction": 0.05,
            "min_sortino": 0.9,
            "mean_sortino": 1.1,
            "min_return_pct": 1.5,
            "mean_return_pct": 2.8,
            "min_num_buys": 3,
            "output": "/tmp/concentrated-best.json",
        },
        {
            "entry_allocator_mode": "concentrated",
            "metric": "sharpe",
            "lookback_days": 10,
            "selection_mode": "winner",
            "switch_margin": 0.0,
            "min_score_gap": 0.0,
            "recency_halflife_days": 0.0,
            "sit_out_threshold": 0.3,
            "trade_amount_scale": 110.0,
            "min_buy_amount": 0.0,
            "entry_intensity_power": 1.1,
            "entry_min_intensity_fraction": 0.0,
            "long_intensity_multiplier": 1.0,
            "short_intensity_multiplier": 1.0,
            "entry_allocator_edge_power": 2.0,
            "entry_allocator_max_single_position_fraction": 0.6,
            "entry_allocator_reserve_fraction": 0.1,
            "min_sortino": 0.4,
            "mean_sortino": 0.9,
            "min_return_pct": 0.5,
            "mean_return_pct": 1.9,
            "min_num_buys": 1,
            "output": "/tmp/concentrated-other.json",
        },
    ]

    summary = build_entry_allocator_mode_summary(rows)

    assert [row["entry_allocator_mode"] for row in summary] == ["concentrated", "legacy"]
    assert summary[0]["count"] == 2
    assert summary[0]["best"]["output"] == "/tmp/concentrated-best.json"
    assert summary[0]["mean_min_sortino"] == 0.65
    assert summary[0]["mean_mean_return_pct"] == 2.3499999999999996
    assert summary[1]["count"] == 1
    assert summary[1]["best"]["output"] == "/tmp/legacy.json"


def test_iter_entry_allocator_grid_collapses_legacy_duplicates() -> None:
    assert iter_entry_allocator_grid(
        mode="legacy",
        edge_powers=[1.0, 2.0, 3.0],
        max_single_position_fractions=[0.4, 0.8],
        reserve_fractions=[0.05, 0.1],
    ) == [(1.0, 0.4, 0.05)]
    assert iter_entry_allocator_grid(
        mode="concentrated",
        edge_powers=[1.0, 2.0],
        max_single_position_fractions=[0.4],
        reserve_fractions=[0.05, 0.1],
    ) == [(1.0, 0.4, 0.05), (1.0, 0.4, 0.1), (2.0, 0.4, 0.05), (2.0, 0.4, 0.1)]


def test_build_deploy_command_includes_sizing_fields() -> None:
    best = {
        "edge": 0.0055,
        "trade_amount_scale": 100.0,
        "min_buy_amount": 0.0,
        "entry_intensity_power": 0.8,
        "entry_min_intensity_fraction": 0.15,
        "long_intensity_multiplier": 1.0,
        "short_intensity_multiplier": 2.0,
        "entry_allocator_mode": "concentrated",
        "entry_allocator_edge_power": 1.7,
        "entry_allocator_max_single_position_fraction": 0.55,
        "entry_allocator_reserve_fraction": 0.15,
        "metric": "sharpe",
        "lookback_days": 14,
        "selection_mode": "winner",
        "switch_margin": 0.0,
        "min_score_gap": 0.0,
        "recency_halflife_days": 2.0,
        "sit_out_threshold": 0.2,
    }
    cmd = build_deploy_command(
        strategy_specs=["s1=/tmp/a:1", "s2=/tmp/b:2"],
        symbols="NVDA,MTCH",
        decision_lag_bars=2,
        entry_selection_mode="first_trigger",
        max_hold_hours=6,
        max_positions=7,
        bar_margin=0.0013,
        entry_order_ttl_hours=0,
        fee_rate=0.001,
        margin_rate=0.0625,
        market_order_entry=True,
        best=best,
    )
    assert "--entry-intensity-power 0.8" in cmd
    assert "--entry-min-intensity-fraction 0.15" in cmd
    assert "--short-intensity-multiplier 2.0" in cmd
    assert "--meta-recency-halflife-days 2.0" in cmd
    assert "--trade-amount-scale 100.0" in cmd
    assert "--decision-lag-bars 2" in cmd
    assert "--entry-selection-mode first_trigger" in cmd
    assert "--entry-allocator-mode concentrated" in cmd
    assert "--entry-allocator-edge-power 1.7" in cmd
    assert "--entry-allocator-max-single-position-fraction 0.55" in cmd
    assert "--entry-allocator-reserve-fraction 0.15" in cmd
    assert "--market-order-entry" in cmd


def _fake_sweep_payload(recency_halflife_days: float) -> dict:
    return {
        "generated_at_utc": "2026-03-05T00:00:00+00:00",
        "strategies": ["a=/tmp/a:1", "b=/tmp/b:2"],
        "symbols": ["NVDA", "PLTR"],
        "config": {},
        "best": {
            "metric": "sharpe",
            "lookback_days": 10,
            "selection_mode": "winner",
            "entry_selection_mode": "edge_rank",
            "switch_margin": 0.0,
            "min_score_gap": 0.0,
            "recency_halflife_days": recency_halflife_days,
            "min_sortino": 0.5 + recency_halflife_days,
            "mean_sortino": 0.8 + recency_halflife_days,
            "min_return_pct": 0.2,
            "mean_return_pct": 0.3,
            "mean_max_drawdown_pct": 0.1,
            "min_num_buys": 2,
            "mean_num_buys": 2.0,
        },
        "strategy_baseline_summaries": [],
        "strategy_baseline_results": [],
        "summaries": [],
        "results": [],
    }


def _build_args(tmp_path: Path, *, skip_existing: bool) -> argparse.Namespace:
    return argparse.Namespace(
        strategy=["a=/tmp/a:1", "b=/tmp/b:2"],
        symbols="NVDA,PLTR",
        metrics="sharpe",
        lookback_days="10",
        holdout_days="30",
        min_edges="0.0065",
        sit_out_thresholds="0.25",
        selection_modes="winner",
        switch_margins="0.0",
        min_score_gaps="0.0",
        recency_halflife_days="0.0,1.0",
        min_num_buys=2,
        max_positions=5,
        max_hold_hours=5,
        trade_amount_scales="100.0",
        min_buy_amounts="0.0",
        entry_intensity_powers="1.0",
        entry_min_intensity_fractions="0.0",
        long_intensity_multipliers="1.0",
        short_intensity_multipliers="1.0",
        entry_allocator_modes="legacy,concentrated",
        entry_allocator_edge_powers="2.0",
        entry_allocator_max_single_position_fractions="0.6",
        entry_allocator_reserve_fractions="0.1",
        decision_lag_bars=1,
        entry_selection_mode="edge_rank",
        market_order_entry=False,
        bar_margin=0.0013,
        execution_bar_margins="0.0005,0.0013",
        entry_order_ttl_hours=0,
        execution_entry_order_ttls="0,1",
        leverage=2.0,
        fee_rate=0.001,
        margin_rate=0.0625,
        sim_backend="python",
        action_cache_dir=None,
        skip_existing=skip_existing,
        output_dir=tmp_path,
    )


def test_run_once_skip_existing_resumes_without_rerunning(monkeypatch, tmp_path: Path) -> None:
    existing = (
        tmp_path
        / "meta_edge0p0065_th0p25_mwinner_sm0p0_mg0p0_hl0p0_tas100p0_mba0p0_pow1p0_minf0p0_lm1p0_smul1p0_eamlegacy_eaep2p0_easp0p6_earm0p1.json"
    )
    existing.write_text(json.dumps(_fake_sweep_payload(0.0)))

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        calls.append(cmd)
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_text(json.dumps(_fake_sweep_payload(1.0)))

    monkeypatch.setattr("unified_hourly_experiment.auto_meta_optimize.subprocess.run", fake_run)

    rec = run_once(_build_args(tmp_path, skip_existing=True))

    assert len(calls) == 3
    assert "--execution-bar-margins" in calls[0]
    assert calls[0][calls[0].index("--execution-bar-margins") + 1] == "0.0005,0.0013"
    assert "--execution-entry-order-ttls" in calls[0]
    assert calls[0][calls[0].index("--execution-entry-order-ttls") + 1] == "0,1"
    assert "--action-cache-dir" in calls[0]
    assert calls[0][calls[0].index("--action-cache-dir") + 1] == str(tmp_path / "action_cache")
    assert "--entry-selection-mode" in calls[0]
    assert calls[0][calls[0].index("--entry-selection-mode") + 1] == "edge_rank"
    assert "--entry-allocator-mode" in calls[0]
    assert "--entry-allocator-edge-power" in calls[0]
    assert rec["best"]["recency_halflife_days"] == 1.0
    assert rec["search_space"]["execution_bar_margins"] == [0.0005, 0.0013]
    assert rec["search_space"]["execution_entry_order_ttl_hours"] == [0, 1]
    assert rec["search_space"]["action_cache_dir"] == str(tmp_path / "action_cache")
    assert rec["search_space"]["entry_selection_mode"] == "edge_rank"
    assert rec["search_space"]["entry_allocator_modes"] == ["legacy", "concentrated"]
    assert [row["entry_allocator_mode"] for row in rec["entry_allocator_mode_summary"]] == [
        "concentrated",
        "legacy",
    ]
    assert rec["entry_allocator_mode_summary"][0]["best"]["output"].endswith("_eamconcentrated_eaep2p0_easp0p6_earm0p1.json")
    assert (tmp_path / "auto_meta_recommendation.json").exists()


def test_run_once_collapses_legacy_allocator_duplicates(monkeypatch, tmp_path: Path) -> None:
    args = _build_args(tmp_path, skip_existing=False)
    args.recency_halflife_days = "0.0"
    args.entry_allocator_edge_powers = "1.0,2.0"

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        calls.append(cmd)
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_text(json.dumps(_fake_sweep_payload(0.0)))

    monkeypatch.setattr("unified_hourly_experiment.auto_meta_optimize.subprocess.run", fake_run)

    run_once(args)

    assert len(calls) == 3
    legacy_calls = [cmd for cmd in calls if cmd[cmd.index("--entry-allocator-mode") + 1] == "legacy"]
    concentrated_calls = [cmd for cmd in calls if cmd[cmd.index("--entry-allocator-mode") + 1] == "concentrated"]
    assert len(legacy_calls) == 1
    assert len(concentrated_calls) == 2
