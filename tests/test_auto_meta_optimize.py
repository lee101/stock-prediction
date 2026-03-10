from __future__ import annotations

import argparse
import json
from pathlib import Path

from unified_hourly_experiment.auto_meta_optimize import (
    build_deploy_command,
    eligible_summary,
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


def test_build_deploy_command_includes_sizing_fields() -> None:
    best = {
        "edge": 0.0055,
        "trade_amount_scale": 100.0,
        "min_buy_amount": 0.0,
        "entry_intensity_power": 0.8,
        "entry_min_intensity_fraction": 0.15,
        "long_intensity_multiplier": 1.0,
        "short_intensity_multiplier": 2.0,
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
        skip_existing=skip_existing,
        output_dir=tmp_path,
    )


def test_run_once_skip_existing_resumes_without_rerunning(monkeypatch, tmp_path: Path) -> None:
    existing = (
        tmp_path
        / "meta_edge0p0065_th0p25_mwinner_sm0p0_mg0p0_hl0p0_tas100p0_mba0p0_pow1p0_minf0p0_lm1p0_smul1p0.json"
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

    assert len(calls) == 1
    assert "--execution-bar-margins" in calls[0]
    assert calls[0][calls[0].index("--execution-bar-margins") + 1] == "0.0005,0.0013"
    assert "--execution-entry-order-ttls" in calls[0]
    assert calls[0][calls[0].index("--execution-entry-order-ttls") + 1] == "0,1"
    assert "--entry-selection-mode" in calls[0]
    assert calls[0][calls[0].index("--entry-selection-mode") + 1] == "edge_rank"
    assert rec["best"]["recency_halflife_days"] == 1.0
    assert rec["search_space"]["execution_bar_margins"] == [0.0005, 0.0013]
    assert rec["search_space"]["execution_entry_order_ttl_hours"] == [0, 1]
    assert rec["search_space"]["entry_selection_mode"] == "edge_rank"
    assert (tmp_path / "auto_meta_recommendation.json").exists()
