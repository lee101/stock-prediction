from __future__ import annotations

import csv
from pathlib import Path

from scripts.run_binance_pack_rolling_validation import (
    load_candidate_configs,
    summarize_results,
)


def _candidate_row(**overrides):
    row = {
        "risk_penalty": 0.8,
        "cvar_weight": 1.0,
        "entry_gap_bps": 65.0,
        "entry_alpha": 0.5,
        "exit_alpha": 0.8,
        "edge_threshold": 0.011,
        "edge_to_full_size": 0.02,
        "min_close_ret": 0.002,
        "close_edge_weight": 0.5,
        "min_upside_downside_ratio": 0.0,
        "max_positions": 5,
        "max_pending_entries": 12,
        "entry_ttl_hours": 6,
        "max_hold_hours": 24,
        "max_leverage": 1.0,
        "entry_selection_mode": "edge_rank",
        "entry_allocator_mode": "concentrated",
        "entry_allocator_edge_power": 2.0,
        "selection_score": 100.0,
        "monthly_return_pct": 4.0,
        "num_sells": 30,
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_candidate_configs_defaults_new_regime_gates_and_dedupes(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    _write_csv(
        csv_path,
        [
            _candidate_row(selection_score=100.0),
            _candidate_row(selection_score=99.0),
        ],
    )

    candidates = load_candidate_configs(
        [csv_path],
        top_k_per_file=10,
        rank_by="selection_score",
        min_source_sells=0,
    )

    assert len(candidates) == 1
    cfg = candidates[0].cfg
    assert cfg.min_recent_ret_24h == -1.0
    assert cfg.min_recent_ret_72h == -1.0
    assert cfg.max_recent_vol_72h == 0.0
    assert cfg.regime_cs_skew_min == -1_000_000_000.0
    assert cfg.vol_target_ann == 0.0
    assert cfg.inv_vol_target_ann == 0.0
    assert cfg.inv_vol_floor == 0.05
    assert cfg.inv_vol_cap == 3.0


def test_load_candidate_configs_applies_top_k_after_ranking(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    _write_csv(
        csv_path,
        [
            _candidate_row(selection_score=1.0, entry_gap_bps=65.0),
            _candidate_row(selection_score=10.0, entry_gap_bps=85.0),
        ],
    )

    candidates = load_candidate_configs(
        [csv_path],
        top_k_per_file=1,
        rank_by="selection_score",
        min_source_sells=0,
    )

    assert len(candidates) == 1
    assert candidates[0].cfg.entry_gap_bps == 85.0


def test_summarize_results_penalizes_negative_cells():
    rows = [
        {
            "candidate_id": "stable",
            "source": "a.csv",
            "source_row": 0,
            "monthly_return_pct": 1.0,
            "total_return_pct": 4.0,
            "sortino": 2.0,
            "max_drawdown_pct": 2.0,
            "num_sells": 30,
            "config_json": "{}",
        },
        {
            "candidate_id": "stable",
            "source": "a.csv",
            "source_row": 0,
            "monthly_return_pct": 0.5,
            "total_return_pct": 2.0,
            "sortino": 1.0,
            "max_drawdown_pct": 3.0,
            "num_sells": 25,
            "config_json": "{}",
        },
        {
            "candidate_id": "fragile",
            "source": "b.csv",
            "source_row": 1,
            "monthly_return_pct": 6.0,
            "total_return_pct": 27.0,
            "sortino": 3.0,
            "max_drawdown_pct": 2.0,
            "num_sells": 30,
            "config_json": "{}",
        },
        {
            "candidate_id": "fragile",
            "source": "b.csv",
            "source_row": 1,
            "monthly_return_pct": -1.0,
            "total_return_pct": -4.0,
            "sortino": 3.0,
            "max_drawdown_pct": 2.0,
            "num_sells": 30,
            "config_json": "{}",
        },
    ]

    summary = summarize_results(rows, min_trades=20)

    assert summary[0]["candidate_id"] == "stable"
    assert summary[1]["negative_cells"] == 1
