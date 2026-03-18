from __future__ import annotations

import json

from src.alpaca_stock_expansion import (
    StockExpansionCandidate,
    candidate_lora_command,
    count_candidate_history_rows,
    default_stock_expansion_candidates,
    filter_candidates_with_hourly_data,
    load_stock_expansion_manifest,
    split_candidates_by_history,
    stock_expansion_sort_key,
    summarize_reforecast_result,
    write_stock_expansion_manifest,
)


def test_write_and_load_stock_expansion_manifest_round_trip(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    candidates = [
        StockExpansionCandidate("AVGO", priority=8, sector="electronic_technology"),
        StockExpansionCandidate("PANW", priority=7, sector="technology_services", side="both"),
    ]
    write_stock_expansion_manifest(
        manifest,
        base_stock_universe="stock19",
        default_checkpoint="binanceneural/checkpoints/demo.pt",
        candidates=candidates,
    )

    base_universe, default_checkpoint, loaded = load_stock_expansion_manifest(manifest)
    assert base_universe == "stock19"
    assert default_checkpoint == "binanceneural/checkpoints/demo.pt"
    assert [candidate.symbol for candidate in loaded] == ["AVGO", "PANW"]
    assert loaded[0].priority == 8
    assert loaded[1].side == "both"


def test_stock_expansion_candidate_normalized_accepts_aliases() -> None:
    candidate = StockExpansionCandidate("aal", side="long_short", priority=3).normalized()
    assert candidate.symbol == "AAL"
    assert candidate.side == "both"


def test_filter_candidates_with_hourly_data_separates_missing(tmp_path) -> None:
    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir(parents=True)
    (stocks_dir / "AVGO.csv").write_text("timestamp,open,high,low,close\n2026-03-17T00:00:00Z,1,1,1,1\n")

    ready, missing = filter_candidates_with_hourly_data(
        [
            StockExpansionCandidate("AVGO"),
            StockExpansionCandidate("APP"),
        ],
        data_root=tmp_path,
    )

    assert [candidate.symbol for candidate in ready] == ["AVGO"]
    assert [candidate.symbol for candidate in missing] == ["APP"]


def test_split_candidates_by_history_flags_insufficient_rows(tmp_path) -> None:
    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir(parents=True)
    (stocks_dir / "AVGO.csv").write_text(
        "timestamp,open,high,low,close\n"
        "2026-03-17T00:00:00Z,1,1,1,1\n"
        "2026-03-17T01:00:00Z,1,1,1,1\n"
    )

    ready, insufficient, missing = split_candidates_by_history(
        [
            StockExpansionCandidate("AVGO"),
            StockExpansionCandidate("APP"),
        ],
        data_root=tmp_path,
        min_history_rows=3,
    )

    assert ready == []
    assert [candidate.symbol for candidate in insufficient] == ["AVGO"]
    assert [candidate.symbol for candidate in missing] == ["APP"]
    assert count_candidate_history_rows("AVGO", data_root=tmp_path) == 2


def test_summarize_reforecast_result_extracts_flat_metrics() -> None:
    summary = {
        "best_mode": "baseline",
        "best_mode_metrics": {"total_return": 0.1, "sortino": 1.5, "max_drawdown": -0.02},
        "modes": [
            {
                "best_scenario": "short_NVDA",
                "scenarios": [
                    {
                        "scenario": "flat",
                        "metrics": {
                            "total_return": 0.03,
                            "sortino": 0.7,
                            "max_drawdown": -0.01,
                            "pnl_abs": 3.0,
                            "periods": 30,
                            "terminated_early": True,
                            "termination_reason": "gate",
                        },
                    }
                ],
            }
        ],
    }

    row = summarize_reforecast_result(summary)
    assert row["best_mode"] == "baseline"
    assert row["best_scenario"] == "short_NVDA"
    assert row["best_total_return"] == 0.1
    assert row["flat"]["sortino"] == 0.7
    assert row["flat"]["termination_reason"] == "gate"


def test_stock_expansion_sort_key_prefers_flat_sortino_then_return() -> None:
    better = {"flat": {"sortino": 1.2, "total_return": 0.01}, "best_sortino": 0.0, "best_total_return": 0.0}
    worse = {"flat": {"sortino": 1.1, "total_return": 0.5}, "best_sortino": 10.0, "best_total_return": 1.0}
    assert stock_expansion_sort_key(better) > stock_expansion_sort_key(worse)


def test_candidate_lora_command_contains_symbol_and_defaults() -> None:
    command = candidate_lora_command("avgo")
    assert "--symbol AVGO" in command
    assert "--context-length 1024" in command
    assert "--save-name-suffix stockexp" in command


def test_default_stock_expansion_candidates_include_new_tech_names() -> None:
    symbols = [candidate.symbol for candidate in default_stock_expansion_candidates()]
    assert "AVGO" in symbols
    assert "PANW" in symbols
    assert "NOW" in symbols
