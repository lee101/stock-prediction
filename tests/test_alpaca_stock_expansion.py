from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.alpaca_stock_expansion import (
    StockExpansionCandidate,
    build_hourly_return_correlation_cohorts,
    candidate_hourly_tuning_command,
    candidate_lora_command,
    candidate_training_plan,
    count_candidate_history_rows,
    default_stock_expansion_candidates,
    filter_candidates_with_hourly_data,
    has_hourly_hyperparams,
    load_stock_expansion_manifest,
    manifest_side_defaults,
    resolve_hourly_hyperparams_path,
    split_candidates_by_history,
    stock_expansion_sort_key,
    summarize_reforecast_result,
    write_stock_expansion_manifest,
)


def _write_hourly_csv(path: Path, prices: list[float], start: str = "2026-01-01T00:00:00Z") -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(prices), freq="h", tz="UTC"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_hourly_csv_with_offset(path: Path, prices: list[float], start: str) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(prices), freq="h", tz="UTC"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


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


def test_manifest_side_defaults_respects_top_level_policy() -> None:
    defaults = manifest_side_defaults(
        {
            "side_policy": {
                "long_only_symbols": ["NVDA", "MU"],
                "evaluate_both_sides_symbols": ["F", "PFE"],
                "short_only_symbols": ["AAL"],
            }
        }
    )
    assert defaults["NVDA"] == "long"
    assert defaults["MU"] == "long"
    assert defaults["F"] == "both"
    assert defaults["PFE"] == "both"
    assert defaults["AAL"] == "short"


def test_load_stock_expansion_manifest_uses_top_level_side_policy_for_legacy_rows(tmp_path) -> None:
    manifest = tmp_path / "legacy_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "base_stock_universe": "live20260318",
                "default_checkpoint": "demo.pt",
                "side_policy": {
                    "long_only_symbols": ["MU"],
                    "evaluate_both_sides_symbols": ["F", "PFE"],
                },
                "candidates": [
                    {"symbol": "MU", "priority": 10},
                    {"symbol": "F", "priority": 9},
                    {"symbol": "PFE", "priority": 8},
                ],
            }
        )
    )

    _base_universe, _checkpoint, loaded = load_stock_expansion_manifest(manifest)
    assert [candidate.side for candidate in loaded] == ["long", "both", "both"]


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


def test_candidate_lora_command_adds_covariates_when_present() -> None:
    command = candidate_lora_command("avgo", covariate_symbols=["msft", "goog"])
    assert "--covariate-symbols MSFT,GOOG" in command
    assert "--covariate-cols close" in command


def test_candidate_hourly_tuning_command_contains_quick_tuning_flags() -> None:
    command = candidate_hourly_tuning_command("ttd")
    assert "--symbols TTD" in command
    assert "--quick" in command
    assert "--save-hyperparams" in command


def test_candidate_training_plan_prefers_hourly_tune_without_config(tmp_path) -> None:
    _write_hourly_csv(tmp_path / "stocks" / "AAA.csv", [100.0 + i for i in range(96)])
    _write_hourly_csv(tmp_path / "stocks" / "BBB.csv", [101.0 + i * 1.01 for i in range(96)])
    _write_hourly_csv(tmp_path / "stocks" / "ZZZ.csv", [50.0 + i * 0.1 for i in range(32)], start="2025-12-01T00:00:00Z")

    cohorts = build_hourly_return_correlation_cohorts(
        ["AAA", "BBB", "ZZZ"],
        data_root=tmp_path,
        lookback_hours=96,
        min_periods=24,
        max_size=2,
        min_abs_corr=0.2,
    )
    assert cohorts["AAA"][:1] == ("BBB",)

    plan = candidate_training_plan(
        "AAA",
        comparison_symbols=["BBB", "ZZZ"],
        data_root=tmp_path,
        hyperparam_root=tmp_path / "hyperparams",
        correlation_cohorts=cohorts,
        max_cohort_size=2,
        min_abs_corr=0.2,
    )

    assert plan["hourly_config_exists"] is False
    assert plan["recommended_next_step"] == "hourly_tune"
    assert plan["recommended_peer_symbols"] == ["BBB"]
    assert "--covariate-symbols BBB" in str(plan["lora_command"])


def test_build_hourly_return_correlation_cohorts_normalizes_hour_offsets(tmp_path) -> None:
    _write_hourly_csv(tmp_path / "stocks" / "AAA.csv", [100.0 + i for i in range(72)])
    _write_hourly_csv_with_offset(
        tmp_path / "stocks" / "BBB.csv",
        [100.5 + i * 1.01 for i in range(72)],
        start="2026-01-01T00:30:00Z",
    )

    cohorts = build_hourly_return_correlation_cohorts(
        ["AAA", "BBB"],
        data_root=tmp_path,
        lookback_hours=72,
        min_periods=24,
        max_size=1,
        min_abs_corr=0.2,
    )

    assert cohorts["AAA"] == ("BBB",)


def test_candidate_training_plan_prefers_multivariate_lora_with_existing_config(tmp_path) -> None:
    _write_hourly_csv(tmp_path / "stocks" / "AAA.csv", [100.0 + i for i in range(96)])
    _write_hourly_csv(tmp_path / "stocks" / "BBB.csv", [100.5 + i * 1.02 for i in range(96)])
    hyperparam_path = resolve_hourly_hyperparams_path("AAA", hyperparam_root=tmp_path / "hyperparams")
    hyperparam_path.parent.mkdir(parents=True, exist_ok=True)
    hyperparam_path.write_text(
        json.dumps(
            {
                "symbol": "AAA",
                "config": {
                    "context_length": 2048,
                },
            }
        )
    )

    assert has_hourly_hyperparams("AAA", hyperparam_root=tmp_path / "hyperparams") is True
    plan = candidate_training_plan(
        "AAA",
        comparison_symbols=["BBB"],
        data_root=tmp_path,
        hyperparam_root=tmp_path / "hyperparams",
        min_periods=24,
        max_cohort_size=2,
        min_abs_corr=0.2,
    )

    assert plan["recommended_next_step"] == "multivariate_lora"
    assert "--context-length 2048" in str(plan["lora_command"])
    assert "--covariate-symbols BBB" in str(plan["lora_command"])


def test_default_stock_expansion_candidates_include_new_tech_names() -> None:
    symbols = [candidate.symbol for candidate in default_stock_expansion_candidates()]
    assert "AVGO" in symbols
    assert "PANW" in symbols
    assert "NOW" in symbols
