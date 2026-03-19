from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_alpaca_stock_expansion import (
    _assess_promotion,
    _build_promotion_summary,
    _candidate_mae_gate_reason,
    _candidate_mae_snapshot,
    _load_forecast_cache_mae,
    _resolve_baseline_metrics_path,
    _resolve_forecast_cache_mae_path,
    _candidate_direction_lists,
    _resolve_base_symbols,
    _resolve_cache_symbols,
    _skipped_result_row,
)
from src.alpaca_stock_expansion import StockExpansionCandidate


def test_resolve_base_symbols_supports_live_alias() -> None:
    symbols = _resolve_base_symbols("live20260318")
    assert symbols[0] == "NVDA"
    assert "MSFT" in symbols
    assert "EXPE" in symbols
    assert "ITUB" not in symbols


def test_resolve_base_symbols_supports_current_live_alias() -> None:
    symbols = _resolve_base_symbols("live20260319")
    assert symbols[0] == "NVDA"
    assert "EXPE" in symbols
    assert symbols[-1] == "ITUB"
    assert "BTG" not in symbols


def test_resolve_base_symbols_supports_post_btg_live_alias() -> None:
    symbols = _resolve_base_symbols("live20260319_post_btg")
    assert symbols[0] == "NVDA"
    assert "ITUB" in symbols
    assert symbols[-1] == "BTG"
    assert "ABEV" not in symbols


def test_resolve_base_symbols_supports_post_abev_live_alias() -> None:
    symbols = _resolve_base_symbols("live20260319_post_abev")
    assert symbols[0] == "NVDA"
    assert "BTG" in symbols
    assert symbols[-1] == "ABEV"


def test_candidate_direction_lists_append_only_the_candidate_side() -> None:
    long_only, short_only = _candidate_direction_lists(
        StockExpansionCandidate("PFE", side="long"),
        base_long_only_symbols=["NVDA"],
        base_short_only_symbols=["AAL"],
    )
    assert long_only == ["NVDA", "PFE"]
    assert short_only == ["AAL"]

    long_only, short_only = _candidate_direction_lists(
        StockExpansionCandidate("F", side="short"),
        base_long_only_symbols=["NVDA"],
        base_short_only_symbols=["AAL"],
    )
    assert long_only == ["NVDA"]
    assert short_only == ["AAL", "F"]

    long_only, short_only = _candidate_direction_lists(
        StockExpansionCandidate("BTG", side="both"),
        base_long_only_symbols=["NVDA"],
        base_short_only_symbols=["AAL"],
    )
    assert long_only == ["NVDA"]
    assert short_only == ["AAL"]


def test_resolve_cache_symbols_supports_candidate_only_mode() -> None:
    ready = [StockExpansionCandidate("PFE"), StockExpansionCandidate("F")]
    assert _resolve_cache_symbols(
        base_symbols=["NVDA", "MSFT"],
        ready_candidates=ready,
        candidate_only_cache_build=False,
    ) == ["NVDA", "MSFT", "PFE", "F"]
    assert _resolve_cache_symbols(
        base_symbols=["NVDA", "MSFT"],
        ready_candidates=ready,
        candidate_only_cache_build=True,
    ) == ["PFE", "F"]


def test_resolve_baseline_metrics_path_supports_external_baseline_source(tmp_path) -> None:
    output_dir = tmp_path / "trial"
    external = tmp_path / "baseline_source"
    assert _resolve_baseline_metrics_path(
        output_dir=output_dir,
        baseline_source_dir=None,
    ) == output_dir / "baseline" / "metrics.json"
    assert _resolve_baseline_metrics_path(
        output_dir=output_dir,
        baseline_source_dir=external,
    ) == external / "metrics.json"


def test_assess_promotion_requires_non_regressing_candidate() -> None:
    promotable = _assess_promotion(
        {
            "symbol": "PFE",
            "total_return_delta_vs_baseline": 0.02,
            "sortino_delta_vs_baseline": 0.4,
            "max_drawdown_delta_vs_baseline": -0.01,
            "terminated_early": False,
        },
        min_return_delta=0.0,
        min_sortino_delta=0.0,
        max_drawdown_delta=0.0,
    )
    assert promotable["promotable"] is True

    rejected = _assess_promotion(
        {
            "symbol": "SOFI",
            "total_return_delta_vs_baseline": 0.001,
            "sortino_delta_vs_baseline": -0.7,
            "max_drawdown_delta_vs_baseline": -0.0008,
            "terminated_early": False,
        },
        min_return_delta=0.0,
        min_sortino_delta=0.0,
        max_drawdown_delta=0.0,
    )
    assert rejected["promotable"] is False
    assert "sortino delta" in str(rejected["promotion_reason"])


def test_assess_promotion_rejects_early_terminated_candidate() -> None:
    rejected = _assess_promotion(
        {
            "symbol": "TTD",
            "total_return_delta_vs_baseline": 0.03,
            "sortino_delta_vs_baseline": 0.2,
            "max_drawdown_delta_vs_baseline": -0.002,
            "terminated_early": True,
            "termination_reason": "baseline gate: sortino too weak",
        },
        min_return_delta=0.0,
        min_sortino_delta=0.0,
        max_drawdown_delta=0.0,
    )
    assert rejected["promotable"] is False
    assert "baseline gate" in str(rejected["promotion_reason"])


def test_load_forecast_cache_mae_and_snapshot(tmp_path) -> None:
    path = tmp_path / "forecast_cache_mae.json"
    path.write_text(
        """
{
  "metrics": [
    {"symbol": "MU", "horizon_hours": 1, "mae_percent": 0.8},
    {"symbol": "MU", "horizon_hours": 24, "mae_percent": 15.2},
    {"symbol": "PFE", "horizon_hours": 24, "mae_percent": 2.2}
  ]
}
""".strip()
    )
    payload = _load_forecast_cache_mae(path)
    assert payload["MU"][1] == 0.8
    assert payload["MU"][24] == 15.2
    assert _candidate_mae_snapshot(payload, "MU") == {
        "candidate_mae_h1_percent": 0.8,
        "candidate_mae_h24_percent": 15.2,
    }


def test_resolve_forecast_cache_mae_path_reuses_external_source(tmp_path) -> None:
    output_dir = tmp_path / "trial"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "forecast_cache_mae.json"
    source_path.write_text(
        """
{
  "metrics": [
    {"symbol": "F", "horizon_hours": 1, "mae_percent": 4.0},
    {"symbol": "F", "horizon_hours": 24, "mae_percent": 4.8}
  ]
}
""".strip()
    )
    resolved = _resolve_forecast_cache_mae_path(
        output_dir=output_dir,
        skip_cache_build=True,
        forecast_cache_mae_source=source_dir,
        require_existing_summary=True,
    )
    assert resolved == output_dir / "forecast_cache_mae.json"
    assert resolved.read_text() == source_path.read_text()


def test_resolve_forecast_cache_mae_path_requires_summary_when_skipping_with_gate(tmp_path) -> None:
    with pytest.raises(SystemExit, match="Missing forecast cache MAE summary"):
        _resolve_forecast_cache_mae_path(
            output_dir=tmp_path / "trial",
            skip_cache_build=True,
            forecast_cache_mae_source=None,
            require_existing_summary=True,
        )


def test_candidate_mae_gate_reason_blocks_bad_cache_quality() -> None:
    reason = _candidate_mae_gate_reason(
        symbol="MU",
        mae_snapshot={"candidate_mae_h1_percent": 0.8, "candidate_mae_h24_percent": 15.2},
        max_h1_mae_percent=0.0,
        max_h24_mae_percent=10.0,
    )
    assert "h24 MAE 15.2000%" in reason

    ok = _candidate_mae_gate_reason(
        symbol="PFE",
        mae_snapshot={"candidate_mae_h1_percent": 1.9, "candidate_mae_h24_percent": 2.2},
        max_h1_mae_percent=0.0,
        max_h24_mae_percent=10.0,
    )
    assert ok == ""


def test_skipped_result_row_marks_candidate_as_not_run() -> None:
    row = _skipped_result_row(
        candidate=StockExpansionCandidate("MU", side="long", priority=88, thesis="memory"),
        baseline_metrics={"total_return": -0.04, "sortino": -6.7, "max_drawdown": 0.04},
        summary_path=Path("/tmp/cache_gate.json"),
        mae_snapshot={"candidate_mae_h1_percent": 0.8, "candidate_mae_h24_percent": 15.2},
        gate_reason="forecast cache gate for MU: h24 MAE 15.2000% exceeds gate 10.0000%",
    )
    assert row["sim_ran"] is False
    assert row["terminated_early"] is True
    assert row["candidate_mae_h24_percent"] == 15.2
    assert "forecast cache gate" in str(row["termination_reason"])


def test_build_promotion_summary_selects_first_promotable_candidate() -> None:
    rows = [
        {
            "symbol": "BASE",
            "promotable": False,
            "promotion_reason": "Baseline row is not a candidate for promotion.",
        },
        {
            "symbol": "PFE",
            "promotable": True,
            "promotion_reason": "return delta 0.020000, sortino delta 0.4000, max drawdown delta -0.010000",
        },
        {
            "symbol": "F",
            "promotable": True,
            "promotion_reason": "return delta 0.010000, sortino delta 0.1000, max drawdown delta -0.005000",
        },
    ]
    summary = _build_promotion_summary(
        rows,
        min_return_delta=0.0,
        min_sortino_delta=0.0,
        max_drawdown_delta=0.0,
    )
    assert summary["promote"] is True
    assert summary["promoted_symbol"] == "PFE"
    assert summary["promotable_candidates"] == ["PFE", "F"]
