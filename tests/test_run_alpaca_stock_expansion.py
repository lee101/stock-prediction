from __future__ import annotations

from scripts.run_alpaca_stock_expansion import (
    _assess_promotion,
    _build_promotion_summary,
    _candidate_direction_lists,
    _resolve_base_symbols,
    _resolve_cache_symbols,
)
from src.alpaca_stock_expansion import StockExpansionCandidate


def test_resolve_base_symbols_supports_live_alias() -> None:
    symbols = _resolve_base_symbols("live20260318")
    assert symbols[0] == "NVDA"
    assert "MSFT" in symbols
    assert "EXPE" in symbols


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
