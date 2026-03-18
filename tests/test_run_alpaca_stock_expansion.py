from __future__ import annotations

from scripts.run_alpaca_stock_expansion import _candidate_direction_lists, _resolve_base_symbols
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
