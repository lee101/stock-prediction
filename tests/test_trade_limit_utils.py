import pytest

from scripts.trade_limit_utils import (
    entry_limit_to_trade_limit,
    parse_entry_limit_map,
    resolve_entry_limit,
)


def test_parse_entry_limit_map_supports_symbol_and_strategy():
    raw = "NVDA@maxdiff:2,AAPL:3,GENERIC@momentum:4"
    parsed = parse_entry_limit_map(raw)
    assert parsed[("nvda", "maxdiff")] == 2
    assert parsed[("aapl", None)] == 3
    assert parsed[("generic", "momentum")] == 4


def test_resolve_entry_limit_falls_back_to_symbol_only():
    parsed = parse_entry_limit_map("AAPL:3,MAXDIFF:5")
    assert resolve_entry_limit(parsed, "AAPL", "maxdiff") == 3
    assert resolve_entry_limit(parsed, "MAXDIFF", "maxdiff") == 5
    assert resolve_entry_limit(parsed, "MSFT", "unknown") is None


def test_entry_limit_to_trade_limit_converts_entries():
    assert entry_limit_to_trade_limit(3) == pytest.approx(6.0)
    assert entry_limit_to_trade_limit(None) is None
    assert entry_limit_to_trade_limit(0) == 0.0
