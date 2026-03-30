"""Tests for hybrid_prompt allocation response parsing.

Covers: parse_allocation_response, _coerce_allocation_number, _alloc_fields_for_symbol.
These handle real Gemini LLM output and are critical for trade decisions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl-trading-agent-binance"))

from hybrid_prompt import (
    AllocationPlan,
    _alloc_fields_for_symbol,
    _coerce_allocation_number,
    parse_allocation_response,
)


class TestCoerceAllocationNumber:
    def test_none_returns_zero(self):
        assert _coerce_allocation_number(None) == 0.0

    def test_int_returns_float(self):
        assert _coerce_allocation_number(25) == 25.0

    def test_float_passthrough(self):
        assert _coerce_allocation_number(12.5) == 12.5

    def test_empty_string_returns_zero(self):
        assert _coerce_allocation_number("") == 0.0
        assert _coerce_allocation_number("  ") == 0.0

    def test_na_variants_return_zero(self):
        for val in ["N/A", "na", "None", "null", "NA"]:
            assert _coerce_allocation_number(val) == 0.0, f"Failed for {val!r}"

    def test_strips_dollar_sign(self):
        assert _coerce_allocation_number("$100.50") == 100.50

    def test_strips_percent_sign(self):
        assert _coerce_allocation_number("25%") == 25.0

    def test_strips_commas(self):
        assert _coerce_allocation_number("1,234.56") == 1234.56

    def test_garbage_returns_zero(self):
        assert _coerce_allocation_number("not_a_number") == 0.0

    def test_negative_number(self):
        assert _coerce_allocation_number("-5.0") == -5.0

    def test_zero_string(self):
        assert _coerce_allocation_number("0") == 0.0


class TestAllocFieldsForSymbol:
    def test_btcusd(self):
        assert _alloc_fields_for_symbol("BTCUSD") == ("btc_pct", "btc_entry", "btc_exit")

    def test_ethusd(self):
        assert _alloc_fields_for_symbol("ETHUSD") == ("eth_pct", "eth_entry", "eth_exit")

    def test_dogeusd(self):
        assert _alloc_fields_for_symbol("DOGEUSD") == ("doge_pct", "doge_entry", "doge_exit")

    def test_unknown_symbol_uses_full_prefix(self):
        pct, entry, exit_ = _alloc_fields_for_symbol("FOOUSD")
        assert pct == "foo_pct"


class TestParseAllocationResponse:
    def test_valid_json_with_allocations(self):
        response = json.dumps({
            "btc_pct": "25",
            "btc_entry": "82000",
            "btc_exit": "85000",
            "eth_pct": "15",
            "eth_entry": "1900",
            "eth_exit": "2100",
            "reasoning": "Bullish on BTC and ETH",
        })
        plan = parse_allocation_response(response)
        assert plan.allocations["BTCUSD"] == 25.0
        assert plan.allocations["ETHUSD"] == 15.0
        assert plan.entry_prices["BTCUSD"] == 82000.0
        assert plan.exit_prices["ETHUSD"] == 2100.0
        assert "Bullish" in plan.reasoning

    def test_zero_allocation_excluded(self):
        response = json.dumps({
            "btc_pct": "0",
            "eth_pct": "30",
            "reasoning": "Only ETH",
        })
        plan = parse_allocation_response(response)
        assert "BTCUSD" not in plan.allocations
        assert plan.allocations["ETHUSD"] == 30.0

    def test_total_over_100_scales_down(self):
        response = json.dumps({
            "btc_pct": "60",
            "eth_pct": "60",
            "reasoning": "Over-allocated",
        })
        plan = parse_allocation_response(response)
        total = sum(plan.allocations.values())
        assert total == pytest.approx(100.0)
        assert plan.allocations["BTCUSD"] == pytest.approx(50.0)
        assert plan.allocations["ETHUSD"] == pytest.approx(50.0)

    def test_json_in_markdown_code_block(self):
        response = 'Some preamble\n```json\n{"btc_pct": "20", "reasoning": "test"}\n```\n'
        plan = parse_allocation_response(response)
        assert plan.allocations["BTCUSD"] == 20.0

    def test_json_embedded_in_text(self):
        response = 'Here is my plan: {"btc_pct": "15", "reasoning": "embedded"} end'
        plan = parse_allocation_response(response)
        assert plan.allocations["BTCUSD"] == 15.0

    def test_completely_unparseable_returns_empty_plan(self):
        plan = parse_allocation_response("This is just natural language with no JSON")
        assert plan.allocations == {}
        assert plan.reasoning != ""

    def test_empty_json_returns_empty_allocations(self):
        plan = parse_allocation_response('{"reasoning": "Nothing to trade"}')
        assert plan.allocations == {}
        assert plan.reasoning == "Nothing to trade"

    def test_na_values_treated_as_zero(self):
        response = json.dumps({
            "btc_pct": "N/A",
            "eth_pct": "20",
            "reasoning": "BTC is N/A",
        })
        plan = parse_allocation_response(response)
        assert "BTCUSD" not in plan.allocations
        assert plan.allocations["ETHUSD"] == 20.0

    def test_allocation_capped_at_100(self):
        response = json.dumps({
            "btc_pct": "150",
            "reasoning": "Over-confident",
        })
        plan = parse_allocation_response(response)
        assert plan.allocations["BTCUSD"] == 100.0

    def test_unknown_symbol_pct_keys_parsed(self):
        response = json.dumps({
            "zec_pct": "10",
            "zec_entry": "230",
            "zec_exit": "280",
            "reasoning": "ZEC dip buy",
        })
        plan = parse_allocation_response(response)
        assert plan.allocations["ZECUSD"] == 10.0
        assert plan.entry_prices["ZECUSD"] == 230.0

    def test_dollar_sign_in_entry_prices(self):
        response = json.dumps({
            "btc_pct": "25",
            "btc_entry": "$82,000",
            "btc_exit": "$85,000.50",
            "reasoning": "Dollar formatted",
        })
        plan = parse_allocation_response(response)
        assert plan.entry_prices["BTCUSD"] == 82000.0
        assert plan.exit_prices["BTCUSD"] == 85000.50

    def test_allocation_plan_dataclass_defaults(self):
        plan = AllocationPlan()
        assert plan.allocations == {}
        assert plan.entry_prices == {}
        assert plan.exit_prices == {}
        assert plan.reasoning == ""
