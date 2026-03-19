"""Tests for reprompt/review machinery in providers.py and trade_binance_live.py wiring."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_hourly_trader.gemini_wrapper import TradePlan
from llm_hourly_trader.providers import (
    _build_reprompt_prompt,
    _plan_has_entry,
    _plan_is_actionable,
    _should_run_reprompt,
    _passes_reprompt_filters,
    call_llm,
)


# -- helpers --

def _make_hold_plan():
    return TradePlan("hold", 0.0, 0.0, 0.0, "no edge")


def _make_long_plan():
    return TradePlan("long", 100.0, 105.0, 0.8, "bullish momentum")


# -- unit tests for policy helpers --

class TestPlanHelpers:
    def test_plan_has_entry_true(self):
        assert _plan_has_entry(TradePlan("long", 100.0, 105.0, 0.8, "")) is True

    def test_plan_has_entry_false_hold(self):
        assert _plan_has_entry(TradePlan("hold", 0.0, 0.0, 0.0, "")) is False

    def test_plan_has_entry_false_sell_only(self):
        assert _plan_has_entry(TradePlan("hold", 0.0, 50.0, 0.5, "")) is False

    def test_plan_is_actionable_long(self):
        assert _plan_is_actionable(TradePlan("long", 100.0, 0.0, 0.5, "")) is True

    def test_plan_is_actionable_hold_with_sell(self):
        assert _plan_is_actionable(TradePlan("hold", 0.0, 50.0, 0.5, "")) is True

    def test_plan_is_actionable_pure_hold(self):
        assert _plan_is_actionable(TradePlan("hold", 0.0, 0.0, 0.0, "")) is False


class TestShouldRunReprompt:
    def test_always_policy(self):
        assert _should_run_reprompt(_make_hold_plan(), "always") is True
        assert _should_run_reprompt(_make_long_plan(), "always") is True

    def test_entry_only_policy_long(self):
        assert _should_run_reprompt(_make_long_plan(), "entry_only") is True

    def test_entry_only_policy_hold(self):
        assert _should_run_reprompt(_make_hold_plan(), "entry_only") is False

    def test_actionable_policy_hold(self):
        assert _should_run_reprompt(_make_hold_plan(), "actionable") is False

    def test_actionable_policy_sell_only(self):
        plan = TradePlan("hold", 0.0, 50.0, 0.5, "")
        assert _should_run_reprompt(plan, "actionable") is True

    def test_invalid_policy(self):
        with pytest.raises(ValueError, match="reprompt_policy"):
            _should_run_reprompt(_make_hold_plan(), "bogus")


class TestPassesRepromptFilters:
    def test_no_filter(self):
        assert _passes_reprompt_filters(_make_long_plan()) is True

    def test_max_confidence_blocks_high(self):
        plan = TradePlan("long", 100.0, 105.0, 0.9, "")
        assert _passes_reprompt_filters(plan, review_max_confidence=0.7) is False

    def test_max_confidence_allows_low(self):
        plan = TradePlan("long", 100.0, 105.0, 0.5, "")
        assert _passes_reprompt_filters(plan, review_max_confidence=0.7) is True


class TestBuildRepromptPrompt:
    def test_contains_original_and_prior(self):
        plan = _make_long_plan()
        result = _build_reprompt_prompt("original task text", plan, pass_index=2, total_passes=3)
        assert "original task text" in result
        assert "PRIOR PLAN JSON" in result
        assert '"direction": "long"' in result
        assert "review pass 2 of 3" in result

    def test_hold_plan_serialized(self):
        plan = _make_hold_plan()
        result = _build_reprompt_prompt("task", plan, pass_index=2, total_passes=2)
        assert '"direction": "hold"' in result


# -- integration tests for call_llm reprompt loop --

class TestCallLlmReprompt:
    @patch("llm_hourly_trader.providers.get_cached", return_value=None)
    @patch("llm_hourly_trader.providers.set_cached")
    def test_single_pass_unchanged(self, mock_set, mock_get):
        """Single pass (reprompt_passes=1) calls provider once, returns plan."""
        plan_data = {"direction": "long", "buy_price": 100, "sell_price": 105,
                     "confidence": 0.8, "reasoning": "test"}
        mock_get.side_effect = [None, plan_data]

        with patch("llm_hourly_trader.providers.PROVIDER_FNS", {
            "gemini": MagicMock(return_value=TradePlan(**plan_data)),
        }):
            result = call_llm("prompt", model="gemini-2.5-flash", reprompt_passes=1)
        assert result.direction == "long"
        assert result.buy_price == 100

    @patch("llm_hourly_trader.providers.get_cached", return_value=None)
    @patch("llm_hourly_trader.providers.set_cached")
    def test_two_pass_calls_twice(self, mock_set, mock_get):
        """2-pass reprompt calls provider twice."""
        call_count = []
        def fake_gemini(prompt, model=None, cache_model=None, provider_call_models=None, **kw):
            call_count.append(1)
            if len(call_count) == 1:
                return TradePlan("long", 100, 105, 0.6, "initial")
            return TradePlan("long", 99, 104, 0.75, "reviewed")

        with patch("llm_hourly_trader.providers.PROVIDER_FNS", {"gemini": fake_gemini}):
            result = call_llm(
                "prompt", model="gemini-2.5-flash",
                reprompt_passes=2, reprompt_policy="always",
            )
        assert len(call_count) == 2
        assert result.confidence == 0.75
        assert result.reasoning == "reviewed"

    @patch("llm_hourly_trader.providers.get_cached", return_value=None)
    @patch("llm_hourly_trader.providers.set_cached")
    def test_entry_only_skips_hold(self, mock_set, mock_get):
        """entry_only policy skips reprompt when first pass returns hold."""
        call_count = []
        def fake_gemini(prompt, model=None, cache_model=None, provider_call_models=None, **kw):
            call_count.append(1)
            return TradePlan("hold", 0, 0, 0, "no trade")

        with patch("llm_hourly_trader.providers.PROVIDER_FNS", {"gemini": fake_gemini}):
            result = call_llm(
                "prompt", model="gemini-2.5-flash",
                reprompt_passes=3, reprompt_policy="entry_only",
            )
        assert len(call_count) == 1
        assert result.direction == "hold"

    @patch("llm_hourly_trader.providers.get_cached", return_value=None)
    @patch("llm_hourly_trader.providers.set_cached")
    def test_review_model_used_on_pass2(self, mock_set, mock_get):
        """Review model is used for pass 2+ when specified."""
        models_called = []
        def fake_gemini(prompt, model=None, cache_model=None, provider_call_models=None, **kw):
            models_called.append(model)
            if len(models_called) == 1:
                return TradePlan("long", 100, 105, 0.6, "initial")
            return TradePlan("long", 99, 104, 0.8, "reviewed by pro")

        with patch("llm_hourly_trader.providers.PROVIDER_FNS", {"gemini": fake_gemini}):
            result = call_llm(
                "prompt", model="gemini-2.5-flash",
                reprompt_passes=2, reprompt_policy="always",
                review_model="gemini-2.5-pro",
            )
        assert models_called[0] == "gemini-2.5-flash"
        assert models_called[1] == "gemini-2.5-pro"
        assert result.reasoning == "reviewed by pro"

    @patch("llm_hourly_trader.providers.get_cached", return_value=None)
    @patch("llm_hourly_trader.providers.set_cached")
    def test_review_cache_namespace_separation(self, mock_set, mock_get):
        """Review responses use a separate cache namespace."""
        cache_models_used = []
        def fake_gemini(prompt, model=None, cache_model=None, provider_call_models=None, **kw):
            cache_models_used.append(cache_model)
            return TradePlan("long", 100, 105, 0.7, "plan")

        with patch("llm_hourly_trader.providers.PROVIDER_FNS", {"gemini": fake_gemini}):
            call_llm(
                "prompt", model="gemini-2.5-flash",
                reprompt_passes=2, reprompt_policy="always",
                review_model="gemini-2.5-pro",
                review_cache_namespace="review_ns",
            )
        assert cache_models_used[0] == "gemini-2.5-flash"
        assert "review_ns" in cache_models_used[1]

    def test_reprompt_passes_zero_raises(self):
        with pytest.raises(ValueError, match="reprompt_passes must be >= 1"):
            call_llm("prompt", model="gemini-2.5-flash", reprompt_passes=0)

    @patch("llm_hourly_trader.providers.get_cached", return_value=None)
    @patch("llm_hourly_trader.providers.set_cached")
    def test_three_pass_reprompt(self, mock_set, mock_get):
        """3-pass reprompt calls provider three times."""
        call_count = []
        def fake_gemini(prompt, model=None, cache_model=None, provider_call_models=None, **kw):
            call_count.append(1)
            return TradePlan("long", 100, 105, 0.5 + len(call_count) * 0.1, f"pass {len(call_count)}")

        with patch("llm_hourly_trader.providers.PROVIDER_FNS", {"gemini": fake_gemini}):
            result = call_llm(
                "prompt", model="gemini-2.5-flash",
                reprompt_passes=3, reprompt_policy="always",
            )
        assert len(call_count) == 3
        assert result.reasoning == "pass 3"


# -- CLI arg wiring tests --

class TestCLIArgWiring:
    def test_argparse_accepts_reprompt_args(self):
        """Verify argparse accepts --reprompt-passes, --review-model, --reprompt-policy."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--reprompt-passes", type=int, default=1)
        parser.add_argument("--review-model", type=str, default=None)
        parser.add_argument("--reprompt-policy", type=str, default="entry_only",
                            choices=["always", "entry_only", "actionable"])
        args = parser.parse_args(["--reprompt-passes", "2", "--review-model", "gemini-2.5-pro",
                                  "--reprompt-policy", "always"])
        assert args.reprompt_passes == 2
        assert args.review_model == "gemini-2.5-pro"
        assert args.reprompt_policy == "always"

    def test_default_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--reprompt-passes", type=int, default=1)
        parser.add_argument("--review-model", type=str, default=None)
        parser.add_argument("--reprompt-policy", type=str, default="entry_only")
        args = parser.parse_args([])
        assert args.reprompt_passes == 1
        assert args.review_model is None
        assert args.reprompt_policy == "entry_only"
