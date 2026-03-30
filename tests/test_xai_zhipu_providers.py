"""Tests for xAI Grok and Zhipu GLM provider integration."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.providers import (
    call_xai,
    call_xai_openai,
    call_zhipu,
    _resolve_provider_and_model,
    PROVIDER_FNS,
    MODEL_PROVIDERS,
    _normalize_confidence,
)
from llm_hourly_trader.gemini_wrapper import TradePlan


class TestProviderRegistration:
    """Ensure xAI and Zhipu are properly registered."""

    def test_xai_in_provider_fns(self):
        assert "xai" in PROVIDER_FNS
        assert "xai_openai" in PROVIDER_FNS

    def test_zhipu_in_provider_fns(self):
        assert "zhipu" in PROVIDER_FNS

    def test_grok_models_registered(self):
        assert MODEL_PROVIDERS["grok-4-1-fast"] == "xai"
        assert MODEL_PROVIDERS["grok-4.20-multi-agent-0309"] == "xai"
        assert MODEL_PROVIDERS["grok-4.20-reasoning"] == "xai"

    def test_glm_models_registered(self):
        assert MODEL_PROVIDERS["glm-5.1"] == "zhipu"
        assert MODEL_PROVIDERS["glm-4-plus"] == "zhipu"

    def test_resolve_grok_auto(self):
        provider, model = _resolve_provider_and_model("grok-4-1-fast")
        assert provider == "xai"
        assert model == "grok-4-1-fast"

    def test_resolve_grok_unknown_auto(self):
        provider, model = _resolve_provider_and_model("grok-99-turbo")
        assert provider == "xai"

    def test_resolve_glm_auto(self):
        provider, model = _resolve_provider_and_model("glm-5.1")
        assert provider == "zhipu"

    def test_resolve_glm_unknown_auto(self):
        provider, model = _resolve_provider_and_model("glm-6-preview")
        assert provider == "zhipu"


class TestXAIProvider:
    """Test call_xai with mocked xai_sdk."""

    def test_returns_hold_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            plan = call_xai("test prompt", max_retries=1)
            assert plan.direction == "hold"
            assert "XAI_API_KEY" in plan.reasoning

    def test_cache_hit(self, tmp_path):
        cached = {
            "direction": "long",
            "buy_price": 100.0,
            "sell_price": 105.0,
            "confidence": 0.8,
            "reasoning": "cached",
        }
        with patch("llm_hourly_trader.providers.get_cached", return_value=cached):
            plan = call_xai("test prompt")
            assert plan.direction == "long"
            assert plan.buy_price == 100.0

    def test_tradeplan_fields(self):
        """TradePlan should have allocation_pct field."""
        plan = TradePlan("long", 100, 105, 0.8, "test", allocation_pct=50.0)
        assert plan.allocation_pct == 50.0


class TestXAIOpenAIProvider:
    """Test call_xai_openai fallback."""

    def test_returns_hold_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            plan = call_xai_openai("test prompt", max_retries=1)
            assert plan.direction == "hold"


class TestZhipuProvider:
    """Test call_zhipu with mocked zhipuai."""

    def test_returns_hold_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZHIPU_API_KEY", None)
            plan = call_zhipu("test prompt", max_retries=1)
            assert plan.direction == "hold"
            assert "ZHIPU_API_KEY" in plan.reasoning

    def test_cache_hit(self):
        cached = {
            "direction": "short",
            "buy_price": 0.0,
            "sell_price": 95.0,
            "confidence": 0.6,
            "reasoning": "cached glm",
        }
        with patch("llm_hourly_trader.providers.get_cached", return_value=cached):
            plan = call_zhipu("test prompt")
            assert plan.direction == "short"
            assert plan.sell_price == 95.0


class TestNormalizeConfidence:
    """Confidence normalization edge cases."""

    def test_normal_range(self):
        assert _normalize_confidence(0.75) == 0.75

    def test_percentage_to_decimal(self):
        assert abs(_normalize_confidence(85) - 0.85) < 0.01

    def test_string_input(self):
        assert abs(_normalize_confidence("0.65") - 0.65) < 0.01

    def test_none_input(self):
        assert _normalize_confidence(None) == 0.0  # None → float(0) = 0.0

    def test_percentage_85(self):
        # 85 > 1.0 → 85/100 = 0.85
        assert abs(_normalize_confidence(85) - 0.85) < 0.01

    def test_clamp_high(self):
        # 1.5 > 1.0 → 1.5/100 = 0.015
        assert abs(_normalize_confidence(1.5) - 0.015) < 0.001

    def test_clamp_zero(self):
        assert _normalize_confidence(0) == 0.0


@pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="XAI_API_KEY not set")
class TestXAILive:
    """Live API tests — only run when key is available."""

    def test_grok_fast_simple_trade(self):
        plan = call_xai(
            "BTC at $87,000. Up 2% in 12h. SMA above. Should I go long or hold?",
            model="grok-4-1-fast",
            max_retries=2,
        )
        assert plan.direction in ("long", "short", "hold")
        assert 0 <= plan.confidence <= 1
        assert isinstance(plan.reasoning, str)


@pytest.mark.skipif(not os.environ.get("ZHIPU_API_KEY"), reason="ZHIPU_API_KEY not set")
class TestZhipuLive:
    """Live API tests — only run when key is available."""

    def test_glm_simple_trade(self):
        plan = call_zhipu(
            "BTC at $87,000. Up 2% in 12h. Long or hold?",
            model="glm-4-plus",
            max_retries=2,
        )
        assert plan.direction in ("long", "short", "hold")
        assert 0 <= plan.confidence <= 1
