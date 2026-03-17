from __future__ import annotations

import pytest

from llm_hourly_trader import cache
from llm_hourly_trader.providers import (
    CacheMissError,
    PROVIDER_FNS,
    _build_reprompt_prompt,
    _gemini_timeout_ms,
    call_llm,
)
from llm_hourly_trader.gemini_wrapper import TradePlan


def test_call_llm_cache_only_returns_cached_plan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.set_cached(
        "gemini-3.1-flash-lite-preview",
        "prompt",
        {
            "direction": "long",
            "buy_price": 100.0,
            "sell_price": 101.0,
            "confidence": 0.7,
            "reasoning": "cached",
        },
    )

    def _provider_should_not_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("provider should not run in cache-only mode")

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _provider_should_not_run)

    plan = call_llm("prompt", model="gemini-3.1-flash-lite-preview", cache_only=True)
    assert plan.direction == "long"
    assert plan.buy_price == 100.0
    assert plan.sell_price == 101.0
    assert plan.confidence == 0.7
    assert plan.reasoning == "cached"


def test_call_llm_cache_only_raises_on_cache_miss(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    def _provider_should_not_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("provider should not run in cache-only mode")

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _provider_should_not_run)

    with pytest.raises(CacheMissError, match="No cached response"):
        call_llm("missing", model="gemini-3.1-flash-lite-preview", cache_only=True)


def test_call_llm_cache_only_reprompt_returns_final_cached_plan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    first_plan = TradePlan("long", 100.0, 104.0, 0.7, "first pass", 0.0)
    second_prompt = _build_reprompt_prompt(
        "prompt",
        first_plan,
        pass_index=2,
        total_passes=2,
    )
    cache.set_cached(
        "gemini-3.1-flash-lite-preview",
        "prompt",
        first_plan.__dict__,
    )
    cache.set_cached(
        "gemini-3.1-flash-lite-preview",
        second_prompt,
        {
            "direction": "hold",
            "buy_price": 0.0,
            "sell_price": 105.0,
            "confidence": 0.8,
            "reasoning": "second pass",
        },
    )

    def _provider_should_not_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("provider should not run in cache-only mode")

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _provider_should_not_run)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        cache_only=True,
        reprompt_passes=2,
    )

    assert plan.direction == "hold"
    assert plan.sell_price == 105.0
    assert plan.reasoning == "second pass"


def test_call_llm_reprompt_passes_reviews_prior_plan(monkeypatch) -> None:
    prompts: list[str] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        if provider_call_models is not None:
            provider_call_models.append(model)
        prompts.append(prompt)
        if len(prompts) == 1:
            return TradePlan("long", 100.0, 104.0, 0.7, "first pass", 0.0)
        return TradePlan("hold", 0.0, 105.0, 0.8, "second pass", 0.0)

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
    )

    assert len(prompts) == 2
    assert prompts[0] == "prompt"
    assert "ORIGINAL TASK:\nprompt" in prompts[1]
    assert '"buy_price": 100.0' in prompts[1]
    assert "review pass 2 of 2" in prompts[1]
    assert plan.direction == "hold"
    assert plan.reasoning == "second pass"


def test_call_llm_actionable_reprompt_policy_skips_flat_hold(monkeypatch) -> None:
    prompts: list[str] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        if provider_call_models is not None:
            provider_call_models.append(model)
        prompts.append(prompt)
        return TradePlan("hold", 0.0, 0.0, 0.4, "flat hold", 0.0)

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        reprompt_policy="actionable",
    )

    assert len(prompts) == 1
    assert plan.reasoning == "flat hold"


def test_call_llm_entry_only_reprompt_skips_exit_only_plan(monkeypatch) -> None:
    prompts: list[str] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        if provider_call_models is not None:
            provider_call_models.append(model)
        prompts.append(prompt)
        return TradePlan("hold", 0.0, 105.0, 0.6, "manage exit", 0.0)

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        reprompt_policy="entry_only",
    )

    assert len(prompts) == 1
    assert plan.reasoning == "manage exit"


def test_call_llm_review_max_confidence_skips_high_conf_entry(monkeypatch) -> None:
    prompts: list[str] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        if provider_call_models is not None:
            provider_call_models.append(model)
        prompts.append(prompt)
        return TradePlan("long", 100.0, 104.0, 0.7, "high confidence", 0.0)

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        reprompt_policy="entry_only",
        review_max_confidence=0.6,
    )

    assert len(prompts) == 1
    assert plan.reasoning == "high confidence"


def test_call_llm_review_model_and_call_trace(monkeypatch) -> None:
    seen_models: list[str] = []
    call_models: list[str] = []
    provider_models: list[str] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        if provider_call_models is not None:
            provider_call_models.append(model)
        seen_models.append(model)
        if len(seen_models) == 1:
            return TradePlan("long", 100.0, 104.0, 0.7, "first pass", 0.0)
        return TradePlan("hold", 0.0, 105.0, 0.8, "second pass", 0.0)

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        review_model="gemini-2.5-flash",
        call_models=call_models,
        provider_models=provider_models,
    )

    assert seen_models == ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash"]
    assert call_models == ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash"]
    assert provider_models == ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash"]
    assert plan.reasoning == "second pass"


def test_call_llm_review_thinking_level_overrides_second_pass(monkeypatch) -> None:
    seen_thinking_levels: list[str | None] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        if provider_call_models is not None:
            provider_call_models.append(model)
        seen_thinking_levels.append(thinking_level)
        if len(seen_thinking_levels) == 1:
            return TradePlan("long", 100.0, 104.0, 0.7, "first pass", 0.0)
        return TradePlan("hold", 0.0, 105.0, 0.8, "second pass", 0.0)

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    plan = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        review_thinking_level="LOW",
        reprompt_passes=2,
    )

    assert seen_thinking_levels == ["HIGH", "LOW"]
    assert plan.reasoning == "second pass"


def test_call_llm_review_cache_namespace_isolates_second_pass_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    first_plan = TradePlan("long", 100.0, 104.0, 0.7, "first pass", 0.0)
    call_cache_models: list[str | None] = []

    def _fake_gemini(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        cache_model: str | None = None,
        provider_call_models: list[str] | None = None,
    ):  # noqa: ANN202
        call_cache_models.append(cache_model)
        if provider_call_models is not None:
            provider_call_models.append(model)
        plan = first_plan if prompt == "prompt" else TradePlan("hold", 0.0, 105.0, 0.8, "second pass", 0.0)
        cache.set_cached(cache_model or model, prompt, plan.__dict__)
        return plan

    monkeypatch.setitem(PROVIDER_FNS, "gemini", _fake_gemini)

    first = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        review_cache_namespace="ns_a",
    )
    second = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        review_cache_namespace="ns_a",
    )
    third = call_llm(
        "prompt",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reprompt_passes=2,
        review_cache_namespace="ns_b",
    )

    assert first.reasoning == "second pass"
    assert second.reasoning == "second pass"
    assert third.reasoning == "second pass"
    assert call_cache_models == [
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-lite-preview::ns_a",
        "gemini-3.1-flash-lite-preview::ns_b",
    ]


def test_gemini_timeout_ms_defaults_and_disables(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_HTTP_TIMEOUT_MS", raising=False)
    assert _gemini_timeout_ms() == 120000

    monkeypatch.setenv("GEMINI_HTTP_TIMEOUT_MS", "0")
    assert _gemini_timeout_ms() is None

    monkeypatch.setenv("GEMINI_HTTP_TIMEOUT_MS", "bad")
    assert _gemini_timeout_ms() == 120000
