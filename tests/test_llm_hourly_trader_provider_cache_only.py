from __future__ import annotations

import pytest

from llm_hourly_trader import cache
from llm_hourly_trader.providers import CacheMissError, PROVIDER_FNS, call_llm


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
