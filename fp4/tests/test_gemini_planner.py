"""Tests for the Gemini re-planner — stub path only by default.

Real API is only hit if the user passes ``--live-llm`` AND GEMINI_API_KEY is
set. CI / marketsim runs never bill.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from fp4.gemini_planner import GeminiPlanner, _stub_plan, _validate_plan
from fp4.planner_cache import PlannerCache


def pytest_addoption(parser):  # pragma: no cover - only used if conftest picks it up
    parser.addoption("--live-llm", action="store_true", default=False)


@pytest.fixture
def tmp_cache(tmp_path: Path) -> PlannerCache:
    return PlannerCache(tmp_path / "cache.sqlite")


def test_stub_plan_shape():
    plan = _stub_plan(["BTC", "ETH"])
    assert plan["universe"] == ["BTC", "ETH"]
    assert set(plan["hints"].keys()) == {"BTC", "ETH"}
    for h in plan["hints"].values():
        assert set(h.keys()) == {"dir", "size_frac", "max_lev", "confidence"}


def test_validate_plan_clamps_and_rejects():
    good = {
        "universe": ["BTC"],
        "hints": {"BTC": {"dir": 2.5, "size_frac": -0.3, "max_lev": 99.0, "confidence": 0.5}},
    }
    v = _validate_plan(good, ["BTC"])
    assert v["hints"]["BTC"]["dir"] == 1.0
    assert v["hints"]["BTC"]["size_frac"] == 0.0
    assert v["hints"]["BTC"]["max_lev"] == 5.0

    with pytest.raises(ValueError):
        _validate_plan({"universe": "BTC", "hints": {}}, ["BTC"])
    with pytest.raises(ValueError):
        _validate_plan({"universe": ["BTC"], "hints": []}, ["BTC"])


def test_planner_defaults_to_stub_without_api_key(tmp_cache, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = GeminiPlanner(api_key=None, universe=["BTC", "ETH"], cache=tmp_cache)
    assert not p.is_live
    plan = p.plan_for_day("2026-04-07", {"cash": 10000}, {"spx": 0.01})
    assert plan["universe"] == ["BTC", "ETH"]
    assert "BTC" in plan["hints"]


def test_cache_hit_skips_compute(tmp_cache, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = GeminiPlanner(api_key=None, universe=["BTC"], cache=tmp_cache)

    calls = {"n": 0}
    real_goc = tmp_cache.get_or_compute

    def counting_goc(date_iso, portfolio_state, universe, compute_fn):
        def wrapped():
            calls["n"] += 1
            return compute_fn()
        return real_goc(date_iso, portfolio_state, universe, wrapped)

    tmp_cache.get_or_compute = counting_goc  # type: ignore

    state = {"cash": 10000, "pos": {}}
    summary = {"spx": 0.01}
    p1 = p.plan_for_day("2026-04-07", state, summary)
    p2 = p.plan_for_day("2026-04-07", state, summary)
    assert p1 == p2
    assert calls["n"] == 1  # second call served from cache
    assert tmp_cache.hits == 1
    assert tmp_cache.misses == 1


def test_cache_key_changes_on_state_change(tmp_cache, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = GeminiPlanner(api_key=None, universe=["BTC"], cache=tmp_cache)
    p.plan_for_day("2026-04-07", {"cash": 10000}, {})
    p.plan_for_day("2026-04-07", {"cash": 20000}, {})
    assert tmp_cache.misses == 2
    assert tmp_cache.hits == 0


def test_fallback_plan_applied(tmp_cache, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fallback = {"BTC": {"dir": 0.7, "size_frac": 0.3, "max_lev": 2.0, "confidence": 0.8}}
    p = GeminiPlanner(
        api_key=None, universe=["BTC"], cache=tmp_cache, fallback_plan=fallback
    )
    plan = p.plan_for_day("2026-04-07", {}, {})
    assert plan["hints"]["BTC"]["dir"] == 0.7
    assert plan["hints"]["BTC"]["max_lev"] == 2.0


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") or os.environ.get("FP4_LIVE_LLM") != "1",
    reason="live Gemini API disabled (set GEMINI_API_KEY and FP4_LIVE_LLM=1)",
)
def test_live_gemini_smoke():  # pragma: no cover
    p = GeminiPlanner(universe=["BTC", "ETH"])
    plan = p.plan_for_day("2026-04-07", {"cash": 10000}, {"spx": 0.01})
    assert "BTC" in plan["hints"]
