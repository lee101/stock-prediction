"""Gemini 3.x daily re-planner for the fp4 RL trading stack.

Default OFF: the trainer only calls this when cfg["gemini_replanner"] is True
AND GEMINI_API_KEY is set in the environment. Otherwise the stub path returns
a deterministic plan derived from cfg["fallback_plan"] so tests and marketsim
runs never hit the real API.

Mirrors the client style from ``llm_hourly_trader/gemini_wrapper.py`` which
uses ``from google import genai; client = genai.Client(api_key=...)``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .planner_cache import PlannerCache

try:  # pragma: no cover - import guarded
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

log = logging.getLogger(__name__)

DEFAULT_UNIVERSE = ["BTC", "ETH", "SOL", "SPY", "QQQ", "NVDA", "TSLA", "MSFT"]
DEFAULT_MODEL = "gemini-3-pro"
MAX_RETRIES = 3


def _default_hint() -> dict:
    return {"dir": 0.0, "size_frac": 0.0, "max_lev": 1.0, "confidence": 0.0}


def _stub_plan(universe: list[str], fallback: dict | None = None) -> dict:
    fallback = fallback or {}
    hints: dict[str, dict] = {}
    for t in universe:
        h = _default_hint()
        h.update(fallback.get(t, {}))
        hints[t] = h
    return {"universe": list(universe), "hints": hints}


def _validate_plan(raw: Any, universe: list[str]) -> dict:
    """Strict JSON schema enforcement: raises ValueError on mismatch."""
    if not isinstance(raw, dict):
        raise ValueError("plan must be a dict")
    uni = raw.get("universe")
    hints = raw.get("hints")
    if not isinstance(uni, list) or not all(isinstance(t, str) for t in uni):
        raise ValueError("plan.universe must be list[str]")
    if not isinstance(hints, dict):
        raise ValueError("plan.hints must be dict")
    out_hints: dict[str, dict] = {}
    for t in uni:
        h = hints.get(t, {})
        if not isinstance(h, dict):
            raise ValueError(f"hint for {t} must be dict")
        try:
            out_hints[t] = {
                "dir": float(h.get("dir", 0.0)),
                "size_frac": float(h.get("size_frac", 0.0)),
                "max_lev": float(h.get("max_lev", 1.0)),
                "confidence": float(h.get("confidence", 0.0)),
            }
        except (TypeError, ValueError) as e:
            raise ValueError(f"hint for {t}: {e}") from e
        # clamp to safe ranges
        out_hints[t]["dir"] = max(-1.0, min(1.0, out_hints[t]["dir"]))
        out_hints[t]["size_frac"] = max(0.0, min(1.0, out_hints[t]["size_frac"]))
        out_hints[t]["max_lev"] = max(0.0, min(5.0, out_hints[t]["max_lev"]))
        out_hints[t]["confidence"] = max(0.0, min(1.0, out_hints[t]["confidence"]))
    return {"universe": list(uni), "hints": out_hints}


@dataclass
class GeminiPlanner:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    fallback_plan: dict | None = None
    cache: PlannerCache | None = None
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = PlannerCache()
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")
        if genai is not None and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:  # pragma: no cover
                log.warning("Gemini client init failed, falling back to stub: %s", e)
                self._client = None
        else:
            self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def _build_prompt(self, date: str, portfolio_state: dict, market_summary: dict) -> str:
        return (
            "You are a portfolio planner for a leveraged RL trading bot. "
            "Return STRICT JSON with schema:\n"
            '{"universe": [ticker,...], "hints": {ticker: {"dir": float in [-1,1], '
            '"size_frac": float in [0,1], "max_lev": float in [0,5], '
            '"confidence": float in [0,1]}}}.\n'
            f"Date: {date}\n"
            f"Universe candidates: {self.universe}\n"
            f"Portfolio state: {json.dumps(portfolio_state, default=str)}\n"
            f"Yesterday's market summary: {json.dumps(market_summary, default=str)}\n"
            "Respond with JSON only."
        )

    def _call_live(self, prompt: str) -> dict:  # pragma: no cover - needs real API
        assert self._client is not None and genai_types is not None
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                temperature = max(0.0, 0.4 - 0.15 * attempt)
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                )
                text = getattr(resp, "text", None) or ""
                raw = json.loads(text)
                return _validate_plan(raw, self.universe)
            except Exception as e:
                last_err = e
                log.warning("Gemini call attempt %d failed: %s", attempt + 1, e)
        log.error("Gemini call failed after %d retries: %s", MAX_RETRIES, last_err)
        return _stub_plan(self.universe, self.fallback_plan)

    def plan_for_day(
        self,
        date: str,
        portfolio_state: dict,
        market_summary: dict,
    ) -> dict:
        """Return a cached plan for ``date``. Computes & caches on miss."""
        def _compute() -> dict:
            if self._client is None:
                return _stub_plan(self.universe, self.fallback_plan)
            prompt = self._build_prompt(date, portfolio_state, market_summary)
            return self._call_live(prompt)

        assert self.cache is not None
        return self.cache.get_or_compute(
            date_iso=date,
            portfolio_state=portfolio_state,
            universe=self.universe,
            compute_fn=_compute,
        )
