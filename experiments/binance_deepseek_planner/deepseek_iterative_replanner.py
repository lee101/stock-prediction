#!/usr/bin/env python3
"""DeepSeek-first simulator-in-the-loop daily crypto replanner.

This experiment asks DeepSeek for candidate daily plan templates, scores those
templates on trailing calibration days, applies simple local mutations, and
executes only the selected template on the next unseen 24-hour Binance window.
It is causal: the current execution day is not used to choose the candidate.

Production inference can be flipped to OpenAI strict Structured Outputs with
`--prod-inference`, but research defaults to DeepSeek.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.binance_openai_planner.gpt55_week_replanner import (
    DEFAULT_INITIAL_CASH,
    DailyPlan,
    PlanLeg,
    _line_encoding_description,
    _metrics,
    _market_line_tokens,
    simulate_day,
)
from experiments.binance_transition_graph.graph_policy import (
    DEFAULT_SYMBOLS,
    close_matrix,
    load_hourly_panel,
)


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_PROD_MODEL = "gpt-5.5"
DEFAULT_PROD_REASONING = "xhigh"

SYSTEM_PROMPT = """You are a quantitative crypto portfolio planner.
Return JSON only. You are not placing live orders.

The local harness will score your candidate templates on trailing calibration
days with fees, slippage, two-hour decision lag, one entry per symbol per day,
and daily forced flattening. It will then execute the best trailing candidate on
the next unseen day.

Known failure mode: one-shot plans that looked coherent lost money because they
were mostly correlated crypto-beta longs with tight stops. Fees/slippage and
selloff hours overwhelmed the winners. Prefer post-fee expectancy, lower
turnover, explicit cash, selective shorts/hedges, and stop distances that
survive normal hourly noise."""


def _system_prompt(prompt_mode: str) -> str:
    json_example = """
Example JSON:
{"candidates":[{"name":"example_shape_DO_NOT_COPY","rationale":"example only","plan":{"day_index":0,"cash_pct":70,"gross_exposure_pct":30,"risk_notes":"example only","legs":[{"symbol":"BTCUSDT","side":"long","allocation_pct":15,"entry_offset_bps":-40,"take_profit_bps":160,"stop_loss_bps":110,"confidence":0.5,"rationale":"example only"},{"symbol":"ETHUSDT","side":"short","allocation_pct":15,"entry_offset_bps":40,"take_profit_bps":160,"stop_loss_bps":110,"confidence":0.5,"rationale":"example only"}]}}]}
Do not copy the example name, rationale, values, sides, symbols, or allocation.
Use the provided market tokens and include all requested symbols."""
    if prompt_mode == "no_think":
        return (
            "Return compact valid json only. Use only provided OHLC tokens. No analysis. "
            "No tools. No web. No code. No external facts. Do not explain. "
            "Pick robust levels after fees.\n" + json_example
        )
    if prompt_mode == "minimal":
        return (
            "Return compact valid json only. Use only the provided OHLC tokens/features. "
            "No web search, no tools, no code, no external facts, no explanation. "
            "Avoid correlated beta longs with tight stops; optimize post-fee expectancy.\n" + json_example
        )
    return SYSTEM_PROMPT + "\nReturn valid json only.\n" + json_example


@dataclass(frozen=True)
class TemplateLeg:
    symbol: str
    side: str
    allocation_pct: float
    entry_offset_bps: float
    take_profit_bps: float
    stop_loss_bps: float
    confidence: float
    rationale: str


@dataclass(frozen=True)
class PlanTemplate:
    name: str
    rationale: str
    day_index: int
    cash_pct: float
    gross_exposure_pct: float
    risk_notes: str
    legs: list[TemplateLeg]


def _load_env_real() -> None:
    """Best-effort import of env_real.py without printing secrets."""
    try:
        env_real = importlib.import_module("env_real")
    except Exception:
        return
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "DEEPSEEK_MODEL", "OPENAI_API_KEY"):
        value = getattr(env_real, key, None)
        if value and not os.getenv(key):
            os.environ[key] = str(value)


def template_batch_schema() -> dict[str, Any]:
    leg = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "symbol",
            "side",
            "allocation_pct",
            "entry_offset_bps",
            "take_profit_bps",
            "stop_loss_bps",
            "confidence",
            "rationale",
        ],
        "properties": {
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["long", "short", "flat"]},
            "allocation_pct": {"type": "number"},
            "entry_offset_bps": {"type": "number"},
            "take_profit_bps": {"type": "number"},
            "stop_loss_bps": {"type": "number"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
    }
    plan = {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_index", "cash_pct", "gross_exposure_pct", "risk_notes", "legs"],
        "properties": {
            "day_index": {"type": "integer"},
            "cash_pct": {"type": "number"},
            "gross_exposure_pct": {"type": "number"},
            "risk_notes": {"type": "string"},
            "legs": {"type": "array", "minItems": 1, "items": leg},
        },
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "rationale", "plan"],
        "properties": {
            "name": {"type": "string"},
            "rationale": {"type": "string"},
            "plan": plan,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {"type": "array", "minItems": 1, "items": candidate},
        },
    }


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _window_return(series: pd.Series, hours: int) -> float:
    if len(series) <= hours:
        return 0.0
    p0 = float(series.iloc[-hours - 1])
    p1 = float(series.iloc[-1])
    return float(p1 / p0 - 1.0) if p0 > 0 else 0.0


def _market_features(
    frames: Mapping[str, pd.DataFrame],
    *,
    day_start_i: int,
    history_hours: int,
    feature_style: str,
    line_bars: int,
) -> dict[str, Any]:
    close = close_matrix(frames)
    hist_start = max(0, day_start_i - int(history_hours))
    hist_close = close.iloc[hist_start:day_start_i]
    returns = hist_close.pct_change().replace([np.inf, -np.inf], np.nan)
    per_symbol: dict[str, Any] = {}
    for symbol in close.columns:
        series = hist_close[symbol].dropna()
        if series.empty:
            continue
        high_168 = float(series.tail(168).max())
        per_symbol[symbol] = {
            "last_close": float(series.iloc[-1]),
            "ret_6h": _window_return(series, 6),
            "ret_24h": _window_return(series, 24),
            "ret_72h": _window_return(series, 72),
            "ret_168h": _window_return(series, 168),
            "vol_24h": float(returns[symbol].tail(24).std(ddof=0) or 0.0),
            "vol_168h": float(returns[symbol].tail(168).std(ddof=0) or 0.0),
            "drawdown_168h": float(series.iloc[-1] / high_168 - 1.0) if high_168 > 0 else 0.0,
            "last_12h_closes": [float(x) for x in series.tail(12).to_numpy(dtype=float)],
        }
    payload: dict[str, Any] = {
        "timestamp": close.index[day_start_i - 1].isoformat(),
        "execution_start": close.index[day_start_i].isoformat(),
        "feature_style": str(feature_style),
        "symbols": list(close.columns),
        "per_symbol": per_symbol,
        "correlation_tail_168h": returns.tail(168).corr().fillna(0.0).round(3).to_dict(),
    }
    if feature_style in {"line_pct", "ohlc_mid"}:
        payload["line_encoding"] = _line_encoding_description(feature_style)
        payload["line_tokens"] = _market_line_tokens(
            frames,
            day_start_i=day_start_i,
            bars=int(line_bars),
            feature_style=feature_style,
        )
    return payload


def _prompt_payload(
    frames: Mapping[str, pd.DataFrame],
    *,
    day: int,
    day_start_i: int,
    history_hours: int,
    equity: float,
    candidates: int,
    calibration_days: int,
    feature_style: str,
    line_bars: int,
    prompt_mode: str,
    previous_feedback: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "task": "Return candidate daily portfolio templates for Binance hourly execution.",
        "prompt_mode": str(prompt_mode),
        "day_index": int(day),
        "num_candidates": int(candidates),
        "market": _market_features(
            frames,
            day_start_i=day_start_i,
            history_hours=history_hours,
            feature_style=feature_style,
            line_bars=line_bars,
        ),
        "portfolio": {
            "equity_usd": float(equity),
            "max_gross_exposure_pct": 150.0,
            "max_single_symbol_allocation_pct": 60.0,
            "cash_pct_minimum": 10.0,
            "shorts_allowed": True,
            "daily_forced_flatten": True,
        },
        "execution_assumptions": {
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "decision_lag_hours": 2,
            "one_entry_per_symbol_per_day": True,
            "long_template": "entry is last_close * (1 + entry_offset_bps/10000); use entry_offset_bps <= 0 for pullback limit buys",
            "short_template": "entry is last_close * (1 + entry_offset_bps/10000); use entry_offset_bps >= 0 for rally limit shorts",
            "take_profit_bps": "positive distance from entry",
            "stop_loss_bps": "positive distance from entry; avoid stops tighter than normal hourly noise",
        },
        "calibration_protocol": {
            "trailing_days_scored_before_execution": int(calibration_days),
            "selection_rule": "local harness mutates candidates, scores trailing days only, then executes best candidate on next unseen day",
            "no_future_leakage": True,
        },
        "known_failure_modes_to_avoid": [
            "Mostly correlated crypto-beta longs with tight stops.",
            "High gross exposure when all selected symbols are highly correlated.",
            "Stops that are inside ordinary hourly noise and churn after fees/slippage.",
            "Plausible narrative plans that ignore post-fee fill mechanics.",
        ],
        "output_rules": [
            "Return one leg per symbol.",
            "Use side=flat and allocation_pct=0 for symbols not traded.",
            "Keep gross_exposure_pct equal to non-flat allocation_pct sum.",
            "Keep entry_offset_bps between -250 and 0 for long pullback entries.",
            "Keep entry_offset_bps between 0 and 250 for short rally entries.",
            "Use take_profit_bps and stop_loss_bps large enough to beat 15 bps all-in friction.",
            "Do not use web search, tool use, code interpreter, or external narratives.",
            "If prompt_mode=minimal, do not include analysis or extra text.",
        ],
        "previous_feedback": list(previous_feedback)[-5:],
        "schema": template_batch_schema(),
    }


def call_deepseek_templates(
    payload: Mapping[str, Any],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    prompt_mode: str,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for DeepSeek calls") from exc
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; pass --load-env-real or export it")
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    chosen_model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout_s))
    request = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": _system_prompt(prompt_mode)},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(**request)
    except TypeError:
        request.pop("response_format", None)
        response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek response was empty")
    return str(content)


def call_openai_templates(
    payload: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    timeout_s: float,
    prompt_mode: str,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for OpenAI production inference") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; pass --load-env-real or export it")
    client = OpenAI(api_key=api_key, timeout=float(timeout_s))
    response = client.responses.create(
        model=str(model),
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": _system_prompt(prompt_mode)}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, indent=2)}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "binance_template_candidates",
                "strict": True,
                "schema": template_batch_schema(),
            }
        },
        reasoning={"effort": str(reasoning_effort)},
        max_output_tokens=int(max_output_tokens),
        store=False,
    )
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenAI response had no output_text")
    return text


def _parse_templates(raw_text: str, *, expected_day: int, allowed_symbols: Sequence[str]) -> list[PlanTemplate]:
    data = _extract_json(raw_text)
    raw_candidates = data.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("template response must include candidates list")
    symbols = {symbol.upper() for symbol in allowed_symbols}
    parsed: list[PlanTemplate] = []
    for idx, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping):
            continue
        plan = candidate.get("plan", {})
        if not isinstance(plan, Mapping):
            continue
        legs: list[TemplateLeg] = []
        for raw_leg in plan.get("legs", []):
            if not isinstance(raw_leg, Mapping):
                continue
            symbol = str(raw_leg.get("symbol", "")).upper()
            if symbol not in symbols:
                continue
            side = str(raw_leg.get("side", "flat")).lower()
            if side not in {"long", "short", "flat"}:
                side = "flat"
            allocation = max(0.0, min(100.0, float(raw_leg.get("allocation_pct", 0.0))))
            if side == "flat":
                allocation = 0.0
            entry_offset = max(-250.0, min(250.0, float(raw_leg.get("entry_offset_bps", 0.0))))
            if side == "long":
                entry_offset = min(0.0, entry_offset)
            elif side == "short":
                entry_offset = max(0.0, entry_offset)
            legs.append(
                TemplateLeg(
                    symbol=symbol,
                    side=side,
                    allocation_pct=allocation,
                    entry_offset_bps=entry_offset,
                    take_profit_bps=max(20.0, min(2_000.0, float(raw_leg.get("take_profit_bps", 100.0)))),
                    stop_loss_bps=max(20.0, min(2_000.0, float(raw_leg.get("stop_loss_bps", 100.0)))),
                    confidence=max(0.0, min(1.0, float(raw_leg.get("confidence", 0.5)))),
                    rationale=str(raw_leg.get("rationale", "")),
                )
            )
        if not legs:
            continue
        gross = sum(leg.allocation_pct for leg in legs if leg.side != "flat")
        if gross > 150.0:
            scale = 150.0 / gross
            legs = [
                TemplateLeg(**{**asdict(leg), "allocation_pct": leg.allocation_pct * scale})
                if leg.side != "flat"
                else leg
                for leg in legs
            ]
            gross = 150.0
        parsed.append(
            PlanTemplate(
                name=str(candidate.get("name") or f"deepseek_template_{idx}"),
                rationale=str(candidate.get("rationale") or ""),
                day_index=int(plan.get("day_index", expected_day)),
                cash_pct=max(0.0, float(plan.get("cash_pct", max(0.0, 100.0 - gross)))),
                gross_exposure_pct=float(plan.get("gross_exposure_pct", gross)),
                risk_notes=str(plan.get("risk_notes", "")),
                legs=legs,
            )
        )
    if not parsed:
        raise ValueError("No usable templates found")
    return parsed


def _cash_template(symbols: Sequence[str], *, day: int) -> PlanTemplate:
    return PlanTemplate(
        name="cash_guard",
        rationale="Deterministic no-trade guard if all candidates score poorly on trailing simulation.",
        day_index=int(day),
        cash_pct=100.0,
        gross_exposure_pct=0.0,
        risk_notes="Stay flat because calibration did not justify risk.",
        legs=[
            TemplateLeg(
                symbol=str(symbol),
                side="flat",
                allocation_pct=0.0,
                entry_offset_bps=0.0,
                take_profit_bps=100.0,
                stop_loss_bps=100.0,
                confidence=1.0,
                rationale="cash guard",
            )
            for symbol in symbols
        ],
    )


def _mutate_templates(
    templates: Sequence[PlanTemplate],
    symbols: Sequence[str],
    *,
    day: int,
    include_cash_guard: bool,
) -> list[PlanTemplate]:
    mutated: list[PlanTemplate] = []
    seen: set[str] = set()
    if include_cash_guard:
        cash = _cash_template(symbols, day=day)
        mutated.append(cash)
        seen.add(cash.name)

    def add(template: PlanTemplate) -> None:
        if template.name in seen:
            return
        seen.add(template.name)
        mutated.append(template)

    for template in templates:
        add(template)
        for suffix, alloc_mult, tp_mult, sl_mult in (
            ("risk50", 0.50, 1.00, 1.00),
            ("risk75", 0.75, 1.00, 1.00),
            ("wide_stop", 0.80, 1.25, 1.50),
        ):
            legs: list[TemplateLeg] = []
            for leg in template.legs:
                if leg.side == "flat":
                    legs.append(leg)
                    continue
                legs.append(
                    TemplateLeg(
                        **{
                            **asdict(leg),
                            "allocation_pct": leg.allocation_pct * alloc_mult,
                            "take_profit_bps": max(20.0, min(2_000.0, leg.take_profit_bps * tp_mult)),
                            "stop_loss_bps": max(20.0, min(2_000.0, leg.stop_loss_bps * sl_mult)),
                        }
                    )
                )
            gross = sum(leg.allocation_pct for leg in legs if leg.side != "flat")
            add(
                PlanTemplate(
                    name=f"{template.name}_{suffix}",
                    rationale=f"Local mutation of {template.name}: alloc x{alloc_mult}, tp x{tp_mult}, stop x{sl_mult}.",
                    day_index=template.day_index,
                    cash_pct=max(0.0, 100.0 - gross),
                    gross_exposure_pct=gross,
                    risk_notes=template.risk_notes,
                    legs=legs,
                )
            )
    return mutated


def instantiate_plan(
    template: PlanTemplate,
    frames: Mapping[str, pd.DataFrame],
    *,
    day_start_i: int,
    day: int,
) -> DailyPlan:
    close = close_matrix(frames)
    legs: list[PlanLeg] = []
    for leg in template.legs:
        last_close = float(close[leg.symbol].iloc[day_start_i - 1])
        if leg.side == "flat" or leg.allocation_pct <= 0.0:
            legs.append(
                PlanLeg(
                    symbol=leg.symbol,
                    side="flat",
                    allocation_pct=0.0,
                    buy_price=0.0,
                    sell_price=0.0,
                    stop_price=0.0,
                    confidence=leg.confidence,
                    rationale=leg.rationale,
                )
            )
            continue
        entry = max(1e-12, last_close * (1.0 + leg.entry_offset_bps / 10_000.0))
        tp = leg.take_profit_bps / 10_000.0
        sl = leg.stop_loss_bps / 10_000.0
        if leg.side == "long":
            buy_price = entry
            sell_price = entry * (1.0 + tp)
            stop_price = max(1e-12, entry * (1.0 - sl))
        else:
            sell_price = entry
            buy_price = max(1e-12, entry * (1.0 - tp))
            stop_price = entry * (1.0 + sl)
        legs.append(
            PlanLeg(
                symbol=leg.symbol,
                side=leg.side,
                allocation_pct=leg.allocation_pct,
                buy_price=buy_price,
                sell_price=sell_price,
                stop_price=stop_price,
                confidence=leg.confidence,
                rationale=leg.rationale,
            )
        )
    gross = sum(leg.allocation_pct for leg in legs if leg.side != "flat")
    return DailyPlan(
        day_index=int(day),
        plan_timestamp=close.index[day_start_i - 1].isoformat(),
        cash_pct=max(0.0, 100.0 - gross),
        gross_exposure_pct=gross,
        risk_notes=template.risk_notes,
        legs=legs,
    )


def evaluate_template(
    frames: Mapping[str, pd.DataFrame],
    template: PlanTemplate,
    *,
    day_start_i: int,
    day: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    plan = instantiate_plan(template, frames, day_start_i=day_start_i, day=day)
    equity, trades, rows = simulate_day(
        frames,
        plan,
        start_i=day_start_i,
        end_i=day_start_i + 24,
        starting_equity=float(initial_cash),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )
    metrics = _metrics(rows, float(initial_cash))
    return {
        "template_name": template.name,
        "start_index": int(day_start_i),
        "start_timestamp": rows[0]["timestamp"] if rows else "",
        "final_equity": float(equity),
        "total_return": float(metrics["total_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "sortino": float(metrics["sortino"]),
        "trades": len(trades),
    }


def score_templates(
    frames: Mapping[str, pd.DataFrame],
    templates: Sequence[PlanTemplate],
    *,
    current_day_start_i: int,
    day: int,
    calibration_days: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    starts = [
        current_day_start_i - 24 * offset
        for offset in range(int(calibration_days), 0, -1)
        if current_day_start_i - 24 * offset >= 1
    ]
    for template in templates:
        evals = [
            evaluate_template(
                frames,
                template,
                day_start_i=start,
                day=day,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            for start in starts
        ]
        returns = np.array([row["total_return"] for row in evals], dtype=float) if evals else np.array([0.0])
        dds = np.array([row["max_drawdown"] for row in evals], dtype=float) if evals else np.array([0.0])
        trades = np.array([row["trades"] for row in evals], dtype=float) if evals else np.array([0.0])
        median_return = float(np.median(returns))
        p25_return = float(np.quantile(returns, 0.25))
        worst_return = float(np.min(returns))
        max_dd = float(np.max(dds))
        avg_trades = float(np.mean(trades))
        # Penalize unstable, drawdown-heavy, and high-turnover candidates.
        score = median_return + 0.50 * p25_return + 0.25 * worst_return - 0.75 * max_dd - 0.0002 * avg_trades
        rows.append(
            {
                "template_name": template.name,
                "score": float(score),
                "median_return": median_return,
                "p25_return": p25_return,
                "worst_return": worst_return,
                "max_drawdown": max_dd,
                "avg_trades": avg_trades,
                "calibration_evals": evals,
            }
        )
    rows.sort(key=lambda row: (row["score"], row["median_return"], -row["max_drawdown"]), reverse=True)
    return rows


def _offline_templates(symbols: Sequence[str], *, day: int) -> list[PlanTemplate]:
    return [
        PlanTemplate(
            name="offline_momentum_pullback",
            rationale="Offline smoke candidate.",
            day_index=day,
            cash_pct=40.0,
            gross_exposure_pct=60.0,
            risk_notes="Small long pullback smoke template.",
            legs=[
                TemplateLeg(
                    symbol=str(symbol),
                    side="long" if idx < 3 else "flat",
                    allocation_pct=20.0 if idx < 3 else 0.0,
                    entry_offset_bps=-40.0,
                    take_profit_bps=140.0,
                    stop_loss_bps=90.0,
                    confidence=0.5,
                    rationale="offline smoke",
                )
                for idx, symbol in enumerate(symbols)
            ],
        )
    ]


def _provider_call(args: argparse.Namespace, payload: Mapping[str, Any]) -> str:
    if args.provider == "deepseek":
        return call_deepseek_templates(
            payload,
            model=str(args.model),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            timeout_s=float(args.api_timeout_s),
            prompt_mode=str(args.prompt_mode),
        )
    if args.provider == "openai":
        return call_openai_templates(
            payload,
            model=str(args.model),
            reasoning_effort=str(args.reasoning_effort),
            max_output_tokens=int(args.max_tokens),
            timeout_s=float(args.api_timeout_s),
            prompt_mode=str(args.prompt_mode),
        )
    raise ValueError(f"unsupported provider: {args.provider}")


def _call_and_parse_templates(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    stamp: str,
    day: int,
    raw_paths: list[str],
) -> list[PlanTemplate]:
    replay_paths = list(getattr(args, "raw_response_paths", None) or [])
    if replay_paths:
        if day >= len(replay_paths):
            raise RuntimeError(f"missing replay raw response for day {day}")
        raw_path = Path(replay_paths[day])
        raw_paths.append(str(raw_path))
        return _parse_templates(raw_path.read_text(encoding="utf-8"), expected_day=day, allowed_symbols=args.symbols)

    last_error: Exception | None = None
    for attempt in range(int(args.retries) + 1):
        request_payload = dict(payload)
        if attempt > 0:
            request_payload["num_candidates"] = 1
            request_payload["retry_attempt"] = attempt
            request_payload["retry_instruction"] = (
                "Previous response was malformed or truncated. Return compact valid JSON only. "
                "Return exactly one candidate and no prose."
            )
        try:
            if args.offline:
                raw = json.dumps(
                    {
                        "candidates": [
                            asdict(t)
                            | {"plan": {k: v for k, v in asdict(t).items() if k not in {"name", "rationale"}}}
                            for t in _offline_templates(args.symbols, day=day)
                        ]
                    }
                )
            else:
                raw = _provider_call(args, request_payload)
        except Exception as exc:
            last_error = exc
            print(f"day={day} attempt={attempt} provider failed: {exc}", flush=True)
            continue
        raw_path = output_dir / f"deepseek_iter_raw_{stamp}_day{day}_attempt{attempt}.json"
        raw_path.write_text(raw, encoding="utf-8")
        raw_paths.append(str(raw_path))
        try:
            return _parse_templates(raw, expected_day=day, allowed_symbols=args.symbols)
        except Exception as exc:
            last_error = exc
            print(f"day={day} attempt={attempt} parse failed: {exc}", flush=True)
    raise RuntimeError(f"day {day} failed after retries: {last_error}")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.load_env_real:
        _load_env_real()
    if args.prod_inference:
        args.provider = "openai"
        args.model = args.prod_model
        args.reasoning_effort = args.prod_reasoning_effort

    total_hours = int(args.history_hours) + int(args.days) * 24 + int(args.decision_lag_hours)
    frames = load_hourly_panel(args.data_root, args.symbols, tail_hours=total_hours)
    common = close_matrix(frames).index
    start_i = int(args.history_hours) + int(args.decision_lag_hours)
    equity = float(args.initial_cash)
    all_trades: list[dict[str, Any]] = []
    all_equity_rows: list[dict[str, Any]] = []
    selected_plans: list[dict[str, Any]] = []
    selected_templates: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    raw_paths: list[str] = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for day in range(int(args.days)):
        day_start = start_i + day * 24
        day_end = min(day_start + 24, len(common))
        if day_end <= day_start:
            break
        print(
            f"day={day} provider={args.provider} equity=${equity:.2f} "
            f"window={common[day_start].isoformat()}",
            flush=True,
        )
        payload = _prompt_payload(
            frames,
            day=day,
            day_start_i=day_start,
            history_hours=int(args.history_hours),
            equity=equity,
            candidates=int(args.candidates),
            calibration_days=int(args.calibration_days),
            feature_style=str(args.feature_style),
            line_bars=int(args.line_bars),
            prompt_mode=str(args.prompt_mode),
            previous_feedback=feedback,
        )
        templates = _call_and_parse_templates(
            args,
            payload,
            output_dir=output_dir,
            stamp=stamp,
            day=day,
            raw_paths=raw_paths,
        )
        print(f"day={day} received candidates", flush=True)
        candidates = _mutate_templates(
            templates,
            args.symbols,
            day=day,
            include_cash_guard=not bool(args.disable_cash_guard),
        )
        scored = score_templates(
            frames,
            candidates,
            current_day_start_i=day_start,
            day=day,
            calibration_days=int(args.calibration_days),
            initial_cash=float(args.initial_cash),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        best_score = scored[0]
        template_by_name = {template.name: template for template in candidates}
        chosen_template = template_by_name[best_score["template_name"]]
        chosen_plan = instantiate_plan(chosen_template, frames, day_start_i=day_start, day=day)
        day_start_equity = equity
        equity, trades, rows = simulate_day(
            frames,
            chosen_plan,
            start_i=day_start,
            end_i=day_end,
            starting_equity=equity,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        day_metrics = _metrics(rows, day_start_equity)
        trade_dicts = [asdict(trade) for trade in trades]
        all_trades.extend(trade_dicts)
        all_equity_rows.extend(rows)
        selected_templates.append(asdict(chosen_template))
        selected_plans.append({**asdict(chosen_plan), "legs": [asdict(leg) for leg in chosen_plan.legs]})
        calibration_rows.append({"day": day, "selected": best_score, "top5": scored[:5]})
        feedback.append(
            {
                "day": day,
                "selected_template": chosen_template.name,
                "calibration_score": best_score["score"],
                "calibration_median_return": best_score["median_return"],
                "actual_day_return": day_metrics["total_return"],
                "actual_day_max_drawdown": day_metrics["max_drawdown"],
                "actual_trades": len(trades),
                "lesson": (
                    "reduce correlated beta and turnover if actual return is negative"
                    if day_metrics["total_return"] < 0
                    else "candidate worked; preserve post-fee edge but avoid over-sizing"
                ),
            }
        )
        print(
            f"day={day} selected={chosen_template.name} "
            f"actual={day_metrics['total_return'] * 100:+.2f}% "
            f"equity=${equity:.2f}",
            flush=True,
        )

    metrics = _metrics(all_equity_rows, float(args.initial_cash))
    gate_passes = metrics["monthly_return"] >= float(args.monthly_target) and metrics["max_drawdown"] <= float(args.max_drawdown_target)
    return {
        "provider": args.provider,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort if args.provider == "openai" else None,
        "prod_inference": bool(args.prod_inference),
        "prod_model": args.prod_model,
        "prod_reasoning_effort": args.prod_reasoning_effort,
        "production_gate": {
            "passes_gate": bool(gate_passes),
            "monthly_target": float(args.monthly_target),
            "max_drawdown_target": float(args.max_drawdown_target),
            "route_to_prod_model_if_gate_passes": bool(gate_passes),
            "prod_model": args.prod_model,
            "prod_reasoning_effort": args.prod_reasoning_effort,
            "structured_outputs_required": True,
        },
        "symbols": list(args.symbols),
        "days": int(args.days),
        "history_hours": int(args.history_hours),
        "calibration_days": int(args.calibration_days),
        "feature_style": str(args.feature_style),
        "line_bars": int(args.line_bars),
        "prompt_mode": str(args.prompt_mode),
        "cash_guard_enabled": not bool(args.disable_cash_guard),
        "initial_cash": float(args.initial_cash),
        "final_equity": float(equity),
        "metrics": metrics,
        "selected_templates": selected_templates,
        "selected_plans": selected_plans,
        "calibration": calibration_rows,
        "feedback": feedback,
        "trades": all_trades,
        "equity_curve": all_equity_rows,
        "raw_response_paths": raw_paths,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    gate = result["production_gate"]
    lines = [
        "# DeepSeek Iterative Binance Replanner",
        "",
        f"- Provider: `{result['provider']}`",
        f"- Model: `{result['model']}`",
        f"- Prompt mode: `{result.get('prompt_mode', 'standard')}`",
        f"- Feature style: `{result.get('feature_style', 'summary')}`",
        f"- Production inference mode: `{result['prod_inference']}`",
        f"- Prod model if gated: `{result['prod_model']}` / `{result['prod_reasoning_effort']}`",
        f"- Symbols: `{', '.join(result['symbols'])}`",
        f"- Cash guard enabled: `{result.get('cash_guard_enabled', True)}`",
        f"- Initial cash: `${result['initial_cash']:,.2f}`",
        f"- Final equity: `${result['final_equity']:,.2f}`",
        f"- Total return: `{metrics['total_return'] * 100:+.2f}%`",
        f"- Monthly-equivalent return: `{metrics['monthly_return'] * 100:+.2f}%`",
        f"- Max drawdown: `{metrics['max_drawdown'] * 100:.2f}%`",
        f"- Sortino: `{metrics['sortino']:.2f}`",
        f"- Trades: `{len(result['trades'])}`",
        f"- Production gate passed: `{gate['passes_gate']}`",
        "",
        "## Method",
        "",
        "- DeepSeek proposes candidate templates.",
        "- Local code mutates allocation/stop variants.",
        "- Candidates are scored only on trailing calibration days.",
        "- The best trailing candidate is executed on the next unseen day.",
        "",
        "## Selected Templates",
        "",
    ]
    feedback_by_day = {item["day"]: item for item in result["feedback"]}
    for template in result["selected_templates"]:
        active = [leg for leg in template["legs"] if leg["side"] != "flat" and leg["allocation_pct"] > 0]
        summary = ", ".join(
            f"{leg['side']} {leg['symbol']} {leg['allocation_pct']:.1f}% off={leg['entry_offset_bps']:.0f}bps"
            for leg in active
        ) or "flat"
        fb = feedback_by_day.get(template["day_index"], {})
        lines.append(
            f"- Day {template['day_index']}: `{template['name']}` {summary}; "
            f"actual={float(fb.get('actual_day_return', 0.0)) * 100:+.2f}%, "
            f"cal_score={float(fb.get('calibration_score', 0.0)):.4f}"
        )
    lines.extend(["", "## Trades", ""])
    for trade in result["trades"]:
        lines.append(
            "- "
            f"{trade['timestamp']} {trade['event']} {trade['side']} {trade['symbol']} "
            f"notional=${trade['notional']:.2f} price={trade['price']:.6g} "
            f"pnl=${trade['pnl']:.2f} equity=${trade['equity_after']:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_PROD_REASONING)
    parser.add_argument("--prod-inference", action="store_true")
    parser.add_argument("--prod-model", default=DEFAULT_PROD_MODEL)
    parser.add_argument("--prod-reasoning-effort", default=DEFAULT_PROD_REASONING)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--feature-style", choices=["summary", "line_pct", "ohlc_mid"], default="summary")
    parser.add_argument("--line-bars", type=int, default=48)
    parser.add_argument("--prompt-mode", choices=["standard", "minimal", "no_think"], default="standard")
    parser.add_argument("--calibration-days", type=int, default=4)
    parser.add_argument("--decision-lag-hours", type=int, default=2)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--api-timeout-s", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--disable-cash-guard", action="store_true")
    parser.add_argument("--raw-response-paths", nargs="*")
    parser.add_argument("--monthly-target", type=float, default=0.27)
    parser.add_argument("--max-drawdown-target", type=float, default=0.20)
    parser.add_argument("--load-env-real", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default="experiments/binance_deepseek_planner/results")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"deepseek_iter_replanner_{stamp}.json"
    md_path = output_dir / f"deepseek_iter_replanner_{stamp}.md"
    try:
        result = run_experiment(args)
    except Exception as exc:
        error = {
            "provider": args.provider,
            "model": args.model,
            "prod_inference": bool(args.prod_inference),
            "error": str(exc),
        }
        json_path.write_text(json.dumps(error, indent=2, sort_keys=True), encoding="utf-8")
        print(f"experiment failed: {exc}")
        print(f"wrote {json_path}")
        return 2
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(md_path, result)
    metrics = result["metrics"]
    print(
        f"final=${result['final_equity']:.2f} "
        f"return={metrics['total_return'] * 100:+.2f}% "
        f"monthly={metrics['monthly_return'] * 100:+.2f}% "
        f"dd={metrics['max_drawdown'] * 100:.2f}% "
        f"trades={len(result['trades'])} "
        f"prod_route={result['production_gate']['route_to_prod_model_if_gate_passes']}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
