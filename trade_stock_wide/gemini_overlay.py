from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd
import requests

from .planner import STRATEGY_SPECS, WidePlannerConfig, build_wide_plan, validate_long_price_levels
from .types import WideCandidate

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiWidePlan:
    action: str
    buy_price: float = 0.0
    sell_price: float = 0.0
    allocation_pct: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""


@dataclass(frozen=True)
class GeminiOverlayStats:
    prompt_count: int = 0
    cache_hits: int = 0
    adjusted_count: int = 0
    skipped_count: int = 0
    invalid_count: int = 0


if genai is not None:
    WIDE_SCHEMA = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["action", "buy_price", "sell_price", "allocation_pct", "confidence"],
        properties={
            "action": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="One of: buy, skip",
            ),
            "buy_price": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Adjusted long entry price as a number string, or 0 if skipping",
            ),
            "sell_price": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Adjusted take-profit price as a number string, or 0 if skipping",
            ),
            "allocation_pct": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Position size as a percent of account equity, from 0 to 50",
            ),
            "confidence": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Confidence from 0.0 to 1.0",
            ),
            "reasoning": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Brief explanation",
            ),
        },
    )
else:  # pragma: no cover
    WIDE_SCHEMA = None


def _sanitize_model_name(model: str) -> str:
    text = str(model or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _cache_path(*, cache_root: Path, model: str, prompt: str) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    path = Path(cache_root) / _sanitize_model_name(model) / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached_plan(*, cache_root: Path, model: str, prompt: str) -> Optional[GeminiWidePlan]:
    path = _cache_path(cache_root=cache_root, model=model, prompt=prompt)
    if not path.exists():
        return None
    try:
        return GeminiWidePlan(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _store_cached_plan(*, cache_root: Path, model: str, prompt: str, plan: GeminiWidePlan) -> None:
    path = _cache_path(cache_root=cache_root, model=model, prompt=prompt)
    path.write_text(json.dumps(plan.__dict__, indent=2) + "\n", encoding="utf-8")


def _frame_sort_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working = working.dropna(subset=["timestamp"]).sort_values("timestamp")
    return working.reset_index(drop=True)


def _candidate_row(frame: pd.DataFrame, candidate: WideCandidate) -> Mapping[str, object]:
    working = _frame_sort_by_time(frame)
    if candidate.session_date and "timestamp" in working.columns:
        day = pd.Timestamp(candidate.session_date, tz="UTC")
        next_day = day + pd.Timedelta(days=1)
        matches = working[(working["timestamp"] >= day) & (working["timestamp"] < next_day)]
        if not matches.empty:
            return matches.iloc[-1].to_dict()
    index = int(candidate.day_index)
    if 0 <= index < len(working):
        return working.iloc[index].to_dict()
    return working.iloc[-1].to_dict()


def _history_window(frame: pd.DataFrame, candidate: WideCandidate, lookback: int) -> pd.DataFrame:
    working = _frame_sort_by_time(frame)
    if candidate.session_date and "timestamp" in working.columns:
        day = pd.Timestamp(candidate.session_date, tz="UTC")
        next_day = day + pd.Timedelta(days=1)
        hist = working[working["timestamp"] < next_day].tail(max(int(lookback), 1))
        if not hist.empty:
            return hist.reset_index(drop=True)
    index = min(max(int(candidate.day_index), 0), max(len(working) - 1, 0))
    start = max(0, index - max(int(lookback), 1) + 1)
    return working.iloc[start : index + 1].reset_index(drop=True)


def _extract_strategy_snapshot(row: Mapping[str, object]) -> str:
    lines: list[str] = []
    for spec in STRATEGY_SPECS:
        pnl = None
        avg = None
        for key in spec.forecast_keys:
            try:
                value = float(row.get(key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            pnl = value
            break
        for key in spec.avg_return_keys:
            try:
                value = float(row.get(key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            avg = value
            break
        if pnl is None and avg is None:
            continue
        pnl_text = f"{pnl:+.4f}" if pnl is not None else "n/a"
        avg_text = f"{avg:+.4f}" if avg is not None else "n/a"
        lines.append(f"  {spec.name:12s} forecast_pnl={pnl_text} avg_return={avg_text}")
    return "\n".join(lines) if lines else "  (no per-strategy snapshot available)"


def _extract_strategy_history(frame: pd.DataFrame, candidate: WideCandidate, lookback: int = 5) -> str:
    working = _frame_sort_by_time(frame)
    if working.empty:
        return "  (no strategy history available)"
    spec = next((item for item in STRATEGY_SPECS if item.name == candidate.strategy), None)
    if spec is None:
        return "  (no strategy history available)"
    subset = _history_window(working, candidate, lookback)
    lines: list[str] = []
    for _, row in subset.iterrows():
        pnl = None
        avg = None
        for key in spec.forecast_keys:
            try:
                pnl = float(row.get(key))
            except (TypeError, ValueError):
                continue
            break
        for key in spec.avg_return_keys:
            try:
                avg = float(row.get(key))
            except (TypeError, ValueError):
                continue
            break
        ts = str(pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce").date()) if "timestamp" in row else "n/a"
        if pnl is None and avg is None:
            continue
        lines.append(
            f"  {ts}: forecast_pnl={pnl:+.4f} avg_return={avg:+.4f}"
            if pnl is not None and avg is not None
            else f"  {ts}: forecast_pnl={pnl if pnl is not None else 'n/a'} avg_return={avg if avg is not None else 'n/a'}"
        )
    return "\n".join(lines) if lines else "  (no strategy history available)"


def build_wide_gemini_prompt(
    *,
    candidate: WideCandidate,
    history: pd.DataFrame,
    current_row: Mapping[str, object],
    rank: int,
    top_k: int,
) -> str:
    bars = _history_window(history, candidate, lookback=20)
    price_lines: list[str] = []
    for _, row in bars.tail(20).iterrows():
        ts = str(pd.to_datetime(row["timestamp"], utc=True, errors="coerce"))[:10] if "timestamp" in row else "n/a"
        volume = row["volume"] if "volume" in row else 0.0
        price_lines.append(
            f"  {ts}: O={float(row['open']):.2f} H={float(row['high']):.2f} "
            f"L={float(row['low']):.2f} C={float(row['close']):.2f} V={float(volume):.0f}"
        )

    current_price = float(candidate.last_close)
    forecast_close = current_row.get("predicted_close_p50", current_price)
    try:
        forecast_close_text = f"{float(forecast_close):.2f}"
    except (TypeError, ValueError):
        forecast_close_text = f"{current_price:.2f}"

    return f"""You are refining a daily long-only stock trade plan for a work-stealing strategy.

Only adjust the entry and take-profit for this already-selected candidate. Do not invent shorts.
If the trade should be skipped entirely, return action=skip and prices=0.

UNIVERSE RANK:
  rank={int(rank)} of top {int(top_k)}

CURRENT CANDIDATE:
  symbol={candidate.symbol}
  selected_strategy={candidate.strategy}
  current_price={candidate.last_close:.2f}
  chronos_entry_low={candidate.entry_price:.2f}
  chronos_take_profit_high={candidate.take_profit_price:.2f}
  chronos_predicted_close={forecast_close_text}
  chronos_predicted_high={candidate.predicted_high:.2f}
  chronos_predicted_low={candidate.predicted_low:.2f}
  forecasted_pnl={candidate.forecasted_pnl:+.4f}
  avg_return={candidate.avg_return:+.4f}
  expected_return_pct={candidate.expected_return_pct * 100:.2f}%

CURRENT DAY STRATEGY SNAPSHOT:
{_extract_strategy_snapshot(current_row)}

RECENT SELECTED STRATEGY HISTORY:
{_extract_strategy_history(history, candidate, lookback=5)}

RECENT DAILY OHLCV:
{chr(10).join(price_lines)}

TASK:
Return JSON only with keys action, buy_price, sell_price, allocation_pct, confidence, reasoning.
- action must be buy or skip
- buy_price must be a realistic long entry below or near the current price
- sell_price must be above buy_price if action=buy
- allocation_pct must be from 0 to 50 and represents percent of account equity to reserve for this watch
- prefer tighter, realistic levels if Chronos seems too optimistic
- if the setup looks weak after recent price action and strategy PnL context, set action=skip
"""


def call_gemini_wide(
    prompt: str,
    *,
    model: str,
    api_key: str | None = None,
    fallback_plan: GeminiWidePlan | None = None,
    temperature: float = 0.2,
) -> Optional[GeminiWidePlan]:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        logger.warning("GEMINI_API_KEY not set; Gemini wide overlay disabled")
        return fallback_plan

    def _plan_from_payload(payload: Mapping[str, object]) -> GeminiWidePlan:
        return GeminiWidePlan(
            action=str(payload.get("action", "skip")).strip().lower() or "skip",
            buy_price=float(payload.get("buy_price", 0.0) or 0.0),
            sell_price=float(payload.get("sell_price", 0.0) or 0.0),
            allocation_pct=float(payload.get("allocation_pct", 0.0) or 0.0),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            reasoning=str(payload.get("reasoning", "")),
        )

    if genai is not None:
        client = genai.Client(api_key=key)
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=WIDE_SCHEMA,
        )
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            payload = json.loads((resp.text or "").strip())
            return _plan_from_payload(payload)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Gemini SDK call failed for model=%s: %s", model, exc)

    # SDK-free fallback for environments that have the API key but not google.genai.
    model_name = model if str(model).startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        response_payload = response.json()
        candidates = response_payload.get("candidates") or []
        if not candidates:
            raise ValueError(f"No candidates returned: {response_payload}")
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise ValueError(f"No text returned: {response_payload}")
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return _plan_from_payload(json.loads(text))
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("Gemini HTTP fallback failed for model=%s: %s", model, exc)
        return fallback_plan


def refine_daily_candidates_with_gemini(
    candidates: Sequence[WideCandidate],
    *,
    account_equity: float,
    planner: WidePlannerConfig,
    backtests_by_symbol: Mapping[str, pd.DataFrame],
    cache_root: Path,
    model: str,
    api_key: str | None = None,
    min_confidence: float = 0.35,
) -> tuple[list[WideCandidate], GeminiOverlayStats]:
    base_orders = build_wide_plan(candidates, account_equity=account_equity, config=planner)
    refined: list[WideCandidate] = []
    stats = GeminiOverlayStats()

    for order in base_orders:
        candidate = order.candidate
        frame = backtests_by_symbol.get(candidate.symbol)
        if frame is None or frame.empty:
            refined.append(candidate)
            continue
        current_row = _candidate_row(frame, candidate)
        prompt = build_wide_gemini_prompt(
            candidate=candidate,
            history=frame,
            current_row=current_row,
            rank=order.rank,
            top_k=planner.top_k,
        )
        plan = _load_cached_plan(cache_root=cache_root, model=model, prompt=prompt)
        cache_hit = plan is not None
        if cache_hit:
            stats = replace(stats, cache_hits=stats.cache_hits + 1)
        else:
            stats = replace(stats, prompt_count=stats.prompt_count + 1)
            plan = call_gemini_wide(prompt, model=model, api_key=api_key)
            if plan is not None:
                _store_cached_plan(cache_root=cache_root, model=model, prompt=prompt, plan=plan)
        if plan is None:
            refined.append(candidate)
            continue
        if str(plan.action).strip().lower() == "skip" and float(plan.confidence) >= float(min_confidence):
            stats = replace(stats, skipped_count=stats.skipped_count + 1)
            continue
        if (
            str(plan.action).strip().lower() != "buy"
            or float(plan.confidence) < float(min_confidence)
            or plan.buy_price <= 0.0
            or plan.sell_price <= 0.0
        ):
            refined.append(candidate)
            continue
        allocation_fraction = min(
            max(float(plan.allocation_pct) / 100.0, 0.0),
            max(float(planner.max_pair_notional_fraction), 0.0),
        )
        try:
            validate_long_price_levels(
                symbol=candidate.symbol,
                entry_price=float(plan.buy_price),
                take_profit_price=float(plan.sell_price),
            )
            if float(plan.buy_price) >= float(candidate.last_close):
                raise ValueError("entry above current price")
            if allocation_fraction <= 0.0:
                raise ValueError("allocation below or equal to zero")
        except ValueError:
            stats = replace(stats, invalid_count=stats.invalid_count + 1)
            refined.append(candidate)
            continue
        refined.append(
            replace(
                candidate,
                entry_price=float(plan.buy_price),
                take_profit_price=float(plan.sell_price),
                allocation_fraction_of_equity=allocation_fraction,
            )
        )
        stats = replace(stats, adjusted_count=stats.adjusted_count + 1)
    return refined, stats


def refine_candidate_days_with_gemini(
    candidate_days: Sequence[Sequence[WideCandidate]],
    *,
    starting_equity: float,
    planner: WidePlannerConfig,
    backtests_by_symbol: Mapping[str, pd.DataFrame],
    cache_root: Path,
    model: str,
    api_key: str | None = None,
    min_confidence: float = 0.35,
) -> tuple[list[list[WideCandidate]], GeminiOverlayStats]:
    all_stats = GeminiOverlayStats()
    equity = float(starting_equity)
    refined_days: list[list[WideCandidate]] = []
    for day_candidates in candidate_days:
        refined, stats = refine_daily_candidates_with_gemini(
            day_candidates,
            account_equity=equity,
            planner=planner,
            backtests_by_symbol=backtests_by_symbol,
            cache_root=cache_root,
            model=model,
            api_key=api_key,
            min_confidence=min_confidence,
        )
        refined_days.append(refined)
        all_stats = GeminiOverlayStats(
            prompt_count=all_stats.prompt_count + stats.prompt_count,
            cache_hits=all_stats.cache_hits + stats.cache_hits,
            adjusted_count=all_stats.adjusted_count + stats.adjusted_count,
            skipped_count=all_stats.skipped_count + stats.skipped_count,
            invalid_count=all_stats.invalid_count + stats.invalid_count,
        )
    return refined_days, all_stats


__all__ = [
    "GeminiOverlayStats",
    "GeminiWidePlan",
    "build_wide_gemini_prompt",
    "call_gemini_wide",
    "refine_candidate_days_with_gemini",
    "refine_daily_candidates_with_gemini",
]
