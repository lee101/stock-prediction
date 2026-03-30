from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency in some envs
    from src.models.chronos2_postprocessing import repair_forecast_ohlc
except Exception:  # pragma: no cover
    repair_forecast_ohlc = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SUPPORTED_NARRATIVE_BACKENDS = ("off", "heuristic", "openai", "anthropic", "xai_tools")
_SUMMARY_MODEL_DEFAULTS = {
    "heuristic": "heuristic",
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-4-5",
    "xai_tools": "grok-4.20-multi-agent-0309",
}


def normalize_narrative_backend(value: object) -> str:
    backend = str(value or "off").strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "disabled": "off",
        "xai": "xai_tools",
        "grok": "xai_tools",
        "grok_tools": "xai_tools",
    }
    backend = aliases.get(backend, backend)
    if backend not in SUPPORTED_NARRATIVE_BACKENDS:
        raise ValueError(
            f"Unsupported narrative backend '{value}'. Expected one of {SUPPORTED_NARRATIVE_BACKENDS}."
        )
    return backend


def resolve_summary_cache_dir(
    *,
    forecast_cache_dir: Path,
    summary_cache_dir: Path | None,
) -> Path:
    if summary_cache_dir is not None:
        return Path(summary_cache_dir)
    forecast_cache_dir = Path(forecast_cache_dir)
    return forecast_cache_dir.parent / "_narrative_summaries" / forecast_cache_dir.name


def resolve_horizon_summary_cache_dir(
    *,
    cache_root: Path,
    horizon: int,
    summary_cache_root: Path | None,
) -> Path:
    forecast_cache_dir = Path(cache_root) / f"h{int(horizon)}"
    explicit_summary_cache_dir = (
        None if summary_cache_root is None else Path(summary_cache_root) / forecast_cache_dir.name
    )
    return resolve_summary_cache_dir(
        forecast_cache_dir=forecast_cache_dir,
        summary_cache_dir=explicit_summary_cache_dir,
    )


class NarrativeSummaryCache:
    """Parquet-backed cache for per-symbol, per-horizon narrative summaries."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        safe = str(symbol).upper().replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to read narrative summary cache %s: %s", path, exc)
            return pd.DataFrame()
        return _normalize_summary_cache_frame(frame)

    def write(self, symbol: str, frame: pd.DataFrame) -> None:
        path = self._path(symbol)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized = _dedupe_summary_cache_frame(frame)
            normalized.to_parquet(path, index=False)
        except Exception as exc:
            logger.warning("Failed to write narrative summary cache %s: %s", path, exc)


def _normalize_summary_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for col in ("timestamp", "issued_at", "target_timestamp", "generated_at"):
        if col in normalized.columns:
            normalized[col] = pd.to_datetime(normalized[col], utc=True, errors="coerce")
    if "horizon_hours" in normalized.columns:
        normalized["horizon_hours"] = pd.to_numeric(normalized["horizon_hours"], errors="coerce")
    return normalized


def _summary_cache_sort_columns(frame: pd.DataFrame) -> list[str]:
    sort_columns = ["timestamp"]
    if "horizon_hours" in frame.columns:
        sort_columns.append("horizon_hours")
    return sort_columns


def _summary_cache_dedupe_columns(frame: pd.DataFrame) -> list[str]:
    dedupe_columns = ["timestamp", "symbol"]
    if "horizon_hours" in frame.columns:
        dedupe_columns.append("horizon_hours")
    return dedupe_columns


def _dedupe_summary_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_summary_cache_frame(frame)
    if normalized.empty:
        return normalized.reset_index(drop=True)
    return (
        normalized
        .drop_duplicates(subset=_summary_cache_dedupe_columns(normalized), keep="last")
        .sort_values(_summary_cache_sort_columns(normalized))
        .reset_index(drop=True)
    )


def _append_summary_cache_frame(existing: pd.DataFrame, payload: Mapping[str, object]) -> pd.DataFrame:
    payload_frame = _normalize_summary_cache_frame(pd.DataFrame([dict(payload)]))
    if existing.empty:
        return payload_frame.reset_index(drop=True)
    return pd.concat([existing, payload_frame], ignore_index=True)


@dataclass(frozen=True)
class NarrativeSummaryRecord:
    timestamp: pd.Timestamp
    symbol: str
    issued_at: pd.Timestamp
    target_timestamp: pd.Timestamp
    horizon_hours: int
    provider: str
    model: str
    factual_summary: str
    predictive_signals: str
    summary_text: str
    signal_strength: float
    confidence: float
    expected_move_pct: float
    recent_return_24h: float
    recent_return_168h: float
    realized_vol_24h: float
    volume_z_24h: float
    generated_at: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "issued_at": self.issued_at,
            "target_timestamp": self.target_timestamp,
            "horizon_hours": int(self.horizon_hours),
            "provider": self.provider,
            "model": self.model,
            "factual_summary": self.factual_summary,
            "predictive_signals": self.predictive_signals,
            "summary_text": self.summary_text,
            "signal_strength": float(self.signal_strength),
            "confidence": float(self.confidence),
            "expected_move_pct": float(self.expected_move_pct),
            "recent_return_24h": float(self.recent_return_24h),
            "recent_return_168h": float(self.recent_return_168h),
            "realized_vol_24h": float(self.realized_vol_24h),
            "volume_z_24h": float(self.volume_z_24h),
            "generated_at": self.generated_at,
        }


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _history_slice(
    history: pd.DataFrame,
    *,
    issued_at: pd.Timestamp,
    context_hours: int,
) -> pd.DataFrame:
    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    issued_at = pd.to_datetime(issued_at, utc=True)
    start = issued_at - pd.Timedelta(hours=max(1, int(context_hours)))
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= issued_at)].copy()
    return frame.reset_index(drop=True)


def _heuristic_inputs(
    history: pd.DataFrame,
    *,
    issued_at: pd.Timestamp,
    horizon_hours: int,
    context_hours: int,
    prepared_context: pd.DataFrame | None = None,
) -> dict[str, float]:
    context = prepared_context if prepared_context is not None else _history_slice(
        history,
        issued_at=issued_at,
        context_hours=context_hours,
    )
    if context.empty:
        raise ValueError("Narrative summary requires non-empty history context.")

    close = pd.to_numeric(context["close"], errors="coerce")
    volume = pd.to_numeric(context.get("volume", pd.Series(index=context.index, dtype=float)), errors="coerce")
    recent_close = _safe_float(close.iloc[-1], default=0.0)

    ret_1h = _safe_float(close.pct_change().iloc[-1], default=0.0) if len(close) > 1 else 0.0
    ret_24h = _safe_float((recent_close / close.iloc[-25]) - 1.0, default=0.0) if len(close) > 24 and close.iloc[-25] else 0.0
    ret_72h = _safe_float((recent_close / close.iloc[-73]) - 1.0, default=0.0) if len(close) > 72 and close.iloc[-73] else 0.0
    ret_168h = _safe_float((recent_close / close.iloc[-169]) - 1.0, default=ret_72h) if len(close) > 168 and close.iloc[-169] else ret_72h

    returns = close.pct_change().dropna()
    vol_24h = _safe_float(returns.tail(24).std(ddof=0), default=0.0)
    vol_72h = _safe_float(returns.tail(72).std(ddof=0), default=vol_24h)

    highs = pd.to_numeric(context.get("high", close), errors="coerce")
    lows = pd.to_numeric(context.get("low", close), errors="coerce")
    range_pct = _safe_float(((highs.tail(24).max() - lows.tail(24).min()) / max(recent_close, 1e-8)), default=0.0)

    volume_mean = _safe_float(volume.tail(24).mean(), default=0.0)
    volume_std = _safe_float(volume.tail(24).std(ddof=0), default=0.0)
    volume_z = 0.0
    if volume_mean > 0.0 and volume_std > 1e-8:
        volume_z = _clip(((_safe_float(volume.iloc[-1], default=volume_mean) - volume_mean) / volume_std), -4.0, 4.0)

    trend_consistency = 1.0 if (ret_24h >= 0.0) == (ret_168h >= 0.0) else -1.0
    horizon_scale = min(math.sqrt(max(1.0, float(horizon_hours))), 4.0)

    vol_anchor = max(vol_24h, 0.25 * vol_72h, 0.0015)
    signal_strength = np.tanh(
        2.2 * ret_24h
        + 1.1 * ret_168h
        + 0.35 * ret_1h
        + 0.05 * volume_z * trend_consistency
    )
    confidence = _clip(
        0.20
        + 0.45 * abs(signal_strength)
        + 0.15 * max(0.0, trend_consistency)
        + 0.10 * min(1.0, range_pct / max(vol_anchor, 1e-8)),
        0.05,
        0.95,
    )
    expected_move_pct = _clip(float(signal_strength) * vol_anchor * horizon_scale, -0.12, 0.12)

    return {
        "last_close": recent_close,
        "ret_1h": ret_1h,
        "ret_24h": ret_24h,
        "ret_72h": ret_72h,
        "ret_168h": ret_168h,
        "vol_24h": vol_24h,
        "vol_72h": vol_72h,
        "range_pct": range_pct,
        "volume_z": volume_z,
        "signal_strength": float(signal_strength),
        "confidence": confidence,
        "expected_move_pct": expected_move_pct,
    }


def _format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def _build_summary_text(factual_summary: str, predictive_signals: str) -> str:
    return f"FACTUAL SUMMARY:\n{factual_summary.strip()}\n\nPREDICTIVE SIGNALS:\n{predictive_signals.strip()}"


def _heuristic_summary(
    *,
    symbol: str,
    history: pd.DataFrame,
    issued_at: pd.Timestamp,
    target_timestamp: pd.Timestamp,
    horizon_hours: int,
    context_hours: int,
    prepared_context: pd.DataFrame | None = None,
) -> NarrativeSummaryRecord:
    stats = _heuristic_inputs(
        history,
        issued_at=issued_at,
        horizon_hours=horizon_hours,
        context_hours=context_hours,
        prepared_context=prepared_context,
    )
    directional_word = "bullish" if stats["signal_strength"] > 0.15 else "bearish" if stats["signal_strength"] < -0.15 else "neutral"
    intensity_word = "mildly" if abs(stats["signal_strength"]) < 0.35 else "moderately" if abs(stats["signal_strength"]) < 0.65 else "strongly"
    factual_summary = (
        f"{symbol} last closed near {stats['last_close']:.4f}. "
        f"Recent momentum was 1h {_format_pct(stats['ret_1h'])}, 24h {_format_pct(stats['ret_24h'])}, "
        f"72h {_format_pct(stats['ret_72h'])}, and 7d {_format_pct(stats['ret_168h'])}. "
        f"Realized 24h volatility was {stats['vol_24h'] * 100.0:.2f}%, 24h range was {stats['range_pct'] * 100.0:.2f}%, "
        f"and volume z-score was {stats['volume_z']:+.2f}."
    )
    predictive_signals = (
        f"Near-term narrative is {intensity_word} {directional_word}. "
        f"Expected move over roughly {int(horizon_hours)}h is {_format_pct(stats['expected_move_pct'])} with "
        f"confidence {stats['confidence']:.2f}. "
        f"Signal strength is {stats['signal_strength']:+.2f} on a -1 to +1 scale, with higher conviction when recent trend and participation stay aligned."
    )
    summary_text = _build_summary_text(factual_summary, predictive_signals)
    return NarrativeSummaryRecord(
        timestamp=pd.to_datetime(target_timestamp, utc=True),
        symbol=str(symbol).upper(),
        issued_at=pd.to_datetime(issued_at, utc=True),
        target_timestamp=pd.to_datetime(target_timestamp, utc=True),
        horizon_hours=int(horizon_hours),
        provider="heuristic",
        model=_SUMMARY_MODEL_DEFAULTS["heuristic"],
        factual_summary=factual_summary,
        predictive_signals=predictive_signals,
        summary_text=summary_text,
        signal_strength=float(stats["signal_strength"]),
        confidence=float(stats["confidence"]),
        expected_move_pct=float(stats["expected_move_pct"]),
        recent_return_24h=float(stats["ret_24h"]),
        recent_return_168h=float(stats["ret_168h"]),
        realized_vol_24h=float(stats["vol_24h"]),
        volume_z_24h=float(stats["volume_z"]),
        generated_at=pd.Timestamp.now(tz="UTC"),
    )


def _json_payload_from_text(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("Empty summary response.")
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _summary_prompt(
    *,
    symbol: str,
    issued_at: pd.Timestamp,
    horizon_hours: int,
    stats: Mapping[str, float],
) -> str:
    return (
        "You are generating narrative context for a time-series forecast.\n"
        "Return JSON with these keys only: factual_summary, predictive_signals, signal_strength, confidence, expected_move_pct.\n"
        "Constraints:\n"
        "- factual_summary: concise description of what happened, numeric where possible.\n"
        "- predictive_signals: concise forward-looking outlook using relative language.\n"
        "- signal_strength: number from -1.0 to 1.0.\n"
        "- confidence: number from 0.0 to 1.0.\n"
        "- expected_move_pct: decimal percent move over the forecast horizon, e.g. 0.012 for +1.2%.\n"
        "- Do not include markdown fences.\n\n"
        f"Symbol: {symbol}\n"
        f"Issued at: {pd.to_datetime(issued_at, utc=True).isoformat()}\n"
        f"Horizon hours: {int(horizon_hours)}\n"
        f"Last close: {stats['last_close']:.6f}\n"
        f"Return 1h: {_format_pct(stats['ret_1h'])}\n"
        f"Return 24h: {_format_pct(stats['ret_24h'])}\n"
        f"Return 72h: {_format_pct(stats['ret_72h'])}\n"
        f"Return 7d: {_format_pct(stats['ret_168h'])}\n"
        f"Realized vol 24h: {stats['vol_24h'] * 100.0:.3f}%\n"
        f"Realized vol 72h: {stats['vol_72h'] * 100.0:.3f}%\n"
        f"Range 24h: {stats['range_pct'] * 100.0:.3f}%\n"
        f"Volume z-score 24h: {stats['volume_z']:+.3f}\n"
        f"Heuristic prior signal: {stats['signal_strength']:+.3f}\n"
        f"Heuristic prior confidence: {stats['confidence']:.3f}\n"
        f"Heuristic prior expected move: {_format_pct(stats['expected_move_pct'])}\n"
    )


def _call_openai_summary(prompt: str, *, model: str) -> dict[str, Any]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Produce concise JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return _json_payload_from_text(content)


def _call_anthropic_summary(prompt: str, *, model: str) -> dict[str, Any]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts: list[str] = []
    for block in getattr(response, "content", []):
        maybe_text = getattr(block, "text", None)
        if maybe_text:
            text_parts.append(str(maybe_text))
    return _json_payload_from_text("\n".join(text_parts))


def _call_xai_tools_summary(prompt: str, *, model: str) -> dict[str, Any]:
    from pydantic import BaseModel, Field
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    from xai_sdk.tools import web_search, x_search

    class NarrativePayload(BaseModel):
        factual_summary: str = Field(description="What happened")
        predictive_signals: str = Field(description="Forward-looking outlook")
        signal_strength: float = Field(description="Range -1.0 to 1.0")
        confidence: float = Field(description="Range 0.0 to 1.0")
        expected_move_pct: float = Field(description="Decimal expected move over the horizon")

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set.")
    client = Client(api_key=api_key)
    chat = client.chat.create(model=model, tools=[web_search(), x_search()])
    chat.append(system(
        "You are generating narrative context for a financial time-series forecast. "
        "Use web and X search if useful for current context, then return structured JSON."
    ))
    chat.append(user(prompt))
    _response, parsed = chat.parse(NarrativePayload)
    return parsed.model_dump()


def _generate_llm_summary(
    *,
    backend: str,
    model: str,
    symbol: str,
    history: pd.DataFrame,
    issued_at: pd.Timestamp,
    target_timestamp: pd.Timestamp,
    horizon_hours: int,
    context_hours: int,
    prepared_context: pd.DataFrame | None = None,
) -> NarrativeSummaryRecord:
    stats = _heuristic_inputs(
        history,
        issued_at=issued_at,
        horizon_hours=horizon_hours,
        context_hours=context_hours,
        prepared_context=prepared_context,
    )
    prompt = _summary_prompt(
        symbol=symbol,
        issued_at=issued_at,
        horizon_hours=horizon_hours,
        stats=stats,
    )
    if backend == "openai":
        payload = _call_openai_summary(prompt, model=model)
    elif backend == "anthropic":
        payload = _call_anthropic_summary(prompt, model=model)
    elif backend == "xai_tools":
        payload = _call_xai_tools_summary(prompt, model=model)
    else:  # pragma: no cover - caller guards this
        raise ValueError(f"Unsupported LLM backend '{backend}'.")

    factual_summary = str(payload.get("factual_summary", "")).strip() or "No factual summary returned."
    predictive_signals = str(payload.get("predictive_signals", "")).strip() or "No predictive signals returned."
    signal_strength = _clip(_safe_float(payload.get("signal_strength"), default=stats["signal_strength"]), -1.0, 1.0)
    confidence = _clip(_safe_float(payload.get("confidence"), default=stats["confidence"]), 0.0, 1.0)
    expected_move_pct = _clip(
        _safe_float(payload.get("expected_move_pct"), default=stats["expected_move_pct"]),
        -0.20,
        0.20,
    )
    summary_text = _build_summary_text(factual_summary, predictive_signals)
    return NarrativeSummaryRecord(
        timestamp=pd.to_datetime(target_timestamp, utc=True),
        symbol=str(symbol).upper(),
        issued_at=pd.to_datetime(issued_at, utc=True),
        target_timestamp=pd.to_datetime(target_timestamp, utc=True),
        horizon_hours=int(horizon_hours),
        provider=backend,
        model=model,
        factual_summary=factual_summary,
        predictive_signals=predictive_signals,
        summary_text=summary_text,
        signal_strength=signal_strength,
        confidence=confidence,
        expected_move_pct=expected_move_pct,
        recent_return_24h=float(stats["ret_24h"]),
        recent_return_168h=float(stats["ret_168h"]),
        realized_vol_24h=float(stats["vol_24h"]),
        volume_z_24h=float(stats["volume_z"]),
        generated_at=pd.Timestamp.now(tz="UTC"),
    )


def generate_summary_record(
    *,
    backend: str,
    model: str | None,
    symbol: str,
    history: pd.DataFrame,
    issued_at: pd.Timestamp,
    target_timestamp: pd.Timestamp,
    horizon_hours: int,
    context_hours: int,
    prepared_context: pd.DataFrame | None = None,
) -> NarrativeSummaryRecord:
    resolved_backend = normalize_narrative_backend(backend)
    resolved_model = str(model or _SUMMARY_MODEL_DEFAULTS.get(resolved_backend, "heuristic")).strip()
    if resolved_backend == "off":
        raise ValueError("Narrative backend 'off' does not generate summaries.")
    if resolved_backend == "heuristic":
        return _heuristic_summary(
            symbol=symbol,
            history=history,
            issued_at=issued_at,
            target_timestamp=target_timestamp,
            horizon_hours=horizon_hours,
            context_hours=context_hours,
            prepared_context=prepared_context,
        )
    try:
        return _generate_llm_summary(
            backend=resolved_backend,
            model=resolved_model,
            symbol=symbol,
            history=history,
            issued_at=issued_at,
            target_timestamp=target_timestamp,
            horizon_hours=horizon_hours,
            context_hours=context_hours,
            prepared_context=prepared_context,
        )
    except Exception as exc:
        logger.warning(
            "Narrative summary backend=%s failed for %s at %s: %s. Falling back to heuristic.",
            resolved_backend,
            symbol,
            pd.to_datetime(target_timestamp, utc=True),
            exc,
        )
        return _heuristic_summary(
            symbol=symbol,
            history=history,
            issued_at=issued_at,
            target_timestamp=target_timestamp,
            horizon_hours=horizon_hours,
            context_hours=context_hours,
            prepared_context=prepared_context,
        )


def _record_from_row(
    row: Mapping[str, Any],
    *,
    default_horizon_hours: int | None = None,
) -> NarrativeSummaryRecord:
    factual_summary = str(row.get("factual_summary", "")).strip()
    predictive_signals = str(row.get("predictive_signals", "")).strip()
    summary_text = str(row.get("summary_text", "")).strip()
    if not summary_text and (factual_summary or predictive_signals):
        summary_text = _build_summary_text(factual_summary, predictive_signals)
    if not summary_text:
        raise ValueError("summary_text")
    horizon_value = row.get("horizon_hours", default_horizon_hours)
    if horizon_value is None or pd.isna(horizon_value):
        if default_horizon_hours is None:
            raise KeyError("horizon_hours")
        horizon_hours = int(default_horizon_hours)
    else:
        horizon_hours = int(horizon_value)
    return NarrativeSummaryRecord(
        timestamp=pd.to_datetime(row["timestamp"], utc=True),
        symbol=str(row["symbol"]).upper(),
        issued_at=pd.to_datetime(row["issued_at"], utc=True),
        target_timestamp=pd.to_datetime(row["target_timestamp"], utc=True),
        horizon_hours=horizon_hours,
        provider=str(row.get("provider", "heuristic")),
        model=str(row.get("model", "heuristic")),
        factual_summary=factual_summary,
        predictive_signals=predictive_signals,
        summary_text=summary_text,
        signal_strength=_safe_float(row.get("signal_strength"), default=0.0),
        confidence=_clip(_safe_float(row.get("confidence"), default=0.0), 0.0, 1.0),
        expected_move_pct=_safe_float(row.get("expected_move_pct"), default=0.0),
        recent_return_24h=_safe_float(row.get("recent_return_24h"), default=0.0),
        recent_return_168h=_safe_float(row.get("recent_return_168h"), default=0.0),
        realized_vol_24h=_safe_float(row.get("realized_vol_24h"), default=0.0),
        volume_z_24h=_safe_float(row.get("volume_z_24h"), default=0.0),
        generated_at=pd.to_datetime(row.get("generated_at"), utc=True, errors="coerce"),
    )


def _merge_summary_cache_frame(existing: pd.DataFrame, payload: Mapping[str, object]) -> pd.DataFrame:
    return _dedupe_summary_cache_frame(_append_summary_cache_frame(existing, payload))


def summarize_forecast_row(
    *,
    symbol: str,
    history: pd.DataFrame,
    row: Mapping[str, Any],
    backend: str,
    model: str | None,
    summary_cache_dir: Path,
    context_hours: int,
    force_rebuild: bool = False,
) -> NarrativeSummaryRecord:
    cache = NarrativeSummaryCache(summary_cache_dir)
    record, _updated, _dirty = _summarize_forecast_row_with_cache(
        symbol=symbol,
        history=history,
        row=row,
        backend=backend,
        model=model,
        cache=cache,
        existing=cache.load(symbol),
        context_hours=context_hours,
        force_rebuild=force_rebuild,
    )
    return record


def _summarize_forecast_row_with_cache(
    *,
    symbol: str,
    history: pd.DataFrame,
    row: Mapping[str, Any],
    backend: str,
    model: str | None,
    cache: NarrativeSummaryCache,
    existing: pd.DataFrame,
    context_hours: int,
    force_rebuild: bool = False,
    persist: bool = True,
    prepared_context: pd.DataFrame | None = None,
) -> tuple[NarrativeSummaryRecord, pd.DataFrame, bool]:
    target_ts = pd.to_datetime(row["timestamp"], utc=True)
    issued_at = pd.to_datetime(row.get("issued_at", target_ts), utc=True)
    target_timestamp = pd.to_datetime(row.get("target_timestamp", target_ts), utc=True)
    horizon_hours = int(row.get("horizon_hours", 1))

    if not force_rebuild and not existing.empty and "timestamp" in existing.columns:
        match = existing[existing["timestamp"] == target_ts]
        if not match.empty and "horizon_hours" in match.columns:
            horizon_values = pd.to_numeric(match["horizon_hours"], errors="coerce")
            exact_match = match[horizon_values == horizon_hours]
            if exact_match.empty:
                exact_match = match[horizon_values.isna()]
            match = exact_match
        if not match.empty:
            try:
                return _record_from_row(
                    match.iloc[-1].to_dict(),
                    default_horizon_hours=horizon_hours,
                ), existing, False
            except Exception as exc:
                logger.warning(
                    "Ignoring invalid narrative summary cache row for %s at %s: %s",
                    str(symbol).upper(),
                    target_ts,
                    exc,
                )

    record = generate_summary_record(
        backend=backend,
        model=model,
        symbol=symbol,
        history=history,
        issued_at=issued_at,
        target_timestamp=target_timestamp,
        horizon_hours=horizon_hours,
        context_hours=context_hours,
        prepared_context=prepared_context,
    )

    payload = record.to_dict()
    combined = (
        _merge_summary_cache_frame(existing, payload)
        if persist
        else _append_summary_cache_frame(existing, payload)
    )
    if persist:
        cache.write(symbol, combined)
    return record, combined, True


def apply_summary_to_forecast_row(
    row: Mapping[str, Any],
    *,
    last_close: float,
    summary: NarrativeSummaryRecord,
) -> dict[str, object]:
    out = dict(row)
    if not math.isfinite(last_close) or last_close <= 0.0:
        return out

    def _get(name: str) -> float | None:
        if name not in row:
            return None
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    base_close = _get("predicted_close_p50")
    base_high = _get("predicted_high_p50")
    base_low = _get("predicted_low_p50")
    if base_close is None or base_high is None or base_low is None:
        return out

    blend = _clip(0.15 + 0.55 * float(summary.confidence), 0.15, 0.70)
    base_delta = (base_close / last_close) - 1.0
    target_delta = base_delta + float(summary.expected_move_pct)
    adjusted_delta = ((1.0 - blend) * base_delta) + (blend * target_delta)
    adjusted_close = max(1e-8, last_close * (1.0 + adjusted_delta))

    up_spread = max(0.0, base_high - base_close)
    down_spread = max(0.0, base_close - base_low)
    spread_scale = 1.0 + (0.35 * float(summary.confidence))
    adjusted_high = max(adjusted_close, adjusted_close + (up_spread * spread_scale))
    adjusted_low = min(adjusted_close, adjusted_close - (down_spread * spread_scale))

    out["base_predicted_close_p50"] = base_close
    out["base_predicted_high_p50"] = base_high
    out["base_predicted_low_p50"] = base_low
    out["predicted_close_p50"] = adjusted_close
    out["predicted_high_p50"] = adjusted_high
    out["predicted_low_p50"] = adjusted_low
    out["narrative_provider"] = summary.provider
    out["narrative_model"] = summary.model
    out["narrative_summary"] = summary.summary_text
    out["narrative_factual_summary"] = summary.factual_summary
    out["narrative_predictive_signals"] = summary.predictive_signals
    out["narrative_signal_strength"] = float(summary.signal_strength)
    out["narrative_confidence"] = float(summary.confidence)
    out["narrative_expected_move_pct"] = float(summary.expected_move_pct)

    for quantile_name in ("predicted_close_p10", "predicted_close_p90"):
        quantile_value = _get(quantile_name)
        if quantile_value is None:
            continue
        out[f"base_{quantile_name}"] = quantile_value
        quantile_delta = (quantile_value / last_close) - 1.0
        out[quantile_name] = max(
            1e-8,
            last_close * (1.0 + (((1.0 - blend) * quantile_delta) + (blend * (quantile_delta + float(summary.expected_move_pct))))),
        )

    if repair_forecast_ohlc is not None:
        repaired = repair_forecast_ohlc(
            last_close=last_close,
            close_p50=out.get("predicted_close_p50"),
            close_p10=out.get("predicted_close_p10"),
            close_p90=out.get("predicted_close_p90"),
            high_p50=out.get("predicted_high_p50"),
            low_p50=out.get("predicted_low_p50"),
        )
        out["predicted_close_p10"] = repaired.close_p10
        out["predicted_close_p50"] = repaired.close_p50
        out["predicted_close_p90"] = repaired.close_p90
        out["predicted_high_p50"] = repaired.high_p50
        out["predicted_low_p50"] = repaired.low_p50
    else:
        out["predicted_high_p50"] = max(
            float(out["predicted_high_p50"]),
            float(out["predicted_close_p50"]),
            float(out["predicted_low_p50"]),
        )
        out["predicted_low_p50"] = min(
            float(out["predicted_low_p50"]),
            float(out["predicted_close_p50"]),
            float(out["predicted_high_p50"]),
        )
    return out


def apply_narrative_overlay(
    forecast_frame: pd.DataFrame,
    *,
    symbol: str,
    history: pd.DataFrame,
    backend: str,
    model: str | None,
    forecast_cache_dir: Path,
    summary_cache_dir: Path | None = None,
    context_hours: int = 24 * 7,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    resolved_backend = normalize_narrative_backend(backend)
    if resolved_backend == "off" or forecast_frame.empty:
        return forecast_frame

    history_frame = history.copy()
    history_frame["timestamp"] = pd.to_datetime(history_frame["timestamp"], utc=True, errors="coerce")
    history_frame = history_frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    summary_dir = resolve_summary_cache_dir(
        forecast_cache_dir=Path(forecast_cache_dir),
        summary_cache_dir=summary_cache_dir,
    )
    cache = NarrativeSummaryCache(summary_dir)
    existing_summaries = cache.load(symbol)
    cache_dirty = False
    prepared_contexts: dict[pd.Timestamp, pd.DataFrame] = {}

    adjusted_rows: list[dict[str, object]] = []
    for item in forecast_frame.to_dict(orient="records"):
        try:
            target_ts = pd.to_datetime(item["timestamp"], utc=True)
            issued_at = pd.to_datetime(item.get("issued_at", target_ts), utc=True)
            context = prepared_contexts.get(issued_at)
            if context is None:
                context = _history_slice(
                    history_frame,
                    issued_at=issued_at,
                    context_hours=context_hours,
                )
                prepared_contexts[issued_at] = context
            if context.empty:
                adjusted_rows.append(dict(item))
                continue
            last_close = _safe_float(context.iloc[-1].get("close"), default=0.0)
            summary, existing_summaries, row_dirty = _summarize_forecast_row_with_cache(
                symbol=symbol,
                history=history_frame,
                row=item,
                backend=resolved_backend,
                model=model,
                cache=cache,
                existing=existing_summaries,
                context_hours=context_hours,
                force_rebuild=force_rebuild,
                persist=False,
                prepared_context=context,
            )
            cache_dirty = cache_dirty or row_dirty
            adjusted_rows.append(
                apply_summary_to_forecast_row(
                    item,
                    last_close=last_close,
                    summary=summary,
                )
            )
        except Exception as exc:
            row_symbol = str(item.get("symbol", symbol)).upper()
            logger.warning(
                "Skipping narrative overlay for %s row %r: %s",
                row_symbol,
                item.get("timestamp"),
                exc,
            )
            adjusted_rows.append(dict(item))
    if cache_dirty:
        cache.write(symbol, existing_summaries)
    return pd.DataFrame(adjusted_rows)


__all__ = [
    "NarrativeSummaryCache",
    "NarrativeSummaryRecord",
    "SUPPORTED_NARRATIVE_BACKENDS",
    "apply_narrative_overlay",
    "apply_summary_to_forecast_row",
    "generate_summary_record",
    "normalize_narrative_backend",
    "resolve_horizon_summary_cache_dir",
    "resolve_summary_cache_dir",
    "summarize_forecast_row",
]
