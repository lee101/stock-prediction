from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.robust_trading_metrics import summarize_scenario_results


DEFAULT_ACTIVE_CHECKPOINT_LINK = Path("neuraldailytraining") / "checkpoints" / "active_latest.pt"
DEFAULT_ACTIVE_CONFIG_PATH = Path("neuraldailytraining") / "checkpoints" / "active_latest.json"


def parse_optional_float_grid(
    values: str | Sequence[str] | None,
    *,
    allow_none: bool = False,
) -> tuple[float | None, ...]:
    if values is None:
        return ()
    raw_items: list[str] = []
    if isinstance(values, str):
        raw_items.extend(values.split(","))
    else:
        for value in values:
            raw_items.extend(str(value).split(","))

    resolved: list[float | None] = []
    seen: set[float | None] = set()
    for item in raw_items:
        token = item.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in {"none", "null", "off", "disabled"}:
            if not allow_none:
                raise ValueError("None values are not allowed in this grid.")
            if None not in seen:
                seen.add(None)
                resolved.append(None)
            continue
        numeric = float(token)
        if numeric not in seen:
            seen.add(numeric)
            resolved.append(numeric)
    return tuple(resolved)


def build_recent_window_start_dates(
    available_dates: Sequence[pd.Timestamp],
    *,
    window_days: int,
    max_windows: int,
    stride_days: int,
    explicit_start_dates: Sequence[str] | None = None,
) -> list[str]:
    if explicit_start_dates:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in explicit_start_dates:
            token = pd.to_datetime(value, utc=True).strftime("%Y-%m-%d")
            if token not in seen:
                seen.add(token)
                normalized.append(token)
        return normalized

    ordered = sorted(pd.to_datetime(list(available_dates), utc=True))
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if stride_days <= 0:
        raise ValueError("stride_days must be positive")
    if len(ordered) < window_days:
        raise ValueError(f"Need at least {window_days} dates, found {len(ordered)}")

    latest_start_idx = len(ordered) - window_days
    starts: list[str] = []
    cursor = latest_start_idx
    while cursor >= 0 and len(starts) < max_windows:
        starts.append(ordered[cursor].strftime("%Y-%m-%d"))
        cursor -= stride_days
    starts.reverse()
    return starts


def annualized_return_pct(total_return: float, periods: int) -> float:
    if periods <= 0:
        return 0.0
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -100.0
    return float(((base ** (365.0 / float(periods))) - 1.0) * 100.0)


def build_scenario_row(
    *,
    start_date: str,
    days: int,
    summary: Mapping[str, Any],
    risk_threshold: float | None,
    confidence_threshold: float | None,
) -> dict[str, Any]:
    total_return = float(summary.get("total_return", 0.0) or 0.0)
    return {
        "start_date": str(start_date),
        "days": int(days),
        "risk_threshold": None if risk_threshold is None else float(risk_threshold),
        "confidence_threshold": None if confidence_threshold is None else float(confidence_threshold),
        "return_pct": total_return * 100.0,
        "annualized_return_pct": annualized_return_pct(total_return, int(days)),
        "sortino": float(summary.get("sortino", 0.0) or 0.0),
        "max_drawdown_pct": float(summary.get("max_drawdown", 0.0) or 0.0) * 100.0,
        "pnl_smoothness": float(summary.get("pnl_smoothness", 0.0) or 0.0),
        "trade_count": float(summary.get("trade_count", 0.0) or 0.0),
        "goodness_score": float(summary.get("goodness_score", 0.0) or 0.0),
        "final_equity": float(summary.get("final_equity", 0.0) or 0.0),
        "pnl": float(summary.get("pnl", 0.0) or 0.0),
    }


def summarize_threshold_scenarios(
    rows: Sequence[Mapping[str, Any]],
    *,
    sortino_clip: float = 10.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float | None, float | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            None if row.get("risk_threshold") is None else float(row["risk_threshold"]),
            None if row.get("confidence_threshold") is None else float(row["confidence_threshold"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (risk_threshold, confidence_threshold), group_rows in grouped.items():
        summary = summarize_scenario_results(group_rows, sortino_clip=sortino_clip)
        goodness_values = np.asarray([float(row.get("goodness_score", 0.0) or 0.0) for row in group_rows], dtype=float)
        summary.update(
            {
                "risk_threshold": risk_threshold,
                "confidence_threshold": confidence_threshold,
                "scenario_count": len(group_rows),
                "goodness_score_mean": float(goodness_values.mean()) if goodness_values.size else 0.0,
                "goodness_score_min": float(goodness_values.min()) if goodness_values.size else 0.0,
            }
        )
        summaries.append(summary)

    summaries.sort(
        key=lambda item: (
            float(item.get("robust_score", 0.0) or 0.0),
            float(item.get("return_p25_pct", 0.0) or 0.0),
            float(item.get("sortino_p25", 0.0) or 0.0),
            -float(item.get("pnl_smoothness_mean", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return summaries


def selection_metric_value(summary: Mapping[str, Any], selection_metric: str) -> float:
    value = float(summary.get(selection_metric, 0.0) or 0.0)
    if selection_metric in {"pnl_smoothness_mean", "max_drawdown_mean_pct", "max_drawdown_worst_pct"}:
        return -value
    return value


def selection_metric_sort_key(summary: Mapping[str, Any], selection_metric: str) -> tuple[float, float, float, float]:
    return (
        selection_metric_value(summary, selection_metric),
        float(summary.get("robust_score", 0.0) or 0.0),
        float(summary.get("sortino_p25", 0.0) or 0.0),
        -float(summary.get("pnl_smoothness_mean", 0.0) or 0.0),
    )


def should_promote_candidate(
    candidate_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any] | None,
    *,
    min_robust_improvement: float = 0.0,
    min_return_p25_pct: float = 0.0,
    min_sortino_p25: float = 0.0,
) -> bool:
    candidate_robust = float(candidate_summary.get("robust_score", 0.0) or 0.0)
    baseline_robust = float("-inf")
    if baseline_summary is not None:
        baseline_robust = float(baseline_summary.get("robust_score", 0.0) or 0.0)
    return (
        candidate_robust >= baseline_robust + float(min_robust_improvement)
        and float(candidate_summary.get("return_p25_pct", 0.0) or 0.0) >= float(min_return_p25_pct)
        and float(candidate_summary.get("sortino_p25", 0.0) or 0.0) >= float(min_sortino_p25)
    )


def write_deployment_config(
    path: str | Path,
    *,
    checkpoint: str | Path,
    risk_threshold: float | None,
    confidence_threshold: float | None,
    symbols: Sequence[str],
    selection_metric: str,
    selection_value: float,
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "risk_threshold": None if risk_threshold is None else float(risk_threshold),
        "confidence_threshold": None if confidence_threshold is None else float(confidence_threshold),
        "symbols": [str(symbol).upper() for symbol in symbols],
        "selection_metric": str(selection_metric),
        "selection_value": float(selection_value),
        "summary": dict(summary),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def load_deployment_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Deployment config must contain a JSON object.")
    return payload


def resolve_deployment_settings(
    *,
    deployment_payload: Mapping[str, Any] | None,
    checkpoint: str | None = None,
    symbols: Sequence[str] | None = None,
    risk_threshold: float | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    payload = dict(deployment_payload or {})
    resolved_checkpoint = checkpoint or payload.get("checkpoint")
    if not resolved_checkpoint:
        raise ValueError("Either checkpoint or deployment_payload['checkpoint'] is required.")

    if symbols:
        resolved_symbols = tuple(str(symbol).upper() for symbol in symbols)
    else:
        resolved_symbols = tuple(str(symbol).upper() for symbol in payload.get("symbols", []) if symbol)

    resolved_risk_threshold = (
        float(risk_threshold)
        if risk_threshold is not None
        else (
            None
            if payload.get("risk_threshold") is None
            else float(payload["risk_threshold"])
        )
    )
    resolved_confidence_threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else (
            None
            if payload.get("confidence_threshold") is None
            else float(payload["confidence_threshold"])
        )
    )
    return {
        "checkpoint": str(resolved_checkpoint),
        "symbols": resolved_symbols,
        "risk_threshold": resolved_risk_threshold,
        "confidence_threshold": resolved_confidence_threshold,
    }
