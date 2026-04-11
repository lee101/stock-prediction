from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalize_checkpoint_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _normalize_symbol_list(symbols: Any) -> list[str]:
    if not isinstance(symbols, list):
        return []
    return [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]


def _parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ManifestEvalTargetSpec:
    symbols: tuple[str, ...]
    leverage: float | None
    preferred_target_label: str | None = "launch_target"


def evaluation_has_target_metadata(evaluation: dict[str, Any]) -> bool:
    if str(evaluation.get("target_label") or "").strip():
        return True
    if _normalize_symbol_list(evaluation.get("symbols")):
        return True
    return _parse_optional_float(evaluation.get("leverage")) is not None


def evaluation_matches_target_spec(
    evaluation: dict[str, Any],
    target_spec: ManifestEvalTargetSpec,
) -> bool:
    if target_spec.preferred_target_label is not None:
        evaluation_target_label = str(evaluation.get("target_label") or "").strip()
        if evaluation_target_label and evaluation_target_label != target_spec.preferred_target_label:
            return False

    evaluation_symbols = _normalize_symbol_list(evaluation.get("symbols"))
    if target_spec.symbols and evaluation_symbols and evaluation_symbols != list(target_spec.symbols):
        return False

    evaluation_leverage = _parse_optional_float(evaluation.get("leverage"))
    if target_spec.leverage is None or evaluation_leverage is None:
        return True
    return math.isclose(
        evaluation_leverage,
        float(target_spec.leverage),
        rel_tol=0.0,
        abs_tol=1e-9,
    )




def requested_checkpoints(launch_checkpoint: str | Path | None, candidate_checkpoints: list[str | Path]) -> tuple[str, ...]:
    checkpoints: list[str] = []
    seen: set[str] = set()
    if launch_checkpoint:
        normalized_launch = _normalize_checkpoint_path(launch_checkpoint)
        seen.add(normalized_launch)
        checkpoints.append(normalized_launch)
    for checkpoint in candidate_checkpoints:
        normalized_checkpoint = _normalize_checkpoint_path(checkpoint)
        if normalized_checkpoint in seen:
            continue
        seen.add(normalized_checkpoint)
        checkpoints.append(normalized_checkpoint)
    return tuple(checkpoints)

def find_target_evaluation(
    payload: dict[str, Any],
    checkpoint: str | Path,
    *,
    target_spec: ManifestEvalTargetSpec,
) -> dict[str, Any] | None:
    target = _normalize_checkpoint_path(checkpoint)
    fallback: dict[str, Any] | None = None
    has_explicit_target_metadata = False
    for evaluation in payload.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        if _normalize_checkpoint_path(str(evaluation.get("checkpoint", ""))) != target:
            continue
        if fallback is None:
            fallback = evaluation
        has_explicit_target_metadata = has_explicit_target_metadata or evaluation_has_target_metadata(evaluation)
        if evaluation_matches_target_spec(evaluation, target_spec):
            return evaluation
    return None if has_explicit_target_metadata else fallback


def select_target_evaluations(
    payload: dict[str, Any],
    *,
    target_spec: ManifestEvalTargetSpec,
) -> list[dict[str, Any]]:
    first_by_checkpoint: dict[str, dict[str, Any]] = {}
    matched_by_checkpoint: dict[str, dict[str, Any]] = {}
    metadata_by_checkpoint: dict[str, bool] = {}
    ordered_checkpoints: list[str] = []
    for evaluation in payload.get("evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        checkpoint = _normalize_checkpoint_path(str(evaluation.get("checkpoint", "")))
        if not checkpoint:
            continue
        if checkpoint not in first_by_checkpoint:
            first_by_checkpoint[checkpoint] = evaluation
            ordered_checkpoints.append(checkpoint)
        metadata_by_checkpoint[checkpoint] = metadata_by_checkpoint.get(checkpoint, False) or evaluation_has_target_metadata(
            evaluation
        )
        if checkpoint not in matched_by_checkpoint and evaluation_matches_target_spec(evaluation, target_spec):
            matched_by_checkpoint[checkpoint] = evaluation
    selected: list[dict[str, Any]] = []
    for checkpoint in ordered_checkpoints:
        matched = matched_by_checkpoint.get(checkpoint)
        if matched is not None:
            selected.append(matched)
            continue
        if not metadata_by_checkpoint.get(checkpoint, False):
            selected.append(first_by_checkpoint[checkpoint])
    return selected
