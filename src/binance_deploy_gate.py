from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.binance_hybrid_eval_defaults import build_expected_prod_eval_config
from src.binance_hybrid_launch import (
    DEFAULT_LAUNCH_SCRIPT,
    BinanceHybridLaunchConfig,
    parse_launch_script,
    resolve_target_launch_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "analysis"
_METRIC_PRIORITY: tuple[tuple[str, str, str], ...] = (
    ("replay", "hourly_goodness_score", "replay.hourly_goodness_score"),
    ("replay", "hourly_total_return", "replay.hourly_total_return"),
    ("replay", "hourly_sortino", "replay.hourly_sortino"),
    ("root", "median_total_return", "holdout.median_total_return"),
    ("root", "median_sortino", "holdout.median_sortino"),
)
_MIN_METRIC_TO_DEPLOY_NEW_CHECKPOINT = 0.0
DEFAULT_MAX_MANIFEST_AGE_HOURS = 24.0


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str
    launch_script: str
    current_checkpoint: str
    candidate_checkpoint: str
    manifest_path: str | None = None
    metric_name: str | None = None
    current_metric: float | None = None
    candidate_metric: float | None = None


@dataclass(frozen=True)
class BestDeployCandidateResult:
    checkpoint: str | None
    reason: str
    launch_script: str
    current_checkpoint: str
    manifest_path: str | None = None
    metric_name: str | None = None
    current_metric: float | None = None
    candidate_metric: float | None = None


@dataclass(frozen=True)
class BestGridDeployCandidateResult(BestDeployCandidateResult):
    symbols_override: str = ""
    leverage_override: float | None = None


def _normalize_checkpoint_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _normalize_symbol_list(symbols: Any) -> list[str]:
    if not isinstance(symbols, list):
        return []
    return [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]


def _normalize_path_value(path: Any) -> str:
    if path is None:
        return ""
    return str(Path(str(path)).expanduser().resolve(strict=False))


def _target_deploy_config_differs_from_live_launch(
    launch_cfg: BinanceHybridLaunchConfig,
    target_cfg: BinanceHybridLaunchConfig,
) -> bool:
    return (
        launch_cfg.symbols != target_cfg.symbols
        or launch_cfg.execution_mode != target_cfg.execution_mode
        or not math.isclose(float(launch_cfg.leverage), float(target_cfg.leverage), rel_tol=0.0, abs_tol=1e-9)
        or launch_cfg.interval != target_cfg.interval
        or launch_cfg.fallback_mode != target_cfg.fallback_mode
        or launch_cfg.model != target_cfg.model
    )


def _manifest_launch_compatibility_reason(
    payload: dict[str, Any],
    launch_config: BinanceHybridLaunchConfig,
) -> str | None:
    manifest_launch = payload.get("launch_config")
    if not isinstance(manifest_launch, dict):
        return "manifest missing launch_config"

    comparisons: tuple[tuple[str, Any, Any], ...] = (
        ("trade_script", manifest_launch.get("trade_script"), launch_config.trade_script),
        ("model", manifest_launch.get("model"), launch_config.model),
        ("execution_mode", manifest_launch.get("execution_mode"), launch_config.execution_mode),
        ("fallback_mode", manifest_launch.get("fallback_mode"), launch_config.fallback_mode),
        ("interval", manifest_launch.get("interval"), launch_config.interval),
    )
    for field, manifest_value, launch_value in comparisons:
        if manifest_value != launch_value:
            return f"manifest launch config mismatch for {field}"

    manifest_symbols = _normalize_symbol_list(manifest_launch.get("symbols"))
    if manifest_symbols != launch_config.symbols:
        return "manifest launch config mismatch for symbols"

    manifest_leverage = manifest_launch.get("leverage")
    try:
        manifest_leverage_value = float(manifest_leverage)
    except (TypeError, ValueError):
        return "manifest launch config missing leverage"
    if not math.isclose(manifest_leverage_value, float(launch_config.leverage), rel_tol=0.0, abs_tol=1e-9):
        return "manifest launch config mismatch for leverage"

    return None


def _manifest_runtime_baseline_reason(payload: dict[str, Any]) -> str | None:
    runtime_audit_issues = payload.get("current_runtime_audit_issues")
    if isinstance(runtime_audit_issues, list) and runtime_audit_issues:
        return "manifest baseline runtime drift issues are present"
    runtime_health_issues = payload.get("current_runtime_health_issues")
    if isinstance(runtime_health_issues, list) and runtime_health_issues:
        return "manifest baseline runtime health issues are present"
    return None


def _manifest_freshness_reason(
    payload: dict[str, Any],
    *,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> str | None:
    reason: str | None = None
    max_age_hours: float | None = None
    if max_manifest_age_hours is not None:
        try:
            max_age_hours = float(max_manifest_age_hours)
        except (TypeError, ValueError):
            reason = "manifest max age is invalid"

    if reason is None and max_age_hours is not None and max_age_hours > 0.0:
        generated_at_raw = payload.get("generated_at")
        if not isinstance(generated_at_raw, str) or not generated_at_raw.strip():
            reason = "manifest missing generated_at"
        else:
            try:
                generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
            except ValueError:
                reason = "manifest generated_at is invalid"
            else:
                if generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=UTC)
                else:
                    generated_at = generated_at.astimezone(UTC)

                now_utc = datetime.now(UTC)
                if generated_at > now_utc + timedelta(minutes=5):
                    reason = "manifest generated_at is in the future"
                else:
                    age_hours = max(0.0, (now_utc - generated_at).total_seconds() / 3600.0)
                    if age_hours > max_age_hours:
                        reason = (
                            "manifest is older than maximum allowed age "
                            f"({age_hours:.1f}h > {max_age_hours:g}h)"
                        )

    return reason


def _manifest_eval_compatibility_reason(payload: dict[str, Any]) -> str | None:
    manifest_eval = payload.get("eval_config")
    if not isinstance(manifest_eval, dict):
        return "manifest missing eval_config"

    expected_eval = build_expected_prod_eval_config()
    float_fields = {
        "fee_rate",
        "slippage_bps",
        "fill_buffer_bps",
        "periods_per_year",
        "replay_eval_fill_buffer_bps",
        "replay_eval_hourly_periods_per_year",
    }
    int_fields = {
        "eval_hours",
        "n_windows",
        "seed",
        "decision_lag",
    }
    bool_fields = {
        "allow_shorts",
        "skip_replay_eval",
    }

    mismatch_reason: str | None = None
    for field, expected_value in expected_eval.items():
        if mismatch_reason is not None:
            break
        manifest_value = manifest_eval.get(field)
        if field == "data_path":
            if _normalize_path_value(manifest_value) != _normalize_path_value(expected_value):
                mismatch_reason = f"manifest eval config mismatch for {field}"
            continue
        if field in float_fields:
            try:
                manifest_float = float(manifest_value)
            except (TypeError, ValueError):
                mismatch_reason = f"manifest eval config mismatch for {field}"
            else:
                if not math.isclose(manifest_float, float(expected_value), rel_tol=0.0, abs_tol=1e-9):
                    mismatch_reason = f"manifest eval config mismatch for {field}"
            continue
        if field in int_fields:
            try:
                manifest_int = int(manifest_value)
            except (TypeError, ValueError):
                mismatch_reason = f"manifest eval config mismatch for {field}"
            else:
                if manifest_int != int(expected_value):
                    mismatch_reason = f"manifest eval config mismatch for {field}"
            continue
        if field in bool_fields:
            if bool(manifest_value) is not bool(expected_value):
                mismatch_reason = f"manifest eval config mismatch for {field}"
            continue
        if manifest_value != expected_value:
            mismatch_reason = f"manifest eval config mismatch for {field}"

    return mismatch_reason


def manifest_matches_launch_config(
    payload: dict[str, Any],
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    *,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
) -> tuple[bool, str | None]:
    launch_cfg = resolve_target_launch_config(
        launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
    )
    reason = _manifest_launch_compatibility_reason(payload, launch_cfg)
    return reason is None, reason


def manifest_matches_deploy_config(
    payload: dict[str, Any],
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    *,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> tuple[bool, str | None]:
    launch_ok, launch_reason = manifest_matches_launch_config(
        payload,
        launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
    )
    if not launch_ok:
        return False, launch_reason
    eval_reason = _manifest_eval_compatibility_reason(payload)
    if eval_reason is not None:
        return False, eval_reason
    freshness_reason = _manifest_freshness_reason(
        payload,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    if freshness_reason is not None:
        return False, freshness_reason
    runtime_reason = _manifest_runtime_baseline_reason(payload)
    return runtime_reason is None, runtime_reason


def find_latest_prod_eval_manifest(
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    *,
    launch_script: str | Path | None = None,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> Path | None:
    root = Path(analysis_root)
    manifests = sorted(
        root.glob("current_binance_prod_eval_*/prod_launch_eval_manifest.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if launch_script is None:
        return manifests[0] if manifests else None
    for manifest_path in manifests:
        try:
            payload = _load_manifest(manifest_path)
        except (json.JSONDecodeError, OSError):
            continue
        compatible, _reason = manifest_matches_deploy_config(
            payload,
            launch_script,
            symbols_override=symbols_override,
            leverage_override=leverage_override,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if compatible:
            return manifest_path
    return None


def resolve_prod_eval_manifest(
    manifest_path: str | Path | None = None,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    launch_script: str | Path | None = None,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> Path | None:
    if manifest_path:
        return Path(manifest_path)
    return find_latest_prod_eval_manifest(
        analysis_root,
        launch_script=launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
        max_manifest_age_hours=max_manifest_age_hours,
    )


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _find_evaluation(payload: dict[str, Any], checkpoint: str) -> dict[str, Any] | None:
    target = _normalize_checkpoint_path(checkpoint)
    for evaluation in payload.get("evaluations", []):
        if _normalize_checkpoint_path(str(evaluation.get("checkpoint", ""))) == target:
            return evaluation
    return None


def load_checkpoint_evaluation(
    checkpoint: str | Path,
    *,
    manifest_path: str | Path | None = None,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    launch_script: str | Path | None = None,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> tuple[Path, dict[str, Any]] | None:
    resolved_manifest = resolve_prod_eval_manifest(
        manifest_path,
        analysis_root,
        launch_script=launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    if resolved_manifest is None or not resolved_manifest.exists():
        return None
    payload = _load_manifest(resolved_manifest)
    if launch_script is not None:
        compatible, _reason = manifest_matches_deploy_config(
            payload,
            launch_script,
            symbols_override=symbols_override,
            leverage_override=leverage_override,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if not compatible:
            return None
    evaluation = _find_evaluation(payload, str(checkpoint))
    if evaluation is None:
        return None
    return resolved_manifest, evaluation


def load_launch_checkpoint_evaluation(
    *,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    manifest_path: str | Path | None = None,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> tuple[Path, dict[str, Any]] | None:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    if not launch_cfg.rl_checkpoint:
        return None
    return load_checkpoint_evaluation(
        launch_cfg.rl_checkpoint,
        manifest_path=manifest_path,
        analysis_root=analysis_root,
        launch_script=launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
        max_manifest_age_hours=max_manifest_age_hours,
    )


def _extract_metric(evaluation: dict[str, Any], scope: str, field: str) -> float | None:
    if scope == "root":
        value = evaluation.get(field)
    else:
        nested = evaluation.get(scope)
        value = nested.get(field) if isinstance(nested, dict) else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_metric(current_eval: dict[str, Any], candidate_eval: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    for scope, field, label in _METRIC_PRIORITY:
        current_value = _extract_metric(current_eval, scope, field)
        candidate_value = _extract_metric(candidate_eval, scope, field)
        if current_value is None or candidate_value is None:
            continue
        return label, current_value, candidate_value
    return None, None, None


def _pick_candidate_metric(candidate_eval: dict[str, Any]) -> tuple[str | None, float | None]:
    for scope, field, label in _METRIC_PRIORITY:
        value = _extract_metric(candidate_eval, scope, field)
        if value is not None:
            return label, value
    return None, None


def pick_candidate_production_metric(candidate_eval: dict[str, Any]) -> tuple[str | None, float | None]:
    return _pick_candidate_metric(candidate_eval)


def _metric_priority_index(metric_name: str | None) -> int:
    if metric_name is None:
        return len(_METRIC_PRIORITY)
    for index, (_scope, _field, label) in enumerate(_METRIC_PRIORITY):
        if label == metric_name:
            return index
    return len(_METRIC_PRIORITY)


def production_metric_sort_key(metric_name: str | None, metric_value: float | None) -> tuple[int, float]:
    value = metric_value if metric_value is not None else -float("inf")
    return -_metric_priority_index(metric_name), value


def _best_manifest_candidate_against_current(
    payload: dict[str, Any],
    *,
    current_checkpoint: str,
    current_eval: dict[str, Any],
) -> tuple[str | None, str | None, float | None]:
    best_checkpoint: str | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    best_priority = len(_METRIC_PRIORITY)
    for evaluation in payload.get("evaluations", []):
        checkpoint = _normalize_checkpoint_path(str(evaluation.get("checkpoint", "")))
        if not checkpoint or checkpoint == current_checkpoint:
            continue
        metric_name, current_metric, metric_value = _pick_metric(current_eval, evaluation)
        if metric_name is None or metric_value is None or current_metric is None:
            continue
        if metric_value <= current_metric or not _candidate_clears_new_checkpoint_floor(metric_value):
            continue
        priority = _metric_priority_index(metric_name)
        if (
            best_checkpoint is None
            or priority < best_priority
            or (priority == best_priority and metric_value > (best_metric_value if best_metric_value is not None else -float("inf")))
        ):
            best_checkpoint = checkpoint
            best_metric_name = metric_name
            best_metric_value = metric_value
            best_priority = priority
    return best_checkpoint, best_metric_name, best_metric_value


def _best_manifest_candidate_without_current(
    payload: dict[str, Any],
) -> tuple[str | None, str | None, float | None]:
    best_checkpoint: str | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    best_priority = len(_METRIC_PRIORITY)
    for evaluation in payload.get("evaluations", []):
        checkpoint = _normalize_checkpoint_path(str(evaluation.get("checkpoint", "")))
        if not checkpoint:
            continue
        metric_name, metric_value = _pick_candidate_metric(evaluation)
        if metric_name is None or metric_value is None:
            continue
        priority = _metric_priority_index(metric_name)
        if (
            best_checkpoint is None
            or priority < best_priority
            or (priority == best_priority and metric_value > (best_metric_value if best_metric_value is not None else -float("inf")))
        ):
            best_checkpoint = checkpoint
            best_metric_name = metric_name
            best_metric_value = metric_value
            best_priority = priority
    return best_checkpoint, best_metric_name, best_metric_value


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.6f}"


def _candidate_clears_new_checkpoint_floor(candidate_metric: float | None) -> bool:
    return candidate_metric is not None and candidate_metric > _MIN_METRIC_TO_DEPLOY_NEW_CHECKPOINT


def _parse_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_grid_best_rows(path: str | Path) -> list[dict[str, str]]:
    grid_path = Path(path)
    best_rows_path = grid_path if grid_path.suffix.lower() == ".csv" else grid_path / "best_by_config.csv"
    if not best_rows_path.exists():
        raise FileNotFoundError(best_rows_path)
    with best_rows_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_best_grid_deploy_candidate(
    *,
    grid_output_path: str | Path,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    require_outperform: bool = True,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> BestGridDeployCandidateResult:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    current_checkpoint = (
        _normalize_checkpoint_path(launch_cfg.rl_checkpoint)
        if launch_cfg.rl_checkpoint
        else ""
    )
    launch_script_resolved = str(Path(launch_script).resolve())

    try:
        best_rows = _load_grid_best_rows(grid_output_path)
    except FileNotFoundError as exc:
        return BestGridDeployCandidateResult(
            checkpoint=None,
            reason=f"grid search results missing best_by_config.csv: {exc}",
            launch_script=launch_script_resolved,
            current_checkpoint=current_checkpoint,
        )

    first_reason = "grid search results contained no deployable rows"
    first_manifest_path: str | None = None
    for row in best_rows:
        checkpoint = str(row.get("checkpoint") or "").strip()
        manifest_path = str(row.get("manifest_path") or "").strip()
        symbols_override = str(row.get("symbols") or "").strip()
        leverage_override = _parse_optional_float(row.get("leverage"))
        if not checkpoint or not manifest_path or leverage_override is None:
            continue
        gate_result = gate_deploy_candidate(
            candidate_checkpoint=checkpoint,
            launch_script=launch_script,
            manifest_path=manifest_path,
            require_outperform=require_outperform,
            symbols_override=symbols_override or None,
            leverage_override=leverage_override,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if gate_result.allowed:
            return BestGridDeployCandidateResult(
                checkpoint=gate_result.candidate_checkpoint,
                reason="best deployable production config from grid search",
                launch_script=launch_script_resolved,
                current_checkpoint=current_checkpoint,
                manifest_path=gate_result.manifest_path,
                metric_name=gate_result.metric_name,
                current_metric=gate_result.current_metric,
                candidate_metric=gate_result.candidate_metric,
                symbols_override=symbols_override,
                leverage_override=leverage_override,
            )
        if first_reason == "grid search results contained no deployable rows":
            first_reason = gate_result.reason
            first_manifest_path = gate_result.manifest_path or manifest_path

    return BestGridDeployCandidateResult(
        checkpoint=None,
        reason=first_reason,
        launch_script=launch_script_resolved,
        current_checkpoint=current_checkpoint,
        manifest_path=first_manifest_path,
    )


def resolve_best_deploy_candidate(
    *,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    manifest_path: str | Path | None = None,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    require_outperform: bool = True,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> BestDeployCandidateResult:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    current_checkpoint = (
        _normalize_checkpoint_path(launch_cfg.rl_checkpoint)
        if launch_cfg.rl_checkpoint
        else ""
    )
    launch_script_resolved = str(Path(launch_script).resolve())
    resolved_manifest = resolve_prod_eval_manifest(
        manifest_path,
        analysis_root,
        launch_script=launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    manifest_path_str: str | None = str(resolved_manifest) if resolved_manifest is not None else None
    checkpoint: str | None = None
    reason: str
    metric_name: str | None = None
    current_metric: float | None = None
    candidate_metric: float | None = None

    if resolved_manifest is None or not resolved_manifest.exists():
        latest_any_manifest = find_latest_prod_eval_manifest(analysis_root)
        if latest_any_manifest is not None:
            manifest_path_str = str(latest_any_manifest)
            try:
                latest_payload = _load_manifest(latest_any_manifest)
            except (json.JSONDecodeError, OSError):
                reason = "no compatible production evaluation manifest found for current deploy config"
            else:
                _compatible, compatibility_reason = manifest_matches_deploy_config(
                    latest_payload,
                    launch_script,
                    symbols_override=symbols_override,
                    leverage_override=leverage_override,
                    max_manifest_age_hours=max_manifest_age_hours,
                )
                reason = compatibility_reason or "no compatible production evaluation manifest found for current deploy config"
        else:
            reason = "no production evaluation manifest found"
    else:
        payload = _load_manifest(resolved_manifest)
        compatible, compatibility_reason = manifest_matches_deploy_config(
            payload,
            launch_script,
            symbols_override=symbols_override,
            leverage_override=leverage_override,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if not compatible:
            reason = compatibility_reason or "manifest deploy config does not match current deploy config"
        elif current_checkpoint:
            current_eval = _find_evaluation(payload, current_checkpoint)
            if current_eval is None:
                reason = "current live checkpoint missing from manifest"
            else:
                best_eval: dict[str, Any] | None = None
                best_priority = len(_METRIC_PRIORITY)
                best_value = -float("inf")
                for evaluation in payload.get("evaluations", []):
                    if not isinstance(evaluation, dict):
                        continue
                    checkpoint_value = _normalize_checkpoint_path(str(evaluation.get("checkpoint", "")))
                    if not checkpoint_value or checkpoint_value == current_checkpoint:
                        continue
                    candidate_metric_name, candidate_current_metric, candidate_metric_value = _pick_metric(current_eval, evaluation)
                    if candidate_metric_name is None or candidate_metric_value is None or candidate_current_metric is None:
                        continue
                    priority = _metric_priority_index(candidate_metric_name)
                    if (
                        best_eval is None
                        or priority < best_priority
                        or (priority == best_priority and candidate_metric_value > best_value)
                    ):
                        best_eval = evaluation
                        best_priority = priority
                        best_value = candidate_metric_value
                if best_eval is None:
                    reason = "no comparable production-faithful candidate found in manifest"
                else:
                    best_checkpoint = _normalize_checkpoint_path(str(best_eval.get("checkpoint", "")))
                    metric_name, current_metric, candidate_metric = _pick_metric(current_eval, best_eval)
                    if metric_name is None:
                        reason = "no comparable production-faithful metric found"
                    elif (
                        require_outperform
                        and candidate_metric is not None
                        and current_metric is not None
                        and candidate_metric <= current_metric
                    ):
                        reason = "no candidate beats current live checkpoint"
                    elif not _candidate_clears_new_checkpoint_floor(candidate_metric):
                        reason = "no candidate clears minimum production-faithful metric required to replace current live checkpoint"
                    else:
                        checkpoint = best_checkpoint
                        reason = "best production-evaluated checkpoint in manifest"
        else:
            best_checkpoint, metric_name, candidate_metric = _best_manifest_candidate_without_current(payload)
            if best_checkpoint is None or metric_name is None:
                reason = "no comparable production-faithful candidate found in manifest"
            elif not _candidate_clears_new_checkpoint_floor(candidate_metric):
                reason = "no candidate clears minimum production-faithful metric required to re-enable RL"
            else:
                checkpoint = best_checkpoint
                reason = "best production-evaluated checkpoint in manifest (current launch has no RL checkpoint)"

    return BestDeployCandidateResult(
        checkpoint=checkpoint,
        reason=reason,
        launch_script=launch_script_resolved,
        current_checkpoint=current_checkpoint,
        manifest_path=manifest_path_str,
        metric_name=metric_name,
        current_metric=current_metric,
        candidate_metric=candidate_metric,
    )


def gate_deploy_candidate(
    *,
    candidate_checkpoint: str | Path,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    manifest_path: str | Path | None = None,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    require_outperform: bool = True,
    require_manifest_best: bool = True,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> GateResult:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    target_launch_cfg = resolve_target_launch_config(
        launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
    )
    current_checkpoint = (
        _normalize_checkpoint_path(launch_cfg.rl_checkpoint)
        if launch_cfg.rl_checkpoint
        else ""
    )
    candidate_checkpoint_norm = _normalize_checkpoint_path(candidate_checkpoint)
    launch_script_resolved = str(Path(launch_script).resolve())

    if (
        current_checkpoint
        and candidate_checkpoint_norm == current_checkpoint
        and not _target_deploy_config_differs_from_live_launch(launch_cfg, target_launch_cfg)
    ):
        return GateResult(
            allowed=True,
            reason="candidate already matches current live checkpoint",
            launch_script=launch_script_resolved,
            current_checkpoint=current_checkpoint,
            candidate_checkpoint=candidate_checkpoint_norm,
        )

    resolved_manifest = resolve_prod_eval_manifest(
        manifest_path,
        analysis_root,
        launch_script=launch_script,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    allowed = True
    reason = "candidate passes deploy gate"
    manifest_path_str: str | None = str(resolved_manifest) if resolved_manifest is not None else None
    metric_name: str | None = None
    current_metric: float | None = None
    candidate_metric: float | None = None

    if resolved_manifest is None or not resolved_manifest.exists():
        allowed = False
        latest_any_manifest = find_latest_prod_eval_manifest(analysis_root)
        if latest_any_manifest is not None:
            manifest_path_str = str(latest_any_manifest)
            try:
                latest_payload = _load_manifest(latest_any_manifest)
            except (json.JSONDecodeError, OSError):
                reason = "no compatible production evaluation manifest found for current deploy config"
            else:
                _compatible, compatibility_reason = manifest_matches_deploy_config(
                    latest_payload,
                    launch_script,
                    symbols_override=symbols_override,
                    leverage_override=leverage_override,
                    max_manifest_age_hours=max_manifest_age_hours,
                )
                reason = compatibility_reason or "no compatible production evaluation manifest found for current deploy config"
        else:
            reason = "no production evaluation manifest found"
    else:
        payload = _load_manifest(resolved_manifest)
        compatible, compatibility_reason = manifest_matches_deploy_config(
            payload,
            launch_script,
            symbols_override=symbols_override,
            leverage_override=leverage_override,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if not compatible:
            allowed = False
            reason = compatibility_reason or "manifest deploy config does not match current deploy config"
        else:
            candidate_eval = _find_evaluation(payload, candidate_checkpoint_norm)
            current_eval = _find_evaluation(payload, current_checkpoint) if current_checkpoint else None
            if candidate_eval is None:
                allowed = False
                reason = "candidate checkpoint missing from manifest"
            elif not current_checkpoint:
                metric_name, candidate_metric = _pick_candidate_metric(candidate_eval)
                if metric_name is None:
                    allowed = False
                    reason = "no comparable production-faithful metric found"
                elif not _candidate_clears_new_checkpoint_floor(candidate_metric):
                    allowed = False
                    reason = "candidate does not clear minimum production-faithful metric required to re-enable RL"
                elif require_manifest_best:
                    best_checkpoint, best_metric_name, best_metric_value = _best_manifest_candidate_without_current(payload)
                    if best_checkpoint and best_checkpoint != candidate_checkpoint_norm:
                        allowed = False
                        reason = (
                            "candidate is not best production-evaluated checkpoint in manifest "
                            f"({best_metric_name or 'metric'} best={_fmt_metric(best_metric_value)} "
                            f"checkpoint={best_checkpoint})"
                        )
                    else:
                        reason = "candidate passes deploy gate (current launch has no RL checkpoint)"
                else:
                    reason = "candidate passes deploy gate (current launch has no RL checkpoint)"
            elif current_eval is None:
                allowed = False
                reason = "current live checkpoint missing from manifest"
            else:
                metric_name, current_metric, candidate_metric = _pick_metric(current_eval, candidate_eval)
                candidate_is_current_model = candidate_checkpoint_norm == current_checkpoint
                if metric_name is None:
                    allowed = False
                    reason = "no comparable production-faithful metric found"
                elif (
                    require_outperform
                    and candidate_metric is not None
                    and current_metric is not None
                    and candidate_metric <= current_metric
                    and not candidate_is_current_model
                ):
                    allowed = False
                    reason = "candidate does not beat current live checkpoint"
                elif not _candidate_clears_new_checkpoint_floor(candidate_metric) and not candidate_is_current_model:
                    allowed = False
                    reason = "candidate does not clear minimum production-faithful metric required to replace current live checkpoint"
                elif require_manifest_best:
                    best_checkpoint, best_metric_name, best_metric_value = _best_manifest_candidate_against_current(
                        payload,
                        current_checkpoint=current_checkpoint,
                        current_eval=current_eval,
                    )
                    if best_checkpoint and best_checkpoint != candidate_checkpoint_norm:
                        allowed = False
                        reason = (
                            "candidate is not best production-evaluated checkpoint in manifest "
                            f"({best_metric_name or 'metric'} best={_fmt_metric(best_metric_value)} "
                            f"checkpoint={best_checkpoint})"
                        )

    return GateResult(
        allowed=allowed,
        reason=reason,
        launch_script=launch_script_resolved,
        current_checkpoint=current_checkpoint,
        candidate_checkpoint=candidate_checkpoint_norm,
        manifest_path=manifest_path_str,
        metric_name=metric_name,
        current_metric=current_metric,
        candidate_metric=candidate_metric,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether a Binance deploy candidate is production-evaluated.")
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--launch-script", default=str(DEFAULT_LAUNCH_SCRIPT))
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    parser.add_argument("--allow-worse-than-current", action="store_true")
    parser.add_argument("--symbols-override", default="")
    parser.add_argument("--leverage-override", type=float, default=None)
    parser.add_argument("--manifest-max-age-hours", type=float, default=DEFAULT_MAX_MANIFEST_AGE_HOURS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = gate_deploy_candidate(
        candidate_checkpoint=args.candidate_checkpoint,
        launch_script=args.launch_script,
        manifest_path=args.manifest_path or None,
        analysis_root=args.analysis_root,
        require_outperform=not args.allow_worse_than_current,
        symbols_override=args.symbols_override or None,
        leverage_override=args.leverage_override,
        max_manifest_age_hours=args.manifest_max_age_hours,
    )

    print(f"Launch:   {result.launch_script}")
    print(f"Current:  {result.current_checkpoint}")
    print(f"Candidate:{' ' if result.candidate_checkpoint else ''} {result.candidate_checkpoint}")
    if result.manifest_path:
        print(f"Manifest: {result.manifest_path}")
    if result.metric_name is not None:
        print(
            f"Metric:   {result.metric_name} "
            f"current={_fmt_metric(result.current_metric)} candidate={_fmt_metric(result.candidate_metric)}"
        )
    print(f"Decision: {'ALLOW' if result.allowed else 'BLOCK'} -- {result.reason}")
    return 0 if result.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
