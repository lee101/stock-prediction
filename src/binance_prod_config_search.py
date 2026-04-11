from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

from src.binance_deploy_gate import (
    DEFAULT_MAX_MANIFEST_AGE_HOURS,
    manifest_matches_deploy_config,
)
from src.binance_hybrid_launch import (
    BinanceHybridLaunchConfig,
    parse_launch_script,
    resolve_target_launch_config,
)
from src.binance_hybrid_process_audit import audit_running_hybrid_process_matches_launch
from src.binance_prod_manifest_targets import (
    ManifestEvalTargetSpec,
    find_target_evaluation,
    requested_checkpoints,
    select_target_evaluations,
)


DEFAULT_MAX_VARIANTS = 64
DEFAULT_MAX_CHECKPOINT_CONFIG_EVALS = DEFAULT_MAX_VARIANTS * 3
T = TypeVar("T")


@dataclass(frozen=True)
class ConfigVariant:
    symbols: tuple[str, ...]
    leverage: float
    output_dir: str
    slug: str


@dataclass(frozen=True)
class PendingCheckpointEvalPlan:
    current_running_config: BinanceHybridLaunchConfig | None
    pending_by_variant: dict[ConfigVariant, int]


@dataclass(frozen=True)
class VariantCheckpointCoverage:
    full_recompute: bool
    covered_launch_target_checkpoints: frozenset[str]
    has_running_hybrid_baseline: bool


@dataclass(frozen=True)
class PendingCheckpointCoveragePlan:
    current_running_config: BinanceHybridLaunchConfig | None
    coverage_by_variant: dict[ConfigVariant, VariantCheckpointCoverage]


def _parse_symbol_set(raw: str) -> tuple[str, ...]:
    tokens = [token.strip().upper() for token in raw.replace(",", " ").split() if token.strip()]
    if not tokens:
        raise ValueError("symbol set must contain at least one symbol")
    return tuple(tokens)


def _slugify_symbols(symbols: tuple[str, ...]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9]+", "", symbol).lower() for symbol in symbols)


def _slugify_leverage(leverage: float) -> str:
    return str(float(leverage)).replace("-", "m").replace(".", "p")


def _is_launch_variant(variant: ConfigVariant, launch_config: BinanceHybridLaunchConfig) -> bool:
    return tuple(variant.symbols) == tuple(launch_config.symbols) and float(variant.leverage) == float(launch_config.leverage)


def build_config_variants(
    *,
    launch_script: str | Path,
    symbol_set_specs: list[str],
    symbol_subset_sizes: list[int],
    leverage_options: list[float],
    output_root: str | Path,
    include_launch_variant: bool = False,
) -> list[ConfigVariant]:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    symbol_sets = [_parse_symbol_set(spec) for spec in symbol_set_specs]
    launch_symbols = tuple(launch_cfg.symbols)
    for subset_size in symbol_subset_sizes:
        if subset_size < 1:
            raise ValueError("symbol subset size must be at least 1")
        if subset_size > len(launch_symbols):
            raise ValueError(
                f"symbol subset size {subset_size} exceeds launch symbol count {len(launch_symbols)}"
            )
        symbol_sets.extend(itertools.combinations(launch_symbols, subset_size))
    if not symbol_sets:
        symbol_sets = [launch_symbols]
    leverages = [float(leverage) for leverage in leverage_options] or [float(launch_cfg.leverage)]

    output_root_path = Path(output_root)
    variants: list[ConfigVariant] = []
    seen: set[tuple[tuple[str, ...], float]] = set()
    for symbols in symbol_sets:
        for leverage in leverages:
            target_cfg = resolve_target_launch_config(
                launch_script,
                symbols_override=",".join(symbols),
                leverage_override=leverage,
            )
            key = (tuple(target_cfg.symbols), float(target_cfg.leverage))
            if key in seen:
                continue
            seen.add(key)
            slug = f"{_slugify_symbols(tuple(target_cfg.symbols))}__lev_{_slugify_leverage(float(target_cfg.leverage))}"
            variants.append(
                ConfigVariant(
                    symbols=tuple(target_cfg.symbols),
                    leverage=float(target_cfg.leverage),
                    output_dir=str((output_root_path / slug).resolve()),
                    slug=slug,
                )
            )
    if include_launch_variant:
        launch_key = (tuple(launch_cfg.symbols), float(launch_cfg.leverage))
        if launch_key not in seen:
            slug = f"{_slugify_symbols(tuple(launch_cfg.symbols))}__lev_{_slugify_leverage(float(launch_cfg.leverage))}"
            variants.append(
                ConfigVariant(
                    symbols=tuple(launch_cfg.symbols),
                    leverage=float(launch_cfg.leverage),
                    output_dir=str((output_root_path / slug).resolve()),
                    slug=slug,
                )
            )
    return variants


def limit_config_variants(
    variants: list[ConfigVariant],
    *,
    launch_config: BinanceHybridLaunchConfig,
    max_variants: int | None,
    variant_offset: int = 0,
) -> list[ConfigVariant]:
    if variant_offset < 0:
        raise ValueError("variant_offset must be >= 0")
    if max_variants is None or max_variants == 0 or len(variants) <= max_variants:
        return list(variants)
    if max_variants < 0:
        raise ValueError("max_variants must be >= 0")
    start = variant_offset % len(variants)
    rotated = list(variants[start:]) + list(variants[:start])
    limited = rotated[:max_variants]
    if any(_is_launch_variant(variant, launch_config) for variant in limited):
        return limited
    launch_variant = next((variant for variant in variants if _is_launch_variant(variant, launch_config)), None)
    if launch_variant is not None and limited:
        limited[-1] = launch_variant
    return limited


def estimate_checkpoint_evals_per_variant(
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    *,
    include_running_hybrid_baseline: bool = False,
) -> int:
    count = len(requested_checkpoints(launch_config.rl_checkpoint, candidate_checkpoints))
    if include_running_hybrid_baseline:
        count += 1
    return max(1, count)


def _normalize_checkpoint_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _configs_match(left: BinanceHybridLaunchConfig, right: BinanceHybridLaunchConfig) -> bool:
    return (
        str(left.model) == str(right.model)
        and list(left.symbols) == list(right.symbols)
        and str(left.execution_mode) == str(right.execution_mode)
        and math.isclose(float(left.leverage), float(right.leverage), rel_tol=0.0, abs_tol=1e-9)
        and left.interval == right.interval
        and str(left.fallback_mode) == str(right.fallback_mode)
        and str(left.rl_checkpoint or "") == str(right.rl_checkpoint or "")
    )


def _variant_target_config(
    variant: ConfigVariant,
    launch_config: BinanceHybridLaunchConfig,
) -> BinanceHybridLaunchConfig:
    return replace(
        launch_config,
        symbols=list(variant.symbols),
        leverage=float(variant.leverage),
    )


def _resolve_current_running_hybrid_config(launch_script: str | Path) -> BinanceHybridLaunchConfig | None:
    try:
        result = audit_running_hybrid_process_matches_launch(launch_script)
    except Exception:
        return None
    running_config = result.running_config
    if running_config is None or not running_config.rl_checkpoint:
        return None
    return running_config


def _variant_requires_running_hybrid_baseline(
    variant: ConfigVariant,
    *,
    launch_config: BinanceHybridLaunchConfig,
    current_running_config: BinanceHybridLaunchConfig | None,
) -> bool:
    if current_running_config is None or not current_running_config.rl_checkpoint:
        return False
    return not _configs_match(_variant_target_config(variant, launch_config), current_running_config)


def _target_eval_spec(
    variant: ConfigVariant,
    *,
    preferred_target_label: str,
) -> ManifestEvalTargetSpec:
    return ManifestEvalTargetSpec(
        symbols=tuple(variant.symbols),
        leverage=float(variant.leverage),
        preferred_target_label=preferred_target_label,
    )


def _config_matches_target(manifest_config: object, target_config: BinanceHybridLaunchConfig) -> bool:
    if not isinstance(manifest_config, dict):
        return False
    comparisons = (
        ("trade_script", target_config.trade_script),
        ("model", target_config.model),
        ("execution_mode", target_config.execution_mode),
        ("fallback_mode", target_config.fallback_mode),
        ("interval", target_config.interval),
        ("rl_checkpoint", target_config.rl_checkpoint),
    )
    for field, expected_value in comparisons:
        if manifest_config.get(field) != expected_value:
            return False
    manifest_symbols = [str(symbol).strip().upper() for symbol in manifest_config.get("symbols", []) if str(symbol).strip()]
    if manifest_symbols != list(target_config.symbols):
        return False
    try:
        manifest_leverage = float(manifest_config.get("leverage"))
    except (TypeError, ValueError):
        return False
    return math.isclose(manifest_leverage, float(target_config.leverage), rel_tol=0.0, abs_tol=1e-9)


def _manifest_has_current_running_hybrid_baseline(
    payload: dict[str, object],
    *,
    current_running_config: BinanceHybridLaunchConfig | None,
) -> bool:
    if current_running_config is None or not current_running_config.rl_checkpoint:
        return True
    if not _config_matches_target(payload.get("current_running_hybrid_config"), current_running_config):
        return False
    return (
        find_target_evaluation(
            payload,
            current_running_config.rl_checkpoint,
            target_spec=ManifestEvalTargetSpec(
                symbols=tuple(current_running_config.symbols),
                leverage=float(current_running_config.leverage),
                preferred_target_label="running_hybrid",
            ),
        )
        is not None
    )


def _covered_launch_target_checkpoints(
    payload: dict[str, object],
    *,
    variant: ConfigVariant,
) -> frozenset[str]:
    target_spec = _target_eval_spec(variant, preferred_target_label="launch_target")
    return frozenset(
        _normalize_checkpoint_path(str(evaluation.get("checkpoint", "")))
        for evaluation in select_target_evaluations(payload, target_spec=target_spec)
        if str(evaluation.get("checkpoint", "")).strip()
    )


def _missing_requested_checkpoints(
    payload: dict[str, object],
    *,
    variant: ConfigVariant,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> tuple[str, ...]:
    required_checkpoints = requested_checkpoints(launch_config.rl_checkpoint, candidate_checkpoints)
    target_spec = _target_eval_spec(variant, preferred_target_label="launch_target")
    return tuple(
        checkpoint
        for checkpoint in required_checkpoints
        if find_target_evaluation(payload, checkpoint, target_spec=target_spec) is None
    )


def build_pending_checkpoint_coverage_plan(
    variants: list[ConfigVariant],
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> PendingCheckpointCoveragePlan:
    current_running_config = _resolve_current_running_hybrid_config(launch_script)
    coverage_by_variant: dict[ConfigVariant, VariantCheckpointCoverage] = {}
    for variant in variants:
        include_running_hybrid_baseline = _variant_requires_running_hybrid_baseline(
            variant,
            launch_config=launch_config,
            current_running_config=current_running_config,
        )
        manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
        if not manifest_path.exists():
            coverage_by_variant[variant] = VariantCheckpointCoverage(
                full_recompute=True,
                covered_launch_target_checkpoints=frozenset(),
                has_running_hybrid_baseline=False,
            )
            continue
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            coverage_by_variant[variant] = VariantCheckpointCoverage(
                full_recompute=True,
                covered_launch_target_checkpoints=frozenset(),
                has_running_hybrid_baseline=False,
            )
            continue
        compatible, _reason = manifest_matches_deploy_config(
            payload,
            launch_script,
            symbols_override=",".join(variant.symbols),
            leverage_override=variant.leverage,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if not compatible:
            coverage_by_variant[variant] = VariantCheckpointCoverage(
                full_recompute=True,
                covered_launch_target_checkpoints=frozenset(),
                has_running_hybrid_baseline=False,
            )
            continue
        coverage_by_variant[variant] = VariantCheckpointCoverage(
            full_recompute=False,
            covered_launch_target_checkpoints=_covered_launch_target_checkpoints(payload, variant=variant),
            has_running_hybrid_baseline=_manifest_has_current_running_hybrid_baseline(
                payload,
                current_running_config=current_running_config if include_running_hybrid_baseline else None,
            ),
        )
    return PendingCheckpointCoveragePlan(
        current_running_config=current_running_config,
        coverage_by_variant=coverage_by_variant,
    )


def estimate_pending_checkpoint_evals_for_variant_from_coverage(
    variant: ConfigVariant,
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    pending_checkpoint_coverage_plan: PendingCheckpointCoveragePlan,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> int:
    current_running_config = pending_checkpoint_coverage_plan.current_running_config
    include_running_hybrid_baseline = _variant_requires_running_hybrid_baseline(
        variant,
        launch_config=launch_config,
        current_running_config=current_running_config,
    )
    coverage = pending_checkpoint_coverage_plan.coverage_by_variant.get(variant)
    if coverage is None:
        return estimate_pending_checkpoint_evals_for_variant(
            variant,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
            max_manifest_age_hours=max_manifest_age_hours,
            current_running_config=current_running_config,
        )
    if coverage.full_recompute:
        return estimate_checkpoint_evals_per_variant(
            launch_config,
            candidate_checkpoints,
            include_running_hybrid_baseline=include_running_hybrid_baseline,
        )
    required_checkpoints = requested_checkpoints(launch_config.rl_checkpoint, candidate_checkpoints)
    extra_running_hybrid_eval = (
        1 if include_running_hybrid_baseline and not coverage.has_running_hybrid_baseline else 0
    )
    missing_checkpoint_evals = sum(
        1
        for checkpoint in required_checkpoints
        if checkpoint not in coverage.covered_launch_target_checkpoints
    )
    return missing_checkpoint_evals + extra_running_hybrid_eval


def estimate_pending_checkpoint_config_evals_with_coverage(
    variants: list[ConfigVariant],
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    pending_checkpoint_coverage_plan: PendingCheckpointCoveragePlan,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
) -> int:
    return sum(
        estimate_pending_checkpoint_evals_for_variant_from_coverage(
            variant,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
            pending_checkpoint_coverage_plan=pending_checkpoint_coverage_plan,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        for variant in variants
    )


def estimate_pending_checkpoint_evals_for_variant(
    variant: ConfigVariant,
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
    current_running_config: BinanceHybridLaunchConfig | None = None,
) -> int:
    if current_running_config is None:
        current_running_config = _resolve_current_running_hybrid_config(launch_script)
    include_running_hybrid_baseline = _variant_requires_running_hybrid_baseline(
        variant,
        launch_config=launch_config,
        current_running_config=current_running_config,
    )

    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    if not manifest_path.exists():
        return estimate_checkpoint_evals_per_variant(
            launch_config,
            candidate_checkpoints,
            include_running_hybrid_baseline=include_running_hybrid_baseline,
        )

    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return estimate_checkpoint_evals_per_variant(
            launch_config,
            candidate_checkpoints,
            include_running_hybrid_baseline=include_running_hybrid_baseline,
        )

    compatible, _reason = manifest_matches_deploy_config(
        payload,
        launch_script,
        symbols_override=",".join(variant.symbols),
        leverage_override=variant.leverage,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    if not compatible:
        return estimate_checkpoint_evals_per_variant(
            launch_config,
            candidate_checkpoints,
            include_running_hybrid_baseline=include_running_hybrid_baseline,
        )

    missing_requested = _missing_requested_checkpoints(
        payload,
        variant=variant,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
    )
    manifest_has_running_hybrid_baseline = _manifest_has_current_running_hybrid_baseline(
        payload,
        current_running_config=current_running_config if include_running_hybrid_baseline else None,
    )
    if not missing_requested and manifest_has_running_hybrid_baseline:
        return 0

    extra_running_hybrid_eval = (
        1 if include_running_hybrid_baseline and not manifest_has_running_hybrid_baseline else 0
    )
    return len(missing_requested) + extra_running_hybrid_eval


def _estimate_full_recompute_pending_evals_for_variant(
    variant: ConfigVariant,
    *,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    current_running_config: BinanceHybridLaunchConfig | None,
) -> int:
    include_running_hybrid_baseline = _variant_requires_running_hybrid_baseline(
        variant,
        launch_config=launch_config,
        current_running_config=current_running_config,
    )
    return estimate_checkpoint_evals_per_variant(
        launch_config,
        candidate_checkpoints,
        include_running_hybrid_baseline=include_running_hybrid_baseline,
    )


def build_pending_checkpoint_eval_plan(
    variants: list[ConfigVariant],
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
    allow_manifest_reuse: bool = True,
) -> PendingCheckpointEvalPlan:
    current_running_config = _resolve_current_running_hybrid_config(launch_script)
    if allow_manifest_reuse:
        pending_by_variant = {
            variant: estimate_pending_checkpoint_evals_for_variant(
                variant,
                launch_script=launch_script,
                launch_config=launch_config,
                candidate_checkpoints=candidate_checkpoints,
                max_manifest_age_hours=max_manifest_age_hours,
                current_running_config=current_running_config,
            )
            for variant in variants
        }
    else:
        pending_by_variant = {
            variant: _estimate_full_recompute_pending_evals_for_variant(
                variant,
                launch_config=launch_config,
                candidate_checkpoints=candidate_checkpoints,
                current_running_config=current_running_config,
            )
            for variant in variants
        }
    return PendingCheckpointEvalPlan(
        current_running_config=current_running_config,
        pending_by_variant=pending_by_variant,
    )


def _pending_evals_for_variant_from_plan(
    variant: ConfigVariant,
    *,
    pending_eval_plan: PendingCheckpointEvalPlan,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_manifest_age_hours: float | None,
) -> int:
    pending_evals = pending_eval_plan.pending_by_variant.get(variant)
    if pending_evals is not None:
        return pending_evals
    return estimate_pending_checkpoint_evals_for_variant(
        variant,
        launch_script=launch_script,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
        max_manifest_age_hours=max_manifest_age_hours,
        current_running_config=pending_eval_plan.current_running_config,
    )


def estimate_pending_checkpoint_config_evals(
    variants: list[ConfigVariant],
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
    pending_eval_plan: PendingCheckpointEvalPlan | None = None,
) -> int:
    pending_eval_plan = pending_eval_plan or build_pending_checkpoint_eval_plan(
        variants,
        launch_script=launch_script,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    return sum(
        _pending_evals_for_variant_from_plan(
            variant,
            pending_eval_plan=pending_eval_plan,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        for variant in variants
    )


def order_config_variants_by_pending_eval_cost(
    variants: list[ConfigVariant],
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
    pending_eval_plan: PendingCheckpointEvalPlan | None = None,
) -> list[ConfigVariant]:
    pending_eval_plan = pending_eval_plan or build_pending_checkpoint_eval_plan(
        variants,
        launch_script=launch_script,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    indexed_variants = list(enumerate(variants))
    indexed_variants.sort(
        key=lambda item: (
            0 if _is_launch_variant(item[1], launch_config) else 1,
            _pending_evals_for_variant_from_plan(
                item[1],
                pending_eval_plan=pending_eval_plan,
                launch_script=launch_script,
                launch_config=launch_config,
                candidate_checkpoints=candidate_checkpoints,
                max_manifest_age_hours=max_manifest_age_hours,
            ),
            item[0],
        )
    )
    return [variant for _index, variant in indexed_variants]


def limit_config_variants_by_pending_eval_budget(
    variants: list[ConfigVariant],
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_checkpoint_config_evals: int | None,
    max_manifest_age_hours: float | None = DEFAULT_MAX_MANIFEST_AGE_HOURS,
    pending_eval_plan: PendingCheckpointEvalPlan | None = None,
) -> list[ConfigVariant]:
    if max_checkpoint_config_evals is None or max_checkpoint_config_evals == 0:
        return list(variants)
    if max_checkpoint_config_evals < 0:
        raise ValueError("max_checkpoint_config_evals must be >= 0")

    pending_eval_plan = pending_eval_plan or build_pending_checkpoint_eval_plan(
        variants,
        launch_script=launch_script,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
        max_manifest_age_hours=max_manifest_age_hours,
    )

    remaining_budget = max_checkpoint_config_evals
    selected_variants: set[ConfigVariant] = set()
    launch_variant = next((variant for variant in variants if _is_launch_variant(variant, launch_config)), None)
    if launch_variant is not None:
        launch_pending_evals = _pending_evals_for_variant_from_plan(
            launch_variant,
            pending_eval_plan=pending_eval_plan,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if launch_pending_evals <= remaining_budget:
            selected_variants.add(launch_variant)
            remaining_budget -= launch_pending_evals

    indexed_variants = list(enumerate(variants))
    indexed_variants.sort(
        key=lambda item: (
            _pending_evals_for_variant_from_plan(
                item[1],
                pending_eval_plan=pending_eval_plan,
                launch_script=launch_script,
                launch_config=launch_config,
                candidate_checkpoints=candidate_checkpoints,
                max_manifest_age_hours=max_manifest_age_hours,
            ),
            item[0],
        )
    )
    for _index, variant in indexed_variants:
        if variant in selected_variants:
            continue
        pending_evals = _pending_evals_for_variant_from_plan(
            variant,
            pending_eval_plan=pending_eval_plan,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
            max_manifest_age_hours=max_manifest_age_hours,
        )
        if pending_evals > remaining_budget:
            continue
        selected_variants.add(variant)
        remaining_budget -= pending_evals

    return [variant for variant in variants if variant in selected_variants]


def max_variants_for_checkpoint_eval_budget(
    launch_config: BinanceHybridLaunchConfig,
    *,
    candidate_checkpoints: list[str | Path],
    max_checkpoint_config_evals: int | None,
) -> int | None:
    if max_checkpoint_config_evals is None:
        return None
    if max_checkpoint_config_evals < 0:
        raise ValueError("max_checkpoint_config_evals must be >= 0")
    if max_checkpoint_config_evals == 0:
        return None
    per_variant_evals = estimate_checkpoint_evals_per_variant(launch_config, candidate_checkpoints)
    return max_checkpoint_config_evals // per_variant_evals


def max_candidate_checkpoints_for_budget(
    launch_config: BinanceHybridLaunchConfig,
    *,
    variant_count: int,
    max_checkpoint_config_evals: int | None,
) -> int | None:
    if max_checkpoint_config_evals is None:
        return None
    if max_checkpoint_config_evals < 0:
        raise ValueError("max_checkpoint_config_evals must be >= 0")
    if max_checkpoint_config_evals == 0:
        return None
    if variant_count < 1:
        return 0
    per_variant_budget = max_checkpoint_config_evals // variant_count
    baseline_eval_slots = 1 if launch_config.rl_checkpoint else 0
    return max(0, per_variant_budget - baseline_eval_slots)


def limit_ranked_candidates(  # noqa: UP047
    ranked_items: list[T],
    *,
    max_candidates: int | None,
    offset: int = 0,
    preserve_first: bool = True,
) -> list[T]:
    if offset < 0:
        raise ValueError("candidate offset must be >= 0")
    if max_candidates is None or max_candidates == 0 or len(ranked_items) <= max_candidates:
        return list(ranked_items)
    if max_candidates < 0:
        raise ValueError("max_candidates must be >= 0")
    if max_candidates == 1 or not ranked_items:
        return list(ranked_items[:1])
    if not preserve_first:
        start = offset % len(ranked_items)
        rotated = list(ranked_items[start:]) + list(ranked_items[:start])
        return rotated[:max_candidates]

    leader = ranked_items[0]
    tail = list(ranked_items[1:])
    if not tail:
        return [leader]
    start = offset % len(tail)
    rotated_tail = tail[start:] + tail[:start]
    return [leader, *rotated_tail[: max_candidates - 1]]
