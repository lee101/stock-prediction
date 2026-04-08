#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binance_deploy_gate import (  # noqa: E402
    DEFAULT_MAX_MANIFEST_AGE_HOURS,
    gate_deploy_candidate,
    manifest_matches_deploy_config,
    pick_candidate_production_metric,
    production_metric_sort_key,
)
from src.binance_hybrid_launch import (  # noqa: E402
    DEFAULT_LAUNCH_SCRIPT,
    BinanceHybridLaunchConfig,
    parse_launch_script,
)
from src.binance_prod_config_search import (  # noqa: E402
    DEFAULT_MAX_CHECKPOINT_CONFIG_EVALS,
    DEFAULT_MAX_VARIANTS,
    ConfigVariant,
    build_config_variants,
    estimate_checkpoint_evals_per_variant,
    limit_config_variants,
    max_variants_for_checkpoint_eval_budget,
)


PROD_EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_binance_hybrid_prod.py"


@dataclass(frozen=True)
class ConfigSearchRow:
    config_slug: str
    symbols: str
    leverage: float
    checkpoint: str
    is_launch_checkpoint: bool
    metric_name: str | None
    metric_value: float | None
    median_total_return: float | None
    median_sortino: float | None
    replay_hourly_goodness_score: float | None
    manifest_path: str
    gate_metric_name: str | None = None
    gate_current_metric: float | None = None
    gate_candidate_metric: float | None = None
    gate_allowed: bool | None = None
    gate_reason: str | None = None


@dataclass(frozen=True)
class VariantEvalResult:
    returncode: int
    rows: list[ConfigSearchRow]


@dataclass(frozen=True)
class ManifestReusePlan:
    manifest_path: Path
    payload: dict[str, object]
    rows: list[ConfigSearchRow]
    missing_requested_checkpoints: tuple[str, ...]


CSV_FIELDNAMES = list(
    asdict(
        ConfigSearchRow(
            "",
            "",
            0.0,
            "",
            False,
            None,
            None,
            None,
            None,
            None,
            "",
        )
    ).keys()
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search a grid of Binance production symbol/leverage configs using production-faithful evaluation."
    )
    parser.add_argument("--launch-script", default=str(DEFAULT_LAUNCH_SCRIPT))
    parser.add_argument("--candidate-checkpoint", action="append", default=[])
    parser.add_argument(
        "--symbols-set",
        action="append",
        default=[],
        help="Repeatable comma/space separated tradable symbol set. Defaults to the current launch symbols.",
    )
    parser.add_argument(
        "--symbols-subset-size",
        action="append",
        type=int,
        default=[],
        help="Repeatable exact subset size to generate from the current launch symbol universe.",
    )
    parser.add_argument(
        "--leverage-option",
        action="append",
        type=float,
        default=[],
        help="Repeatable leverage value to evaluate. Defaults to the current launch leverage.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include-launch-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reuse-manifests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest-max-age-hours", type=float, default=DEFAULT_MAX_MANIFEST_AGE_HOURS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-variants", type=int, default=DEFAULT_MAX_VARIANTS, help="maximum number of config variants to evaluate; use 0 to disable")
    parser.add_argument("--max-checkpoint-config-evals", type=int, default=DEFAULT_MAX_CHECKPOINT_CONFIG_EVALS, help="maximum total checkpoint-config evaluations to schedule; use 0 to disable")
    parser.add_argument("--variant-offset", type=int, default=0, help="offset into the config variant list before applying caps; useful for rotating through large searches")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "analysis" / f"binance_prod_config_grid_{stamp}"


def _slugify_symbols(symbols: tuple[str, ...]) -> str:
    return "_".join("".join(ch for ch in symbol.lower() if ch.isalnum()) for symbol in symbols)


def _slugify_leverage(leverage: float) -> str:
    return str(float(leverage)).replace("-", "m").replace(".", "p")


def _normalize_checkpoint_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def build_variant_command(
    variant: ConfigVariant,
    *,
    launch_script: str | Path,
    python_bin: str | Path,
    candidate_checkpoints: list[str | Path],
) -> list[str]:
    cmd = [
        str(python_bin),
        str(PROD_EVAL_SCRIPT),
        "--launch-script",
        str(launch_script),
        "--output-dir",
        variant.output_dir,
        "--symbols",
        ",".join(variant.symbols),
        "--leverage",
        str(float(variant.leverage)),
    ]
    for checkpoint in candidate_checkpoints:
        cmd.extend(["--candidate-checkpoint", str(Path(checkpoint).expanduser().resolve(strict=False))])
    return cmd


def _safe_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _annotate_row_with_gate(
    row: ConfigSearchRow,
    *,
    launch_script: str | Path | None,
    require_manifest_best: bool = False,
) -> ConfigSearchRow:
    if launch_script is None or not row.checkpoint:
        return row
    gate_result = gate_deploy_candidate(
        candidate_checkpoint=row.checkpoint,
        launch_script=launch_script,
        manifest_path=row.manifest_path,
        require_manifest_best=require_manifest_best,
        symbols_override=row.symbols,
        leverage_override=row.leverage,
    )
    return ConfigSearchRow(
        config_slug=row.config_slug,
        symbols=row.symbols,
        leverage=row.leverage,
        checkpoint=row.checkpoint,
        is_launch_checkpoint=row.is_launch_checkpoint,
        metric_name=row.metric_name,
        metric_value=row.metric_value,
        median_total_return=row.median_total_return,
        median_sortino=row.median_sortino,
        replay_hourly_goodness_score=row.replay_hourly_goodness_score,
        manifest_path=row.manifest_path,
        gate_metric_name=gate_result.metric_name,
        gate_current_metric=gate_result.current_metric,
        gate_candidate_metric=gate_result.candidate_metric,
        gate_allowed=gate_result.allowed,
        gate_reason=gate_result.reason,
    )


def load_manifest_rows(
    manifest_path: str | Path,
    *,
    launch_script: str | Path | None = None,
    require_manifest_best: bool = False,
) -> list[ConfigSearchRow]:
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text())
    launch_config = payload.get("launch_config") if isinstance(payload.get("launch_config"), dict) else {}
    evaluations = payload.get("evaluations") if isinstance(payload.get("evaluations"), list) else []
    launch_checkpoint = str(launch_config.get("rl_checkpoint") or "")
    symbols = launch_config.get("symbols") if isinstance(launch_config.get("symbols"), list) else []
    leverage = _safe_float(launch_config.get("leverage")) or 0.0
    config_slug = f"{_slugify_symbols(tuple(str(symbol) for symbol in symbols))}__lev_{_slugify_leverage(leverage)}"

    rows: list[ConfigSearchRow] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        checkpoint = str(evaluation.get("checkpoint") or "")
        metric_name, metric_value = pick_candidate_production_metric(evaluation)
        replay = evaluation.get("replay") if isinstance(evaluation.get("replay"), dict) else {}
        row = ConfigSearchRow(
            config_slug=config_slug,
            symbols=" ".join(str(symbol) for symbol in symbols),
            leverage=leverage,
            checkpoint=checkpoint,
            is_launch_checkpoint=bool(launch_checkpoint and checkpoint == launch_checkpoint),
            metric_name=metric_name,
            metric_value=metric_value,
            median_total_return=_safe_float(evaluation.get("median_total_return")),
            median_sortino=_safe_float(evaluation.get("median_sortino")),
            replay_hourly_goodness_score=_safe_float(replay.get("hourly_goodness_score")),
            manifest_path=str(manifest_file.resolve()),
        )
        rows.append(
            _annotate_row_with_gate(
                row,
                launch_script=launch_script,
                require_manifest_best=require_manifest_best,
            )
        )
    return rows


def _requested_checkpoints(
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> tuple[str, ...]:
    checkpoints: list[str] = []
    seen: set[str] = set()
    if launch_config.rl_checkpoint:
        normalized_launch = _normalize_checkpoint_path(launch_config.rl_checkpoint)
        seen.add(normalized_launch)
        checkpoints.append(normalized_launch)
    for checkpoint in candidate_checkpoints:
        normalized = _normalize_checkpoint_path(checkpoint)
        if normalized in seen:
            continue
        seen.add(normalized)
        checkpoints.append(normalized)
    return tuple(checkpoints)


def _manifest_checkpoint_paths(payload: dict[str, object]) -> set[str]:
    manifest_evaluations = payload.get("evaluations")
    if not isinstance(manifest_evaluations, list):
        return set()
    return {
        _normalize_checkpoint_path(str(evaluation.get("checkpoint", "")))
        for evaluation in manifest_evaluations
        if isinstance(evaluation, dict) and str(evaluation.get("checkpoint", "")).strip()
    }


def _missing_requested_checkpoints(
    payload: dict[str, object],
    *,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> tuple[str, ...]:
    required_checkpoints = _requested_checkpoints(launch_config, candidate_checkpoints)
    manifest_checkpoints = _manifest_checkpoint_paths(payload)
    return tuple(checkpoint for checkpoint in required_checkpoints if checkpoint not in manifest_checkpoints)


def _missing_candidate_checkpoints(
    missing_requested_checkpoints: tuple[str, ...],
    *,
    launch_config: BinanceHybridLaunchConfig,
) -> list[str]:
    launch_checkpoint = (
        _normalize_checkpoint_path(launch_config.rl_checkpoint)
        if launch_config.rl_checkpoint
        else None
    )
    return [
        checkpoint
        for checkpoint in missing_requested_checkpoints
        if checkpoint != launch_checkpoint
    ]


def _filter_rows_for_requested_checkpoints(
    rows: list[ConfigSearchRow],
    *,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> list[ConfigSearchRow]:
    requested = set(_requested_checkpoints(launch_config, candidate_checkpoints))
    return [
        row
        for row in rows
        if _normalize_checkpoint_path(row.checkpoint) in requested
    ]


def _filter_manifest_payload_for_requested_checkpoints(
    payload: dict[str, object],
    *,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> dict[str, object]:
    requested = set(_requested_checkpoints(launch_config, candidate_checkpoints))
    filtered_payload = dict(payload)
    evaluations = payload.get("evaluations")
    filtered_payload["evaluations"] = [
        evaluation
        for evaluation in evaluations
        if isinstance(evaluation, dict)
        and _normalize_checkpoint_path(str(evaluation.get("checkpoint", ""))) in requested
    ] if isinstance(evaluations, list) else []
    return filtered_payload


def _requested_manifest_path(
    variant: ConfigVariant,
    *,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> Path:
    requested = _requested_checkpoints(launch_config, candidate_checkpoints)
    digest = hashlib.sha256("\n".join(requested).encode("utf-8")).hexdigest()[:16]
    return Path(variant.output_dir) / f"prod_launch_eval_manifest.requested.{digest}.json"


def _load_requested_manifest_rows(
    source_manifest_path: Path,
    *,
    variant: ConfigVariant,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> list[ConfigSearchRow]:
    payload = json.loads(source_manifest_path.read_text())
    requested_payload = _filter_manifest_payload_for_requested_checkpoints(
        payload,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
    )
    requested_manifest_path = _requested_manifest_path(
        variant,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
    )
    requested_manifest_path.write_text(json.dumps(requested_payload, indent=2, sort_keys=True))
    return load_manifest_rows(requested_manifest_path, launch_script=launch_script)


def _merge_manifest_payloads(
    existing_payload: dict[str, object],
    incremental_payload: dict[str, object],
) -> dict[str, object]:
    existing_evaluations = existing_payload.get("evaluations")
    incremental_evaluations = incremental_payload.get("evaluations")
    merged_by_checkpoint: dict[str, dict[str, object]] = {}
    ordered_checkpoints: list[str] = []

    for evaluations in (existing_evaluations, incremental_evaluations):
        if not isinstance(evaluations, list):
            continue
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            checkpoint = str(evaluation.get("checkpoint", "")).strip()
            if not checkpoint:
                continue
            normalized = _normalize_checkpoint_path(checkpoint)
            if normalized not in ordered_checkpoints:
                ordered_checkpoints.append(normalized)
            merged_by_checkpoint[normalized] = evaluation

    merged_payload = dict(incremental_payload)
    merged_payload["evaluations"] = [merged_by_checkpoint[checkpoint] for checkpoint in ordered_checkpoints]
    return merged_payload


def _load_reusable_manifest_plan(
    variant: ConfigVariant,
    *,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
    max_manifest_age_hours: float | None,
) -> ManifestReusePlan | None:
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    compatible, _reason = manifest_matches_deploy_config(
        payload,
        launch_script,
        symbols_override=",".join(variant.symbols),
        leverage_override=variant.leverage,
        max_manifest_age_hours=max_manifest_age_hours,
    )
    if not compatible:
        return None
    return ManifestReusePlan(
        manifest_path=manifest_path,
        payload=payload,
        rows=_load_requested_manifest_rows(
            manifest_path,
            variant=variant,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
        ),
        missing_requested_checkpoints=_missing_requested_checkpoints(
            payload,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
        ),
    )


def _row_metric_name(row: ConfigSearchRow) -> str | None:
    return row.gate_metric_name or row.metric_name


def _row_metric_value(row: ConfigSearchRow) -> float | None:
    return row.gate_candidate_metric if row.gate_candidate_metric is not None else row.metric_value


def _row_sort_key(row: ConfigSearchRow) -> tuple[int, float]:
    return production_metric_sort_key(_row_metric_name(row), _row_metric_value(row))


def _row_selection_sort_key(row: ConfigSearchRow) -> tuple[int, int, float]:
    metric_priority, metric_value = _row_sort_key(row)
    gate_allowed = 1 if row.gate_allowed is True else 0
    return gate_allowed, metric_priority, metric_value


def select_best_rows(rows: list[ConfigSearchRow], *, candidate_checkpoints_requested: bool, include_launch_checkpoint: bool = False) -> list[ConfigSearchRow]:
    grouped: dict[str, list[ConfigSearchRow]] = {}
    for row in rows:
        grouped.setdefault(row.config_slug, []).append(row)

    best_rows: list[ConfigSearchRow] = []
    for config_rows in grouped.values():
        selectable_rows = config_rows
        if candidate_checkpoints_requested and not include_launch_checkpoint:
            candidate_rows = [row for row in config_rows if not row.is_launch_checkpoint]
            if candidate_rows:
                selectable_rows = candidate_rows
        best_rows.append(max(selectable_rows, key=_row_selection_sort_key))
    return sorted(best_rows, key=_row_selection_sort_key, reverse=True)


def _write_rows_csv(path: Path, rows: list[ConfigSearchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _merge_incremental_manifest_rows(
    reuse_plan: ManifestReusePlan,
    *,
    variant: ConfigVariant,
    launch_script: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> list[ConfigSearchRow]:
    incremental_payload = json.loads(reuse_plan.manifest_path.read_text())
    merged_payload = _merge_manifest_payloads(reuse_plan.payload, incremental_payload)
    reuse_plan.manifest_path.write_text(json.dumps(merged_payload, indent=2, sort_keys=True))
    return _load_requested_manifest_rows(
        reuse_plan.manifest_path,
        variant=variant,
        launch_script=launch_script,
        launch_config=launch_config,
        candidate_checkpoints=candidate_checkpoints,
    )


def _evaluate_variant(
    variant: ConfigVariant,
    *,
    launch_script: str | Path,
    python_bin: str | Path,
    launch_config: BinanceHybridLaunchConfig,
    candidate_checkpoints: list[str | Path],
) -> VariantEvalResult:
    cmd = build_variant_command(
        variant,
        launch_script=launch_script,
        python_bin=python_bin,
        candidate_checkpoints=candidate_checkpoints,
    )
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return VariantEvalResult(returncode=completed.returncode, rows=[])
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    if not manifest_path.exists():
        sys.stderr.write(f"missing manifest: {manifest_path}\n")
        return VariantEvalResult(returncode=2, rows=[])
    return VariantEvalResult(
        returncode=0,
        rows=_load_requested_manifest_rows(
            manifest_path,
            variant=variant,
            launch_script=launch_script,
            launch_config=launch_config,
            candidate_checkpoints=candidate_checkpoints,
        ),
    )


def run_grid_search(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if args.jobs < 1:
        sys.stderr.write("--jobs must be at least 1\n")
        return 2

    output_root = Path(args.output_dir) if args.output_dir else _default_output_dir()
    launch_config = parse_launch_script(args.launch_script, require_rl_checkpoint=False)
    try:
        requested_variants = build_config_variants(
            launch_script=args.launch_script,
            symbol_set_specs=list(args.symbols_set),
            symbol_subset_sizes=list(args.symbols_subset_size),
            leverage_options=list(args.leverage_option),
            output_root=output_root,
            include_launch_variant=args.include_launch_checkpoint,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    candidate_checkpoints_requested = bool(args.candidate_checkpoint)
    variant_offset = int(args.variant_offset)
    if variant_offset < 0:
        sys.stderr.write("--variant-offset must be >= 0\n")
        return 2
    raw_max_variants = int(args.max_variants)
    if raw_max_variants < 0:
        sys.stderr.write("--max-variants must be >= 0\n")
        return 2
    max_variants = None if raw_max_variants == 0 else raw_max_variants
    raw_max_checkpoint_config_evals = int(args.max_checkpoint_config_evals)
    if raw_max_checkpoint_config_evals < 0:
        sys.stderr.write("--max-checkpoint-config-evals must be >= 0\n")
        return 2
    max_checkpoint_config_evals = None if raw_max_checkpoint_config_evals == 0 else raw_max_checkpoint_config_evals
    per_variant_checkpoint_evals = estimate_checkpoint_evals_per_variant(
        launch_config,
        list(args.candidate_checkpoint),
    )
    requested_checkpoint_config_evals = len(requested_variants) * per_variant_checkpoint_evals
    max_variants_by_checkpoint_budget = max_variants_for_checkpoint_eval_budget(
        launch_config,
        candidate_checkpoints=list(args.candidate_checkpoint),
        max_checkpoint_config_evals=max_checkpoint_config_evals,
    )
    if max_variants_by_checkpoint_budget is not None and max_variants_by_checkpoint_budget < 1 and requested_variants:
        sys.stderr.write(
            f"estimated checkpoint-config eval count {requested_checkpoint_config_evals} exceeds max_checkpoint_config_evals {max_checkpoint_config_evals}; budget too small to evaluate even one config variant\n"
        )
        return 2
    effective_max_variants = max_variants
    if max_variants_by_checkpoint_budget is not None:
        effective_max_variants = (
            max_variants_by_checkpoint_budget
            if effective_max_variants is None
            else min(effective_max_variants, max_variants_by_checkpoint_budget)
        )
    variants = limit_config_variants(
        requested_variants,
        launch_config=launch_config,
        max_variants=effective_max_variants,
        variant_offset=variant_offset,
    )
    if max_variants is not None and len(requested_variants) > max_variants:
        sys.stderr.write(
            f"variant count {len(requested_variants)} exceeds max_variants {raw_max_variants}; pruning to {len(variants)} variant(s)\n"
        )
    if max_variants_by_checkpoint_budget is not None and len(requested_variants) > max_variants_by_checkpoint_budget:
        sys.stderr.write(
            f"estimated checkpoint-config eval count {requested_checkpoint_config_evals} exceeds max_checkpoint_config_evals {max_checkpoint_config_evals}; pruning to {len(variants)} variant(s)\n"
        )
    estimated_checkpoint_config_evals = len(variants) * per_variant_checkpoint_evals

    print("Binance production config grid search")
    print(f"  launch:        {Path(args.launch_script).resolve()}")
    print(f"  variants:      {len(variants)}")
    if len(variants) != len(requested_variants):
        print(f"  requested variants: {len(requested_variants)}")
    print(f"  output root:   {output_root}")
    if args.candidate_checkpoint:
        print(f"  candidates:    {len(args.candidate_checkpoint)}")
    else:
        print("  candidates:    <launch checkpoint only>")
    print(f"  jobs:          {args.jobs}")
    print(f"  reuse manifest:{bool(args.reuse_manifests)}")
    print(f"  include launch checkpoint: {bool(args.include_launch_checkpoint)}")
    print(f"  max variants:  {args.max_variants}")
    print(f"  variant offset: {args.variant_offset}")
    print(f"  est checkpoint-config evals: {estimated_checkpoint_config_evals}")
    print(f"  max checkpoint-config evals: {args.max_checkpoint_config_evals}")
    if args.symbols_subset_size:
        print(f"  subset sizes:  {', '.join(str(size) for size in args.symbols_subset_size)}")
    print("")

    for index, variant in enumerate(variants, start=1):
        cmd = build_variant_command(
            variant,
            launch_script=args.launch_script,
            python_bin=args.python_bin,
            candidate_checkpoints=list(args.candidate_checkpoint),
        )
        print(f"[{index}/{len(variants)}] symbols={' '.join(variant.symbols)} leverage={variant.leverage:g}")
        print(f"  {' '.join(cmd)}")

    if args.dry_run:
        print("\nDRY RUN -- no evaluations executed")
        return 0

    all_rows: list[ConfigSearchRow] = []
    pending_variants: list[tuple[ConfigVariant, ManifestReusePlan | None, list[str | Path]]] = []
    fully_reused_variants = 0
    partially_reused_variants = 0
    if args.reuse_manifests:
        for variant in variants:
            reuse_plan = _load_reusable_manifest_plan(
                variant,
                launch_script=args.launch_script,
                launch_config=launch_config,
                candidate_checkpoints=list(args.candidate_checkpoint),
                max_manifest_age_hours=args.manifest_max_age_hours,
            )
            if reuse_plan is None:
                pending_variants.append((variant, None, list(args.candidate_checkpoint)))
                continue
            if not reuse_plan.missing_requested_checkpoints:
                all_rows.extend(reuse_plan.rows)
                fully_reused_variants += 1
                continue
            pending_variants.append(
                (
                    variant,
                    reuse_plan,
                    _missing_candidate_checkpoints(
                        reuse_plan.missing_requested_checkpoints,
                        launch_config=launch_config,
                    ),
                )
            )
            partially_reused_variants += 1
    else:
        pending_variants = [
            (variant, None, list(args.candidate_checkpoint))
            for variant in variants
        ]

    if args.jobs == 1:
        for variant, reuse_plan, variant_candidate_checkpoints in pending_variants:
            result = _evaluate_variant(
                variant,
                launch_script=args.launch_script,
                python_bin=args.python_bin,
                launch_config=launch_config,
                candidate_checkpoints=variant_candidate_checkpoints,
            )
            if result.returncode != 0:
                return result.returncode
            if reuse_plan is not None:
                all_rows.extend(
                    _merge_incremental_manifest_rows(
                        reuse_plan,
                        variant=variant,
                        launch_script=args.launch_script,
                        launch_config=launch_config,
                        candidate_checkpoints=list(args.candidate_checkpoint),
                    )
                )
            else:
                all_rows.extend(result.rows)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    _evaluate_variant,
                    variant,
                    launch_script=args.launch_script,
                    python_bin=args.python_bin,
                    launch_config=launch_config,
                    candidate_checkpoints=variant_candidate_checkpoints,
                ): (variant, reuse_plan)
                for variant, reuse_plan, variant_candidate_checkpoints in pending_variants
            }
            for future in concurrent.futures.as_completed(futures):
                variant, reuse_plan = futures[future]
                result = future.result()
                if result.returncode != 0:
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()
                    return result.returncode
                if reuse_plan is not None:
                    all_rows.extend(
                        _merge_incremental_manifest_rows(
                            reuse_plan,
                            variant=variant,
                            launch_script=args.launch_script,
                            launch_config=launch_config,
                            candidate_checkpoints=list(args.candidate_checkpoint),
                        )
                    )
                else:
                    all_rows.extend(result.rows)

    if not all_rows:
        sys.stderr.write("no evaluation rows were produced\n")
        return 2

    best_rows = select_best_rows(
        all_rows,
        candidate_checkpoints_requested=candidate_checkpoints_requested,
        include_launch_checkpoint=args.include_launch_checkpoint,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows_path = output_root / "all_results.csv"
    best_rows_path = output_root / "best_by_config.csv"
    _write_rows_csv(all_rows_path, all_rows)
    _write_rows_csv(best_rows_path, best_rows)

    print("\nTop production configs")
    for index, row in enumerate(best_rows[: max(1, int(args.top_k))], start=1):
        metric_value = _row_metric_value(row)
        metric_value = metric_value if metric_value is not None else float("-inf")
        gate_status = "ALLOW" if row.gate_allowed is True else "BLOCK"
        print(
            f"  {index}. symbols={row.symbols} leverage={row.leverage:g} "
            f"checkpoint={Path(row.checkpoint).name or row.checkpoint} "
            f"metric={_row_metric_name(row) or 'N/A'} value={metric_value:+.4f} gate={gate_status}"
        )
    if fully_reused_variants or partially_reused_variants:
        print(
            f"\nReused manifests: full={fully_reused_variants}/{len(variants)} partial={partially_reused_variants}/{len(variants)}"
        )
    print(f"\nAll rows:   {all_rows_path}")
    print(f"Best rows:  {best_rows_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_grid_search(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
