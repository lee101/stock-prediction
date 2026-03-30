#!/usr/bin/env python3
"""Run the same stock-training config through multiple trainer backends."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.config import TRAINER_BACKENDS, TrainerBackend
from binanceneural.data import MultiSymbolDataModule
from binanceneural.trainer_factory import build_trainer
from unified_hourly_experiment.classic_training_common import (
    ArgsFileParser,
    EFFECTIVE_ARGS_JSON_FILENAME,
    EFFECTIVE_ARGS_TXT_FILENAME,
    build_classic_data_module,
    build_classic_training_config,
    parse_horizons,
    render_classic_run_plan_summary,
    write_effective_args_artifacts,
)
from unified_hourly_experiment.directional_constraints import build_directional_constraints
from unified_hourly_experiment.jax_classic_defaults import (
    DEFAULT_JAX_CLASSIC_SYMBOLS_CSV,
    JAX_CLASSIC_COMPARE_DEFAULT_DRY_TRAIN_STEPS,
    JAX_CLASSIC_COMPARE_DEFAULT_EPOCHS,
    JAX_CLASSIC_COMPARE_DEFAULT_MAX_RUNS,
    JAX_CLASSIC_DEFAULT_BATCH_SIZE,
    JAX_CLASSIC_DEFAULT_DECISION_LAG_BARS,
    JAX_CLASSIC_DEFAULT_FILL_BUFFER_PCT,
    JAX_CLASSIC_DEFAULT_FILL_TEMPERATURE,
    JAX_CLASSIC_DEFAULT_GRAD_CLIP,
    JAX_CLASSIC_DEFAULT_HIDDEN_DIM,
    JAX_CLASSIC_DEFAULT_LEARNING_RATE,
    JAX_CLASSIC_DEFAULT_MAKER_FEE,
    JAX_CLASSIC_DEFAULT_MARGIN_ANNUAL_RATE,
    JAX_CLASSIC_DEFAULT_MAX_HOLD_HOURS,
    JAX_CLASSIC_DEFAULT_MAX_LEVERAGE,
    JAX_CLASSIC_DEFAULT_NUM_HEADS,
    JAX_CLASSIC_DEFAULT_NUM_LAYERS,
    JAX_CLASSIC_DEFAULT_RETURN_WEIGHT,
    JAX_CLASSIC_DEFAULT_SEED,
    JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH,
    JAX_CLASSIC_DEFAULT_SMOOTHNESS_PENALTY,
    JAX_CLASSIC_DEFAULT_VALIDATION_DAYS,
    JAX_CLASSIC_DEFAULT_WEIGHT_DECAY,
)
from unified_hourly_experiment.symbol_validation import parse_symbols
from unified_orchestrator.jsonl_utils import append_jsonl_row

RUN_EVENTS_FILENAME = "run_events.jsonl"


RunEventType = Literal[
    "run_start",
    "plan_error",
    "dataset_prepare_start",
    "dataset_ready",
    "dataset_error",
    "backend_seed_start",
    "backend_seed_complete",
    "backend_seed_error",
    "run_complete",
]


class CompareTrainingSummary(TypedDict):
    epochs: int
    dry_train_steps: int
    batch_size: int
    sequence_length: int
    validation_days: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    use_compile: bool


class CompareRunPlan(TypedDict):
    symbols: list[str]
    backend_count: int
    backends: list[TrainerBackend]
    seed_count: int
    seeds: list[int]
    total_runs: int
    max_runs: int
    allow_large_run: bool
    plan_error: str | None
    forecast_horizons: list[int]
    cache_only: bool
    output_dir: str
    effective_args_path: str
    effective_args_cli_path: str
    preload: str | None
    training: CompareTrainingSummary


class SuccessfulBackendResult(TypedDict):
    backend: TrainerBackend
    seed: int
    status: Literal["ok"]
    duration_sec: float
    history_length: int
    best_checkpoint: str
    stop_reason: str | None
    best_epoch: int | None
    best_val_score: float | None
    best_val_sortino: float | None
    best_val_return: float | None
    best_train_loss: float | None


class FailedBackendResult(TypedDict):
    backend: TrainerBackend
    seed: int
    status: Literal["error"]
    duration_sec: float
    error: str
    traceback: str


BackendResult = SuccessfulBackendResult | FailedBackendResult


class BackendAggregateSummary(TypedDict):
    backend: TrainerBackend
    run_count: int
    ok_runs: int
    error_runs: int
    success_rate: float
    seeds: list[int]
    best_val_score_mean: float | None
    best_val_score_std: float | None
    stability_adjusted_score: float | None
    best_val_sortino_mean: float | None
    best_val_return_mean: float | None
    duration_sec_mean: float | None


class CompareReport(TypedDict):
    run_plan: CompareRunPlan
    run_id: str
    run_events_path: str
    symbols: list[str]
    forecast_horizons: list[int]
    dry_train_steps: int | None
    cache_only: bool
    plan_error: str | None
    results: list[BackendResult]
    backend_summary: list[BackendAggregateSummary]
    winner_backend: TrainerBackend | None
    recommended_backend: TrainerBackend | None
    dataset_error: NotRequired[str]
    effective_args_path: NotRequired[str]
    effective_args_cli_path: NotRequired[str]
    effective_args_warning: NotRequired[str]


class _HistoryEntry(Protocol):
    epoch: int | None
    train_loss: float | None
    val_score: float | None
    val_sortino: float | None
    val_return: float | None


class _TrainerArtifacts(Protocol):
    history: list[_HistoryEntry] | tuple[_HistoryEntry, ...] | None
    best_checkpoint: object
    stop_reason: str | None


def parse_backends(raw: str) -> list[TrainerBackend]:
    requested = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not requested:
        raise ValueError("At least one backend is required.")
    invalid = [backend for backend in requested if backend not in TRAINER_BACKENDS]
    if invalid:
        allowed = ", ".join(TRAINER_BACKENDS)
        raise ValueError(f"Unsupported trainer backend(s): {invalid}. Allowed: {allowed}.")
    deduped: list[TrainerBackend] = []
    for backend in requested:
        typed_backend = cast(TrainerBackend, backend)
        if typed_backend not in deduped:
            deduped.append(typed_backend)
    return deduped


def parse_seeds(raw: str | None, *, fallback_seed: int) -> list[int]:
    if raw is None or not str(raw).strip():
        return [int(fallback_seed)]
    seeds = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    deduped: list[int] = []
    for seed in seeds:
        if seed not in deduped:
            deduped.append(seed)
    return deduped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = ArgsFileParser(
        description="Compare PyTorch and JAX trainer backends on the same stock config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--symbols", default=DEFAULT_JAX_CLASSIC_SYMBOLS_CSV)
    parser.add_argument("--backends", default="torch,jax_classic")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("unified_hourly_experiment/backend_compare"))
    parser.add_argument("--epochs", type=int, default=JAX_CLASSIC_COMPARE_DEFAULT_EPOCHS)
    parser.add_argument("--dry-train-steps", type=int, default=JAX_CLASSIC_COMPARE_DEFAULT_DRY_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=JAX_CLASSIC_DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--validation-days", type=int, default=JAX_CLASSIC_DEFAULT_VALIDATION_DAYS)
    parser.add_argument("--forecast-horizons", default="1")
    parser.add_argument("--cache-only", dest="cache_only", action="store_true", default=True)
    parser.add_argument("--allow-forecast-refresh", dest="cache_only", action="store_false")
    parser.add_argument("--learning-rate", type=float, default=JAX_CLASSIC_DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=JAX_CLASSIC_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=JAX_CLASSIC_DEFAULT_GRAD_CLIP)
    parser.add_argument("--hidden-dim", type=int, default=JAX_CLASSIC_DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=JAX_CLASSIC_DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=JAX_CLASSIC_DEFAULT_NUM_HEADS)
    parser.add_argument("--return-weight", type=float, default=JAX_CLASSIC_DEFAULT_RETURN_WEIGHT)
    parser.add_argument("--smoothness-penalty", type=float, default=JAX_CLASSIC_DEFAULT_SMOOTHNESS_PENALTY)
    parser.add_argument("--maker-fee", type=float, default=JAX_CLASSIC_DEFAULT_MAKER_FEE)
    parser.add_argument("--fill-temperature", type=float, default=JAX_CLASSIC_DEFAULT_FILL_TEMPERATURE)
    parser.add_argument("--max-hold-hours", type=float, default=JAX_CLASSIC_DEFAULT_MAX_HOLD_HOURS)
    parser.add_argument("--max-leverage", type=float, default=JAX_CLASSIC_DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--margin-annual-rate", type=float, default=JAX_CLASSIC_DEFAULT_MARGIN_ANNUAL_RATE)
    parser.add_argument("--decision-lag-bars", type=int, default=JAX_CLASSIC_DEFAULT_DECISION_LAG_BARS)
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--fill-buffer-pct", type=float, default=JAX_CLASSIC_DEFAULT_FILL_BUFFER_PCT)
    parser.add_argument("--seed", type=int, default=JAX_CLASSIC_DEFAULT_SEED)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds to compare per backend.")
    parser.add_argument("--max-runs", type=int, default=JAX_CLASSIC_COMPARE_DEFAULT_MAX_RUNS)
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Allow planned backend x seed runs to exceed --max-runs.",
    )
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--preload", type=Path, default=None)
    parser.add_argument(
        "--describe-run",
        action="store_true",
        help="Print the resolved run plan and exit without preparing data or training.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _make_training_config(
    args: argparse.Namespace,
    backend: TrainerBackend,
    run_dir: Path,
    *,
    seed: int,
):
    return build_classic_training_config(
        args,
        backend=backend,
        checkpoint_root=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
        run_name=f"compare_{backend}_seed{seed}",
        extra_kwargs={
            "seed": int(seed),
            "use_compile": bool(args.use_compile and backend == "torch"),
            "wandb_mode": "disabled",
            "wandb_log_metrics": False,
        },
    )


def _best_history_entry(artifacts: _TrainerArtifacts) -> _HistoryEntry | None:
    history = list(getattr(artifacts, "history", []) or [])
    if not history:
        return None
    return max(history, key=lambda item: getattr(item, "val_score", None) or float("-inf"))


def summarize_backend_result(
    backend: TrainerBackend,
    seed: int,
    artifacts: _TrainerArtifacts,
    duration_sec: float,
) -> SuccessfulBackendResult:
    best = _best_history_entry(artifacts)
    summary: SuccessfulBackendResult = {
        "backend": backend,
        "seed": int(seed),
        "status": "ok",
        "duration_sec": round(float(duration_sec), 3),
        "history_length": len(list(getattr(artifacts, "history", []) or [])),
        "best_checkpoint": str(getattr(artifacts, "best_checkpoint", None) or ""),
        "stop_reason": getattr(artifacts, "stop_reason", None),
        "best_epoch": getattr(best, "epoch", None),
        "best_val_score": getattr(best, "val_score", None),
        "best_val_sortino": getattr(best, "val_sortino", None),
        "best_val_return": getattr(best, "val_return", None),
        "best_train_loss": getattr(best, "train_loss", None),
    }
    return summary


def _append_run_event(
    events_path: Path,
    *,
    event_type: RunEventType,
    run_id: str,
    output_dir: Path,
    payload: dict[str, object] | None = None,
) -> None:
    event: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": str(event_type),
        "run_id": str(run_id),
        "output_dir": str(output_dir),
        "pid": int(os.getpid()),
    }
    if payload:
        event.update(payload)
    try:
        append_jsonl_row(events_path, event, sort_keys=True, default=str)
    except Exception as exc:  # pragma: no cover - best-effort observability
        print(
            f"[compare_trainer_backends] failed to append {event_type!r} event to {events_path}: {exc}",
            file=sys.stderr,
        )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    assert mean is not None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(variance ** 0.5)


def summarize_backend_groups(results: list[BackendResult]) -> list[BackendAggregateSummary]:
    grouped: dict[TrainerBackend, list[BackendResult]] = {}
    for result in results:
        grouped.setdefault(result["backend"], []).append(result)

    summaries: list[BackendAggregateSummary] = []
    for backend, items in grouped.items():
        ok_items = [item for item in items if item.get("status") == "ok"]
        score_values = [float(item["best_val_score"]) for item in ok_items if item.get("best_val_score") is not None]
        score_mean = _mean(score_values)
        score_std = _std(score_values)
        summaries.append(
            {
                "backend": backend,
                "run_count": len(items),
                "ok_runs": len(ok_items),
                "error_runs": len(items) - len(ok_items),
                "success_rate": float(len(ok_items) / len(items)) if items else 0.0,
                "seeds": [int(item["seed"]) for item in items if item.get("seed") is not None],
                "best_val_score_mean": score_mean,
                "best_val_score_std": score_std,
                "stability_adjusted_score": (
                    None if score_mean is None else float(score_mean - (score_std or 0.0))
                ),
                "best_val_sortino_mean": _mean(
                    [float(item["best_val_sortino"]) for item in ok_items if item.get("best_val_sortino") is not None]
                ),
                "best_val_return_mean": _mean(
                    [float(item["best_val_return"]) for item in ok_items if item.get("best_val_return") is not None]
                ),
                "duration_sec_mean": _mean(
                    [float(item["duration_sec"]) for item in ok_items if item.get("duration_sec") is not None]
                ),
            }
    )
    return sorted(summaries, key=lambda item: item["backend"])


def recommend_backend(backend_summary: list[BackendAggregateSummary]) -> TrainerBackend | None:
    eligible = [item for item in backend_summary if item.get("best_val_score_mean") is not None]
    if not eligible:
        return None
    recommendation = max(
        eligible,
        key=lambda item: (
            float(item["success_rate"]),
            float(item["stability_adjusted_score"] or float("-inf")),
            float(item["best_val_score_mean"] or float("-inf")),
        ),
    )
    return recommendation["backend"]


def render_markdown_report(report: CompareReport) -> str:
    run_plan = report.get("run_plan") or {}
    lines = [
        "# Trainer Backend Comparison",
        "",
        f"- Symbols: `{','.join(report['symbols'])}`",
        f"- Forecast horizons: `{','.join(str(item) for item in report['forecast_horizons'])}`",
        f"- Seeds: `{','.join(str(item) for item in run_plan.get('seeds', []))}`",
        f"- Planned runs: `{run_plan.get('total_runs', 0)}`",
        f"- Dry train steps: `{report['dry_train_steps']}`",
        f"- Cache only: `{report['cache_only']}`",
        f"- Output dir: `{run_plan.get('output_dir', '')}`",
        "",
    ]
    if report.get("effective_args_path"):
        lines.insert(-1, f"- Effective args: `{report['effective_args_path']}`")
    if report.get("effective_args_cli_path"):
        lines.insert(-1, f"- Rerun args file: `{report['effective_args_cli_path']}`")
    if report.get("effective_args_warning"):
        lines.insert(-1, f"- Effective args warning: `{report['effective_args_warning']}`")
    if report.get("run_events_path"):
        lines.insert(-1, f"- Run events: `{report['run_events_path']}`")
    plan_error = report.get("plan_error")
    if plan_error:
        lines.extend(
            [
                "## Plan Error",
                "",
                plan_error,
                "",
            ]
        )
    dataset_error = report.get("dataset_error")
    if dataset_error:
        lines.extend(
            [
                "## Dataset Error",
                "",
                dataset_error,
                "",
            ]
        )
    backend_summary = report.get("backend_summary") or []
    if backend_summary:
        lines.extend(
            [
                "## Aggregated Summary",
                "",
                "| Backend | Runs | OK runs | Errors | Success rate | Mean val_score | Std val_score | Stable score | Mean val_sortino | Mean val_return | Mean duration (s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in backend_summary:
            lines.append(
                "| {backend} | {run_count} | {ok_runs} | {error_runs} | {success_rate} | {score_mean} | {score_std} | {stable_score} | {sortino_mean} | "
                "{return_mean} | {duration_mean} |".format(
                    backend=item["backend"],
                    run_count=int(item["run_count"]),
                    ok_runs=int(item["ok_runs"]),
                    error_runs=int(item["error_runs"]),
                    success_rate=_fmt_metric(item.get("success_rate")),
                    score_mean=_fmt_metric(item.get("best_val_score_mean")),
                    score_std=_fmt_metric(item.get("best_val_score_std")),
                    stable_score=_fmt_metric(item.get("stability_adjusted_score")),
                    sortino_mean=_fmt_metric(item.get("best_val_sortino_mean")),
                    return_mean=_fmt_metric(item.get("best_val_return_mean")),
                    duration_mean=_fmt_metric(item.get("duration_sec_mean")),
                )
            )
        lines.append("")
    lines.extend(
        [
        "## Per-Run Results",
        "",
        "| Backend | Seed | Status | Duration (s) | Best val_score | Best val_sortino | Best val_return | Stop reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in report["results"]:
        lines.append(
            "| {backend} | {seed} | {status} | {duration_sec:.3f} | {best_val_score} | {best_val_sortino} | "
            "{best_val_return} | {stop_reason} |".format(
                backend=result["backend"],
                seed=int(result.get("seed", 0)),
                status=result["status"],
                duration_sec=float(result["duration_sec"]),
                best_val_score=_fmt_metric(result.get("best_val_score")),
                best_val_sortino=_fmt_metric(result.get("best_val_sortino")),
                best_val_return=_fmt_metric(result.get("best_val_return")),
                stop_reason=result.get("stop_reason") or "",
            )
        )
    winner = report.get("winner_backend")
    if winner:
        lines.extend(["", f"Winner by `best_val_score`: `{winner}`"])
    recommended = report.get("recommended_backend")
    if recommended:
        lines.extend([f"Recommended backend by reliability: `{recommended}`"])
    return "\n".join(lines) + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def build_run_plan(args: argparse.Namespace) -> CompareRunPlan:
    symbols = parse_symbols(args.symbols)
    horizons = parse_horizons(args.forecast_horizons)
    backends = parse_backends(args.backends)
    seeds = parse_seeds(getattr(args, "seeds", None), fallback_seed=int(args.seed))
    max_runs = int(getattr(args, "max_runs", JAX_CLASSIC_COMPARE_DEFAULT_MAX_RUNS))
    if max_runs <= 0:
        raise ValueError("max_runs must be positive.")
    allow_large_run = bool(getattr(args, "allow_large_run", False))
    total_runs = len(backends) * len(seeds)
    plan_error = None
    if total_runs > max_runs and not allow_large_run:
        plan_error = (
            f"Planned {total_runs} backend runs ({len(backends)} backends x {len(seeds)} seeds), "
            f"which exceeds max_runs={max_runs}. Use --allow-large-run to override."
        )
    output_dir = Path(args.output_dir)
    return {
        "symbols": symbols,
        "backend_count": len(backends),
        "backends": backends,
        "seed_count": len(seeds),
        "seeds": seeds,
        "total_runs": total_runs,
        "max_runs": max_runs,
        "allow_large_run": allow_large_run,
        "plan_error": plan_error,
        "forecast_horizons": list(horizons),
        "cache_only": bool(args.cache_only),
        "output_dir": str(output_dir),
        "effective_args_path": str(output_dir / EFFECTIVE_ARGS_JSON_FILENAME),
        "effective_args_cli_path": str(output_dir / EFFECTIVE_ARGS_TXT_FILENAME),
        "preload": str(args.preload) if args.preload else None,
        "training": {
            "epochs": int(args.epochs),
            "dry_train_steps": int(args.dry_train_steps),
            "batch_size": int(args.batch_size),
            "sequence_length": int(args.sequence_length),
            "validation_days": int(args.validation_days),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "use_compile": bool(args.use_compile),
        },
}


def _build_failed_run_plan(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    plan_error: str,
) -> CompareRunPlan:
    return {
        "symbols": [],
        "backend_count": 0,
        "backends": [],
        "seed_count": 0,
        "seeds": [],
        "total_runs": 0,
        "max_runs": int(getattr(args, "max_runs", JAX_CLASSIC_COMPARE_DEFAULT_MAX_RUNS)),
        "allow_large_run": bool(getattr(args, "allow_large_run", False)),
        "plan_error": str(plan_error),
        "forecast_horizons": [],
        "cache_only": bool(getattr(args, "cache_only", True)),
        "output_dir": str(output_dir),
        "effective_args_path": str(output_dir / EFFECTIVE_ARGS_JSON_FILENAME),
        "effective_args_cli_path": str(output_dir / EFFECTIVE_ARGS_TXT_FILENAME),
        "preload": str(getattr(args, "preload", None)) if getattr(args, "preload", None) else None,
        "training": {
            "epochs": int(getattr(args, "epochs", JAX_CLASSIC_COMPARE_DEFAULT_EPOCHS)),
            "dry_train_steps": int(getattr(args, "dry_train_steps", JAX_CLASSIC_COMPARE_DEFAULT_DRY_TRAIN_STEPS)),
            "batch_size": int(getattr(args, "batch_size", JAX_CLASSIC_DEFAULT_BATCH_SIZE)),
            "sequence_length": int(getattr(args, "sequence_length", JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH)),
            "validation_days": int(getattr(args, "validation_days", JAX_CLASSIC_DEFAULT_VALIDATION_DAYS)),
            "learning_rate": float(getattr(args, "learning_rate", JAX_CLASSIC_DEFAULT_LEARNING_RATE)),
            "weight_decay": float(getattr(args, "weight_decay", JAX_CLASSIC_DEFAULT_WEIGHT_DECAY)),
            "grad_clip": float(getattr(args, "grad_clip", JAX_CLASSIC_DEFAULT_GRAD_CLIP)),
            "hidden_dim": int(getattr(args, "hidden_dim", JAX_CLASSIC_DEFAULT_HIDDEN_DIM)),
            "num_layers": int(getattr(args, "num_layers", JAX_CLASSIC_DEFAULT_NUM_LAYERS)),
            "num_heads": int(getattr(args, "num_heads", JAX_CLASSIC_DEFAULT_NUM_HEADS)),
            "use_compile": bool(getattr(args, "use_compile", False)),
        },
    }


def _ensure_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to prepare output directory {output_dir}: {exc}") from exc


def _attach_effective_args_artifacts(
    report: CompareReport,
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> None:
    parser = build_arg_parser()
    try:
        effective_args_path, effective_args_cli_path = write_effective_args_artifacts(
            parser,
            args,
            output_dir,
            module_name="unified_hourly_experiment.compare_trainer_backends",
        )
    except Exception as exc:
        warning = f"Failed to write effective args artifacts in {output_dir}: {exc}"
        report["effective_args_warning"] = warning
        print(f"[compare_trainer_backends] {warning}", file=sys.stderr)
        return
    report["effective_args_path"] = str(effective_args_path)
    report["effective_args_cli_path"] = str(effective_args_cli_path)


def _write_compare_report_artifacts(
    report: CompareReport,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    _ensure_output_dir(output_dir)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    try:
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        md_path.write_text(render_markdown_report(report), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write comparison report artifacts in {output_dir}: {exc}") from exc
    return json_path, md_path


def run_backend_comparison(args: argparse.Namespace) -> CompareReport:
    plan = build_run_plan(args)
    symbols = list(plan["symbols"])
    horizons = tuple(int(item) for item in plan["forecast_horizons"])
    backends = list(plan["backends"])
    seeds = [int(item) for item in plan["seeds"]]
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_events_path = output_dir / RUN_EVENTS_FILENAME
    run_started_at = datetime.now(timezone.utc)
    run_id = f"{os.getpid()}:{int(run_started_at.timestamp() * 1000)}"
    _append_run_event(
        run_events_path,
        event_type="run_start",
        run_id=run_id,
        output_dir=output_dir,
        payload={
            "symbols": symbols,
            "forecast_horizons": list(horizons),
            "backends": backends,
            "seeds": seeds,
            "cache_only": bool(args.cache_only),
            "total_runs": int(plan["total_runs"]),
        },
    )

    if plan.get("plan_error"):
        _append_run_event(
            run_events_path,
            event_type="plan_error",
            run_id=run_id,
            output_dir=output_dir,
            payload={"error": str(plan["plan_error"])},
        )
        return {
            "run_plan": plan,
            "run_id": run_id,
            "run_events_path": str(run_events_path),
            "symbols": symbols,
            "forecast_horizons": list(horizons),
            "dry_train_steps": args.dry_train_steps,
            "cache_only": bool(args.cache_only),
            "plan_error": plan["plan_error"],
            "results": [],
            "backend_summary": [],
            "winner_backend": None,
            "recommended_backend": None,
        }

    _append_run_event(
        run_events_path,
        event_type="dataset_prepare_start",
        run_id=run_id,
        output_dir=output_dir,
        payload={"symbol_count": len(symbols)},
    )
    try:
        _, data_module = build_classic_data_module(
            args,
            symbols=symbols,
            horizons=horizons,
            data_module_cls=MultiSymbolDataModule,
            directional_constraints=build_directional_constraints(symbols),
        )
    except Exception as exc:
        _append_run_event(
            run_events_path,
            event_type="dataset_error",
            run_id=run_id,
            output_dir=output_dir,
            payload={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return {
            "run_plan": plan,
            "run_id": run_id,
            "run_events_path": str(run_events_path),
            "symbols": symbols,
            "forecast_horizons": list(horizons),
            "dry_train_steps": args.dry_train_steps,
            "cache_only": bool(args.cache_only),
            "plan_error": None,
            "dataset_error": str(exc),
            "results": [],
            "winner_backend": None,
            "recommended_backend": None,
        }
    _append_run_event(
        run_events_path,
        event_type="dataset_ready",
        run_id=run_id,
        output_dir=output_dir,
        payload={"symbol_count": len(symbols)},
    )

    results: list[BackendResult] = []
    for backend in backends:
        for seed in seeds:
            run_dir = output_dir / backend / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            train_cfg = _make_training_config(args, backend, run_dir, seed=seed)
            started = time.perf_counter()
            _append_run_event(
                run_events_path,
                event_type="backend_seed_start",
                run_id=run_id,
                output_dir=output_dir,
                payload={"backend": backend, "seed": int(seed), "run_dir": str(run_dir)},
            )
            try:
                trainer = build_trainer(train_cfg, data_module)
                artifacts = trainer.train()
                result = summarize_backend_result(backend, seed, artifacts, time.perf_counter() - started)
                results.append(result)
                _append_run_event(
                    run_events_path,
                    event_type="backend_seed_complete",
                    run_id=run_id,
                    output_dir=output_dir,
                    payload=result,
                )
            except Exception as exc:
                result: FailedBackendResult = {
                    "backend": backend,
                    "seed": int(seed),
                    "status": "error",
                    "duration_sec": round(time.perf_counter() - started, 3),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                results.append(result)
                _append_run_event(
                    run_events_path,
                    event_type="backend_seed_error",
                    run_id=run_id,
                    output_dir=output_dir,
                    payload={
                        "backend": backend,
                        "seed": int(seed),
                        "run_dir": str(run_dir),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": result["traceback"],
                        "duration_sec": result["duration_sec"],
                    },
                )

    backend_summary = summarize_backend_groups(results)
    winner_backend: TrainerBackend | None = None
    successful = [item for item in backend_summary if item.get("best_val_score_mean") is not None]
    if successful:
        winner_backend = max(successful, key=lambda item: float(item["best_val_score_mean"]))["backend"]
    recommended_backend = recommend_backend(backend_summary)

    report: CompareReport = {
        "run_plan": plan,
        "run_id": run_id,
        "run_events_path": str(run_events_path),
        "symbols": symbols,
        "forecast_horizons": list(horizons),
        "dry_train_steps": args.dry_train_steps,
        "cache_only": bool(args.cache_only),
        "plan_error": None,
        "results": results,
        "backend_summary": backend_summary,
        "winner_backend": winner_backend,
        "recommended_backend": recommended_backend,
    }
    _append_run_event(
        run_events_path,
        event_type="run_complete",
        run_id=run_id,
        output_dir=output_dir,
        payload={
            "winner_backend": winner_backend,
            "recommended_backend": recommended_backend,
            "result_count": len(results),
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 6),
        },
    )
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        plan = build_run_plan(args)
    except Exception as exc:
        _ensure_output_dir(output_dir)
        run_started_at = datetime.now(timezone.utc)
        run_id = f"{os.getpid()}:{int(run_started_at.timestamp() * 1000)}"
        run_events_path = output_dir / RUN_EVENTS_FILENAME
        _append_run_event(
            run_events_path,
            event_type="run_start",
            run_id=run_id,
            output_dir=output_dir,
            payload={
                "symbols": [],
                "forecast_horizons": [],
                "backends": [],
                "seeds": [],
                "cache_only": bool(getattr(args, "cache_only", True)),
                "total_runs": 0,
            },
        )
        _append_run_event(
            run_events_path,
            event_type="plan_error",
            run_id=run_id,
            output_dir=output_dir,
            payload={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        report: CompareReport = {
            "run_plan": _build_failed_run_plan(args, output_dir=output_dir, plan_error=str(exc)),
            "run_id": run_id,
            "run_events_path": str(run_events_path),
            "symbols": [],
            "forecast_horizons": [],
            "dry_train_steps": getattr(args, "dry_train_steps", None),
            "cache_only": bool(getattr(args, "cache_only", True)),
            "plan_error": str(exc),
            "results": [],
            "backend_summary": [],
            "winner_backend": None,
            "recommended_backend": None,
        }
        _attach_effective_args_artifacts(report, args, output_dir=output_dir)
        _write_compare_report_artifacts(report, output_dir=output_dir)
        print(f"Plan error: {exc}")
        raise SystemExit(1)
    if bool(getattr(args, "describe_run", False)):
        print(json.dumps(plan, indent=2))
        return
    print(render_classic_run_plan_summary(plan, title="Backend Compare Plan"))
    report = run_backend_comparison(args)
    _attach_effective_args_artifacts(report, args, output_dir=output_dir)
    report.setdefault("run_events_path", str(output_dir / RUN_EVENTS_FILENAME))
    json_path, md_path = _write_compare_report_artifacts(report, output_dir=output_dir)
    print(json.dumps(report["run_plan"], indent=2))
    effective_args_path = report.get("effective_args_path")
    effective_args_cli_path = report.get("effective_args_cli_path")
    if effective_args_path:
        print(f"Wrote {effective_args_path}")
    if effective_args_cli_path:
        print(f"Wrote {effective_args_cli_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if effective_args_cli_path:
        print(
            "Rerun with: "
            "python -m unified_hourly_experiment.compare_trainer_backends "
            f"@{effective_args_cli_path}"
        )
    if report.get("plan_error"):
        print(f"Plan error: {report['plan_error']}")
        raise SystemExit(1)
    if report.get("dataset_error"):
        print(f"Dataset error: {report['dataset_error']}")
        raise SystemExit(1)
    if report.get("winner_backend"):
        print(f"Winner: {report['winner_backend']}")
    if report.get("recommended_backend"):
        print(f"Recommended: {report['recommended_backend']}")


if __name__ == "__main__":
    main()
