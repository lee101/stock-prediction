"""
Shared evaluation harness for comparing RL checkpoints across training stacks.

This scaffold standardises metadata capture and provides a plug-in system for
module-specific evaluators (hftraining, gymrl, pufferlibtraining, differentiable_market).
It currently records checkpoint stats and baseline references, and is intended to be
extended with full PnL backtests and simulation hooks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "evaltests" / "baseline_pnl_summary.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "evaltests" / "rl_benchmark_results.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalTarget:
    """Configuration for a checkpoint evaluation request."""

    name: str
    module: str
    checkpoint: Path
    config_path: Optional[Path] = None
    notes: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvalTarget":
        """Normalise a JSON payload into an EvalTarget."""
        try:
            name = str(payload["name"])
            module = str(payload["module"])
            checkpoint = Path(payload["checkpoint"])
        except KeyError as exc:  # pragma: no cover - validated via unit tests
            raise ValueError(f"Missing required target field: {exc}") from exc
        config_path = payload.get("config_path")
        notes = payload.get("notes")
        return cls(
            name=name,
            module=module,
            checkpoint=checkpoint,
            config_path=Path(config_path) if config_path else None,
            notes=str(notes) if notes is not None else None,
        )


@dataclass(slots=True)
class EvaluationResult:
    """Container for aggregated evaluation metadata."""

    target: EvalTarget
    status: str
    metrics: Mapping[str, Any]
    warnings: List[str]

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["target"] = {
            "name": self.target.name,
            "module": self.target.module,
            "checkpoint": str(self.target.checkpoint),
            "config_path": str(self.target.config_path) if self.target.config_path else None,
            "notes": self.target.notes,
        }
        return payload


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def load_baseline_summary() -> Mapping[str, Any]:
    """Load the most recent baseline summary if available."""
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE
    if BASELINE_PATH.exists():
        try:
            _BASELINE_CACHE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _BASELINE_CACHE = {"error": f"Failed to parse {BASELINE_PATH.name}: {exc}"}
    else:
        _BASELINE_CACHE = {"warning": "Baseline summary not generated yet."}
    return _BASELINE_CACHE


# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------


Evaluator = Callable[[EvalTarget], EvaluationResult]
_EVALUATORS: Dict[str, Evaluator] = {}
_BASELINE_CACHE: Mapping[str, Any] | None = None


def register_evaluator(module: str) -> Callable[[Evaluator], Evaluator]:
    """Decorator to register evaluators for a given module name."""

    def decorator(func: Evaluator) -> Evaluator:
        _EVALUATORS[module] = func
        return func

    return decorator


def _resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path)


def _checkpoint_metadata(checkpoint_path: Path) -> Mapping[str, Any]:
    if not checkpoint_path.exists():
        return {"exists": False}
    stat = checkpoint_path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _default_evaluator(target: EvalTarget) -> EvaluationResult:
    """Fallback evaluator that records checkpoint metadata only."""
    resolved = _resolve_path(target.checkpoint)
    checkpoint_path = resolved if resolved is not None else target.checkpoint
    checkpoint_path = checkpoint_path if isinstance(checkpoint_path, Path) else Path(checkpoint_path)
    metadata = _checkpoint_metadata(checkpoint_path)
    warnings: List[str] = []
    status = "missing_checkpoint" if not metadata.get("exists") else "pending"
    if status == "missing_checkpoint":
        warnings.append(f"Checkpoint not found at {checkpoint_path}")
    metrics: Dict[str, Any] = {
        "checkpoint": metadata,
        "implementation": "pending",
    }
    return EvaluationResult(target=target, status=status, metrics=metrics, warnings=warnings)


@register_evaluator("hftraining")
def _evaluate_hftraining(target: EvalTarget) -> EvaluationResult:
    checkpoint_path = _resolve_path(target.checkpoint)
    result = _default_evaluator(target)
    metrics = dict(result.metrics)
    warnings = list(result.warnings)

    base_dir = None
    config_path = _resolve_path(target.config_path)
    if config_path and config_path.exists():
        base_dir = config_path.parent
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse hftraining config {config_path}: {exc}")
            config_payload = {}
    else:
        config_payload = {}
        if config_path:
            warnings.append(f"Config path missing: {config_path}")

    if base_dir is None and checkpoint_path:
        base_dir = checkpoint_path.parent

    training_metrics = {}
    status = result.status
    if base_dir:
        metrics_path = base_dir / "training_metrics.json"
        if metrics_path.exists():
            try:
                raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                warnings.append(f"Failed to parse training metrics {metrics_path}: {exc}")
                raw_metrics = []
            if isinstance(raw_metrics, list) and raw_metrics:
                final_eval = next((item for item in reversed(raw_metrics) if item.get("phase") == "eval"), None)
                final_train = next((item for item in reversed(raw_metrics) if item.get("phase") == "train"), None)
                eval_items = [item for item in raw_metrics if item.get("phase") == "eval"]
                best_eval = min(
                    eval_items,
                    key=lambda item: item.get("loss", float("inf")),
                ) if eval_items else None
                training_metrics = {
                    "steps_logged": len(raw_metrics),
                    "final_eval_loss": final_eval.get("loss") if final_eval else None,
                    "final_train_loss": final_train.get("loss") if final_train else None,
                    "final_eval_return": final_eval.get("avg_return") if final_eval else None,
                    "best_eval_loss": best_eval.get("loss") if best_eval else None,
                    "best_eval_step": best_eval.get("step") if best_eval else None,
                }
                status = "evaluated"
            else:
                warnings.append(f"No metrics entries found in {metrics_path}")
        else:
            warnings.append(f"training_metrics.json not found in {base_dir}")
    else:
        warnings.append("Unable to resolve hftraining run directory for metrics analysis.")

    config_summary: Dict[str, Any] = {}
    if isinstance(config_payload, Mapping):
        training_section: Mapping[str, Any] = config_payload
        if "training" in config_payload and isinstance(config_payload["training"], Mapping):
            training_section = config_payload["training"]  # type: ignore[assignment]
        for key in ("max_steps", "learning_rate", "batch_size", "gradient_accumulation_steps", "scheduler"):
            if key in training_section:
                config_summary[key] = training_section[key]
        if "optimizer" in config_payload and isinstance(config_payload["optimizer"], Mapping):
            optimizer_section = config_payload["optimizer"]
            for key in ("name", "weight_decay", "beta1", "beta2"):
                if key in optimizer_section:
                    config_summary[f"optimizer_{key}"] = optimizer_section[key]

    metrics.update(
        {
            "implementation": "hftraining_eval_v0",
            "config": config_summary,
            "training_metrics": training_metrics,
        }
    )
    return EvaluationResult(target=target, status=status, metrics=metrics, warnings=warnings)


@register_evaluator("gymrl")
def _evaluate_gymrl(target: EvalTarget) -> EvaluationResult:
    base_result = _default_evaluator(target)
    metrics = dict(base_result.metrics)
    warnings = list(base_result.warnings)
    status = base_result.status

    metadata_path = _resolve_path(target.config_path)
    metadata: Mapping[str, Any] | None = None
    base_dir: Optional[Path] = None

    if metadata_path and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse GymRL metadata {metadata_path}: {exc}")
        else:
            base_dir = metadata_path.parent
    elif metadata_path:
        warnings.append(f"GymRL metadata path missing: {metadata_path}")

    if metadata is None:
        checkpoint_path = _resolve_path(target.checkpoint)
        if checkpoint_path:
            candidate = checkpoint_path.parent / "training_metadata.json"
            if candidate.exists():
                try:
                    metadata = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    warnings.append(f"Failed to parse GymRL metadata {candidate}: {exc}")
                else:
                    base_dir = candidate.parent
            else:
                warnings.append(f"training_metadata.json not found alongside {checkpoint_path.name}")

    gym_metrics: Dict[str, Any] = {}
    config_summary: Dict[str, Any] = {}
    topk_summary: List[Mapping[str, Any]] = []

    if isinstance(metadata, Mapping):
        status = "evaluated"
        args_section = metadata.get("args", {})
        if isinstance(args_section, Mapping):
            for key in (
                "num_timesteps",
                "learning_rate",
                "batch_size",
                "n_steps",
                "seed",
                "turnover_penalty",
                "weight_cap",
                "allow_short",
                "leverage_cap",
            ):
                if key in args_section:
                    config_summary[key] = args_section[key]

        env_config = metadata.get("env_config", {})
        validation_metrics = metadata.get("validation_metrics", {})
        gym_metrics.update(
            {
                "train_steps": metadata.get("train_steps"),
                "validation_steps": metadata.get("validation_steps"),
                "total_steps": metadata.get("total_steps"),
                "num_assets": metadata.get("num_assets"),
                "num_features": metadata.get("num_features"),
                "forecast_backend_used": metadata.get("forecast_backend_used"),
                "validation_metrics": validation_metrics,
                "env_config": env_config,
            }
        )

        topk = metadata.get("topk_checkpoints", [])
        if isinstance(topk, list):
            for item in topk:
                if isinstance(item, Mapping):
                    topk_summary.append(
                        {
                            "reward": item.get("reward"),
                            "path": item.get("path"),
                        }
                    )
        feature_meta = metadata.get("feature_extra_metadata", {})
        if isinstance(feature_meta, Mapping):
            gym_metrics["feature_backend"] = feature_meta.get("backend_name")
            gym_metrics["feature_errors"] = feature_meta.get("backend_errors")

        forecast_errors = metadata.get("forecast_backend_errors")
        if forecast_errors:
            gym_metrics["forecast_backend_errors"] = forecast_errors

    metrics.update(
        {
            "implementation": "gymrl_eval_v0",
            "config": config_summary,
            "gymrl_metrics": gym_metrics,
            "topk_checkpoints": topk_summary,
        }
    )

    return EvaluationResult(target=target, status=status, metrics=metrics, warnings=warnings)


@register_evaluator("pufferlibtraining")
def _evaluate_pufferlib(target: EvalTarget) -> EvaluationResult:
    base_result = _default_evaluator(target)
    metrics = dict(base_result.metrics)
    warnings = list(base_result.warnings)
    status = base_result.status

    summary_path = _resolve_path(target.config_path)
    summary_data: Mapping[str, Any] | None = None
    if summary_path and summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse PufferLib pipeline summary {summary_path}: {exc}")
    elif summary_path:
        warnings.append(f"Pipeline summary not found: {summary_path}")

    pipeline_info: Dict[str, Any] = {}
    aggregate_info: Dict[str, Any] = {}

    if isinstance(summary_data, Mapping):
        status = "evaluated"
        base_checkpoint = summary_data.get("base_checkpoint")
        specialists = summary_data.get("specialists", {})
        portfolio_pairs = summary_data.get("portfolio_pairs", {})
        pipeline_info["base_checkpoint"] = base_checkpoint
        if isinstance(specialists, Mapping):
            pipeline_info["specialists"] = list(specialists.keys())
        pair_summaries: Dict[str, Any] = {}
        if isinstance(portfolio_pairs, Mapping):
            for pair, payload in portfolio_pairs.items():
                if not isinstance(payload, Mapping):
                    continue
                best_epoch = payload.get("best_epoch")
                pair_summary: Dict[str, Any] = {
                    "best_checkpoint": payload.get("best_checkpoint"),
                    "best_val_profit": payload.get("best_val_profit"),
                    "best_epoch": best_epoch,
                }
                if isinstance(best_epoch, int):
                    profit_key = f"val/profit_epoch_{best_epoch}"
                    sharpe_key = f"val/sharpe_epoch_{best_epoch}"
                    cvar_key = f"val/cvar_epoch_{best_epoch}"
                    pair_summary["best_epoch_profit"] = payload.get(profit_key)
                    pair_summary["best_epoch_sharpe"] = payload.get(sharpe_key)
                    pair_summary["best_epoch_cvar"] = payload.get(cvar_key)
                pair_summaries[str(pair)] = pair_summary
        if pair_summaries:
            pipeline_info["portfolio_pairs"] = pair_summaries

        # Attempt to read aggregate metrics CSV located alongside the summary.
        if summary_path:
            aggregate_path = summary_path.parent / "aggregate_pufferlib_metrics.csv"
            if aggregate_path.exists():
                try:
                    import csv

                    by_pair: Dict[str, Dict[str, float | str]] = {}
                    with aggregate_path.open("r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            pair = row.get("pair")
                            if not pair:
                                continue
                            try:
                                aggregate_entry = {
                                    "run": row.get("run"),
                                    "days": int(float(row["days"])) if row.get("days") else None,
                                    "avg_daily_return": float(row["avg_daily_return"]) if row.get("avg_daily_return") else None,
                                    "annualized_return": float(row["annualized_return"]) if row.get("annualized_return") else None,
                                    "cumulative_return": float(row["cumulative_return"]) if row.get("cumulative_return") else None,
                                }
                            except (ValueError, TypeError):
                                continue
                            by_pair[pair] = aggregate_entry
                    if by_pair:
                        aggregate_info = by_pair
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Failed to parse aggregate metrics {aggregate_path}: {exc}")

    metrics.update(
        {
            "implementation": "pufferlib_eval_v0",
            "pipeline": pipeline_info,
            "aggregate_pair_metrics": aggregate_info,
        }
    )

    return EvaluationResult(target=target, status=status, metrics=metrics, warnings=warnings)


@register_evaluator("differentiable_market")
def _evaluate_diff_market(target: EvalTarget) -> EvaluationResult:
    base_result = _default_evaluator(target)
    metrics = dict(base_result.metrics)
    warnings = list(base_result.warnings)
    status = base_result.status

    config_path = _resolve_path(target.config_path)
    checkpoint_path = _resolve_path(target.checkpoint)

    run_dir: Optional[Path] = None
    config_data: Mapping[str, Any] | None = None

    if config_path and config_path.exists():
        run_dir = config_path.parent
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse differentiable market config {config_path}: {exc}")
    elif config_path:
        warnings.append(f"Differentiable market config missing: {config_path}")

    if run_dir is None and checkpoint_path:
        run_dir = checkpoint_path.parent.parent
        candidate_config = run_dir / "config.json"
        if candidate_config.exists():
            try:
                config_data = json.loads(candidate_config.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                warnings.append(f"Failed to parse differentiable market config {candidate_config}: {exc}")

    config_summary: Dict[str, Any] = {}
    training_summary: Dict[str, Any] = {}
    eval_summary: Dict[str, Any] = {}
    topk_summary: List[Mapping[str, Any]] = []
    report_summary: Mapping[str, Any] | None = None

    if isinstance(config_data, Mapping):
        status = "evaluated"
        train_cfg = config_data.get("train", {})
        env_cfg = config_data.get("env", {})
        eval_cfg = config_data.get("eval", {})

        if isinstance(train_cfg, Mapping):
            for key in (
                "epochs",
                "batch_windows",
                "microbatch_windows",
                "rollout_groups",
                "lookback",
                "lr_muon",
                "lr_adamw",
                "entropy_coef",
                "kl_coef",
                "use_muon",
                "use_compile",
                "gradient_checkpointing",
            ):
                if key in train_cfg:
                    config_summary[key] = train_cfg[key]
        if isinstance(env_cfg, Mapping):
            env_summary = {k: env_cfg.get(k) for k in ("transaction_cost", "risk_aversion", "drawdown_lambda")}
            config_summary["env"] = env_summary
        if isinstance(eval_cfg, Mapping):
            config_summary["eval"] = {
                "window_length": eval_cfg.get("window_length"),
                "stride": eval_cfg.get("stride"),
                "metric": eval_cfg.get("metric"),
            }

    if run_dir:
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            final_eval: Optional[Mapping[str, Any]] = None
            best_eval_by_sharpe: Optional[Mapping[str, Any]] = None
            best_eval_by_objective: Optional[Mapping[str, Any]] = None
            try:
                with metrics_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if entry.get("phase") == "eval":
                            final_eval = entry
                            if entry.get("eval_sharpe") is not None:
                                if (
                                    best_eval_by_sharpe is None
                                    or entry.get("eval_sharpe", float("-inf")) > best_eval_by_sharpe.get("eval_sharpe", float("-inf"))
                                ):
                                    best_eval_by_sharpe = entry
                            if entry.get("eval_objective") is not None:
                                if (
                                    best_eval_by_objective is None
                                    or entry.get("eval_objective", float("inf")) < best_eval_by_objective.get("eval_objective", float("inf"))
                                ):
                                    best_eval_by_objective = entry
                training_summary["metrics_logged"] = True
            except json.JSONDecodeError as exc:
                warnings.append(f"Failed to parse metrics from {metrics_path}: {exc}")
            else:
                if final_eval:
                    eval_summary["final"] = {
                        "step": final_eval.get("step"),
                        "objective": final_eval.get("eval_objective"),
                        "sharpe": final_eval.get("eval_sharpe"),
                        "turnover": final_eval.get("eval_turnover"),
                        "total_return": final_eval.get("eval_total_return"),
                        "annual_return": final_eval.get("eval_annual_return"),
                        "max_drawdown": final_eval.get("eval_max_drawdown"),
                    }
                if best_eval_by_sharpe and best_eval_by_sharpe is not final_eval:
                    eval_summary["best_sharpe"] = {
                        "step": best_eval_by_sharpe.get("step"),
                        "sharpe": best_eval_by_sharpe.get("eval_sharpe"),
                        "objective": best_eval_by_sharpe.get("eval_objective"),
                        "total_return": best_eval_by_sharpe.get("eval_total_return"),
                    }
                if best_eval_by_objective and best_eval_by_objective is not final_eval:
                    eval_summary["best_objective"] = {
                        "step": best_eval_by_objective.get("step"),
                        "objective": best_eval_by_objective.get("eval_objective"),
                        "sharpe": best_eval_by_objective.get("eval_sharpe"),
                        "total_return": best_eval_by_objective.get("eval_total_return"),
                    }

        topk_path = run_dir / "topk_checkpoints.json"
        if topk_path.exists():
            try:
                topk_data = json.loads(topk_path.read_text(encoding="utf-8"))
                if isinstance(topk_data, list):
                    for item in topk_data:
                        if isinstance(item, Mapping):
                            topk_summary.append(
                                {
                                    "rank": item.get("rank"),
                                    "step": item.get("step"),
                                    "loss": item.get("loss"),
                                    "path": item.get("path"),
                                }
                            )
            except json.JSONDecodeError as exc:
                warnings.append(f"Failed to parse top-k checkpoints {topk_path}: {exc}")

    if isinstance(config_data, Mapping):
        eval_cfg = config_data.get("eval", {})
        report_dir = None
        if isinstance(eval_cfg, Mapping):
            report_dir = eval_cfg.get("report_dir")
        if report_dir:
            report_path = _resolve_path(Path(report_dir) / "report.json")
            if report_path and report_path.exists():
                try:
                    report_summary = json.loads(report_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    warnings.append(f"Failed to parse evaluation report {report_path}: {exc}")

    metrics.update(
        {
            "implementation": "diff_market_eval_v0",
            "config": config_summary,
            "training": training_summary,
            "eval_metrics": eval_summary,
            "topk_checkpoints": topk_summary,
            "report_summary": report_summary,
        }
    )

    return EvaluationResult(target=target, status=status, metrics=metrics, warnings=warnings)


def evaluate_target(target: EvalTarget) -> EvaluationResult:
    evaluator = _EVALUATORS.get(target.module, _default_evaluator)
    return evaluator(target)


def run_evaluations(targets: Iterable[EvalTarget]) -> Dict[str, Any]:
    """Execute evaluations and return a serialisable payload."""
    evaluations: List[EvaluationResult] = []
    for target in targets:
        evaluations.append(evaluate_target(target))

    baseline = load_baseline_summary()
    baseline_trade_history = baseline.get("trade_history") if isinstance(baseline, Mapping) else {}
    baseline_realized_pnl = (
        baseline_trade_history.get("total_realized_pnl") if isinstance(baseline_trade_history, Mapping) else None
    )
    baseline_deepseek = baseline.get("deepseek") if isinstance(baseline, Mapping) else {}
    deepseek_reference: Dict[str, Any] = {}
    if isinstance(baseline_deepseek, Mapping):
        for name, payload in baseline_deepseek.items():
            if isinstance(payload, Mapping):
                net = payload.get("net_pnl")
                realized = payload.get("realized_pnl")
                if net is not None or realized is not None:
                    deepseek_reference[name] = {
                        "net_pnl": net,
                        "realized_pnl": realized,
                        "fees": payload.get("fees"),
                    }

    for result in evaluations:
        comparisons: Dict[str, Any] = {}
        if baseline_realized_pnl is not None:
            comparisons["baseline_total_realized_pnl"] = baseline_realized_pnl
        if deepseek_reference:
            comparisons["deepseek_reference"] = deepseek_reference

        if result.target.module == "gymrl":
            gym_metrics = result.metrics.get("gymrl_metrics", {})
            validation = gym_metrics.get("validation_metrics") if isinstance(gym_metrics, Mapping) else {}
            cumulative_return = validation.get("cumulative_return") if isinstance(validation, Mapping) else None
            average_daily_return = validation.get("average_net_return_non_crypto") if isinstance(validation, Mapping) else None
            if cumulative_return is not None:
                comparisons["gymrl_cumulative_return"] = cumulative_return
            if average_daily_return is not None:
                comparisons["gymrl_average_daily_return"] = average_daily_return

        if result.target.module == "differentiable_market":
            eval_metrics = result.metrics.get("eval_metrics", {})
            final_eval = eval_metrics.get("final") if isinstance(eval_metrics, Mapping) else {}
            total_return = final_eval.get("total_return")
            if total_return is not None:
                comparisons["diff_market_total_return"] = total_return

        if result.target.module == "pufferlibtraining":
            aggregate_pairs = result.metrics.get("aggregate_pair_metrics", {})
            if isinstance(aggregate_pairs, Mapping):
                comparisons["pufferlib_pair_cumulative_returns"] = {
                    pair: stats.get("cumulative_return")
                    for pair, stats in aggregate_pairs.items()
                    if isinstance(stats, Mapping) and stats.get("cumulative_return") is not None
                }

        if comparisons:
            result.metrics["comparisons"] = comparisons

    scoreboard: List[Dict[str, Any]] = []
    baseline_per_day = None
    baseline_duration_days = None
    if isinstance(baseline_trade_history, Mapping):
        curve = baseline_trade_history.get("cumulative_curve")
        if isinstance(curve, list) and len(curve) >= 2:
            try:
                start = datetime.fromisoformat(curve[0][0])
                end = datetime.fromisoformat(curve[-1][0])
                duration_seconds = (end - start).total_seconds()
                if duration_seconds > 0:
                    baseline_duration_days = duration_seconds / 86400.0
                    if baseline_realized_pnl is not None:
                        baseline_per_day = baseline_realized_pnl / baseline_duration_days
            except (ValueError, TypeError):
                baseline_duration_days = None

    def _add_score_entry(
        name: str,
        module: str,
        score: Optional[float],
        details: Mapping[str, Any],
        *,
        per_day: Optional[float] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "name": name,
            "module": module,
            "score": score,
            "details": dict(details),
        }
        if per_day is not None:
            entry["score_per_day"] = per_day
            if baseline_per_day not in (None, 0):
                entry["relative_to_baseline"] = per_day / baseline_per_day
        scoreboard.append(entry)

    for result in evaluations:
        module = result.target.module
        metrics_map = result.metrics
        score: Optional[float] = None
        details: Dict[str, Any] = {}
        per_day_score: Optional[float] = None

        if module == "gymrl":
            gym_metrics = metrics_map.get("gymrl_metrics", {})
            if isinstance(gym_metrics, Mapping):
                validation = gym_metrics.get("validation_metrics")
                if isinstance(validation, Mapping):
                    score = validation.get("cumulative_return")
                    details = {
                        "cumulative_return": validation.get("cumulative_return"),
                        "average_daily_return": validation.get("average_net_return_non_crypto"),
                        "sharpe": validation.get("average_log_reward"),
                        "turnover": validation.get("average_turnover"),
                    }
                    per_day_score = validation.get("average_net_return_non_crypto")

        elif module == "differentiable_market":
            eval_metrics = metrics_map.get("eval_metrics", {})
            if isinstance(eval_metrics, Mapping):
                final_eval = eval_metrics.get("final")
                if isinstance(final_eval, Mapping):
                    score = final_eval.get("total_return")
                    details = {
                        "total_return": final_eval.get("total_return"),
                        "annual_return": final_eval.get("annual_return"),
                        "sharpe": final_eval.get("sharpe"),
                        "turnover": final_eval.get("turnover"),
                        "periods_per_year": final_eval.get("eval_periods_per_year"),
                    }
                    periods_per_year = final_eval.get("eval_periods_per_year")
                    if isinstance(periods_per_year, (int, float)) and periods_per_year > 0:
                        per_day_score = final_eval.get("total_return", 0.0) / periods_per_year * 252
                    else:
                        per_day_score = final_eval.get("total_return")
            report_summary = metrics_map.get("report_summary")
            if isinstance(report_summary, Mapping):
                score = report_summary.get("cumulative_return_mean", score)
                per_day_score = report_summary.get("cumulative_return_mean", per_day_score)
                details = {
                    **details,
                    "report_cumulative_return": report_summary.get("cumulative_return_mean"),
                    "report_sharpe": report_summary.get("sharpe_mean"),
                    "report_objective": report_summary.get("objective_mean"),
                }

        elif module == "pufferlibtraining":
            aggregate_pairs = metrics_map.get("aggregate_pair_metrics", {})
            if isinstance(aggregate_pairs, Mapping) and aggregate_pairs:
                best_pair = max(
                    aggregate_pairs.items(),
                    key=lambda item: item[1].get("cumulative_return", float("-inf")) if isinstance(item[1], Mapping) else float("-inf"),
                )
                pair_name, pair_stats = best_pair
                if isinstance(pair_stats, Mapping):
                    score = pair_stats.get("cumulative_return")
                    details = {
                        "best_pair": pair_name,
                        "cumulative_return": pair_stats.get("cumulative_return"),
                        "annualized_return": pair_stats.get("annualized_return"),
                        "avg_daily_return": pair_stats.get("avg_daily_return"),
                        "run": pair_stats.get("run"),
                    }
                    per_day_score = pair_stats.get("avg_daily_return")

        elif module == "hftraining":
            training_metrics = metrics_map.get("training_metrics", {})
            if isinstance(training_metrics, Mapping):
                score = training_metrics.get("final_eval_return")
                details = {
                    "final_eval_return": training_metrics.get("final_eval_return"),
                    "final_eval_loss": training_metrics.get("final_eval_loss"),
                    "best_eval_loss": training_metrics.get("best_eval_loss"),
                }
                per_day_score = training_metrics.get("final_eval_return")

        if score is not None or details:
            _add_score_entry(result.target.name, module, score, details, per_day=per_day_score)

    # Add DeepSeek benchmark entries to scoreboard.
    for name, payload in deepseek_reference.items():
        if isinstance(payload, Mapping):
            score = payload.get("net_pnl")
            per_day_score = None
            if baseline_duration_days and baseline_duration_days > 0 and score is not None:
                per_day_score = score / baseline_duration_days
            _add_score_entry(
                f"deepseek_{name}",
                "deepseek",
                score,
                {
                    "net_pnl": payload.get("net_pnl"),
                    "realized_pnl": payload.get("realized_pnl"),
                    "fees": payload.get("fees"),
                },
                per_day=per_day_score,
            )

    if baseline_realized_pnl is not None:
        per_day = baseline_per_day
        _add_score_entry(
            "baseline_production",
            "baseline",
            baseline_realized_pnl,
            {"total_realized_pnl": baseline_realized_pnl},
            per_day=per_day,
        )

    scoreboard_sorted = sorted(
        scoreboard,
        key=lambda item: (item.get("score") is None, -(item.get("score") or float("-inf"))),
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "results": [item.to_payload() for item in evaluations],
        "scoreboard": scoreboard_sorted,
    }
    return payload


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_targets_from_config(config_path: Path) -> List[EvalTarget]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw_targets = raw.get("targets", [])
    elif isinstance(raw, list):
        raw_targets = raw
    else:
        raise ValueError("Config must be a list or dict with 'targets'.")
    return [EvalTarget.from_mapping(item) for item in raw_targets]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="RL benchmark evaluation harness.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON file describing evaluation targets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the combined evaluation report (default: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args(argv)

    targets = _load_targets_from_config(args.config)
    payload = run_evaluations(targets)

    output_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation summary written to {output_path}")
    render_script = REPO_ROOT / "evaltests" / "render_scoreboard.py"
    if render_script.exists():
        try:
            subprocess.run([sys.executable, str(render_script)], check=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to render scoreboard: {exc}")


if __name__ == "__main__":
    main()
