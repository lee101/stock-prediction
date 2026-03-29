#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPTS_DIR = str(REPO / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from wandb_metrics_reader import _check_api_key, _import_wandb


@dataclass(frozen=True)
class MetricStability:
    count: int
    start: float | None
    end: float | None
    best: float | None
    worst: float | None
    total_change: float
    relative_improvement: float
    slope_per_step: float
    diff_std: float
    direction_match_rate: float
    finish_gap_to_best: float
    smoothness_ratio: float
    finish_gap_ratio: float
    stability_score: float


def compute_metric_stability(values: list[float], *, goal: str) -> MetricStability:
    if goal not in {"min", "max"}:
        raise ValueError("goal must be 'min' or 'max'")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return MetricStability(0, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if arr.size == 1:
        value = float(arr[0])
        return MetricStability(1, value, value, value, value, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    diffs = np.diff(arr)
    start = float(arr[0])
    end = float(arr[-1])
    best = float(np.min(arr)) if goal == "min" else float(np.max(arr))
    worst = float(np.max(arr)) if goal == "min" else float(np.min(arr))
    total_change = end - start
    if goal == "min":
        relative_improvement = (start - end) / max(abs(start), 1e-12)
        direction_match_rate = float(np.mean(diffs <= 0.0))
        finish_gap_to_best = max(0.0, end - best)
    else:
        relative_improvement = (end - start) / max(abs(start), 1e-12)
        direction_match_rate = float(np.mean(diffs >= 0.0))
        finish_gap_to_best = max(0.0, best - end)

    value_range = max(float(np.max(arr) - np.min(arr)), 1e-12)
    diff_std = float(np.std(diffs))
    smoothness_ratio = float(diff_std / value_range)
    finish_gap_ratio = float(finish_gap_to_best / value_range)
    stability_score = float(
        direction_match_rate
        + 0.5 * max(relative_improvement, 0.0)
        - smoothness_ratio
        - finish_gap_ratio
    )
    return MetricStability(
        count=int(arr.size),
        start=start,
        end=end,
        best=best,
        worst=worst,
        total_change=float(total_change),
        relative_improvement=float(relative_improvement),
        slope_per_step=float(total_change / max(arr.size - 1, 1)),
        diff_std=diff_std,
        direction_match_rate=direction_match_rate,
        finish_gap_to_best=float(finish_gap_to_best),
        smoothness_ratio=smoothness_ratio,
        finish_gap_ratio=finish_gap_ratio,
        stability_score=stability_score,
    )


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fetch_run_stability_rows(
    *,
    wandb,
    project: str,
    entity: str | None,
    metric_key: str,
    goal: str,
    run_ids: list[str],
    group: str | None,
    last_n_runs: int,
    history_samples: int,
) -> list[dict[str, Any]]:
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    if run_ids:
        runs = []
        for run_id in run_ids:
            try:
                runs.append(api.run(f"{project_path}/{run_id}"))
            except Exception as exc:
                print(f"Warning: failed to fetch run {project_path}/{run_id}: {exc}", file=sys.stderr)
    else:
        filters = {"group": group} if group else None
        runs = list(api.runs(project_path, filters=filters, order="-created_at"))[:last_n_runs]

    rows: list[dict[str, Any]] = []
    for run in runs:
        history_rows = list(run.history(keys=[metric_key], samples=history_samples))
        values = [_coerce_float(row.get(metric_key)) for row in history_rows]
        clean_values = [float(value) for value in values if value is not None]
        stability = compute_metric_stability(clean_values, goal=goal)
        summary = dict(run.summary) if run.summary else {}
        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "group": getattr(run, "group", None),
                "state": getattr(run, "state", None),
                "url": getattr(run, "url", None),
                "metric_key": metric_key,
                "latest_metric": _coerce_float(summary.get(metric_key)),
                "history_count": len(clean_values),
                "stability": asdict(stability),
            }
        )
    rows.sort(key=lambda row: row["stability"]["stability_score"], reverse=True)
    return rows


def format_markdown(rows: list[dict[str, Any]], *, project: str, entity: str | None, metric_key: str, goal: str) -> str:
    project_path = f"{entity}/{project}" if entity else project
    lines = [
        f"## WandB Stability Report: {project_path}",
        f"*Metric:* `{metric_key}` ({goal})",
        "",
    ]
    if not rows:
        lines.append("*No runs found.*")
        return "\n".join(lines)

    lines.extend(
        [
            "| Run | Stability | Direction | Smoothness | Finish Gap | Count |",
            "|-----|-----------|-----------|------------|------------|-------|",
        ]
    )
    for row in rows:
        stability = row["stability"]
        lines.append(
            "| {name} | {score:.3f} | {direction:.2%} | {smooth:.3f} | {gap:.3f} | {count} |".format(
                name=row["name"],
                score=stability["stability_score"],
                direction=stability["direction_match_rate"],
                smooth=stability["smoothness_ratio"],
                gap=stability["finish_gap_ratio"],
                count=stability["count"],
            )
        )

    best = rows[0]
    best_stability = best["stability"]
    lines.extend(
        [
            "",
            f"### Most Stable Run: {best['name']}",
            "",
            f"- `stability_score`: `{best_stability['stability_score']:.3f}`",
            f"- `direction_match_rate`: `{best_stability['direction_match_rate']:.2%}`",
            f"- `relative_improvement`: `{best_stability['relative_improvement']:.2%}`",
            f"- `smoothness_ratio`: `{best_stability['smoothness_ratio']:.3f}`",
            f"- `finish_gap_ratio`: `{best_stability['finish_gap_ratio']:.3f}`",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize W&B metric stability for recent runs")
    parser.add_argument("--project", default="stock")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--metric-key", required=True)
    parser.add_argument("--goal", choices=["min", "max"], required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--group", default=None)
    parser.add_argument("--last-n-runs", type=int, default=10)
    parser.add_argument("--history-samples", type=int, default=500)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _check_api_key()
    wandb = _import_wandb()
    rows = fetch_run_stability_rows(
        wandb=wandb,
        project=args.project,
        entity=args.entity,
        metric_key=args.metric_key,
        goal=args.goal,
        run_ids=list(args.run_id),
        group=args.group,
        last_n_runs=args.last_n_runs,
        history_samples=args.history_samples,
    )
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return
    print(format_markdown(rows, project=args.project, entity=args.entity, metric_key=args.metric_key, goal=args.goal))


if __name__ == "__main__":
    main()
