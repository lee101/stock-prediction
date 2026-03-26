#!/usr/bin/env python3
"""Scout promising WandB runs and derive sweep suggestions from the top slice."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_scripts_dir = str(REPO / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import wandb_metrics_reader as _reader  # noqa: E402

_check_api_key = _reader._check_api_key
_import_wandb = _reader._import_wandb


DEFAULT_METRIC_KEYS = (
    "val/score",
    "val_score",
    "eval_score",
    "best_val_return",
    "val/return",
    "val_return",
    "eval_return",
    "val/sortino",
    "val_sortino",
    "eval_sortino",
)


def _unwrap_config_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _first_present(mapping: dict[str, Any], keys: list[str] | tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_number(value: Any) -> bool:
    return _coerce_float(value) is not None


def _contains_token(text: Any, needles: list[str]) -> bool:
    if not needles:
        return True
    haystack = str(text or "").lower()
    return any(needle.lower() in haystack for needle in needles)


def _fetch_runs(
    *,
    wandb,
    project: str,
    entity: str | None,
    group: str | None,
    last_n_runs: int,
    metric_keys: list[str],
    extra_metric_keys: list[str],
    states: list[str],
    name_contains: list[str],
    group_contains: list[str],
    exclude_name_contains: list[str],
    exclude_group_contains: list[str],
    min_runtime_sec: float,
    min_steps: int,
) -> list[dict[str, Any]]:
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    filters: dict[str, Any] = {}
    if group:
        filters["group"] = group
    runs_raw = list(api.runs(project_path, filters=filters or None, order="-created_at"))[:last_n_runs]

    rows: list[dict[str, Any]] = []
    for run in runs_raw:
        summary = dict(run.summary) if run.summary else {}
        config_raw = dict(run.config) if run.config else {}
        config = {key: _unwrap_config_value(value) for key, value in config_raw.items()}
        state = str(getattr(run, "state", None) or "")
        group = getattr(run, "group", None)
        runtime_sec = _coerce_float(summary.get("_runtime")) or 0.0
        steps = int(_coerce_float(summary.get("_step")) or _coerce_float(summary.get("global_step")) or 0)
        if states and state.lower() not in states:
            continue
        if name_contains and not _contains_token(run.name, name_contains):
            continue
        if group_contains and not _contains_token(group, group_contains):
            continue
        if exclude_name_contains and _contains_token(run.name, exclude_name_contains):
            continue
        if exclude_group_contains and _contains_token(group, exclude_group_contains):
            continue
        if runtime_sec < min_runtime_sec:
            continue
        if steps < min_steps:
            continue
        primary_metric = _coerce_float(_first_present(summary, metric_keys))
        metrics: dict[str, float] = {}
        for key in extra_metric_keys:
            value = _coerce_float(summary.get(key))
            if value is not None:
                metrics[key] = value
        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "group": group,
                "state": state,
                "created_at": getattr(run, "created_at", None),
                "url": getattr(run, "url", None),
                "runtime_sec": runtime_sec,
                "steps": steps,
                "primary_metric": primary_metric,
                "summary": summary,
                "config": config,
                "metrics": metrics,
            }
        )
    rows.sort(key=lambda row: row["primary_metric"] if row["primary_metric"] is not None else float("-inf"), reverse=True)
    return rows


def _infer_sweep_values(values: list[Any]) -> dict[str, Any]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"kind": "unknown", "values": []}
    if all(isinstance(value, bool) for value in clean):
        counts = Counter(bool(value) for value in clean)
        ordered = [item for item, _ in counts.most_common()]
        return {"kind": "bool", "values": ordered}
    if all(isinstance(value, int) and not isinstance(value, bool) for value in clean):
        ordered = sorted({int(value) for value in clean})
        return {"kind": "int", "values": ordered}
    if all(_is_number(value) for value in clean):
        floats = sorted(float(value) for value in clean)
        lo = floats[0]
        hi = floats[-1]
        mid = statistics.median(floats)
        if lo > 0 and hi / lo >= 4.0:
            candidates = sorted({round(lo, 8), round(mid, 8), round(hi, 8)})
            return {"kind": "log_float", "values": candidates}
        candidates = sorted({round(lo, 8), round(mid, 8), round(hi, 8)})
        return {"kind": "float", "values": candidates}
    counts = Counter(str(value) for value in clean)
    ordered = [item for item, _ in counts.most_common(5)]
    return {"kind": "categorical", "values": ordered}


def build_scout_report(
    *,
    runs: list[dict[str, Any]],
    project: str,
    entity: str | None,
    metric_label: str,
    top_k: int,
    param_names: list[str],
) -> dict[str, Any]:
    top_runs = [row for row in runs if row["primary_metric"] is not None][:top_k]
    param_value_map: dict[str, list[Any]] = defaultdict(list)
    for row in top_runs:
        cfg = row["config"]
        for name in param_names:
            if name in cfg:
                param_value_map[name].append(cfg[name])

    suggestions = {
        name: _infer_sweep_values(values)
        for name, values in param_value_map.items()
        if values
    }

    common_config: dict[str, Any] = {}
    for name, values in param_value_map.items():
        counts = Counter(json.dumps(value, sort_keys=True) for value in values)
        raw = counts.most_common(1)[0][0]
        common_config[name] = json.loads(raw)

    top_rows = []
    for row in top_runs:
        top_rows.append(
            {
                "name": row["name"],
                "group": row["group"],
                "state": row["state"],
                "primary_metric": row["primary_metric"],
                "runtime_sec": row["runtime_sec"],
                "steps": row["steps"],
                "sortino": _coerce_float(_first_present(row["summary"], ["val/sortino", "val_sortino", "eval_sortino"])),
                "return": _coerce_float(_first_present(row["summary"], ["val/return", "val_return", "eval_return", "best_val_return"])),
                "url": row["url"],
                "config_subset": {name: row["config"].get(name) for name in param_names if name in row["config"]},
            }
        )

    return {
        "project": f"{entity}/{project}" if entity else project,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_label": metric_label,
        "top_k": top_k,
        "top_runs": top_rows,
        "common_config": common_config,
        "sweep_suggestions": suggestions,
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"## WandB Run Scout: {report['project']}",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]
    top_runs = report["top_runs"]
    if not top_runs:
        lines.append("*No scored runs found.*")
        return "\n".join(lines)

    lines.extend(
        [
            "### Top Runs",
            "",
            "| Run | Metric | Return | Sortino | Runtime | Steps | Group | Key Config |",
            "|-----|--------|--------|---------|---------|-------|-------|------------|",
        ]
    )
    for row in top_runs:
        cfg = ", ".join(f"{k}={v}" for k, v in row["config_subset"].items()) or "—"
        metric = f"{row['primary_metric']:.4f}" if row["primary_metric"] is not None else "—"
        ret = f"{row['return']:.4f}" if row["return"] is not None else "—"
        sortino = f"{row['sortino']:.4f}" if row["sortino"] is not None else "—"
        group = row["group"] or "—"
        runtime = f"{row['runtime_sec']:.0f}s" if row["runtime_sec"] else "—"
        steps = str(row["steps"]) if row["steps"] else "—"
        lines.append(f"| {row['name']} | {metric} | {ret} | {sortino} | {runtime} | {steps} | {group} | {cfg} |")

    lines.extend(["", "### Common Winning Pattern", ""])
    if report["common_config"]:
        for key, value in report["common_config"].items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("*No common config keys found in the top slice.*")

    lines.extend(["", "### Sweep Suggestions", ""])
    if report["sweep_suggestions"]:
        for key, payload in report["sweep_suggestions"].items():
            values = ", ".join(str(value) for value in payload["values"]) or "—"
            lines.append(f"- `{key}` ({payload['kind']}): {values}")
    else:
        lines.append("*No sweep suggestions were derived.*")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scout promising WandB runs and derive sweep candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--last-n-runs", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--states", default="")
    parser.add_argument("--name-contains", default="")
    parser.add_argument("--group-contains", default="")
    parser.add_argument("--exclude-name-contains", default="")
    parser.add_argument("--exclude-group-contains", default="")
    parser.add_argument("--min-runtime-sec", type=float, default=0.0)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument(
        "--metric-keys",
        default=",".join(DEFAULT_METRIC_KEYS),
        help="Comma-separated summary keys to try in order for the ranking metric.",
    )
    parser.add_argument(
        "--params",
        default="hidden_size,learning_rate,lr,weight_decay,batch_size,optimizer,fill_slippage_bps,ent_coef,return_weight,fill_temperature,sequence_length,transformer_dim,transformer_layers,transformer_heads",
        help="Comma-separated config keys to inspect for sweep suggestions.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _check_api_key()
    wandb = _import_wandb()

    metric_keys = [token.strip() for token in args.metric_keys.split(",") if token.strip()]
    params = [token.strip() for token in args.params.split(",") if token.strip()]
    states = [token.strip().lower() for token in args.states.split(",") if token.strip()]
    name_contains = [token.strip() for token in args.name_contains.split(",") if token.strip()]
    group_contains = [token.strip() for token in args.group_contains.split(",") if token.strip()]
    exclude_name_contains = [token.strip() for token in args.exclude_name_contains.split(",") if token.strip()]
    exclude_group_contains = [token.strip() for token in args.exclude_group_contains.split(",") if token.strip()]
    runs = _fetch_runs(
        wandb=wandb,
        project=args.project,
        entity=args.entity,
        group=args.group,
        last_n_runs=args.last_n_runs,
        metric_keys=metric_keys,
        extra_metric_keys=metric_keys,
        states=states,
        name_contains=name_contains,
        group_contains=group_contains,
        exclude_name_contains=exclude_name_contains,
        exclude_group_contains=exclude_group_contains,
        min_runtime_sec=args.min_runtime_sec,
        min_steps=args.min_steps,
    )
    report = build_scout_report(
        runs=runs,
        project=args.project,
        entity=args.entity,
        metric_label=metric_keys[0],
        top_k=args.top_k,
        param_names=params,
    )

    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)

    if args.format == "json":
        print(payload)
    else:
        print(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
