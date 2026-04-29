#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_pipeline_runs(run_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(run_root.glob("*/run_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_manifest_path"] = str(manifest_path)
        runs.append(payload)
    return runs


def _worst_slip_monthly(run: dict[str, Any]) -> float | None:
    eval_payload = run.get("eval_100d") or {}
    aggregate = eval_payload.get("aggregate") or {}
    if not isinstance(aggregate, dict):
        return None
    value = aggregate.get("worst_slip_monthly")
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _promotion_gate_state(run: dict[str, Any]) -> str:
    eval_payload = run.get("eval_100d") or {}
    gate = eval_payload.get("promotion_gate")
    if not isinstance(gate, dict):
        return "unknown"
    return "pass" if gate.get("passed") is True or gate.get("pass") is True else "fail"


def _promotion_failure_count(run: dict[str, Any]) -> int | None:
    eval_payload = run.get("eval_100d") or {}
    gate = eval_payload.get("promotion_gate")
    if not isinstance(gate, dict):
        return None
    failures = gate.get("failures")
    if isinstance(failures, list):
        return len(failures)
    return None


def render_runs_table(runs: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "status",
        "promotion_gate",
        "gate_failures",
        "symbols",
        "offsets",
        "train_days",
        "val_days",
        "worst_slip_monthly",
        "wandb_run_id",
    ]
    rows: list[dict[str, str]] = []
    for run in runs:
        window = run.get("window") or {}
        offsets = run.get("offsets") or []
        symbols = run.get("symbols") or []
        train_start = window.get("train_start")
        train_end = window.get("train_end")
        if train_start and train_end:
            from pandas import Timestamp

            train_days = str((Timestamp(train_end) - Timestamp(train_start)).days + 1)
        else:
            train_days = "—"
        worst_slip_monthly = _worst_slip_monthly(run)
        promotion_gate = _promotion_gate_state(run)
        failure_count = _promotion_failure_count(run)
        row = {
            "run": str(run.get("run_name", Path(run["_manifest_path"]).parent.name)),
            "status": str(run.get("status", "unknown")),
            "promotion_gate": promotion_gate,
            "gate_failures": "—" if failure_count is None else str(failure_count),
            "symbols": str(len(symbols)),
            "offsets": ",".join(str(value) for value in offsets) if offsets else "—",
            "train_days": train_days,
            "val_days": str((window.get("val_days") if window.get("val_days") is not None else "—")),
            "worst_slip_monthly": (
                f"{100.0 * worst_slip_monthly:+.2f}%" if worst_slip_monthly is not None else "—"
            ),
            "wandb_run_id": str(((run.get("wandb") or {}).get("run_id")) or "—"),
        }
        rows.append(row)

    gate_rank = {"pass": 2, "fail": 1, "unknown": 0}
    rows.sort(
        key=lambda row: (
            gate_rank.get(row["promotion_gate"], 0),
            float(row["worst_slip_monthly"].rstrip("%"))
            if row["worst_slip_monthly"] not in {"—"}
            else float("-inf"),
        ),
        reverse=True,
    )
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows)) if rows else len(header)
        for header in headers
    }
    lines = [
        " ".join(header.ljust(widths[header]) for header in headers),
        " ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append(" ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local dashboard for augmented daily stock C-env pipeline runs.")
    parser.add_argument("--run-root", type=Path, default=Path("analysis/augmented_daily_stock_runs"))
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")
    args = parser.parse_args()

    runs = load_pipeline_runs(Path(args.run_root))
    if args.json:
        print(json.dumps(runs, indent=2, sort_keys=True))
    else:
        print(render_runs_table(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
