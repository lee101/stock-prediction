#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    value = aggregate.get("worst_slip_monthly")
    return float(value) if value is not None else None


def render_runs_table(runs: list[dict[str, Any]]) -> str:
    headers = ["run", "status", "symbols", "offsets", "train_days", "val_days", "worst_slip_monthly", "wandb_run_id"]
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
        row = {
            "run": str(run.get("run_name", Path(run["_manifest_path"]).parent.name)),
            "status": str(run.get("status", "unknown")),
            "symbols": str(len(symbols)),
            "offsets": ",".join(str(value) for value in offsets) if offsets else "—",
            "train_days": train_days,
            "val_days": str((window.get("val_days") if window.get("val_days") is not None else "—")),
            "worst_slip_monthly": (
                f"{100.0 * _worst_slip_monthly(run):+.2f}%"
                if _worst_slip_monthly(run) is not None
                else "—"
            ),
            "wandb_run_id": str(((run.get("wandb") or {}).get("run_id")) or "—"),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: float(row["worst_slip_monthly"].rstrip("%")) if row["worst_slip_monthly"] not in {"—"} else float("-inf"),
        reverse=True,
    )
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) if rows else len(header) for header in headers}
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
