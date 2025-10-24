#!/usr/bin/env python3
"""
Summarise guard-confirmed mock backtests relative to the production baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "evaltests" / "baseline_pnl_summary.json"
BACKTEST_DIR = REPO_ROOT / "evaltests" / "backtests"
OUTPUT_PATH = REPO_ROOT / "evaltests" / "guard_vs_baseline.md"


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    baseline = _load_json(BASELINE_PATH)
    trade_history = baseline.get("trade_history") if isinstance(baseline, Mapping) else {}
    trade_log = baseline.get("trade_log") if isinstance(baseline, Mapping) else {}
    realised_pnl = None
    duration_days = None
    if isinstance(trade_history, Mapping):
        realised_pnl = trade_history.get("total_realized_pnl")
    if isinstance(trade_log, Mapping):
        snapshots = trade_log.get("snapshots")
        if isinstance(snapshots, Mapping):
            duration_days = snapshots.get("duration_days")

    lines = ["# Guard vs Production Baseline", ""]
    if isinstance(realised_pnl, (int, float)) and isinstance(duration_days, (int, float)) and duration_days > 0:
        baseline_avg_daily = realised_pnl / duration_days
        lines.append(f"- Production realised PnL: {realised_pnl:,.2f} over {duration_days:.2f} days.")
        lines.append(f"- Baseline average daily PnL: {baseline_avg_daily:,.2f}.")
    else:
        lines.append("- Production baseline metrics unavailable.")
    lines.append("")

    rows = []
    if BACKTEST_DIR.exists():
        for path in sorted(BACKTEST_DIR.glob("gymrl_guard_confirm_*.json")):
            data = _load_json(path)
            strategies = data.get("strategies")
            if not isinstance(strategies, Mapping):
                continue
            maxdiff = strategies.get("maxdiff")
            simple = strategies.get("simple")
            if not isinstance(maxdiff, Mapping) or not isinstance(simple, Mapping):
                continue
            rows.append(
                {
                    "symbol": data.get("symbol", path.stem.split("_")[-1].upper()),
                    "maxdiff_return": maxdiff.get("return"),
                    "maxdiff_sharpe": maxdiff.get("sharpe"),
                    "simple_return": simple.get("return"),
                }
            )

    if rows:
        lines.append("| Symbol | MaxDiff Return | MaxDiff Sharpe | Simple Return | Δ (MaxDiff - Simple) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        deltas = []
        for row in rows:
            maxdiff_ret = row["maxdiff_return"]
            simple_ret = row["simple_return"]
            delta = (
                maxdiff_ret - simple_ret
                if isinstance(maxdiff_ret, (int, float)) and isinstance(simple_ret, (int, float))
                else None
            )
            if isinstance(delta, (int, float)):
                deltas.append(delta)
            lines.append(
                f"| {row['symbol']} | "
                f"{maxdiff_ret if isinstance(maxdiff_ret, (int, float)) else 'n/a'} | "
                f"{row['maxdiff_sharpe'] if isinstance(row['maxdiff_sharpe'], (int, float)) else 'n/a'} | "
                f"{simple_ret if isinstance(simple_ret, (int, float)) else 'n/a'} | "
                f"{delta if isinstance(delta, (int, float)) else 'n/a'} |"
            )
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            lines.append(f"| **Average** |  |  |  | {avg_delta:.4f} |")
        lines.append("")
    else:
        lines.append("_No guard backtest summaries found._")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
