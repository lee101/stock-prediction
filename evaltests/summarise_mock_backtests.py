#!/usr/bin/env python3
"""
Aggregate guard-confirm mock backtest outputs.

Reads JSON summaries in ``evaltests/backtests/gymrl_guard_confirm_*.json`` and
emits a Markdown table highlighting MaxDiff performance versus the simple
strategy baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

BACKTEST_DIR = Path(__file__).resolve().parent / "backtests"
OUTPUT_MD = Path(__file__).resolve().parent / "guard_mock_backtests.md"


def _load_summary(path: Path) -> Mapping[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    rows: list[Mapping[str, object]] = []
    if BACKTEST_DIR.exists():
        for path in sorted(BACKTEST_DIR.glob("gymrl_guard_confirm_*.json")):
            data = _load_summary(path)
            if not isinstance(data, Mapping):
                continue
            strategies = data.get("strategies")
            if not isinstance(strategies, Mapping):
                continue
            simple = strategies.get("simple") or {}
            maxdiff = strategies.get("maxdiff") or {}
            rows.append(
                {
                    "symbol": data.get("symbol", path.stem.split("_")[-1].upper()),
                    "maxdiff_return": maxdiff.get("return"),
                    "maxdiff_sharpe": maxdiff.get("sharpe"),
                    "simple_return": simple.get("return"),
                }
            )

    lines: list[str] = ["# Guard-Confirmed Mock Backtests", ""]
    if not rows:
        lines.append("_No mock backtest summaries found._")
    else:
        lines.append("| Symbol | MaxDiff Return | MaxDiff Sharpe | Simple Return | Δ (MaxDiff - Simple) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        deltas = []
        maxdiff_returns = []
        simple_returns = []
        for row in rows:
            maxdiff_ret = row["maxdiff_return"]
            maxdiff_sharpe = row["maxdiff_sharpe"]
            simple_ret = row["simple_return"]
            delta = None
            if isinstance(maxdiff_ret, (int, float)) and isinstance(simple_ret, (int, float)):
                delta = maxdiff_ret - simple_ret
                deltas.append(delta)
                maxdiff_returns.append(maxdiff_ret)
                simple_returns.append(simple_ret)
            md_ret = maxdiff_ret if isinstance(maxdiff_ret, (int, float)) else "n/a"
            md_sharpe = maxdiff_sharpe if isinstance(maxdiff_sharpe, (int, float)) else "n/a"
            simple = simple_ret if isinstance(simple_ret, (int, float)) else "n/a"
            md_delta = delta if isinstance(delta, (int, float)) else "n/a"
            lines.append(f"| {row['symbol']} | {md_ret} | {md_sharpe} | {simple} | {md_delta} |")
        if deltas:
            avg_maxdiff = sum(maxdiff_returns) / len(maxdiff_returns)
            avg_simple = sum(simple_returns) / len(simple_returns)
            avg_delta = sum(deltas) / len(deltas)
            lines.append(f"| **Average** | {avg_maxdiff:.4f} | - | {avg_simple:.4f} | {avg_delta:.4f} |")
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
