#!/usr/bin/env python3
"""Compare baseline vs high-sample guard backtests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = REPO_ROOT / "evaltests" / "backtests"
OUTPUT_PATH = REPO_ROOT / "evaltests" / "guard_highsample_comparison.md"

BASELINE_SUFFIX = "_real_full.json"
HIGHSAMPLE_SUFFIX = "_real_full_highsamples.json"


def _load(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_metrics(data: Mapping[str, object]) -> dict[str, float]:
    strategies = data.get("strategies") if isinstance(data, Mapping) else {}
    result: dict[str, float] = {}
    if isinstance(strategies, Mapping):
        maxdiff = strategies.get("maxdiff")
        simple = strategies.get("simple")
        if isinstance(maxdiff, Mapping):
            result["maxdiff_return"] = float(maxdiff.get("return", 0.0))
            result["maxdiff_sharpe"] = float(maxdiff.get("sharpe", 0.0))
        if isinstance(simple, Mapping):
            result["simple_return"] = float(simple.get("return", 0.0))
            result["simple_sharpe"] = float(simple.get("sharpe", 0.0))
        turnover = maxdiff.get("turnover") if isinstance(maxdiff, Mapping) else None
        if turnover is not None:
            result["maxdiff_turnover"] = float(turnover)
    metrics = data.get("metrics") if isinstance(data, Mapping) else {}
    if isinstance(metrics, Mapping):
        val_loss = metrics.get("close_val_loss")
        if isinstance(val_loss, (int, float)):
            result["close_val_loss"] = float(val_loss)
    return result


def main() -> None:
    lines = ["# Guard High-Sample Comparison", ""]
    lines.append("| Symbol | MaxDiff Return Δ | Simple Return Δ | MaxDiff Sharpe Δ | Turnover Δ | Close Val Loss Δ | Notes |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")

    rows = []
    for baseline_path in sorted(BACKTEST_DIR.glob(f"*{BASELINE_SUFFIX}")):
        symbol_prefix = baseline_path.name.replace(BASELINE_SUFFIX, "")
        highsample_path = BACKTEST_DIR / f"{symbol_prefix}{HIGHSAMPLE_SUFFIX.replace('.json','')}.json"
        if not highsample_path.exists():
            continue

        baseline = _load(baseline_path)
        high = _load(highsample_path)
        if not baseline or not high:
            continue

        base_metrics = _extract_metrics(baseline)
        high_metrics = _extract_metrics(high)

        def diff(key: str) -> float | None:
            if key not in base_metrics or key not in high_metrics:
                return None
            return high_metrics[key] - base_metrics[key]

        notes = []
        turnover_delta = diff("maxdiff_turnover")
        if turnover_delta is not None:
            notes.append("↓ turnover" if turnover_delta < 0 else "↑ turnover")

        lines.append(
            "| {symbol} | {mdiff:.4f} | {sdiff:.4f} | {shdiff:.4f} | {tdiff:.4f} | {vdiff:.5f} | {notes} |".format(
                symbol=symbol_prefix.upper(),
                mdiff=diff("maxdiff_return") or 0.0,
                sdiff=diff("simple_return") or 0.0,
                shdiff=diff("maxdiff_sharpe") or 0.0,
                tdiff=turnover_delta or 0.0,
                vdiff=diff("close_val_loss") or 0.0,
                notes=", ".join(notes) if notes else "",
            )
        )

        rows.append(diff("maxdiff_return") or 0.0)

    if not rows:
        lines.append("| _No comparisons found_ | | | | | | |")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
