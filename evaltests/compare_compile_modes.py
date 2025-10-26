#!/usr/bin/env python3
"""Compare Toto torch.compile runs against the standard configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = REPO_ROOT / "evaltests" / "backtests"
OUTPUT_PATH = REPO_ROOT / "evaltests" / "guard_compile_comparison.md"

BASIC_SUFFIX = "_real_full.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Toto torch.compile runs against baseline.")
    parser.add_argument(
        "--compile-suffix",
        default="_real_full_compile.json",
        help="Suffix for compiled backtests (default: _real_full_compile.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output markdown path (default: evaltests/guard_compile_comparison.md).",
    )
    return parser.parse_args()


def _load(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _metric(data: Mapping[str, object] | None, strategy: str, field: str) -> float | None:
    if not data or not isinstance(data, Mapping):
        return None
    strategies = data.get("strategies")
    if not isinstance(strategies, Mapping):
        return None
    strat = strategies.get(strategy)
    if not isinstance(strat, Mapping):
        return None
    value = strat.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _val_loss(data: Mapping[str, object] | None) -> float | None:
    if not data or not isinstance(data, Mapping):
        return None
    metrics = data.get("metrics")
    if isinstance(metrics, Mapping):
        value = metrics.get("close_val_loss")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def fmt(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def main() -> None:
    args = parse_args()
    compile_suffix = args.compile_suffix
    output_path = args.output if isinstance(args.output, Path) else Path(args.output)
    lines = ["# Guard Compile Comparison", ""]
    lines.append("| Symbol | MaxDiff Δ (compile - base) | Simple Δ | Sharpe Δ | Val Loss Δ |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    for base_path in sorted(BACKTEST_DIR.glob(f"*{BASIC_SUFFIX}")):
        symbol_prefix = base_path.name.replace(BASIC_SUFFIX, "")
        compile_path = BACKTEST_DIR / f"{symbol_prefix}{compile_suffix.replace('.json','')}.json"
        if not compile_path.exists():
            continue

        base = _load(base_path)
        compiled = _load(compile_path)

        maxdiff_delta = _diff(_metric(base, "maxdiff", "return"), _metric(compiled, "maxdiff", "return"))
        simple_delta = _diff(_metric(base, "simple", "return"), _metric(compiled, "simple", "return"))
        sharpe_delta = _diff(_metric(base, "maxdiff", "sharpe"), _metric(compiled, "maxdiff", "sharpe"))
        loss_delta = _diff(_val_loss(base), _val_loss(compiled))

        lines.append(
            "| {symbol} | {md} | {sd} | {sh} | {ld} |".format(
                symbol=symbol_prefix.upper(),
                md=fmt(maxdiff_delta),
                sd=fmt(simple_delta),
                sh=fmt(sharpe_delta),
                ld=fmt(loss_delta, precision=5),
            )
        )

    if len(lines) == 3:
        lines.append("| _No compile runs found_ | | | | |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
