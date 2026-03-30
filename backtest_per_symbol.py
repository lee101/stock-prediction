#!/usr/bin/env python3
"""Per-symbol backtest to find best tradable pairs.

Tests each symbol individually with the winning LLM model to find
which symbols generate the most PnL. Deploy only the best ones.

Usage:
  python -u backtest_per_symbol.py --days 30 --model glm-4-plus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from env_real import *  # noqa: F401,F403

from unified_orchestrator.backtest_hybrid import run_backtest

ALL_STOCK_SYMBOLS = [
    "AAPL", "AMZN", "DBX", "GOOG", "META", "MSFT",
    "NET", "NVDA", "NYT", "PLTR", "TRIP", "TSLA", "YELP",
]


def run_per_symbol(
    symbols: list[str],
    days: int,
    model: str,
    cadence: str = "daily",
    min_confidence: float = 0.3,
) -> dict:
    results = {}

    for sym in symbols:
        print(f"\n{'=' * 60}")
        print(f"TESTING: {sym} with {model}")
        print(f"{'=' * 60}")
        t0 = time.time()

        try:
            r = run_backtest(
                symbols=[sym],
                days=days,
                initial_cash=10_000.0,
                modes=["gemini_only"],
                model=model,
                thinking_level="NONE" if "glm" in model or "grok" in model else "HIGH",
                decision_cadence=cadence,
                min_plan_confidence=min_confidence,
            )
            elapsed = time.time() - t0
            data = r.get("gemini_only", {})
            data["symbol"] = sym
            data["elapsed_s"] = elapsed
            results[sym] = data
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[sym] = {"error": str(e), "symbol": sym}

    # Print sorted results
    print(f"\n{'=' * 80}")
    print(f"PER-SYMBOL RESULTS — {model} — {days}d {cadence}")
    print(f"{'=' * 80}")
    print(f"{'Symbol':<10} {'Return%':>10} {'Sortino':>10} {'MaxDD%':>10} {'Fills':>8} {'Grade':>8}")
    print("-" * 60)

    sorted_results = sorted(
        [(sym, d) for sym, d in results.items() if "error" not in d],
        key=lambda x: x[1].get("return_pct", -999),
        reverse=True,
    )

    for sym, data in sorted_results:
        ret = data.get("return_pct", 0)
        sortino = data.get("sortino", 0)
        dd = data.get("max_drawdown", 0)
        fills = data.get("fills", 0)

        if ret > 2 and sortino > 2:
            grade = "A+"
        elif ret > 1:
            grade = "A"
        elif ret > 0:
            grade = "B"
        elif ret > -1:
            grade = "C"
        else:
            grade = "F"

        print(f"{sym:<10} {ret:>+9.2f}% {sortino:>10.2f} {dd:>9.2f}% {fills:>8d} {grade:>8}")

    # Recommend deployment set
    deploy_syms = [sym for sym, d in sorted_results if d.get("return_pct", 0) > 0]
    print(f"\nRECOMMENDED DEPLOY SET ({len(deploy_syms)} symbols): {deploy_syms}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=ALL_STOCK_SYMBOLS)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--model", default="glm-4-plus")
    parser.add_argument("--cadence", choices=["hourly", "daily"], default="daily")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    results = run_per_symbol(
        symbols=args.symbols,
        days=args.days,
        model=args.model,
        cadence=args.cadence,
        min_confidence=args.min_confidence,
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
