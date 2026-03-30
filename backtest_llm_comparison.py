#!/usr/bin/env python3
"""Compare LLM providers on realistic market simulation.

Tests grok-4-1-fast, glm-4-plus, and gemini-3.1-flash-lite-preview
on the same backtest window. Uses the existing backtest_hybrid infrastructure.

Usage:
  # Crypto hourly (default)
  python backtest_llm_comparison.py --days 30

  # Stock daily
  python backtest_llm_comparison.py --days 60 --cadence daily --asset-class stock \
      --symbols NVDA PLTR GOOG META MSFT

  # Just grok vs gemini
  python backtest_llm_comparison.py --models grok-4-1-fast gemini-3.1-flash-lite-preview
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from env_real import *  # noqa: F401,F403 — exports API keys

from unified_orchestrator.backtest_hybrid import run_backtest


DEFAULT_CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
DEFAULT_STOCK_SYMBOLS = ["NVDA", "PLTR", "GOOG", "META", "MSFT"]

# Models to compare: (display_name, model_id, thinking_level)
# Use cheap/fast models for backtesting, deploy the better one in prod
BACKTEST_MODELS = [
    ("Gemini-3.1-flash", "gemini-3.1-flash-lite-preview", "HIGH"),
    ("Grok-4-1-fast", "grok-4-1-fast", None),
    ("GLM-4-plus", "glm-4-plus", None),
]


def run_comparison(
    symbols: list[str],
    days: int,
    cadence: str,
    models: list[tuple[str, str, str | None]],
    start_ts: str | None = None,
    end_ts: str | None = None,
    initial_cash: float = 10_000.0,
    min_confidence: float = 0.4,
) -> dict:
    """Run same backtest window across multiple models, compare results."""
    all_results = {}

    for display_name, model_id, thinking_level in models:
        print(f"\n{'#' * 70}")
        print(f"# MODEL: {display_name} ({model_id})")
        print(f"{'#' * 70}")
        t0 = time.time()

        try:
            result = run_backtest(
                symbols=symbols,
                days=days,
                initial_cash=initial_cash,
                modes=["gemini_only"],  # LLM-only mode (uses call_llm dispatcher)
                model=model_id,
                thinking_level=thinking_level or "NONE",
                decision_cadence=cadence,
                min_plan_confidence=min_confidence,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            elapsed = time.time() - t0
            r = result.get("gemini_only", {})
            r["model"] = model_id
            r["display_name"] = display_name
            r["total_elapsed_s"] = elapsed
            all_results[display_name] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[display_name] = {
                "error": str(e),
                "model": model_id,
                "display_name": display_name,
            }

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("LLM COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Model':<25} {'Return%':>10} {'Sortino':>10} {'MaxDD%':>10} {'Fills':>8} {'API$':>8} {'Time':>8}")
    print("-" * 80)

    best_return = -float("inf")
    best_model = None

    for name, data in all_results.items():
        if "error" in data:
            print(f"{name:<25} {'ERROR':>10} - {data['error'][:40]}")
            continue

        ret = data.get("return_pct", 0)
        sortino = data.get("sortino", 0)
        dd = data.get("max_drawdown", 0)
        fills = data.get("fills", 0)
        api_calls = data.get("api_calls", 0)
        elapsed = data.get("total_elapsed_s", 0)

        marker = ""
        if ret > best_return:
            best_return = ret
            best_model = name

        print(f"{name:<25} {ret:>+9.2f}% {sortino:>10.2f} {dd:>9.2f}% {fills:>8d} {api_calls:>8d} {elapsed:>7.0f}s")

    if best_model:
        print(f"\nBEST: {best_model} ({best_return:+.2f}%)")
        best_data = all_results[best_model]
        print(f"  Model ID: {best_data['model']}")
        print(f"  Sortino: {best_data.get('sortino', 0):.2f}")
        print(f"  Max Drawdown: {best_data.get('max_drawdown', 0):.2f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare LLM providers on market simulation")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to trade (default: crypto=BTC/ETH/SOL, stock=NVDA/PLTR/etc)")
    parser.add_argument("--days", type=int, default=14, help="Backtest window in days")
    parser.add_argument("--cadence", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--asset-class", choices=["crypto", "stock"], default="crypto")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to test (default: grok-4-1-fast, glm-4-plus, gemini-3.1-flash-lite-preview)")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--start-ts", default=None)
    parser.add_argument("--end-ts", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    # Default symbols
    if args.symbols is None:
        args.symbols = DEFAULT_CRYPTO_SYMBOLS if args.asset_class == "crypto" else DEFAULT_STOCK_SYMBOLS

    # Build model list
    if args.models:
        models = []
        for m in args.models:
            # Find display name from defaults
            found = False
            for dn, mid, tl in BACKTEST_MODELS:
                if mid == m:
                    models.append((dn, mid, tl))
                    found = True
                    break
            if not found:
                models.append((m, m, None))
    else:
        models = BACKTEST_MODELS

    print(f"Backtest: {args.days}d {args.cadence} | Symbols: {args.symbols}")
    print(f"Models: {[m[0] for m in models]}")

    results = run_comparison(
        symbols=args.symbols,
        days=args.days,
        cadence=args.cadence,
        models=models,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        initial_cash=args.cash,
        min_confidence=args.min_confidence,
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
