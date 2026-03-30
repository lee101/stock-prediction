#!/usr/bin/env python3
"""Multi-window + multi-pass backtest to find optimal config.

Tests across 120d, 30d, 7d, 2d windows with 1-pass and 2-pass reprompting.

Usage:
  python -u backtest_multiwindow.py
  python -u backtest_multiwindow.py --models glm-4-plus gemini-3.1-flash-lite-preview
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

SYMBOLS = ["YELP", "NET", "DBX", "PLTR", "AMZN"]
WINDOWS = [120, 30, 7, 2]


def run_config(
    symbols: list[str],
    days: int,
    model: str,
    thinking_level: str,
    reprompt_passes: int,
    review_model: str | None = None,
) -> dict:
    """Run one backtest config and return results."""
    try:
        r = run_backtest(
            symbols=symbols,
            days=days,
            initial_cash=10_000.0,
            modes=["gemini_only"],
            model=model,
            thinking_level=thinking_level,
            decision_cadence="daily",
            min_plan_confidence=0.3,
            reprompt_passes=reprompt_passes,
            reprompt_policy="actionable",
            review_model=review_model,
        )
        return r.get("gemini_only", {"error": "no result"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["glm-4-plus", "gemini-3.1-flash-lite-preview"])
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS)
    parser.add_argument("--output-json", default="backtest_multiwindow_results.json")
    args = parser.parse_args()

    # Configs to test: model × window × passes
    configs = []
    for model in args.models:
        thinking = "NONE" if "glm" in model or "grok" in model else "HIGH"
        for days in args.windows:
            # 1-pass (baseline)
            configs.append({
                "label": f"{model}_{days}d_1pass",
                "model": model,
                "days": days,
                "thinking": thinking,
                "passes": 1,
                "review_model": None,
            })
            # 2-pass self-review
            configs.append({
                "label": f"{model}_{days}d_2pass",
                "model": model,
                "days": days,
                "thinking": thinking,
                "passes": 2,
                "review_model": None,
            })

    # Also test cross-model review: GLM → Gemini refine
    for days in args.windows:
        if "glm-4-plus" in args.models and "gemini-3.1-flash-lite-preview" in args.models:
            configs.append({
                "label": f"glm4_gemini_review_{days}d",
                "model": "glm-4-plus",
                "days": days,
                "thinking": "NONE",
                "passes": 2,
                "review_model": "gemini-3.1-flash-lite-preview",
            })

    results = {}
    total = len(configs)

    for i, cfg in enumerate(configs):
        label = cfg["label"]
        print(f"\n{'#' * 70}")
        print(f"# [{i+1}/{total}] {label}")
        print(f"{'#' * 70}")
        t0 = time.time()

        r = run_config(
            symbols=args.symbols,
            days=cfg["days"],
            model=cfg["model"],
            thinking_level=cfg["thinking"],
            reprompt_passes=cfg["passes"],
            review_model=cfg["review_model"],
        )
        r["label"] = label
        r["config"] = cfg
        r["wall_time_s"] = time.time() - t0
        results[label] = r

    # Print summary table
    print(f"\n{'=' * 100}")
    print("MULTI-WINDOW RESULTS")
    print(f"{'=' * 100}")
    print(f"{'Config':<45} {'Return%':>10} {'Sortino':>10} {'MaxDD%':>10} {'Fills':>8} {'Time':>8}")
    print("-" * 100)

    # Group by window
    for days in args.windows:
        print(f"\n--- {days}d window ---")
        window_results = [(k, v) for k, v in results.items() if f"_{days}d" in k]
        window_results.sort(key=lambda x: x[1].get("sortino", -999), reverse=True)
        for label, data in window_results:
            if "error" in data:
                print(f"  {label:<43} ERROR: {str(data['error'])[:40]}")
                continue
            ret = data.get("return_pct", 0)
            sortino = data.get("sortino", 0)
            dd = data.get("max_drawdown", 0)
            fills = data.get("fills", 0)
            wt = data.get("wall_time_s", 0)
            print(f"  {label:<43} {ret:>+9.2f}% {sortino:>10.2f} {dd:>9.2f}% {fills:>8d} {wt:>7.0f}s")

    # Best overall by Sortino
    valid = [(k, v) for k, v in results.items() if "error" not in v and v.get("sortino", 0) != 0]
    if valid:
        best = max(valid, key=lambda x: x[1].get("sortino", -999))
        print(f"\nBEST SORTINO: {best[0]} → {best[1].get('sortino', 0):.2f} ({best[1].get('return_pct', 0):+.2f}%)")

    # Save
    out = Path(args.output_json)
    out.write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
