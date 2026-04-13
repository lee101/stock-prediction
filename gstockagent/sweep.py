#!/usr/bin/env python3
import argparse
import json
from itertools import product
from .config import GStockConfig
from .simulator import run_simulation


def run_sweep(start_date: str, end_date: str, verbose: bool = False):
    leverage_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
    model_grid = ["gemini-3.1-lite", "glm-5"]
    max_pos_grid = [5, 10]

    results = []
    for lev, model, mp in product(leverage_grid, model_grid, max_pos_grid):
        cfg = GStockConfig(leverage=lev, model=model, max_positions=mp)
        label = f"lev={lev} model={model} mp={mp}"
        print(f"\n--- {label} ---")
        try:
            r = run_simulation(cfg, start_date, end_date, use_cache=True, verbose=verbose)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            row = {
                "leverage": lev, "model": model, "max_positions": mp,
                "return": r["total_return_pct"], "monthly": r["monthly_return_pct"],
                "max_dd": r["max_drawdown_pct"], "sortino": r["sortino"],
                "sharpe": r["sharpe"], "trades": r["n_trades"],
                "win_rate": r["win_rate_pct"], "days": r["n_days"],
            }
            results.append(row)
            print(f"  ret={row['return']:+.1f}% dd={row['max_dd']:.1f}% "
                  f"sort={row['sortino']:.2f} trades={row['trades']}")
        except Exception as e:
            print(f"  FAILED: {e}")

    if results:
        print("\n\n=== SWEEP RESULTS ===")
        print(f"{'Lev':>5} {'Model':>12} {'MP':>3} {'Ret%':>8} {'Mo%':>7} "
              f"{'DD%':>7} {'Sort':>6} {'Shrp':>6} {'Trd':>5} {'WR%':>5}")
        print("-" * 72)
        for r in sorted(results, key=lambda x: x["sortino"], reverse=True):
            print(f"{r['leverage']:>5.1f} {r['model']:>12} {r['max_positions']:>3} "
                  f"{r['return']:>+7.1f} {r['monthly']:>+6.1f} "
                  f"{r['max_dd']:>6.1f} {r['sortino']:>6.2f} {r['sharpe']:>6.2f} "
                  f"{r['trades']:>5} {r['win_rate']:>5.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_sweep(args.start, args.end, args.verbose)


if __name__ == "__main__":
    main()
