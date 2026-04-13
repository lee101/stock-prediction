#!/usr/bin/env python3
"""Run leverage/model sweep with incremental JSON output."""
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gstockagent.config import GStockConfig
from gstockagent.simulator import run_simulation

SYMS = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "AAVE", "LTC",
    "XRP", "DOT", "UNI", "NEAR", "APT", "ICP", "BNB",
    "ADA", "FIL", "ARB", "OP", "INJ", "SUI", "TIA", "SEI",
    "ATOM", "ALGO", "BCH", "TRX", "SHIB", "PEPE",
]

OUT_FILE = Path(__file__).parent / "sweep_results.json"

def run():
    leverage_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
    model_grid = ["gemini-3.1-lite", "glm-5"]
    max_pos_grid = [5]
    start, end = "2025-10-01", "2026-01-10"

    # load existing results
    existing = []
    if OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text())
    done_keys = {(r["leverage"], r["model"], r["max_positions"]) for r in existing}

    results = list(existing)

    for model in model_grid:
        for lev in leverage_grid:
            for mp in max_pos_grid:
                key = (lev, model, mp)
                if key in done_keys:
                    print(f"SKIP {model} lev={lev} mp={mp} (already done)")
                    continue

                label = f"{model} lev={lev} mp={mp}"
                print(f"\n--- {label} ---", flush=True)
                t0 = time.time()
                try:
                    cfg = GStockConfig(symbols=SYMS, leverage=lev, model=model,
                                       max_positions=mp, initial_capital=10000)
                    r = run_simulation(cfg, start, end, use_cache=True, verbose=False)
                    if "error" in r:
                        print(f"  ERROR: {r['error']}")
                        continue
                    row = {
                        "leverage": lev, "model": model, "max_positions": mp,
                        "return": r["total_return_pct"], "monthly": r["monthly_return_pct"],
                        "max_dd": r["max_drawdown_pct"], "sortino": r["sortino"],
                        "sharpe": r["sharpe"], "trades": r["n_trades"],
                        "win_rate": r["win_rate_pct"], "days": r["n_days"],
                        "final": r["final_equity"],
                    }
                    results.append(row)
                    OUT_FILE.write_text(json.dumps(results, indent=2))
                    elapsed = time.time() - t0
                    print(f"  ret={row['return']:+.1f}% dd={row['max_dd']:.1f}% "
                          f"sort={row['sortino']:.2f} trades={row['trades']} "
                          f"({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)

    # print summary
    print("\n\n=== SWEEP RESULTS ===")
    print(f"{'Model':>14} {'Lev':>5} {'MP':>3} {'Ret%':>8} {'Mo%':>7} "
          f"{'DD%':>7} {'Sort':>6} {'Shrp':>6} {'Trd':>5} {'WR%':>5}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["sortino"], reverse=True):
        print(f"{r['model']:>14} {r['leverage']:>5.1f} {r['max_positions']:>3} "
              f"{r['return']:>+7.1f} {r['monthly']:>+6.1f} "
              f"{r['max_dd']:>6.1f} {r['sortino']:>6.2f} {r['sharpe']:>6.2f} "
              f"{r['trades']:>5} {r['win_rate']:>5.1f}")


if __name__ == "__main__":
    run()
