#!/usr/bin/env python3
"""A/B comparison: LLM-only vs LLM+RL signals."""
import json
import sys
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

OUT_FILE = Path(__file__).parent / "rl_comparison_results.json"


def run():
    leverage_grid = [0.5, 1.0]
    model_grid = ["gemini-3.1-lite", "glm-5"]
    start, end = "2025-10-01", "2026-01-10"

    existing = []
    if OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text())
    done_keys = {(r["leverage"], r["model"], r["rl_signals"]) for r in existing}
    results = list(existing)

    for model in model_grid:
        for lev in leverage_grid:
            for use_rl in [False, True]:
                key = (lev, model, use_rl)
                if key in done_keys:
                    print(f"SKIP {model} lev={lev} rl={use_rl}")
                    continue
                label = f"{model} lev={lev} rl={use_rl}"
                print(f"\n--- {label} ---", flush=True)
                t0 = time.time()
                try:
                    cfg = GStockConfig(
                        symbols=SYMS, leverage=lev, model=model,
                        max_positions=5, initial_capital=10000,
                    )
                    r = run_simulation(
                        cfg, start, end, use_cache=True,
                        verbose=False, use_rl_signals=use_rl,
                    )
                    if "error" in r:
                        print(f"  ERROR: {r['error']}")
                        continue
                    row = {
                        "leverage": lev,
                        "model": model,
                        "rl_signals": use_rl,
                        "return": r["total_return_pct"],
                        "monthly": r["monthly_return_pct"],
                        "max_dd": r["max_drawdown_pct"],
                        "sortino": r["sortino"],
                        "sharpe": r["sharpe"],
                        "trades": r["n_trades"],
                        "win_rate": r["win_rate_pct"],
                        "days": r["n_days"],
                        "final": r["final_equity"],
                    }
                    results.append(row)
                    OUT_FILE.write_text(json.dumps(results, indent=2))
                    elapsed = time.time() - t0
                    print(
                        f"  ret={row['return']:+.1f}% dd={row['max_dd']:.1f}% "
                        f"sort={row['sortino']:.2f} trades={row['trades']} "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    import traceback

                    traceback.print_exc()

    print("\n\n=== RL COMPARISON ===")
    print(
        f"{'Model':>14} {'Lev':>5} {'RL':>5} {'Ret%':>8} {'Mo%':>7} "
        f"{'DD%':>7} {'Sort':>6} {'Shrp':>6} {'Trd':>5} {'WR%':>5}"
    )
    print("-" * 80)
    for r in sorted(results, key=lambda x: (x["model"], x["leverage"], x["rl_signals"])):
        print(
            f"{r['model']:>14} {r['leverage']:>5.1f} {str(r['rl_signals']):>5} "
            f"{r['return']:>+7.1f} {r['monthly']:>+6.1f} "
            f"{r['max_dd']:>6.1f} {r['sortino']:>6.2f} {r['sharpe']:>6.2f} "
            f"{r['trades']:>5} {r['win_rate']:>5.1f}"
        )


if __name__ == "__main__":
    run()
