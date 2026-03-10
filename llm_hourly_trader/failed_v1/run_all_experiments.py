"""Run all Gemini Flash Lite trading experiments sequentially."""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.backtest import run_backtest

SYMBOLS = ["BTCUSD", "ETHUSD"]
# Free tier: 15 req/min => 4s between calls
RATE_LIMIT = 4.2

experiments = [
    {"symbols": SYMBOLS, "days": 1, "mode": "structured", "label": "1d-structured"},
    {"symbols": SYMBOLS, "days": 1, "mode": "code", "label": "1d-code"},
    {"symbols": SYMBOLS, "days": 7, "mode": "structured", "label": "7d-structured"},
    {"symbols": SYMBOLS, "days": 7, "mode": "code", "label": "7d-code"},
]

all_results = {}
for exp in experiments:
    print(f"\n\n{'#'*70}")
    print(f"# EXPERIMENT: {exp['label']}")
    print(f"{'#'*70}")
    try:
        result = run_backtest(
            symbols=exp["symbols"],
            days=exp["days"],
            mode=exp["mode"],
            rate_limit_seconds=RATE_LIMIT,
        )
        all_results[exp["label"]] = result
    except Exception as e:
        print(f"EXPERIMENT FAILED: {e}")
        all_results[exp["label"]] = {"error": str(e)}

# Final comparison
print(f"\n\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Experiment':<20} {'Return':>10} {'Sortino':>10} {'Trades':>8} {'PnL':>12} {'Fees':>10}")
print("-" * 70)
for label, r in all_results.items():
    if "error" in r:
        print(f"{label:<20} {'ERROR':>10}")
        continue
    print(
        f"{label:<20} "
        f"{r['total_return_pct']:>+9.4f}% "
        f"{r['sortino']:>10.4f} "
        f"{r['total_trades']:>8d} "
        f"${r['realized_pnl']:>+10.2f} "
        f"${r['total_fees']:>9.2f}"
    )
print(f"{'='*70}")
