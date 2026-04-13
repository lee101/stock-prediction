"""Run LLM hourly trader experiments v2.

Uses gemini-3.1-flash-lite-preview.
Runs experiments sequentially to stay within rate limits.
Disk cache avoids repeating API calls on re-runs.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.backtest import run_backtest, GROUPS
from llm_hourly_trader.config import BacktestConfig

MODEL = "gemini-3.1-flash-lite-preview"
# Keep this on the lighter 3.1 lane for batch experiments.
RATE_LIMIT = 6.5  # ~9 req/min to be safe

experiments = [
    # Crypto 30d - 3 prompt variants
    {"group": "crypto", "days": 30, "prompt": "default", "label": "crypto_30d_default"},
    {"group": "crypto", "days": 30, "prompt": "conservative", "label": "crypto_30d_conservative"},
    {"group": "crypto", "days": 30, "prompt": "aggressive", "label": "crypto_30d_aggressive"},
    # AI stocks 14d (limited by forecast availability)
    {"group": "ai_stocks", "days": 14, "prompt": "default", "label": "ai_stocks_14d_default"},
    # Short stocks 14d
    {"group": "short_stocks", "days": 14, "prompt": "default", "label": "short_stocks_14d_default"},
]

all_results = {}
for exp in experiments:
    print(f"\n\n{'#'*70}")
    print(f"# {exp['label']}")
    print(f"{'#'*70}")

    symbols = GROUPS[exp["group"]]
    config = BacktestConfig(
        initial_cash=10_000.0,
        max_hold_hours=6,
        max_position_pct=0.25,
        rate_limit_seconds=RATE_LIMIT,
        model=MODEL,
        prompt_variant=exp["prompt"],
    )
    try:
        result = run_backtest(symbols=symbols, days=exp["days"], config=config)
        all_results[exp["label"]] = result
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results[exp["label"]] = {"error": str(e)}

# Final comparison
print(f"\n\n{'='*90}")
print("FINAL COMPARISON")
print(f"{'='*90}")
print(f"{'Experiment':<30} {'Return':>10} {'MaxDD':>10} {'Sortino':>10} {'Trades':>8} {'PnL':>12} {'Fees':>10}")
print("-" * 90)
for label, r in all_results.items():
    if "error" in r:
        print(f"{label:<30} {'ERROR':>10} - {r.get('error','')[:40]}")
        continue
    print(
        f"{label:<30} "
        f"{r['total_return_pct']:>+9.2f}% "
        f"{r['max_drawdown_pct']:>9.2f}% "
        f"{r['sortino']:>10.2f} "
        f"{r.get('entries',0)+r.get('exits',0):>8d} "
        f"${r['realized_pnl']:>+10.2f} "
        f"${r['total_fees']:>9.2f}"
    )
print(f"{'='*90}")

# Save summary
summary_path = Path(__file__).resolve().parent / "results" / "experiment_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSummary saved to {summary_path}")
