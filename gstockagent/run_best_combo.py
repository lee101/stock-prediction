#!/usr/bin/env python3
"""Test best combination: risk_averse prompt + RL signals."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gstockagent.config import GStockConfig
from gstockagent import simulator as sim_mod
from gstockagent.simulator import run_simulation
from gstockagent import prompt as prompt_mod

SYMS = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "AAVE", "LTC",
    "XRP", "DOT", "UNI", "NEAR", "APT", "ICP", "BNB",
    "ADA", "FIL", "ARB", "OP", "INJ", "SUI", "TIA", "SEI",
    "ATOM", "ALGO", "BCH", "TRX", "SHIB", "PEPE",
]

OUT_FILE = Path(__file__).parent / "best_combo_results.json"

RISK_AVERSE_TEMPLATE = """You are a conservative cryptocurrency portfolio manager on Binance.
Date: {date}
Available capital: ${capital:.2f} USDT
Max leverage: {leverage}x | Max positions: {max_positions}

CURRENT PRICES (last 7 days):
{price_table}

CHRONOS2 24-HOUR FORECASTS:
{forecast_table}
{rl_section}
CURRENT PORTFOLIO:
{portfolio_table}

TASK: Allocate capital CONSERVATIVELY for next 24 hours.
RULES:
- Never allocate more than 15% to a single position
- Always keep at least 40% in cash (USDT)
- Only take positions with clear directional signals
- Stop losses must be tight (within 3% of entry)
- Prefer fewer, higher-conviction trades

Respond with ONLY a JSON object:
```json
{{"allocations": {{"BTC": {{"allocation_pct": 10, "direction": "long", "exit_price": 70000, "stop_price": 68000}}}}, "reasoning": "brief explanation"}}
```"""

_original_build_prompt = prompt_mod.build_prompt


def _make_variant_prompt(variant_template):
    def _build(symbols, data_dir, forecast_dir, as_of, positions,
               current_prices, capital, leverage_limit, max_positions,
               rl_signals_table=""):
        price_table = prompt_mod.build_price_table(symbols, data_dir, as_of)
        forecast_table = prompt_mod.build_forecast_table(symbols, forecast_dir, as_of)
        portfolio_table = prompt_mod.build_portfolio_table(positions, current_prices)
        rl_section = ""
        if rl_signals_table:
            rl_section = f"\nRL SIGNALS:\n{rl_signals_table}\n"
        return variant_template.format(
            date=as_of.strftime("%Y-%m-%d"),
            capital=capital,
            leverage=leverage_limit,
            max_positions=max_positions,
            price_table=price_table,
            forecast_table=forecast_table,
            portfolio_table=portfolio_table,
            rl_section=rl_section,
        )
    return _build


def run():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.1-lite"
    start, end = "2025-10-01", "2026-01-10"

    configs = [
        {"leverage": 0.5, "variant": "default", "rl": False},
        {"leverage": 0.5, "variant": "default", "rl": True},
        {"leverage": 0.5, "variant": "risk_averse", "rl": False},
        {"leverage": 0.5, "variant": "risk_averse", "rl": True},
        {"leverage": 1.0, "variant": "risk_averse", "rl": True},
    ]

    existing = []
    if OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text())
    done_keys = {(r["model"], r["leverage"], r["variant"], r["rl"]) for r in existing}
    results = list(existing)

    variant_fn = _make_variant_prompt(RISK_AVERSE_TEMPLATE)

    for c in configs:
        key = (model, c["leverage"], c["variant"], c["rl"])
        if key in done_keys:
            print(f"SKIP {c}")
            continue

        if c["variant"] == "risk_averse":
            prompt_mod.build_prompt = variant_fn
            sim_mod.build_prompt = variant_fn
        else:
            prompt_mod.build_prompt = _original_build_prompt
            sim_mod.build_prompt = _original_build_prompt

        label = f"{model} lev={c['leverage']} {c['variant']} rl={c['rl']}"
        print(f"\n--- {label} ---", flush=True)
        t0 = time.time()
        try:
            cfg = GStockConfig(
                symbols=SYMS, leverage=c["leverage"], model=model,
                max_positions=5, initial_capital=10000,
            )
            r = run_simulation(cfg, start, end, use_cache=True, verbose=False,
                              use_rl_signals=c["rl"])
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            row = {
                "model": model, "leverage": c["leverage"],
                "variant": c["variant"], "rl": c["rl"],
                "return": r["total_return_pct"], "monthly": r["monthly_return_pct"],
                "max_dd": r["max_drawdown_pct"], "sortino": r["sortino"],
                "sharpe": r["sharpe"], "trades": r["n_trades"],
                "win_rate": r["win_rate_pct"], "days": r["n_days"],
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

    prompt_mod.build_prompt = _original_build_prompt
    sim_mod.build_prompt = _original_build_prompt

    print("\n\n=== BEST COMBO RESULTS ===")
    print(
        f"{'Model':>14} {'Lev':>5} {'Variant':>12} {'RL':>5} {'Ret%':>8} {'Mo%':>7} "
        f"{'DD%':>7} {'Sort':>6} {'Shrp':>6} {'Trd':>5} {'WR%':>5}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['model']:>14} {r['leverage']:>5.1f} {r['variant']:>12} "
            f"{str(r['rl']):>5} {r['return']:>+7.1f} {r['monthly']:>+6.1f} "
            f"{r['max_dd']:>6.1f} {r['sortino']:>6.2f} {r['sharpe']:>6.2f} "
            f"{r['trades']:>5} {r['win_rate']:>5.1f}"
        )


if __name__ == "__main__":
    run()
