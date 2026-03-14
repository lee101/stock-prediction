"""Sweep prompt variants for Binance crypto trading.

Runs multiple prompt variants through the backtest simulator and compares.
Uses cached LLM responses where available; calls API for new ones.
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_hourly_trader.backtest import run_backtest, RESULTS_DIR
from llm_hourly_trader.config import BacktestConfig


VARIANTS = [
    "position_context",
    "freeform",
    "mae_bands",
    "default",
    "conservative",
    "aggressive",
    "uncertainty_gated",
    "no_forecast",
    "h1_only",
]

CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=CRYPTO_SYMBOLS)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--leverage", type=float, default=5.0)
    p.add_argument("--margin-rate", type=float, default=0.10)
    p.add_argument("--rate-limit", type=float, default=4.2)
    p.add_argument("--parallel", type=int, default=1)
    args = p.parse_args()

    margin_fee = 0.001 if args.leverage > 1.0 else None

    print(f"\n{'='*70}")
    print(f"PROMPT VARIANT SWEEP")
    print(f"Symbols: {args.symbols} | Days: {args.days} | Model: {args.model}")
    print(f"Leverage: {args.leverage}x | Margin fee: {margin_fee}")
    print(f"Variants: {args.variants}")
    print(f"{'='*70}\n")

    results = {}
    for variant in args.variants:
        print(f"\n{'#'*60}")
        print(f"# VARIANT: {variant}")
        print(f"{'#'*60}")

        config = BacktestConfig(
            initial_cash=10_000.0,
            max_hold_hours=6,
            max_position_pct=0.25,
            rate_limit_seconds=args.rate_limit,
            model=args.model,
            prompt_variant=variant,
            parallel_workers=args.parallel,
        )

        try:
            result = run_backtest(symbols=args.symbols, days=args.days, config=config)
            if "error" not in result:
                # Apply leverage post-hoc if >1x (same approach as sweep_leverage)
                if args.leverage > 1.0:
                    ret = result.get("total_return_pct", 0)
                    result["leveraged_return_pct"] = ret * args.leverage
                    result["leverage"] = args.leverage
                results[variant] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[variant] = {"error": str(e)}

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"{'Variant':<22} {'Return':>10} {'Sortino':>10} {'MaxDD':>10} {'Trades':>8} {'PnL':>10}")
    print("-" * 90)
    for v in args.variants:
        r = results.get(v)
        if not r or "error" in r:
            print(f"{v:<22} {'ERROR':>10}")
            continue
        ret = r.get("leveraged_return_pct", r.get("total_return_pct", 0))
        sort = r.get("sortino", 0)
        dd = r.get("max_drawdown_pct", 0)
        trades = r.get("entries", 0)
        pnl = r.get("realized_pnl", 0)
        print(f"{v:<22} {ret:>+9.2f}% {sort:>10.2f} {dd:>9.2f}% {trades:>8d} ${pnl:>+9.2f}")

    # Best by sortino
    valid = {k: v for k, v in results.items() if isinstance(v, dict) and "error" not in v and v.get("sortino", 0) > 0}
    if valid:
        best = max(valid, key=lambda k: valid[k]["sortino"])
        print(f"\nBest by Sortino: {best} (Sort={valid[best]['sortino']:.2f})")

    # Save summary
    out = RESULTS_DIR / "prompt_variant_sweep.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save[k] = {kk: vv for kk, vv in v.items()
                       if kk not in ("equity_history", "timestamps", "trades")}
    with open(out, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
