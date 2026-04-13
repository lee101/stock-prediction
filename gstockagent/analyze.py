#!/usr/bin/env python3
"""Analyze simulation results - equity curves, trade stats, drawdowns."""
import argparse
import json
import numpy as np
from .config import GStockConfig
from .simulator import run_simulation


def analyze_run(config: GStockConfig, start: str, end: str, label: str = ""):
    r = run_simulation(config, start, end, use_cache=True, verbose=False)
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return r

    eq = [e["equity"] for e in r["equity_curve"]]
    dates = [e["date"] for e in r["equity_curve"]]

    print(f"\n{'='*60}")
    print(f"  {label or config.model} | lev={config.leverage}x | maxpos={config.max_positions}")
    print(f"{'='*60}")
    print(f"  Period: {dates[0]} to {dates[-1]} ({r['n_days']} days)")
    print(f"  Return: {r['total_return_pct']:+.1f}% (${config.initial_capital:.0f} -> ${r['final_equity']:.0f})")
    print(f"  Monthly: {r['monthly_return_pct']:+.1f}%")
    print(f"  Max DD: {r['max_drawdown_pct']:.1f}%")
    print(f"  Sortino: {r['sortino']:.2f} | Sharpe: {r['sharpe']:.2f}")
    print(f"  Trades: {r['n_trades']} | Win rate: {r['win_rate_pct']:.1f}%")

    # monthly breakdown
    if len(dates) > 30:
        print(f"\n  Monthly breakdown:")
        months = {}
        for e in r["equity_curve"]:
            mo = e["date"][:7]
            if mo not in months:
                months[mo] = []
            months[mo].append(e["equity"])
        prev_end = config.initial_capital
        for mo, eqs in sorted(months.items()):
            mo_ret = (eqs[-1] / prev_end - 1) * 100
            peak = max(eqs)
            dd = (min(eqs) / peak - 1) * 100
            print(f"    {mo}: {mo_ret:+6.1f}% (dd={dd:.1f}%) eq=${eqs[-1]:.0f}")
            prev_end = eqs[-1]

    # trade analysis
    if r["trade_log"]:
        pnls = [t["pnl"] for t in r["trade_log"]]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        print(f"\n  Trade stats:")
        print(f"    Avg win: ${np.mean(wins):.2f}" if wins else "    No wins")
        print(f"    Avg loss: ${np.mean(losses):.2f}" if losses else "    No losses")
        if wins and losses:
            print(f"    Win/loss ratio: {abs(np.mean(wins)/np.mean(losses)):.2f}")
        print(f"    Avg PnL: ${np.mean(pnls):.2f}")

        # per-symbol breakdown
        sym_pnl = {}
        for t in r["trade_log"]:
            sym_pnl.setdefault(t["symbol"], []).append(t["pnl"])
        print(f"\n  Per-symbol PnL:")
        for sym, pnls_s in sorted(sym_pnl.items(), key=lambda x: sum(x[1]), reverse=True):
            total = sum(pnls_s)
            print(f"    {sym:>6}: ${total:+.0f} ({len(pnls_s)} trades, "
                  f"wr={sum(1 for p in pnls_s if p>0)/len(pnls_s)*100:.0f}%)")

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.1-lite")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-01-10")
    args = parser.parse_args()

    cfg = GStockConfig(leverage=args.leverage, model=args.model,
                       max_positions=args.max_positions)
    analyze_run(cfg, args.start, args.end)


if __name__ == "__main__":
    main()
