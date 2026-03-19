#!/usr/bin/env python3
"""Run 120-day backtest of the deployed hybrid RL+Gemini spot strategy.

Wraps backtest_hybrid_rl_gemini.py with deployed config:
  - 6 symbols: BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD
  - Gemini 2.5 Flash with cache
  - ~120 days: 2025-11-15 to 2026-03-19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_SYMBOLS = "BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD"
DEFAULT_START = "2025-11-15"
DEFAULT_END = "2026-03-19"
DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CASH = 10000.0
DEFAULT_REPORT = "reports/120d_hybrid_spot_eval.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="120-day hybrid spot backtest eval")
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cash", type=float, default=DEFAULT_CASH)
    p.add_argument("--mode", choices=["hybrid", "rl_only"], default="hybrid")
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print command without running")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def run_eval(args):
    report_path = REPO / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Hybrid spot 120d eval")
    print(f"  symbols: {args.symbols}")
    print(f"  period:  {args.start_date} -> {args.end_date}")
    print(f"  mode:    {args.mode}")
    print(f"  model:   {args.model}")
    print(f"  report:  {report_path}")
    print()

    if args.dry_run:
        print("DRY RUN -- would run backtest with above config")
        return 0

    sys.path.insert(0, str(REPO))
    from backtest_hybrid_rl_gemini import run_hybrid_backtest

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    ctx, metrics = run_hybrid_backtest(
        symbols=symbols,
        checkpoint=args.checkpoint,
        start_date=args.start_date,
        end_date=args.end_date,
        mode=args.mode,
        initial_cash=args.cash,
        model=args.model,
    )

    result = {
        "config": {
            "symbols": symbols,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "mode": args.mode,
            "model": args.model,
            "cash": args.cash,
            "checkpoint": args.checkpoint,
        },
        "metrics": metrics,
    }

    # per-symbol PnL from trade history
    per_sym = {}
    for t in ctx.trade_history:
        per_sym[t.symbol] = per_sym.get(t.symbol, 0.0) + t.pnl
    result["per_symbol_pnl"] = per_sym

    report_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"120-DAY HYBRID SPOT EVAL ({args.mode.upper()})")
    print(f"{'='*60}")
    print(f"  Total return:   {metrics.get('total_return', 0)*100:+.2f}%")
    print(f"  Annualized:     {metrics.get('annualized_return', 0)*100:+.1f}%")
    print(f"  Sortino:        {metrics.get('sortino', 0):.2f}")
    print(f"  Max drawdown:   {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Trades:         {metrics.get('n_trades', 0)}")
    if per_sym:
        print(f"  Per-symbol PnL:")
        for sym, pnl in sorted(per_sym.items(), key=lambda x: -x[1]):
            print(f"    {sym:8s} ${pnl:+.2f}")
    print(f"\n  Report saved: {report_path}")
    return 0


def main():
    args = parse_args()
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
