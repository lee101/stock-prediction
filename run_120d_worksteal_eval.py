#!/usr/bin/env python3
"""Run 120-day backtest of the deployed work-steal dip-buying strategy.

Wraps binance_worksteal/backtest_gemini.py with deployed config:
  - 30-symbol universe
  - dip=20% tp=15% sl=10% sma=20 trail=3% maxpos=5 maxhold=14d
  - ~120 days: 2025-11-15 to 2026-03-19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_START = "2025-11-15"
DEFAULT_END = "2026-03-19"
DEFAULT_DATA_DIR = "trainingdata/train"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_REPORT = "reports/120d_worksteal_eval.json"

DEPLOYED_CONFIG = {
    "dip_pct": 0.20,
    "proximity_pct": 0.02,
    "profit_target_pct": 0.15,
    "stop_loss_pct": 0.10,
    "max_positions": 5,
    "max_hold_days": 14,
    "lookback_days": 20,
    "ref_price_method": "high",
    "sma_filter_period": 20,
    "trailing_stop_pct": 0.03,
    "max_drawdown_exit": 0.25,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="120-day work-steal backtest eval")
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--rule-only", action="store_true", help="Skip Gemini, rule-only backtest")
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print config without running")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def run_eval(args):
    report_path = REPO / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Work-steal 120d eval")
    print(f"  period:    {args.start_date} -> {args.end_date}")
    print(f"  data:      {args.data_dir}")
    print(f"  rule-only: {args.rule_only}")
    print(f"  model:     {args.model}")
    print(f"  report:    {report_path}")
    print(f"  config:    {DEPLOYED_CONFIG}")
    print()

    if args.dry_run:
        print("DRY RUN -- would run backtest with above config")
        return 0

    sys.path.insert(0, str(REPO))

    from binance_worksteal.backtest import FULL_UNIVERSE
    from binance_worksteal.strategy import (
        WorkStealConfig, load_daily_bars, run_worksteal_backtest,
    )

    data_dir = REPO / args.data_dir
    all_bars = load_daily_bars(str(data_dir), FULL_UNIVERSE)
    print(f"Loaded {len(all_bars)} symbols")

    config = WorkStealConfig(**DEPLOYED_CONFIG)

    if args.rule_only:
        equity_df, trades, metrics = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=args.start_date, end_date=args.end_date,
        )
        mode_label = "RULE-ONLY"
    else:
        from binance_worksteal.backtest_gemini import run_gemini_backtest
        equity_df, trades, metrics = run_gemini_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=args.start_date, end_date=args.end_date,
            model=args.model, use_cache=not args.no_cache,
        )
        mode_label = f"GEMINI ({args.model})"

    # per-symbol PnL
    per_sym = {}
    for t in trades:
        if t.side == "sell":
            pnl = t.pnl if hasattr(t, "pnl") else 0.0
            per_sym[t.symbol] = per_sym.get(t.symbol, 0.0) + pnl

    result = {
        "config": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "mode": "rule_only" if args.rule_only else "gemini",
            "model": args.model if not args.rule_only else None,
            "universe_size": len(FULL_UNIVERSE),
            **DEPLOYED_CONFIG,
        },
        "metrics": metrics,
        "per_symbol_pnl": per_sym,
    }

    report_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"120-DAY WORK-STEAL EVAL ({mode_label})")
    print(f"{'='*60}")
    print(f"  Total return:   {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"  Sortino:        {metrics.get('sortino', 0):.2f}")
    print(f"  Sharpe:         {metrics.get('sharpe', 0):.2f}")
    print(f"  Max drawdown:   {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Win rate:       {metrics.get('win_rate', 0):.1f}%")
    print(f"  Trades:         {metrics.get('n_trades', 0)}")
    print(f"  Final equity:   ${metrics.get('final_equity', 0):.0f}")
    if per_sym:
        print(f"  Per-symbol PnL:")
        for sym, pnl in sorted(per_sym.items(), key=lambda x: -x[1]):
            print(f"    {sym:10s} ${pnl:+.2f}")
    print(f"\n  Report saved: {report_path}")
    return 0


def main():
    args = parse_args()
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
