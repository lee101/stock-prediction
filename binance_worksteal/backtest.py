#!/usr/bin/env python3
"""Backtest the work-stealing dip-buying strategy on daily crypto data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.strategy import (
    WorkStealConfig, load_daily_bars, run_worksteal_backtest, print_results,
)


FULL_UNIVERSE = [
    "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AVAXUSD", "LINKUSD",
    "AAVEUSD", "LTCUSD", "XRPUSD", "DOTUSD", "UNIUSD", "NEARUSD",
    "APTUSD", "ICPUSD", "SHIBUSD", "ADAUSD", "FILUSD", "ARBUSD",
    "OPUSD", "INJUSD", "SUIUSD", "TIAUSD", "SEIUSD", "ATOMUSD",
    "ALGOUSD", "BCHUSD", "BNBUSD", "TRXUSD", "PEPEUSD", "MATICUSD",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override symbol list")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days", type=int, default=30, help="Backtest last N days if no start/end")
    parser.add_argument("--dip-pct", type=float, default=0.10)
    parser.add_argument("--proximity-pct", type=float, default=0.005)
    parser.add_argument("--profit-target", type=float, default=0.05)
    parser.add_argument("--stop-loss", type=float, default=0.08)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ref-method", choices=["high", "sma", "close"], default="high")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--trailing-stop", type=float, default=0.0)
    parser.add_argument("--cooldown", type=int, default=1)
    args = parser.parse_args()

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading data for {len(symbols)} symbols from {args.data_dir}")

    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols with data")

    if not all_bars:
        print("ERROR: No data loaded")
        return 1

    # Auto-compute date range if not specified
    if not args.start_date and not args.end_date:
        latest = max(df["timestamp"].max() for df in all_bars.values())
        import pandas as pd
        args.end_date = str(latest.date())
        args.start_date = str((latest - pd.Timedelta(days=args.days)).date())
        print(f"Auto date range: {args.start_date} to {args.end_date}")

    config = WorkStealConfig(
        dip_pct=args.dip_pct,
        proximity_pct=args.proximity_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
        max_hold_days=args.max_hold_days,
        lookback_days=args.lookback,
        ref_price_method=args.ref_method,
        maker_fee=args.fee,
        initial_cash=args.cash,
        trailing_stop_pct=args.trailing_stop,
        reentry_cooldown_days=args.cooldown,
    )

    print(f"\nConfig: dip={config.dip_pct:.0%} prox={config.proximity_pct:.1%} "
          f"tp={config.profit_target_pct:.0%} sl={config.stop_loss_pct:.0%} "
          f"maxpos={config.max_positions} maxhold={config.max_hold_days}d")

    equity_df, trades, metrics = run_worksteal_backtest(
        all_bars, config,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print_results(equity_df, trades, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
