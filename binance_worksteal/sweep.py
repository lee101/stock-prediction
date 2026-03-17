#!/usr/bin/env python3
"""Auto-research sweep for work-stealing strategy parameters."""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig, load_daily_bars, run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE


SWEEP_GRID = {
    "dip_pct": [0.05, 0.07, 0.10, 0.15, 0.20],
    "proximity_pct": [0.005, 0.01, 0.02, 0.03],
    "profit_target_pct": [0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "stop_loss_pct": [0.05, 0.08, 0.10, 0.15, 0.20],
    "max_positions": [3, 5, 7, 10],
    "max_hold_days": [7, 14, 21, 30],
    "lookback_days": [10, 20, 30],
    "ref_price_method": ["high", "sma", "close"],
    "max_leverage": [1.0, 2.0, 3.0, 5.0],
    "enable_shorts": [False, True],
    "trailing_stop_pct": [0.0, 0.03, 0.05],
    "sma_filter_period": [0, 20, 50],
    "market_breadth_filter": [0.0, 0.6, 0.7, 0.8],
}


def run_sweep(
    all_bars, start_date, end_date, output_csv,
    max_trials=500, cash=10000.0,
):
    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    all_combos = list(itertools.product(*values))

    random.seed(42)
    if len(all_combos) > max_trials:
        combos = random.sample(all_combos, max_trials)
    else:
        combos = all_combos

    print(f"Sweep: {len(combos)} configs (from {len(all_combos)} total)")

    results = []
    best_sortino = -999
    best_config = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        config = WorkStealConfig(initial_cash=cash, **params)

        try:
            equity_df, trades, metrics = run_worksteal_backtest(
                {k: v.copy() for k, v in all_bars.items()},
                config, start_date=start_date, end_date=end_date,
            )
        except Exception:
            continue

        if not metrics:
            continue

        row = {**params, **metrics, "n_trades": len(trades)}
        results.append(row)

        sortino = metrics.get("sortino", 0)
        ret = metrics.get("total_return_pct", 0)
        dd = metrics.get("max_drawdown_pct", 0)
        wr = metrics.get("win_rate", 0)

        if sortino > best_sortino:
            best_sortino = sortino
            best_config = params
            marker = " ***BEST***"
        else:
            marker = ""

        if (i + 1) % 25 == 0 or marker:
            lev = params["max_leverage"]
            sh = "S" if params["enable_shorts"] else "L"
            sma = params["sma_filter_period"]
            mb = params["market_breadth_filter"]
            print(f"  [{i+1}/{len(combos)}] sort={sortino:6.2f} ret={ret:6.2f}% dd={dd:6.2f}% "
                  f"wr={wr:5.1f}% tr={len(trades):3d} "
                  f"dip={params['dip_pct']:.0%} tp={params['profit_target_pct']:.0%} "
                  f"sl={params['stop_loss_pct']:.0%} pos={params['max_positions']} "
                  f"lev={lev:.0f}x {sh} sma={sma} mb={mb}{marker}")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("sortino", ascending=False)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(results)} results to {output_csv}")

        print(f"\nTop 20 by Sortino:")
        cols = ["sortino", "total_return_pct", "max_drawdown_pct", "win_rate", "n_trades",
                "dip_pct", "proximity_pct", "profit_target_pct", "stop_loss_pct",
                "max_positions", "max_hold_days", "lookback_days", "ref_price_method",
                "max_leverage", "enable_shorts", "trailing_stop_pct",
                "sma_filter_period", "market_breadth_filter"]
        for _, r in df.head(20).iterrows():
            sh = "Y" if r.get("enable_shorts", False) else "N"
            print(f"  sort={r['sortino']:6.2f} ret={r['total_return_pct']:7.2f}% "
                  f"dd={r['max_drawdown_pct']:7.2f}% wr={r.get('win_rate',0):5.1f}% "
                  f"tr={r['n_trades']:4.0f} | "
                  f"dip={r['dip_pct']:.0%} tp={r['profit_target_pct']:.0%} "
                  f"sl={r['stop_loss_pct']:.0%} pos={r['max_positions']:.0f} "
                  f"hold={r['max_hold_days']:.0f} look={r['lookback_days']:.0f} "
                  f"ref={r['ref_price_method']} lev={r['max_leverage']:.0f}x {sh} "
                  f"sma={r.get('sma_filter_period',0):.0f} "
                  f"mb={r.get('market_breadth_filter',0):.1f}")

    if best_config:
        print(f"\nBest config (Sortino={best_sortino:.2f}):")
        for k, v in best_config.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--output", default="binance_worksteal/sweep_results.csv")
    args = parser.parse_args()

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading {len(symbols)} symbols from {args.data_dir}")
    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols")

    if not all_bars:
        print("ERROR: No data")
        return 1

    if not args.start_date:
        latest = max(df["timestamp"].max() for df in all_bars.values())
        args.end_date = str(latest.date())
        args.start_date = str((latest - pd.Timedelta(days=args.days)).date())
    print(f"Date range: {args.start_date} to {args.end_date}")

    run_sweep(
        all_bars, args.start_date, args.end_date, args.output,
        max_trials=args.max_trials, cash=args.cash,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
