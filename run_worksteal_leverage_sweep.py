#!/usr/bin/env python3
"""Sweep leverage + stop-loss for the work-stealing daily strategy.

Fixes the best known config and varies only leverage and stop-loss to find
the Pareto-optimal leverage level across multiple time windows.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.strategy import (
    MARGIN_ANNUAL_RATE, WorkStealConfig, load_daily_bars, run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE

LEVERAGE_GRID = [1.0, 2.0, 3.0, 5.0]
STOP_LOSS_GRID = [0.05, 0.08, 0.10]

BASE_CONFIG = dict(
    dip_pct=0.20,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    sma_filter_period=20,
    trailing_stop_pct=0.03,
    max_positions=5,
    max_hold_days=14,
    lookback_days=20,
    ref_price_method="high",
    proximity_pct=0.03,
    sma_check_method="pre_dip",
    maker_fee=0.001,
    initial_cash=10_000.0,
    reentry_cooldown_days=1,
    margin_annual_rate=MARGIN_ANNUAL_RATE,
)

WINDOWS = {
    "30d": 30,
    "60d": 60,
    "105d": 105,
}


def make_config(leverage: float, stop_loss: float) -> WorkStealConfig:
    return WorkStealConfig(**{**BASE_CONFIG, "max_leverage": leverage, "stop_loss_pct": stop_loss})


def compute_windows(all_bars: dict[str, pd.DataFrame]) -> dict[str, tuple[str, str]]:
    latest = max(df["timestamp"].max() for df in all_bars.values())
    end = str(latest.date())
    result = {}
    for name, days in WINDOWS.items():
        start = str((latest - pd.Timedelta(days=days)).date())
        result[name] = (start, end)
    return result


def run_single(all_bars, config, start_date, end_date):
    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    try:
        equity_df, trades, metrics = run_worksteal_backtest(
            bars_copy, config, start_date=start_date, end_date=end_date,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    if not metrics:
        return None
    exits = [t for t in trades if t.side in ("sell", "cover")]
    return {
        "return_pct": metrics.get("total_return_pct", 0),
        "sortino": metrics.get("sortino", 0),
        "max_dd_pct": metrics.get("max_drawdown_pct", 0),
        "num_trades": len(exits),
        "win_rate": metrics.get("win_rate", 0),
        "sharpe": metrics.get("sharpe", 0),
        "final_equity": metrics.get("final_equity", 0),
    }


def run_sweep(all_bars, windows, dry_run=False):
    if dry_run:
        latest = max(df["timestamp"].max() for df in all_bars.values())
        end = str(latest.date())
        start = str((latest - pd.Timedelta(days=7)).date())
        windows = {"7d_dryrun": (start, end)}
        leverage_grid = [1.0, 2.0]
        sl_grid = [0.10]
    else:
        leverage_grid = LEVERAGE_GRID
        sl_grid = STOP_LOSS_GRID

    combos = list(itertools.product(leverage_grid, sl_grid))
    total = len(combos) * len(windows)
    print(f"Sweep: {len(combos)} configs x {len(windows)} windows = {total} runs")

    results = []
    i = 0
    for leverage, stop_loss in combos:
        for wname, (start, end) in windows.items():
            i += 1
            config = make_config(leverage, stop_loss)
            print(f"  [{i}/{total}] lev={leverage:.0f}x sl={stop_loss:.0%} window={wname} ...", end="", flush=True)
            m = run_single(all_bars, config, start, end)
            if m is None:
                print(" SKIP")
                continue
            row = {
                "leverage": leverage,
                "stop_loss": stop_loss,
                "window": wname,
                **m,
            }
            results.append(row)
            print(f" ret={m['return_pct']:+.2f}% sort={m['sortino']:.2f} dd={m['max_dd_pct']:.2f}% "
                  f"trades={m['num_trades']} wr={m['win_rate']:.1f}%")

    return results


def print_summary(results) -> pd.DataFrame | None:
    if not results:
        print("No results.")
        return None
    df = pd.DataFrame(results)
    df = df.sort_values("sortino", ascending=False)
    print(f"\n{'='*100}")
    print(f"LEVERAGE SWEEP RESULTS (sorted by Sortino)")
    print(f"{'='*100}")
    print(f"{'leverage':>8} {'stop_loss':>9} {'window':>10} {'return%':>8} {'sortino':>8} "
          f"{'max_dd%':>8} {'trades':>7} {'win_rate':>8}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{r['leverage']:>8.1f} {r['stop_loss']:>9.2f} {r['window']:>10} "
              f"{r['return_pct']:>+7.2f}% {r['sortino']:>8.2f} {r['max_dd_pct']:>7.2f}% "
              f"{r['num_trades']:>7.0f} {r['win_rate']:>7.1f}%")

    print(f"\n--- Pareto frontier (best Sortino per leverage level) ---")
    for lev in sorted(df["leverage"].unique()):
        sub = df[df["leverage"] == lev]
        best = sub.loc[sub["sortino"].idxmax()]
        print(f"  {lev:.0f}x: sort={best['sortino']:.2f} ret={best['return_pct']:+.2f}% "
              f"dd={best['max_dd_pct']:.2f}% sl={best['stop_loss']:.0%} window={best['window']}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Work-stealing leverage sweep")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--output", default="binance_worksteal/leverage_sweep_results.csv")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 1 window, 2 leverages")
    args = parser.parse_args()

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading {len(symbols)} symbols from {args.data_dir}")
    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols")

    if not all_bars:
        print("ERROR: No data loaded")
        return 1

    windows = compute_windows(all_bars)
    if not args.dry_run:
        for wname, (s, e) in windows.items():
            print(f"  {wname}: {s} to {e}")

    results = run_sweep(all_bars, windows, dry_run=args.dry_run)
    df = print_summary(results)

    if df is not None and not df.empty:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved {len(df)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
