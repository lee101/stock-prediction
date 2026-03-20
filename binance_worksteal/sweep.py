#!/usr/bin/env python3
"""Auto-research sweep for work-stealing strategy parameters.

Supports:
- Multi-window evaluation for robustness (worst-case Sortino selection)
- Production-realistic fill model via --realistic flag
- Optional C simulator for 10x speedup via --use-csim
- Dated CSV output with full metrics
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig, load_daily_bars, run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE


SWEEP_GRID = {
    "dip_pct": [0.05, 0.07, 0.10, 0.15, 0.20],
    "proximity_pct": [0.01, 0.02, 0.03, 0.05],
    "profit_target_pct": [0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "stop_loss_pct": [0.05, 0.08, 0.10, 0.15, 0.20],
    "max_positions": [3, 5, 7, 10],
    "max_hold_days": [7, 14, 21, 30],
    "lookback_days": [10, 20, 30],
    "ref_price_method": ["high", "sma", "close"],
    "max_leverage": [1.0, 2.0, 3.0, 5.0],
    "enable_shorts": [False, True],
    "trailing_stop_pct": [0.0, 0.03, 0.05],
    "sma_filter_period": [0, 10, 20],
    "market_breadth_filter": [0.0, 0.6, 0.7, 0.8],
}


def build_windows(all_bars, window_days=60, n_windows=3):
    latest = max(df["timestamp"].max() for df in all_bars.values())
    windows = []
    for i in range(n_windows):
        end = latest - pd.Timedelta(days=i * window_days)
        start = end - pd.Timedelta(days=window_days)
        windows.append((str(start.date()), str(end.date())))
    return windows


def _try_load_csim():
    try:
        from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast
        return run_worksteal_backtest_fast
    except Exception:
        return None


def eval_config_single_window(all_bars, config, start_date, end_date, use_csim_fn=None):
    if use_csim_fn is not None:
        metrics = use_csim_fn(
            all_bars, config, start_date=start_date, end_date=end_date,
        )
        if not metrics:
            return None
        metrics["n_trades"] = metrics.get("total_trades", 0)
        return metrics
    try:
        equity_df, trades, metrics = run_worksteal_backtest(
            all_bars, config, start_date=start_date, end_date=end_date,
        )
    except Exception:
        return None
    if not metrics:
        return None
    metrics["n_trades"] = len(trades)
    return metrics


def eval_config_multi_window(all_bars, config, windows, use_csim_fn=None):
    window_metrics = []
    for start, end in windows:
        m = eval_config_single_window(all_bars, config, start, end, use_csim_fn)
        if m is None:
            return None
        window_metrics.append(m)

    sortinos = [m.get("sortino", 0) for m in window_metrics]
    returns = [m.get("total_return_pct", 0) for m in window_metrics]
    drawdowns = [m.get("max_drawdown_pct", 0) for m in window_metrics]
    n_trades_list = [m.get("n_trades", 0) for m in window_metrics]
    win_rates = [m.get("win_rate", 0) for m in window_metrics]

    combined = {
        "mean_sortino": float(np.mean(sortinos)),
        "min_sortino": float(np.min(sortinos)),
        "max_sortino": float(np.max(sortinos)),
        "mean_return_pct": float(np.mean(returns)),
        "min_return_pct": float(np.min(returns)),
        "max_return_pct": float(np.max(returns)),
        "mean_drawdown_pct": float(np.mean(drawdowns)),
        "worst_drawdown_pct": float(np.min(drawdowns)),
        "mean_n_trades": float(np.mean(n_trades_list)),
        "total_n_trades": int(np.sum(n_trades_list)),
        "mean_win_rate": float(np.mean(win_rates)),
        "n_windows": len(windows),
    }
    for i, m in enumerate(window_metrics):
        combined[f"w{i}_sortino"] = m.get("sortino", 0)
        combined[f"w{i}_return_pct"] = m.get("total_return_pct", 0)
        combined[f"w{i}_drawdown_pct"] = m.get("max_drawdown_pct", 0)
        combined[f"w{i}_n_trades"] = m.get("n_trades", 0)
    return combined


def run_sweep(
    all_bars, windows, output_csv,
    max_trials=500, cash=10000.0, realistic=False, use_csim=False,
):
    csim_fn = _try_load_csim() if use_csim else None
    if use_csim and csim_fn is None:
        print("WARN: --use-csim requested but C lib not available, falling back to Python")

    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    all_combos = list(itertools.product(*values))

    random.seed(42)
    if len(all_combos) > max_trials:
        combos = random.sample(all_combos, max_trials)
    else:
        combos = all_combos

    print(f"Sweep: {len(combos)} configs (from {len(all_combos)} total)")
    print(f"Windows: {len(windows)}")
    for i, (s, e) in enumerate(windows):
        print(f"  W{i}: {s} to {e}")
    if realistic:
        print("Mode: REALISTIC (strict fill, proximity filter)")

    results = []
    best_min_sortino = -999
    best_config = None
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        config = WorkStealConfig(initial_cash=cash, strict_fill=realistic, **params)

        multi = eval_config_multi_window(all_bars, config, windows, csim_fn)
        if multi is None:
            continue

        row = {**params, **multi}
        results.append(row)

        min_sort = multi["min_sortino"]
        mean_sort = multi["mean_sortino"]
        mean_ret = multi["mean_return_pct"]
        worst_dd = multi["worst_drawdown_pct"]
        mean_wr = multi["mean_win_rate"]
        total_tr = multi["total_n_trades"]

        if min_sort > best_min_sortino:
            best_min_sortino = min_sort
            best_config = params
            marker = " ***BEST***"
        else:
            marker = ""

        if (i + 1) % 25 == 0 or marker:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            lev = params["max_leverage"]
            sh = "S" if params["enable_shorts"] else "L"
            sma = params["sma_filter_period"]
            print(f"  [{i+1}/{len(combos)} {rate:.1f}/s] "
                  f"min_sort={min_sort:6.2f} mean_sort={mean_sort:6.2f} "
                  f"ret={mean_ret:6.2f}% dd={worst_dd:6.2f}% "
                  f"wr={mean_wr:5.1f}% tr={total_tr:3d} "
                  f"dip={params['dip_pct']:.0%} tp={params['profit_target_pct']:.0%} "
                  f"sl={params['stop_loss_pct']:.0%} pos={params['max_positions']} "
                  f"lev={lev:.0f}x {sh} sma={sma}{marker}")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("min_sortino", ascending=False)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(results)} results to {output_csv}")
        elapsed = time.time() - t0
        print(f"Total time: {elapsed:.1f}s ({len(results)/elapsed:.1f} configs/s)")

        print(f"\nTop 10 by WORST-CASE Sortino:")
        for rank, (_, r) in enumerate(df.head(10).iterrows(), 1):
            sh = "Y" if r.get("enable_shorts", False) else "N"
            print(f"  #{rank} min_sort={r['min_sortino']:6.2f} mean_sort={r['mean_sortino']:6.2f} "
                  f"ret={r['mean_return_pct']:7.2f}% dd={r['worst_drawdown_pct']:7.2f}% "
                  f"wr={r['mean_win_rate']:5.1f}% tr={r['total_n_trades']:4.0f} | "
                  f"dip={r['dip_pct']:.0%} tp={r['profit_target_pct']:.0%} "
                  f"sl={r['stop_loss_pct']:.0%} pos={r['max_positions']:.0f} "
                  f"hold={r['max_hold_days']:.0f} look={r['lookback_days']:.0f} "
                  f"ref={r['ref_price_method']} lev={r['max_leverage']:.0f}x {sh} "
                  f"sma={r.get('sma_filter_period',0):.0f} "
                  f"mb={r.get('market_breadth_filter',0):.1f}")
            for w in range(int(r.get("n_windows", 0))):
                ws = r.get(f"w{w}_sortino", 0)
                wr_ = r.get(f"w{w}_return_pct", 0)
                wd = r.get(f"w{w}_drawdown_pct", 0)
                wt = r.get(f"w{w}_n_trades", 0)
                print(f"       W{w}: sort={ws:6.2f} ret={wr_:6.2f}% dd={wd:6.2f}% tr={wt:.0f}")

    if best_config:
        print(f"\nRecommended production config (best worst-case Sortino={best_min_sortino:.2f}):")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
        print(f"\nRationale: selected by worst-case Sortino across {len(windows)} "
              f"non-overlapping {windows[0][0]}..{windows[-1][1]} windows. "
              f"This minimizes regime-dependent overfitting.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--realistic", action="store_true",
                        help="Strict fill model: only fill when low <= buy_target")
    parser.add_argument("--use-csim", action="store_true",
                        help="Use C simulator for ~10x speedup")
    args = parser.parse_args()

    if args.output is None:
        tag = datetime.now().strftime("%Y%m%d")
        mode = "_realistic" if args.realistic else ""
        args.output = f"binance_worksteal/sweep_results_{tag}{mode}.csv"

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading {len(symbols)} symbols from {args.data_dir}")
    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols")

    if not all_bars:
        print("ERROR: No data")
        return 1

    if args.start_date and args.end_date:
        windows = [(args.start_date, args.end_date)]
    else:
        windows = build_windows(all_bars, window_days=args.days, n_windows=args.n_windows)

    run_sweep(
        all_bars, windows, args.output,
        max_trials=args.n_trials, cash=args.cash,
        realistic=args.realistic, use_csim=args.use_csim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
