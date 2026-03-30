#!/usr/bin/env python3
"""C-accelerated parameter sweep v2 - extended windows including crash periods."""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dataclasses import replace

from binance_worksteal.strategy import WorkStealConfig, load_daily_bars
from binance_worksteal.csim.compat import assert_csim_compatible_configs
from binance_worksteal.backtest import FULL_UNIVERSE


GRID = {
    "dip_pct":          [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    "profit_target_pct":[0.08, 0.10, 0.12, 0.15, 0.20],
    "stop_loss_pct":    [0.05, 0.08, 0.10, 0.15],
    "max_positions":    [3, 5, 7],
    "max_hold_days":    [7, 14, 21],
    "trailing_stop_pct":[0.0, 0.03, 0.05],
    "sma_filter_period":[0, 20],
    "max_leverage":     [1.0, 2.0, 3.0],
}

BASE_CONFIG = WorkStealConfig(
    dip_pct=0.20,
    proximity_pct=0.03,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    max_positions=5,
    max_hold_days=14,
    lookback_days=20,
    maker_fee=0.001,
    fdusd_fee=0.0,
    initial_cash=10_000.0,
    trailing_stop_pct=0.03,
    reentry_cooldown_days=1,
    max_leverage=1.0,
    sma_filter_period=20,
    sma_check_method="current",
    margin_annual_rate=0.0625,
    max_drawdown_exit=0.25,
    risk_off_trigger_sma_period=0,
    risk_off_trigger_momentum_period=0,
)

MIN_TRADES = 5
SORTINO_CAP = 50.0


def _get_csim_batch_fn():
    try:
        from binance_worksteal.csim.fast_worksteal import run_worksteal_batch_fast
    except Exception as exc:
        raise RuntimeError("C simulator batch backend is unavailable for csim_sweep2.") from exc
    return run_worksteal_batch_fast


def generate_configs():
    keys = list(GRID.keys())
    vals = [GRID[k] for k in keys]
    configs = []
    for combo in product(*vals):
        kwargs = dict(zip(keys, combo))
        if kwargs["stop_loss_pct"] >= kwargs["profit_target_pct"]:
            continue
        if kwargs["trailing_stop_pct"] >= kwargs["profit_target_pct"]:
            continue
        configs.append(replace(BASE_CONFIG, **kwargs))
    assert_csim_compatible_configs(configs, context="csim_sweep2")
    return configs


def clip_sortino(s):
    return max(min(s, SORTINO_CAP), -SORTINO_CAP)


def run_sweep():
    t0 = time.time()
    run_worksteal_batch_fast = _get_csim_batch_fn()

    print("Loading data...")
    all_bars = load_daily_bars("trainingdata/train", FULL_UNIVERSE)
    print(f"Loaded {len(all_bars)} symbols in {time.time()-t0:.1f}s")

    latest = max(df["timestamp"].max() for df in all_bars.values())
    earliest = min(df["timestamp"].min() for df in all_bars.values())
    print(f"Data range: {earliest.date()} to {latest.date()}")

    configs = generate_configs()
    n_configs = len(configs)
    print(f"Generated {n_configs} configs")

    # Multiple windows including crash periods
    windows = []
    for name, days in [("30d", 30), ("60d", 60), ("90d", 90), ("180d", 180), ("365d", 365)]:
        end = str(latest.date())
        start = str((latest - pd.Timedelta(days=days)).date())
        windows.append((name, start, end))

    # Also add specific challenging periods
    # Dec 2025 crash through Jan 2026
    windows.append(("crash_dec_jan", "2025-12-01", "2026-01-31"))
    # Bull run Jun-Sep 2025
    windows.append(("bull_jun_sep", "2025-06-01", "2025-09-30"))

    print(f"Windows: {len(windows)}")
    for n, s, e in windows:
        print(f"  {n}: {s} to {e}")

    BATCH = 5000

    # Run each window
    window_data = {}
    for wname, wstart, wend in windows:
        print(f"\n--- {wname}: {wstart} to {wend} ---")
        tw0 = time.time()
        window_results = []
        for i in range(0, n_configs, BATCH):
            batch = configs[i:i+BATCH]
            results = run_worksteal_batch_fast(all_bars, batch, start_date=wstart, end_date=wend)
            window_results.extend(results)
        window_data[wname] = window_results
        # Quick stats
        rets = [r["total_return_pct"] for r in window_results]
        trades = [r["total_trades"] for r in window_results]
        print(f"  done {time.time()-tw0:.1f}s | ret: [{min(rets):.1f}%, {np.median(rets):.1f}%, {max(rets):.1f}%] | trades: [{min(trades)}, {np.median(trades):.0f}, {max(trades)}]")

    # Build per-config summary
    wnames = [w[0] for w in windows]
    rows = []

    for ci in range(n_configs):
        cfg = configs[ci]

        # Get results per window
        wd = {wn: window_data[wn][ci] for wn in wnames}

        # Require min trades in the 3 main windows
        if wd["30d"]["total_trades"] < MIN_TRADES:
            continue
        if wd["90d"]["total_trades"] < MIN_TRADES * 2:
            continue

        sortinos = {wn: clip_sortino(wd[wn]["sortino"]) for wn in wnames}
        returns = {wn: wd[wn]["total_return_pct"] for wn in wnames}
        dds = {wn: wd[wn]["max_drawdown_pct"] for wn in wnames}
        trades = {wn: wd[wn]["total_trades"] for wn in wnames}

        # Core metrics: mean across recent 3 windows
        core_windows = ["30d", "60d", "90d"]
        mean_sort_core = np.mean([sortinos[w] for w in core_windows])
        n_pos_core = sum(1 for w in core_windows if sortinos[w] > 0)

        # Extended metrics: include longer windows
        all_sortinos = [sortinos[w] for w in wnames]
        mean_sort_all = np.mean(all_sortinos)
        n_pos_all = sum(1 for s in all_sortinos if s > 0)

        # Safety: core_sortino * consistency / (1 + drawdown)
        worst_dd = min(dds[w] for w in wnames)
        safety_core = mean_sort_core * n_pos_core / (1.0 + abs(worst_dd / 100.0))
        safety_all = mean_sort_all * n_pos_all / (1.0 + abs(worst_dd / 100.0))

        row = {
            "dip_pct": cfg.dip_pct,
            "profit_target_pct": cfg.profit_target_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "max_positions": cfg.max_positions,
            "max_hold_days": cfg.max_hold_days,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "sma_filter_period": cfg.sma_filter_period,
            "max_leverage": cfg.max_leverage,
            "safety_core": round(safety_core, 4),
            "safety_all": round(safety_all, 4),
            "mean_sort_core": round(mean_sort_core, 4),
            "mean_sort_all": round(mean_sort_all, 4),
            "n_pos_core": n_pos_core,
            "n_pos_all": n_pos_all,
            "worst_dd_pct": round(worst_dd, 4),
        }

        for wn in wnames:
            row[f"sort_{wn}"] = round(sortinos[wn], 4)
            row[f"ret_{wn}"] = round(returns[wn], 4)
            row[f"dd_{wn}"] = round(dds[wn], 4)
            row[f"trades_{wn}"] = trades[wn]
            row[f"wr_{wn}"] = round(wd[wn]["win_rate"], 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} configs passed filters (of {n_configs} total)")

    # Sort by safety_all (accounts for crash and bull periods too)
    df = df.sort_values("safety_all", ascending=False)

    top50 = df.head(50)
    outpath = Path("binance_worksteal/csim_sweep2_results.csv")
    top50.to_csv(outpath, index=False)
    print(f"Saved top-50 to {outpath}")

    df.to_csv("binance_worksteal/csim_sweep2_full.csv", index=False)

    # Display results grouped by leverage
    for lev in [1.0, 2.0, 3.0]:
        fl = df[df["max_leverage"] == lev].sort_values("safety_all", ascending=False)
        # Deduplicate by core params
        dedup_cols = ["dip_pct", "profit_target_pct", "stop_loss_pct", "trailing_stop_pct", "sma_filter_period"]
        fl_dedup = fl.drop_duplicates(subset=dedup_cols)

        print(f"\n{'='*140}")
        print(f"TOP 15 at {lev:.0f}x LEVERAGE (ranked by safety_all, deduplicated by core signal params)")
        print(f"{'='*140}")
        show_cols = [
            "dip_pct", "profit_target_pct", "stop_loss_pct", "max_positions", "max_hold_days",
            "trailing_stop_pct", "sma_filter_period",
            "safety_all", "mean_sort_all", "n_pos_all", "worst_dd_pct",
            "sort_30d", "sort_90d", "sort_180d", "sort_365d", "sort_crash_dec_jan", "sort_bull_jun_sep",
            "ret_30d", "ret_90d", "ret_180d", "ret_365d", "ret_crash_dec_jan",
        ]
        existing_cols = [c for c in show_cols if c in fl_dedup.columns]
        with pd.option_context('display.max_columns', 30, 'display.width', 300):
            print(fl_dedup.head(15)[existing_cols].to_string(index=False))

    # Current deployed comparison
    deployed_mask = (
        (df["dip_pct"] == 0.20) &
        (df["profit_target_pct"] == 0.15) &
        (df["stop_loss_pct"] == 0.10) &
        (df["max_positions"] == 5) &
        (df["max_hold_days"] == 14) &
        (df["trailing_stop_pct"] == 0.03) &
        (df["sma_filter_period"] == 20) &
        (df["max_leverage"] == 1.0)
    )
    deployed = df[deployed_mask]
    print(f"\n{'='*140}")
    print(f"CURRENT DEPLOYED CONFIG")
    print(f"{'='*140}")
    if len(deployed) > 0:
        all_cols = [c for c in df.columns if not c.startswith("wr_")]
        print(deployed[all_cols].to_string(index=False))
    else:
        print("(not in filtered results)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return df


if __name__ == "__main__":
    run_sweep()
