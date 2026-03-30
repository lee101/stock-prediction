#!/usr/bin/env python3
"""C-accelerated parameter sweep for work-stealing strategy."""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dataclasses import replace

from binance_worksteal.strategy import WorkStealConfig, load_daily_bars
from binance_worksteal.csim.compat import assert_csim_compatible_configs
from binance_worksteal.backtest import FULL_UNIVERSE


GRID = {
    "dip_pct":          [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    "profit_target_pct":[0.08, 0.10, 0.12, 0.15, 0.20],
    "stop_loss_pct":    [0.05, 0.08, 0.10, 0.15],
    "max_positions":    [3, 5, 7, 10],
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

MIN_TRADES_30D = 5
MIN_TRADES_60D = 8
MIN_TRADES_90D = 10
SORTINO_CAP = 50.0  # cap to avoid degenerate high values from low-trade configs


def _get_csim_batch_fn():
    try:
        from binance_worksteal.csim.fast_worksteal import run_worksteal_batch_fast
    except Exception as exc:
        raise RuntimeError("C simulator batch backend is unavailable for csim_sweep.") from exc
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
    assert_csim_compatible_configs(configs, context="csim_sweep")
    return configs


def compute_windows(all_bars):
    latest = max(df["timestamp"].max() for df in all_bars.values())
    windows = []
    for name, days in [("30d", 30), ("60d", 60), ("90d", 90)]:
        end = str(latest.date())
        start = str((latest - pd.Timedelta(days=days)).date())
        windows.append((name, start, end))
    return windows


def clip_sortino(s):
    return max(min(s, SORTINO_CAP), -SORTINO_CAP)


def run_sweep():
    t0 = time.time()
    run_worksteal_batch_fast = _get_csim_batch_fn()

    print("Loading data...")
    all_bars = load_daily_bars("trainingdata/train", FULL_UNIVERSE)
    print(f"Loaded {len(all_bars)} symbols in {time.time()-t0:.1f}s")

    configs = generate_configs()
    n_configs = len(configs)
    print(f"Generated {n_configs} configs (after filtering nonsensical combos)")

    windows = compute_windows(all_bars)
    print(f"Windows: {[(n,s,e) for n,s,e in windows]}")

    BATCH = 5000

    # Run each window and store results keyed by (config_index, window_name)
    window_data = {}  # wname -> list of result dicts

    for wname, wstart, wend in windows:
        print(f"\n--- Window {wname}: {wstart} to {wend} ---")
        tw0 = time.time()
        window_results = []

        for i in range(0, n_configs, BATCH):
            batch = configs[i:i+BATCH]
            results = run_worksteal_batch_fast(all_bars, batch, start_date=wstart, end_date=wend)
            window_results.extend(results)
            done = min(i + BATCH, n_configs)
            if done % 10000 == 0 or done >= n_configs:
                print(f"  {done}/{n_configs} done ({time.time()-tw0:.1f}s)")

        window_data[wname] = window_results
        print(f"  {wname} done in {time.time()-tw0:.1f}s")

    # Build per-config summary
    rows = []

    for ci in range(n_configs):
        cfg = configs[ci]
        r30 = window_data["30d"][ci]
        r60 = window_data["60d"][ci]
        r90 = window_data["90d"][ci]

        # Filter: require minimum trades in each window
        if r30["total_trades"] < MIN_TRADES_30D:
            continue
        if r60["total_trades"] < MIN_TRADES_60D:
            continue
        if r90["total_trades"] < MIN_TRADES_90D:
            continue

        s30 = clip_sortino(r30["sortino"])
        s60 = clip_sortino(r60["sortino"])
        s90 = clip_sortino(r90["sortino"])
        mean_sort = (s30 + s60 + s90) / 3.0

        n_positive_windows = sum(1 for s in [s30, s60, s90] if s > 0)

        mean_ret = (r30["total_return_pct"] + r60["total_return_pct"] + r90["total_return_pct"]) / 3.0
        worst_dd = min(r30["max_drawdown_pct"], r60["max_drawdown_pct"], r90["max_drawdown_pct"])
        mean_trades = (r30["total_trades"] + r60["total_trades"] + r90["total_trades"]) / 3.0
        mean_wr = (r30["win_rate"] + r60["win_rate"] + r90["win_rate"]) / 3.0

        # Safety score: sortino * (positive windows) / (1 + |worst_dd|)
        safety = mean_sort * n_positive_windows / (1.0 + abs(worst_dd / 100.0))

        rows.append({
            "dip_pct": cfg.dip_pct,
            "profit_target_pct": cfg.profit_target_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "max_positions": cfg.max_positions,
            "max_hold_days": cfg.max_hold_days,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "sma_filter_period": cfg.sma_filter_period,
            "max_leverage": cfg.max_leverage,
            "mean_sortino": round(mean_sort, 4),
            "mean_return_pct": round(mean_ret, 4),
            "worst_dd_pct": round(worst_dd, 4),
            "mean_trades": round(mean_trades, 1),
            "mean_win_rate": round(mean_wr, 2),
            "n_pos_windows": n_positive_windows,
            "safety": round(safety, 4),
            "sort_30d": round(s30, 4),
            "sort_60d": round(s60, 4),
            "sort_90d": round(s90, 4),
            "ret_30d": round(r30["total_return_pct"], 4),
            "ret_60d": round(r60["total_return_pct"], 4),
            "ret_90d": round(r90["total_return_pct"], 4),
            "dd_30d": round(r30["max_drawdown_pct"], 4),
            "dd_60d": round(r60["max_drawdown_pct"], 4),
            "dd_90d": round(r90["max_drawdown_pct"], 4),
            "trades_30d": r30["total_trades"],
            "trades_60d": r60["total_trades"],
            "trades_90d": r90["total_trades"],
        })

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} configs passed minimum trade filters (of {n_configs} total)")

    if len(df) == 0:
        print("No configs passed filters! Lowering thresholds...")
        # Fallback: just show all
        return

    # Sort by safety score (composite of sortino, consistency, drawdown)
    df = df.sort_values("safety", ascending=False)

    top50 = df.head(50)
    outpath = Path("binance_worksteal/csim_sweep_results.csv")
    top50.to_csv(outpath, index=False)
    print(f"Saved top-50 to {outpath}")

    # Full results
    df.to_csv("binance_worksteal/csim_sweep_full.csv", index=False)
    print(f"Saved all {len(df)} passing configs to binance_worksteal/csim_sweep_full.csv")

    print(f"\n{'='*120}")
    print(f"TOP 25 CONFIGS (ranked by safety = sortino * consistency / drawdown)")
    print(f"{'='*120}")
    display_cols = [
        "dip_pct", "profit_target_pct", "stop_loss_pct", "max_positions",
        "max_hold_days", "trailing_stop_pct", "sma_filter_period", "max_leverage",
        "safety", "mean_sortino", "mean_return_pct", "worst_dd_pct",
        "mean_trades", "mean_win_rate", "n_pos_windows",
        "sort_30d", "sort_60d", "sort_90d",
        "ret_30d", "ret_60d", "ret_90d",
    ]
    with pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 12):
        print(top50.head(25)[display_cols].to_string(index=False))

    # Also show top by mean_sortino for comparison
    df_sort = df.sort_values("mean_sortino", ascending=False)
    print(f"\n{'='*120}")
    print(f"TOP 10 by raw mean Sortino (for comparison)")
    print(f"{'='*120}")
    print(df_sort.head(10)[display_cols].to_string(index=False))

    # Show top configs that are robust (all 3 windows positive)
    robust = df[df["n_pos_windows"] == 3].sort_values("safety", ascending=False)
    print(f"\n{'='*120}")
    print(f"TOP 15 ROBUST CONFIGS (positive Sortino in ALL 3 windows)")
    print(f"({len(robust)} total robust configs)")
    print(f"{'='*120}")
    if len(robust) > 0:
        print(robust.head(15)[display_cols].to_string(index=False))
    else:
        print("None found!")

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
    print(f"\n--- Current deployed config ---")
    if len(deployed) > 0:
        print(deployed[display_cols].to_string(index=False))
    else:
        print("(not in filtered results -- too few trades?)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Total configs evaluated: {n_configs}")
    print(f"Total sim runs: {n_configs * len(windows)}")

    return df


if __name__ == "__main__":
    run_sweep()
