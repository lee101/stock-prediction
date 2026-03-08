#!/usr/bin/env python3
"""Thorough sweep of meta-switcher lookback hyperparameters."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceleveragesui.sim_meta_switcher import (
    load_model_and_actions, per_bar_sim, compute_stats,
    DOGE_CKPT, AAVE_CKPT, MARGIN_HOURLY_RATE,
)


def meta_sim_v2(doge_df, aave_df, max_leverage=2.0, lookback_hours=168, metric="sortino"):
    """Enhanced meta-sim with configurable selection metric."""
    common_ts = set(doge_df["timestamp"]) & set(aave_df["timestamp"])
    d = doge_df[doge_df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    a = aave_df[aave_df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    n = len(d)

    d_eq = d["equity"].values
    a_eq = a["equity"].values
    d_in = d["in_position"].values
    a_in = a["in_position"].values

    d_rel = np.ones(n)
    a_rel = np.ones(n)
    for i in range(1, n):
        d_rel[i] = d_eq[i] / (d_eq[i-1] + 1e-10) if d_eq[i-1] > 0 else 1.0
        a_rel[i] = a_eq[i] / (a_eq[i-1] + 1e-10) if a_eq[i-1] > 0 else 1.0

    def score(eq_arr, end_idx, window):
        start = max(0, end_idx - window)
        if end_idx - start < 6:
            return 0.0
        sub = eq_arr[start:end_idx+1]
        rets = np.diff(sub) / (np.abs(sub[:-1]) + 1e-10)
        if len(rets) < 2:
            return 0.0
        if metric == "sortino":
            neg = rets[rets < 0]
            dd = np.std(neg) if len(neg) > 1 else 1e-10
            return float(np.mean(rets)) / (dd + 1e-10)
        elif metric == "return":
            return float(sub[-1] / sub[0] - 1)
        elif metric == "sharpe":
            return float(np.mean(rets)) / (np.std(rets) + 1e-10)
        elif metric == "calmar":
            peak = np.maximum.accumulate(sub)
            dd = ((sub - peak) / (peak + 1e-10)).min()
            ret = sub[-1] / sub[0] - 1
            return ret / (abs(dd) + 1e-10)
        return 0.0

    active = None
    meta_equity = [10000.0]
    switches = 0

    for i in range(n):
        if active is None:
            ds = score(d_eq, i, lookback_hours)
            as_ = score(a_eq, i, lookback_hours)
            active = "aave" if as_ > ds else "doge"

        rel = d_rel[i] if active == "doge" else a_rel[i]
        meta_equity.append(meta_equity[-1] * rel)

        in_pos = d_in[i] if active == "doge" else a_in[i]
        was_in = (d_in[i-1] if active == "doge" else a_in[i-1]) if i > 0 else False

        if was_in and not in_pos:
            ds = score(d_eq, i, lookback_hours)
            as_ = score(a_eq, i, lookback_hours)
            new = "aave" if as_ > ds else "doge"
            if new != active:
                switches += 1
                active = new

    return np.array(meta_equity[1:]), switches


def main():
    logger.info("Loading models and running per-bar sims...")

    lookbacks = [6, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720]
    metrics = ["sortino", "return", "sharpe", "calmar"]
    test_windows = [30, 60, 90]

    all_results = []

    for test_d in test_windows:
        logger.info(f"\n{'='*70}\n{test_d}d TEST WINDOW\n{'='*70}")
        d_bars, d_act = load_model_and_actions(DOGE_CKPT, "DOGEUSD", test_days=test_d)
        a_bars, a_act = load_model_and_actions(AAVE_CKPT, "AAVEUSD", test_days=test_d)
        dp = per_bar_sim(d_bars, d_act, 0.001, 2.0, "DOGEUSD")
        ap = per_bar_sim(a_bars, a_act, 0.001, 2.0, "AAVEUSD")

        ds = compute_stats(dp["equity"].values, "DOGE")
        as_ = compute_stats(ap["equity"].values, "AAVE")
        logger.info(f"  DOGE:  Sort={ds['sortino']:>6.2f} Ret={ds['total_return']:>+8.1f}% DD={ds['max_dd']:>6.1f}%")
        logger.info(f"  AAVE:  Sort={as_['sortino']:>6.2f} Ret={as_['total_return']:>+8.1f}% DD={as_['max_dd']:>6.1f}%")

        best_sort = -999
        best_cfg = ""

        for m in metrics:
            for lb in lookbacks:
                if lb > test_d * 24 * 0.8:
                    continue
                eq, sw = meta_sim_v2(dp, ap, 2.0, lb, m)
                s = compute_stats(eq, f"{m}_{lb}h")
                all_results.append({
                    "test_d": test_d, "metric": m, "lookback": lb,
                    "sortino": s["sortino"], "return": s["total_return"],
                    "max_dd": s["max_dd"], "switches": sw,
                })
                if s["sortino"] > best_sort:
                    best_sort = s["sortino"]
                    best_cfg = f"{m}_{lb}h"

        # Print sorted by sortino for this window
        window_results = [r for r in all_results if r["test_d"] == test_d]
        window_results.sort(key=lambda x: x["sortino"], reverse=True)
        logger.info(f"\n  {'Metric':<10} {'LB':>5} {'Sort':>7} {'Ret':>9} {'DD':>7} {'Sw':>4}")
        logger.info(f"  {'-'*50}")
        for r in window_results[:15]:
            logger.info(f"  {r['metric']:<10} {r['lookback']:>4}h {r['sortino']:>7.2f} {r['return']:>+8.1f}% {r['max_dd']:>6.1f}% {r['switches']:>4}")
        logger.info(f"  ... DOGE baseline: Sort={ds['sortino']:.2f} Ret={ds['total_return']:+.1f}% DD={ds['max_dd']:.1f}%")
        logger.info(f"  ... AAVE baseline: Sort={as_['sortino']:.2f} Ret={as_['total_return']:+.1f}% DD={as_['max_dd']:.1f}%")

    # Cross-window consistency: which configs are good at ALL windows?
    logger.info(f"\n{'='*70}\nCROSS-WINDOW CONSISTENCY\n{'='*70}")
    from collections import defaultdict
    cfg_scores = defaultdict(list)
    for r in all_results:
        key = f"{r['metric']}_{r['lookback']}h"
        cfg_scores[key].append(r["sortino"])

    consistent = []
    for key, scores in cfg_scores.items():
        if len(scores) >= 3:
            consistent.append((key, min(scores), np.mean(scores), max(scores)))
    consistent.sort(key=lambda x: x[1], reverse=True)  # sort by worst-case

    logger.info(f"  {'Config':<20} {'Min':>7} {'Mean':>7} {'Max':>7}")
    logger.info(f"  {'-'*50}")
    for cfg, mn, mean, mx in consistent[:15]:
        logger.info(f"  {cfg:<20} {mn:>7.2f} {mean:>7.2f} {mx:>7.2f}")


if __name__ == "__main__":
    main()
