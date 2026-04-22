"""Validate the top combined candidate across multiple windows and regimes.

Top candidate from find_combined.py:
  - Weekend: sma_mult=1.0, mom_vs_sma_min=0.0, vol_max=0.03, top_k=1, kinds={weekend}
  - Weekday: sma_mult=1.05 (or any), mom_vs_sma_min=0.0, vol_max=0.02, top_k=2, kinds={weekday, holiday}
  - Universe: 10 crypto (BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, LINK, LTC, DOT)
  - fee=10bps, max_gross=1.0

Test robustness across:
  1. 120d target window (2025-12-17 → 2026-04-15)
  2. Prior 120d (2025-08-19 → 2025-12-17)
  3. Prior 240d (2025-04-22 → 2025-12-17)
  4. Full OOS (2022-07-01 → 2026-04-15) — if hourly data goes that far back
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sessions import build_sessions
from ooh_backtest import (
    build_session_panel, apply_signal, session_pnl_series,
    summarize, DEFAULT_SYMBOLS,
)
from find_combined import combine_picks

REPO = Path(__file__).resolve().parent.parent

WEEKEND_CFG = dict(sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
                    vol_max=0.03, top_k=1, kinds={"weekend"})
WEEKDAY_CFG = dict(sma_mult=1.05, mom7_min=0.0, mom_vs_sma_min=0.0,
                   vol_max=0.02, top_k=2, kinds={"weekday", "holiday"})


def run_window(start: str, end: str, *, symbols=DEFAULT_SYMBOLS,
               fee_bps=10.0, max_gross=1.0,
               weekend_cfg=WEEKEND_CFG, weekday_cfg=WEEKDAY_CFG,
               also_weekend_only: bool = True):
    sessions = build_sessions(start, end)
    if not sessions:
        return None
    n_days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).days
    periods_per_month = len(sessions) * (30.0 / n_days)
    panel = build_session_panel(symbols, sessions)

    picks_wknd = apply_signal(panel, **weekend_cfg)
    picks_wd = apply_signal(panel, **weekday_cfg)
    combined = combine_picks(picks_wknd, picks_wd)
    series_comb = session_pnl_series(combined, sessions, fee_bps=fee_bps, max_gross=max_gross)
    s_comb = summarize(series_comb, "combined", periods_per_month=periods_per_month)
    s_comb["window"] = f"{start} → {end}"
    s_comb["n_days"] = n_days

    s_wknd = None
    if also_weekend_only:
        series_wknd = session_pnl_series(picks_wknd, sessions, fee_bps=fee_bps, max_gross=max_gross)
        s_wknd = summarize(series_wknd, "weekend_only", periods_per_month=periods_per_month)
        s_wknd["window"] = f"{start} → {end}"
        s_wknd["n_days"] = n_days

    return s_comb, s_wknd


def print_row(s: dict, name: str):
    print(f"{name:<35s} "
          f"n={s['n_sessions']:>4d} tr={s['n_trade_sessions']:>3d} "
          f"med={s['median_session_pnl_pct']:>+7.3f}% "
          f"mean={s['mean_session_pnl_pct']:>+7.3f}% "
          f"p10={s['p10_session_pnl_pct']:>+7.3f}% "
          f"neg={s['neg_session_rate_pct']:>4.1f}% "
          f"dd={s['max_dd_pct']:>+7.2f}% "
          f"sort={s['sortino']:>+5.2f} "
          f"mo={s['monthly_contribution_pct']:>+6.3f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    args = p.parse_args()

    windows = [
        ("2025-12-17", "2026-04-15", "TARGET_120d"),
        ("2025-08-19", "2025-12-17", "PRIOR_120d"),
        ("2025-04-22", "2025-08-19", "PRIOR_120d_2"),
        ("2024-12-20", "2025-04-22", "PRIOR_120d_3"),
        ("2024-08-22", "2024-12-20", "PRIOR_120d_4"),
        ("2024-04-25", "2024-08-22", "PRIOR_120d_5"),
        ("2023-10-28", "2024-04-25", "PRIOR_180d"),
        ("2023-01-01", "2023-10-28", "PRIOR_10MO"),
        ("2022-07-01", "2023-01-01", "2H22"),
    ]

    results = []
    for start, end, name in windows:
        out = run_window(start, end, fee_bps=args.fee_bps, symbols=args.symbols)
        if out is None:
            continue
        s_comb, s_wknd = out
        print(f"\n=== {name} ({start} → {end}, {s_comb['n_days']}d) ===")
        print_row(s_wknd, "  weekend-only baseline")
        print_row(s_comb, "  COMBINED (week+weekday)")
        results.append({
            "window": name, "start": start, "end": end,
            "combined": s_comb, "weekend_only": s_wknd,
        })

    out_dir = REPO / "crypto_out_of_hours" / "results_validate"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "validate.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_dir}/validate.json")

    # Summary: what fraction of windows does combined strict-beat weekend-only?
    n_win = 0
    n_strict = 0
    n_mo_beat = 0
    for r in results:
        n_win += 1
        c = r["combined"]
        w = r["weekend_only"]
        if c["monthly_contribution_pct"] > w["monthly_contribution_pct"]:
            n_mo_beat += 1
        if (c["monthly_contribution_pct"] > w["monthly_contribution_pct"] and
            c["max_dd_pct"] >= w["max_dd_pct"] - 0.01 and
            c["neg_session_rate_pct"] <= w["neg_session_rate_pct"] + 5.0):
            n_strict += 1
    print(f"\n{n_win} windows total")
    print(f"{n_mo_beat} windows: combined > weekend-only monthly PnL")
    print(f"{n_strict} windows: combined strict-beats weekend-only (mo>, dd>=, neg<=+5)")


if __name__ == "__main__":
    main()
