"""Fair comparison: weekend MVP baseline on the SAME 120d window vs
out-of-hours candidates on the SAME 120d window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "crypto_weekend"))

from backtest import (
    build_weekend_panel,
    weekend_pnl_series,
    add_holdout_rows,
    summarize as mvp_summarize,
)
from backtest_tight import tight_filter

sys.path.insert(0, str(REPO / "crypto_out_of_hours"))
from sessions import build_sessions  # noqa: E402
from ooh_backtest import (  # noqa: E402
    build_session_panel, apply_signal, session_pnl_series,
    summarize as ooh_summarize, DEFAULT_SYMBOLS,
)


def mvp_120d(symbols, start: pd.Timestamp, end: pd.Timestamp, fee_bps=10.0):
    """Reproduce weekend MVP tight_filter on a specific window."""
    panel = build_weekend_panel(symbols)
    picked = tight_filter(panel, sma_mult=1.0, mom7_min=0.0, mom30_min=0.05, vol_max=0.03)
    weekly_raw = weekend_pnl_series(picked, fee_bps=fee_bps, max_gross=1.0)
    all_fridays = sorted(panel["fri_date"].unique())
    weekly = add_holdout_rows(weekly_raw, all_fridays)
    window = weekly[(weekly["fri_date"] >= start) & (weekly["fri_date"] <= end)]
    return mvp_summarize(window, "mvp_120d"), window


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-17")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--fee-bps", type=float, default=10.0)
    args = p.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    n_days = (end - start).days
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # MVP used these 3

    mvp_summary, mvp_weekly = mvp_120d(symbols, start, end, fee_bps=args.fee_bps)
    print("=" * 80)
    print("WEEKEND MVP (tight_filter: BTC+ETH+SOL, sma=1.0, mom_vs_sma=5%, vol=3%)")
    print(f"  Window: {args.start} → {args.end}  (fee={args.fee_bps}bps)")
    print(f"  n_weekends={mvp_summary['n_weekends']}  trade={mvp_summary['n_trade_weekends']}")
    print(f"  mean_weekly%={mvp_summary['mean_weekly_pnl_pct']:+.3f}  "
          f"median_weekly%={mvp_summary['median_weekly_pnl_pct']:+.3f}  "
          f"p10%={mvp_summary['p10_weekly_pnl_pct']:+.3f}")
    print(f"  neg%={mvp_summary['neg_weekend_rate_pct']:.1f}  "
          f"max_dd%={mvp_summary['max_dd_pct']:+.3f}")
    print(f"  sortino_wkly={mvp_summary['sortino_weekly']:+.3f}  "
          f"cum%={mvp_summary['cum_return_pct']:+.3f}  "
          f"monthly%={mvp_summary['monthly_contribution_pct']:+.3f}")

    # MVP used BTC+ETH+SOL. Let's compare OOH with the same 3-symbol universe first,
    # then with the 10-symbol universe.
    for sym_set_name, sym_set in [
        ("BTC+ETH+SOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        ("10-crypto", DEFAULT_SYMBOLS),
    ]:
        sessions = build_sessions(args.start, args.end)
        panel = build_session_panel(sym_set, sessions)
        periods_per_month = len(sessions) * (30.0 / n_days)
        # Mirror of MVP: weekend-only, same filter translated to OOH feature names
        # (mom_vs_sma -> mom_vs_sma_min, vol_20d<=0.03)
        picked = apply_signal(
            panel,
            sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03,
            kinds={"weekend"},
        )
        series = session_pnl_series(picked, sessions, fee_bps=args.fee_bps, max_gross=1.0)
        s = ooh_summarize(series, f"ooh_weekend_only_{sym_set_name}",
                           periods_per_month=periods_per_month)
        print("\n" + "=" * 80)
        print(f"OOH WEEKEND-ONLY ({sym_set_name})")
        print(f"  Window: {args.start} → {args.end}")
        print(f"  n_sessions={s['n_sessions']}  trade={s['n_trade_sessions']}")
        print(f"  mean_session%={s['mean_session_pnl_pct']:+.3f}  "
              f"p10%={s['p10_session_pnl_pct']:+.3f}  "
              f"neg%={s['neg_session_rate_pct']:.1f}")
        print(f"  max_dd%={s['max_dd_pct']:+.3f}  "
              f"sortino={s['sortino']:+.3f}  "
              f"monthly%={s['monthly_contribution_pct']:+.3f}  cum%={s['cum_return_pct']:+.3f}")

        # Extended: add weekday on TOP — using best candidate from grid
        # 1. Combined best weekend trend-follow config on weekend-only (k=1)
        best_tf = dict(sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0, vol_max=0.03, top_k=1,
                       kinds={"weekend"})
        picked_tf = apply_signal(panel, **best_tf)
        series_tf = session_pnl_series(picked_tf, sessions, fee_bps=args.fee_bps, max_gross=1.0)
        s_tf = ooh_summarize(series_tf, f"ooh_weekend_top_trend_{sym_set_name}",
                             periods_per_month=periods_per_month)
        print(f"\nOOH WEEKEND top-trend-follow (msm=0%, vm=3%, k=1) [{sym_set_name}]")
        print(f"  trade={s_tf['n_trade_sessions']}  mean%={s_tf['mean_session_pnl_pct']:+.3f}  "
              f"p10%={s_tf['p10_session_pnl_pct']:+.3f}  neg%={s_tf['neg_session_rate_pct']:.1f}")
        print(f"  dd%={s_tf['max_dd_pct']:+.3f}  sortino={s_tf['sortino']:+.3f}  "
              f"mo%={s_tf['monthly_contribution_pct']:+.3f}  cum%={s_tf['cum_return_pct']:+.3f}")

    # Compare side-by-side — explicit: monthly_contribution AND max_dd
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE on 120d window (mean monthly PnL / max_dd):")
    print(f"  MVP weekend-only (BTC+ETH+SOL):  "
          f"mean={mvp_summary['mean_weekly_pnl_pct']:+.3f}%/wk  "
          f"monthly={mvp_summary['monthly_contribution_pct']:+.3f}%/mo  "
          f"max_dd={mvp_summary['max_dd_pct']:+.3f}%")


if __name__ == "__main__":
    main()
