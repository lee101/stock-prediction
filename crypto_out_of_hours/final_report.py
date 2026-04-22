"""Final comparative report: OOH combined vs weekend MVP baseline.

Run on the TARGET 120d window (2025-12-17 → 2026-04-15) at fees 10/20/50 bps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "crypto_weekend"))
from backtest import (  # noqa: E402
    build_weekend_panel, weekend_pnl_series, add_holdout_rows,
    summarize as mvp_summarize,
)
from backtest_tight import tight_filter  # noqa: E402

sys.path.insert(0, str(REPO / "crypto_out_of_hours"))
from sessions import build_sessions  # noqa: E402
from ooh_backtest import (  # noqa: E402
    build_session_panel, apply_signal, session_pnl_series,
    summarize as ooh_summarize, DEFAULT_SYMBOLS,
)
from find_combined import combine_picks  # noqa: E402


TARGET_START = "2025-12-17"
TARGET_END = "2026-04-15"


def run_mvp(fee_bps: float):
    """Weekend MVP on target 120d window."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    panel = build_weekend_panel(symbols)
    picked = tight_filter(panel, sma_mult=1.0, mom7_min=0.0, mom30_min=0.05, vol_max=0.03)
    weekly_raw = weekend_pnl_series(picked, fee_bps=fee_bps, max_gross=1.0)
    all_fridays = sorted(panel["fri_date"].unique())
    weekly = add_holdout_rows(weekly_raw, all_fridays)
    start = pd.Timestamp(TARGET_START, tz="UTC")
    end = pd.Timestamp(TARGET_END, tz="UTC")
    window = weekly[(weekly["fri_date"] >= start) & (weekly["fri_date"] <= end)]
    s = mvp_summarize(window, f"mvp_{fee_bps}bps")
    n_weeks = len(window)
    periods_per_month = n_weeks * (30.0 / 119.0)
    # Align field names with OOH
    return {
        "name": f"MVP_weekend_{fee_bps}bps",
        "n_sessions": n_weeks,
        "n_trade_sessions": s["n_trade_weekends"],
        "median_session_pnl_pct": s["median_weekly_pnl_pct"],
        "mean_session_pnl_pct": s["mean_weekly_pnl_pct"],
        "p10_session_pnl_pct": s["p10_weekly_pnl_pct"],
        "neg_session_rate_pct": s["neg_weekend_rate_pct"],
        "max_dd_pct": s["max_dd_pct"],
        "sortino": s["sortino_weekly"],
        "monthly_contribution_pct": s["monthly_contribution_pct"],
        "cum_return_pct": s["cum_return_pct"],
    }


def run_ooh_combined(fee_bps: float, symbols=DEFAULT_SYMBOLS):
    sessions = build_sessions(TARGET_START, TARGET_END)
    n_days = (pd.Timestamp(TARGET_END, tz="UTC") - pd.Timestamp(TARGET_START, tz="UTC")).days
    periods_per_month = len(sessions) * (30.0 / n_days)
    panel = build_session_panel(symbols, sessions)
    picks_wknd = apply_signal(
        panel, sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
        vol_max=0.03, top_k=1, kinds={"weekend"},
    )
    picks_wd = apply_signal(
        panel, sma_mult=1.05, mom7_min=0.0, mom_vs_sma_min=0.0,
        vol_max=0.02, top_k=2, kinds={"weekday", "holiday"},
    )
    combined = combine_picks(picks_wknd, picks_wd)
    series = session_pnl_series(combined, sessions, fee_bps=fee_bps, max_gross=1.0)
    s = ooh_summarize(series, f"OOH_combined_{fee_bps}bps",
                      periods_per_month=periods_per_month)
    return s


def run_ooh_weekend_only(fee_bps: float, symbols=DEFAULT_SYMBOLS):
    sessions = build_sessions(TARGET_START, TARGET_END)
    n_days = (pd.Timestamp(TARGET_END, tz="UTC") - pd.Timestamp(TARGET_START, tz="UTC")).days
    periods_per_month = len(sessions) * (30.0 / n_days)
    panel = build_session_panel(symbols, sessions)
    picks = apply_signal(
        panel, sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
        vol_max=0.03, top_k=1, kinds={"weekend"},
    )
    series = session_pnl_series(picks, sessions, fee_bps=fee_bps, max_gross=1.0)
    s = ooh_summarize(series, f"OOH_weekend_only_{fee_bps}bps",
                      periods_per_month=periods_per_month)
    return s


def row(s: dict) -> str:
    return (f"  n={s['n_sessions']:>3d} tr={s['n_trade_sessions']:>3d} "
            f"med={s['median_session_pnl_pct']:>+7.3f}% "
            f"mean={s['mean_session_pnl_pct']:>+7.3f}% "
            f"p10={s['p10_session_pnl_pct']:>+7.3f}% "
            f"neg={s['neg_session_rate_pct']:>4.1f}% "
            f"dd={s['max_dd_pct']:>+7.2f}% "
            f"sort={s['sortino']:>+5.2f} "
            f"mo={s['monthly_contribution_pct']:>+6.2f}% "
            f"cum={s['cum_return_pct']:>+6.2f}%")


def main():
    print("=" * 90)
    print(f"FINAL REPORT — 120-day marketsim ({TARGET_START} → {TARGET_END})")
    print("=" * 90)

    results = []
    for fee_bps in [10.0, 20.0, 50.0]:
        print(f"\n── fee = {fee_bps}bps ──")
        m = run_mvp(fee_bps)
        ow = run_ooh_weekend_only(fee_bps)
        oc = run_ooh_combined(fee_bps)
        print(f"MVP weekend-only (BTC+ETH+SOL, sma=1.0 mom_sma=5% vol=3% gross=1.0):")
        print(row(m))
        print(f"OOH weekend-only top-trend (10-crypto, sma=1.0 msm=0% vol=3% k=1):")
        print(row(ow))
        print(f"OOH COMBINED (weekend k=1 + weekday sma=1.05 msm=0% vol=2% k=2):")
        print(row(oc))
        results.append({"fee_bps": fee_bps, "mvp": m, "ooh_weekend_only": ow, "ooh_combined": oc})

    # Strict-beat check at 10bps
    print("\n" + "=" * 90)
    print("STRICT-BEAT CHECK (at fee=10bps, target 120d window):")
    m = results[0]["mvp"]
    oc = results[0]["ooh_combined"]
    ow = results[0]["ooh_weekend_only"]
    print(f"  MVP: {m['monthly_contribution_pct']:+.3f}%/mo, dd={m['max_dd_pct']:+.3f}%, "
          f"neg%={m['neg_session_rate_pct']:.1f}")
    print(f"  OOH weekend-only: {ow['monthly_contribution_pct']:+.3f}%/mo, dd={ow['max_dd_pct']:+.3f}%, "
          f"neg%={ow['neg_session_rate_pct']:.1f}")
    print(f"  OOH combined: {oc['monthly_contribution_pct']:+.3f}%/mo, dd={oc['max_dd_pct']:+.3f}%, "
          f"neg%={oc['neg_session_rate_pct']:.1f}")

    beats_mo = oc["monthly_contribution_pct"] > m["monthly_contribution_pct"]
    dd_ok = oc["max_dd_pct"] > -7.5
    print(f"\n  Combined beats MVP on monthly PnL: {beats_mo}")
    print(f"  Combined DD ≤ 7.5%: {dd_ok}")
    print(f"  Strict-beat PASS: {beats_mo and dd_ok}")

    out_dir = REPO / "crypto_out_of_hours" / "results_final"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "final_report.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_dir}/final_report.json")


if __name__ == "__main__":
    main()
