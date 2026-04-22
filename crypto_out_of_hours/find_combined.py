"""Find a COMBINED strategy: weekend trend-follow + weekday layer that together
strict-beat the MVP baseline (+0.90%/mo, DD ≤ 7.5%) on the SAME 120d window.

Search over: weekend trend-follow config (fixed best) + weekday overlay config.
The weekday overlay is judged on the *combined* series (not just weekdays)."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sessions import build_sessions
from ooh_backtest import (
    build_session_panel, apply_signal, session_pnl_series,
    summarize, DEFAULT_SYMBOLS,
)
from backtest_grid import apply_mean_rev

REPO = Path(__file__).resolve().parent.parent


def combine_picks(picks_a: pd.DataFrame, picks_b: pd.DataFrame) -> pd.DataFrame:
    """Concatenate and dedupe on (session_start, symbol)."""
    if picks_a.empty and picks_b.empty:
        return picks_a
    if picks_a.empty:
        return picks_b
    if picks_b.empty:
        return picks_a
    combined = pd.concat([picks_a, picks_b], axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["session_start", "symbol"])
    return combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-17")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--max-gross", type=float, default=1.0)
    args = p.parse_args()

    sessions = build_sessions(args.start, args.end)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    n_days = (end - start).days
    periods_per_month = len(sessions) * (30.0 / n_days)

    panel = build_session_panel(args.symbols, sessions)

    # Fixed best weekend config
    weekend_cfg = dict(sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
                        vol_max=0.03, top_k=1, kinds={"weekend"})
    picks_wknd = apply_signal(panel, **weekend_cfg)

    # Baseline: weekend only
    series_wknd = session_pnl_series(picks_wknd, sessions,
                                     fee_bps=args.fee_bps, max_gross=args.max_gross)
    s_wknd = summarize(series_wknd, "weekend_only", periods_per_month=periods_per_month)
    print(f"Weekend-only baseline: monthly={s_wknd['monthly_contribution_pct']:+.3f}% "
          f"p10={s_wknd['p10_session_pnl_pct']:+.3f}% "
          f"neg%={s_wknd['neg_session_rate_pct']:.1f} "
          f"dd={s_wknd['max_dd_pct']:+.3f}% "
          f"sortino={s_wknd['sortino']:+.3f}")

    # Sweep weekday overlay candidates
    results = []

    # Weekday trend-follow — very tight (high sma_mult, strong mom, low vol)
    for sma_mult, mom_vs_sma_min, vol_max, top_k in product(
        [1.00, 1.02, 1.05, 1.08, 1.10],
        [0.0, 0.03, 0.05, 0.08],
        [0.02, 0.03, 0.04, 0.06],
        [1, 2, 3],
    ):
        picks_wd = apply_signal(
            panel, sma_mult=sma_mult, mom7_min=0.0,
            mom_vs_sma_min=mom_vs_sma_min, vol_max=vol_max,
            top_k=top_k, kinds={"weekday", "holiday"},
        )
        combined = combine_picks(picks_wknd, picks_wd)
        series = session_pnl_series(combined, sessions,
                                     fee_bps=args.fee_bps, max_gross=args.max_gross)
        s = summarize(series, "combined",
                      periods_per_month=periods_per_month)
        if s["n_trade_sessions"] < 3:
            continue
        results.append({
            "signal": "trend",
            "sma_mult": sma_mult, "mom_vs_sma_min": mom_vs_sma_min,
            "vol_max": vol_max, "top_k": top_k,
            "trade": s["n_trade_sessions"],
            "med": s["median_session_pnl_pct"],
            "mean": s["mean_session_pnl_pct"],
            "p10": s["p10_session_pnl_pct"],
            "neg": s["neg_session_rate_pct"],
            "dd": s["max_dd_pct"],
            "sortino": s["sortino"],
            "mo": s["monthly_contribution_pct"],
            "cum": s["cum_return_pct"],
        })

    # Mean-reversion weekday overlay (gentler — more restrictive)
    for sma_mult_max, mom7_max, vol_max, top_k in product(
        [0.95, 0.93, 0.90, 0.88],
        [-0.05, -0.08, -0.10, -0.12],
        [0.02, 0.03, 0.04, 0.06],
        [1, 2, 3],
    ):
        picks_wd = apply_mean_rev(
            panel, sma_mult_max=sma_mult_max, mom7_max=mom7_max,
            vol_max=vol_max, top_k=top_k, kinds={"weekday", "holiday"},
        )
        combined = combine_picks(picks_wknd, picks_wd)
        series = session_pnl_series(combined, sessions,
                                     fee_bps=args.fee_bps, max_gross=args.max_gross)
        s = summarize(series, "combined",
                      periods_per_month=periods_per_month)
        if s["n_trade_sessions"] < 3:
            continue
        results.append({
            "signal": "meanrev",
            "sma_mult_max": sma_mult_max, "mom7_max": mom7_max,
            "vol_max": vol_max, "top_k": top_k,
            "trade": s["n_trade_sessions"],
            "med": s["median_session_pnl_pct"],
            "mean": s["mean_session_pnl_pct"],
            "p10": s["p10_session_pnl_pct"],
            "neg": s["neg_session_rate_pct"],
            "dd": s["max_dd_pct"],
            "sortino": s["sortino"],
            "mo": s["monthly_contribution_pct"],
            "cum": s["cum_return_pct"],
        })

    # Filter: must strict-beat weekend-only baseline (mo>baseline AND dd>baseline)
    baseline_mo = s_wknd["monthly_contribution_pct"]
    baseline_dd = s_wknd["max_dd_pct"]
    strict = [r for r in results
              if r["mo"] > baseline_mo and r["dd"] >= baseline_dd - 0.01]

    # Filter: DD must stay above -7.5 (MVP bar)
    hard_gate = [r for r in strict if r["dd"] > -7.5]

    print(f"\nConfigs evaluated: {len(results)}")
    print(f"Strict-beat vs weekend-only ({baseline_mo:+.3f}%/mo, {baseline_dd:+.3f}% dd): {len(strict)}")
    print(f"Passing HARD GATE (mo>baseline, dd>-7.5): {len(hard_gate)}")

    hard_gate.sort(key=lambda r: (r["mo"], r["sortino"]), reverse=True)
    header = (f"{'sig':<8s} {'cfg':<45s} {'trade':>5s} {'med%':>7s} "
              f"{'mean%':>7s} {'p10%':>7s} {'neg%':>5s} {'dd%':>7s} "
              f"{'sort':>6s} {'mo%':>7s} {'cum%':>7s}")
    print("\nTop 20 passing HARD GATE:")
    print(header)
    print("=" * len(header))
    for r in hard_gate[:20]:
        if r["signal"] == "trend":
            cfg = f"sma={r['sma_mult']} msm={r['mom_vs_sma_min']:.2f} vm={r['vol_max']:.2f} k={r['top_k']}"
        else:
            cfg = f"sma<={r['sma_mult_max']} m7<={r['mom7_max']:.2f} vm={r['vol_max']:.2f} k={r['top_k']}"
        print(f"{r['signal']:<8s} {cfg:<45s} {r['trade']:>5d} "
              f"{r['med']:>+7.3f} {r['mean']:>+7.3f} "
              f"{r['p10']:>+7.3f} {r['neg']:>5.1f} {r['dd']:>+7.2f} "
              f"{r['sortino']:>+6.2f} {r['mo']:>+7.3f} {r['cum']:>+7.2f}")

    # Print top-10 by monthly regardless (for diagnostic)
    strict_sorted = sorted(results, key=lambda r: r["mo"], reverse=True)[:15]
    print("\nTop 15 by monthly PnL (NO gate, diagnostic):")
    print(header)
    print("=" * len(header))
    for r in strict_sorted:
        if r["signal"] == "trend":
            cfg = f"sma={r['sma_mult']} msm={r['mom_vs_sma_min']:.2f} vm={r['vol_max']:.2f} k={r['top_k']}"
        else:
            cfg = f"sma<={r['sma_mult_max']} m7<={r['mom7_max']:.2f} vm={r['vol_max']:.2f} k={r['top_k']}"
        print(f"{r['signal']:<8s} {cfg:<45s} {r['trade']:>5d} "
              f"{r['med']:>+7.3f} {r['mean']:>+7.3f} "
              f"{r['p10']:>+7.3f} {r['neg']:>5.1f} {r['dd']:>+7.2f} "
              f"{r['sortino']:>+6.2f} {r['mo']:>+7.3f} {r['cum']:>+7.2f}")

    # Write results
    out_dir = REPO / "crypto_out_of_hours" / "results_combined"
    out_dir.mkdir(exist_ok=True)
    out = {
        "weekend_baseline": s_wknd,
        "hard_gate_top30": hard_gate[:30],
        "by_monthly_top30": strict_sorted[:30],
    }
    (out_dir / "combined_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_dir}/combined_summary.json")


if __name__ == "__main__":
    main()
