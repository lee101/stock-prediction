"""Wider grid search over signal variants — trend-follow AND mean-reversion,
different session kinds, different vol caps, top_k."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ooh_backtest import (
    build_session_panel, apply_signal, session_pnl_series,
    summarize, DEFAULT_SYMBOLS,
)
from sessions import build_sessions

REPO = Path(__file__).resolve().parent.parent


def apply_mean_rev(
    panel: pd.DataFrame,
    *,
    sma_mult_max: float = 1.0,
    mom7_max: float = 0.0,
    vol_max: float = 0.06,
    top_k: int | None = None,
    kinds: set | None = None,
) -> pd.DataFrame:
    """Mean-reversion: BUY only when below SMA AND 7d momentum negative (oversold),
    vol-capped."""
    df = panel.copy()
    if kinds is not None:
        df = df[df["kind"].isin(kinds)]
    df["mom_vs_sma"] = df["entry_price"] / df["sma_20"] - 1.0
    mask = (df["entry_price"] < df["sma_20"] * sma_mult_max) & \
           (df["mom_7d"] < mom7_max) & \
           (df["vol_20d"] <= vol_max)
    df = df[mask]
    if top_k is not None:
        df = df.sort_values(["session_start", "mom_7d"], ascending=[True, True])  # most oversold first
        df = df.groupby("session_start").head(top_k)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2025-12-17")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--output-dir", default=str(REPO / "crypto_out_of_hours" / "results_grid"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = build_sessions(args.start, args.end)
    n_days = (pd.Timestamp(args.end, tz="UTC") - pd.Timestamp(args.start, tz="UTC")).days
    periods_per_month = len(sessions) * (30.0 / n_days)
    panel = build_session_panel(args.symbols, sessions)

    kind_sets = [
        ("all", None),
        ("weekday", {"weekday"}),
        ("weekend", {"weekend"}),
        ("weekend+holiday", {"weekend", "holiday"}),
        ("weekday+holiday", {"weekday", "holiday"}),
    ]
    # Trend-follow grid
    tf_grid = []
    for sma_mult, mom_vs_sma_min, vol_max, top_k in product(
        [1.00, 1.02, 1.05],
        [0.0, 0.03, 0.05, 0.08],
        [0.03, 0.04, 0.06, 0.10],
        [None, 1, 2, 3],
    ):
        tf_grid.append({
            "signal": "trend",
            "sma_mult": sma_mult,
            "mom7_min": 0.0,
            "mom_vs_sma_min": mom_vs_sma_min,
            "vol_max": vol_max,
            "top_k": top_k,
        })

    # Mean-reversion grid
    mr_grid = []
    for sma_mult_max, mom7_max, vol_max, top_k in product(
        [1.00, 0.97, 0.95],
        [0.0, -0.02, -0.05],
        [0.03, 0.04, 0.06, 0.10],
        [None, 1, 2, 3],
    ):
        mr_grid.append({
            "signal": "meanrev",
            "sma_mult_max": sma_mult_max,
            "mom7_max": mom7_max,
            "vol_max": vol_max,
            "top_k": top_k,
        })

    all_results = []
    for kind_name, kinds in kind_sets:
        for cfg in tf_grid:
            c = {**cfg, "kinds": kinds}
            picked = apply_signal(
                panel,
                sma_mult=c["sma_mult"], mom7_min=c["mom7_min"],
                mom_vs_sma_min=c["mom_vs_sma_min"], vol_max=c["vol_max"],
                top_k=c["top_k"], kinds=kinds,
            )
            series = session_pnl_series(picked, sessions, fee_bps=args.fee_bps, max_gross=args.max_gross)
            s = summarize(series, f"{cfg['signal']}_{kind_name}",
                          periods_per_month=periods_per_month)
            s["config"] = c
            s["kind_name"] = kind_name
            if s["n_trade_sessions"] >= 5:
                all_results.append(s)
        for cfg in mr_grid:
            c = {**cfg, "kinds": kinds}
            picked = apply_mean_rev(
                panel,
                sma_mult_max=c["sma_mult_max"], mom7_max=c["mom7_max"],
                vol_max=c["vol_max"], top_k=c["top_k"], kinds=kinds,
            )
            series = session_pnl_series(picked, sessions, fee_bps=args.fee_bps, max_gross=args.max_gross)
            s = summarize(series, f"{cfg['signal']}_{kind_name}",
                          periods_per_month=periods_per_month)
            s["config"] = c
            s["kind_name"] = kind_name
            if s["n_trade_sessions"] >= 5:
                all_results.append(s)

    # Rank by (monthly_contribution AND neg_rate<20 AND max_dd>-7.5)
    def passes(r):
        return (r["monthly_contribution_pct"] > 0 and
                r["neg_session_rate_pct"] < 20.0 and
                r["max_dd_pct"] > -7.5)

    passing = [r for r in all_results if passes(r)]
    passing.sort(key=lambda r: (r["monthly_contribution_pct"], r["sortino"]), reverse=True)

    print(f"\nTotal configs evaluated: {len(all_results)}")
    print(f"Passing strict-dom gate (mo>0, neg<20%, dd>-7.5): {len(passing)}")
    print("\nTop 30 passing:")
    header = (f"{'signal':<10s} {'kind':<16s} {'cfg':<40s} {'trade':>5s} "
              f"{'med%':>7s} {'mean%':>7s} {'p10%':>7s} {'neg%':>5s} "
              f"{'dd%':>7s} {'sort':>6s} {'mo%':>7s} {'cum%':>7s}")
    print(header)
    print("=" * len(header))
    for r in passing[:30]:
        c = r["config"]
        sig = c["signal"]
        if sig == "trend":
            cfg_str = f"sma={c['sma_mult']} msm={c['mom_vs_sma_min']:.2f} vm={c['vol_max']:.2f} k={c['top_k']}"
        else:
            cfg_str = f"sma<={c['sma_mult_max']} m7<={c['mom7_max']:.2f} vm={c['vol_max']:.2f} k={c['top_k']}"
        print(f"{sig:<10s} {r['kind_name']:<16s} {cfg_str:<40s} "
              f"{r['n_trade_sessions']:>5d} "
              f"{r['median_session_pnl_pct']:>+7.3f} "
              f"{r['mean_session_pnl_pct']:>+7.3f} "
              f"{r['p10_session_pnl_pct']:>+7.3f} "
              f"{r['neg_session_rate_pct']:>5.1f} "
              f"{r['max_dd_pct']:>+7.2f} "
              f"{r['sortino']:>+6.2f} "
              f"{r['monthly_contribution_pct']:>+7.3f} "
              f"{r['cum_return_pct']:>+7.2f}")

    # Also show top by monthly_pnl regardless of gate (for signal diagnostic)
    by_mo = sorted(all_results, key=lambda r: r["monthly_contribution_pct"], reverse=True)[:15]
    print(f"\n\nTop 15 by monthly_pnl (no gate):")
    print(header)
    print("=" * len(header))
    for r in by_mo:
        c = r["config"]
        sig = c["signal"]
        if sig == "trend":
            cfg_str = f"sma={c['sma_mult']} msm={c['mom_vs_sma_min']:.2f} vm={c['vol_max']:.2f} k={c['top_k']}"
        else:
            cfg_str = f"sma<={c['sma_mult_max']} m7<={c['mom7_max']:.2f} vm={c['vol_max']:.2f} k={c['top_k']}"
        print(f"{sig:<10s} {r['kind_name']:<16s} {cfg_str:<40s} "
              f"{r['n_trade_sessions']:>5d} "
              f"{r['median_session_pnl_pct']:>+7.3f} "
              f"{r['mean_session_pnl_pct']:>+7.3f} "
              f"{r['p10_session_pnl_pct']:>+7.3f} "
              f"{r['neg_session_rate_pct']:>5.1f} "
              f"{r['max_dd_pct']:>+7.2f} "
              f"{r['sortino']:>+6.2f} "
              f"{r['monthly_contribution_pct']:>+7.3f} "
              f"{r['cum_return_pct']:>+7.2f}")

    summary_out = {
        "meta": {
            "start": args.start, "end": args.end,
            "symbols": args.symbols, "fee_bps": args.fee_bps,
            "max_gross": args.max_gross, "n_sessions": len(sessions),
            "periods_per_month": periods_per_month,
            "n_total_configs": len(all_results),
            "n_passing_gate": len(passing),
        },
        "passing_top30": [{k: v for k, v in r.items() if k != "series"} for r in passing[:30]],
        "top30_by_mo": [{k: v for k, v in r.items() if k != "series"} for r in by_mo[:30]],
    }
    (out_dir / "grid_summary.json").write_text(json.dumps(summary_out, indent=2, default=str))
    print(f"\nWrote {out_dir}/grid_summary.json")


if __name__ == "__main__":
    main()
