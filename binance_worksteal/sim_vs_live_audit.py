#!/usr/bin/env python3
"""Sim-vs-production parity audit for binance_worksteal.

Runs the backtest under production-realistic constraints and reports
where candidates are blocked, helping diagnose why sim trades but
production does not.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig,
    compute_ref_price,
    compute_sma,
    load_daily_bars,
    run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE


def audit_entries(
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    bars = {}
    for sym in list(all_bars.keys()):
        df = all_bars[sym].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        bars[sym] = df

    all_dates = sorted(set().union(*[set(df["timestamp"].tolist()) for df in bars.values()]))
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        all_dates = [d for d in all_dates if d <= end_ts]

    rows = []
    for date in all_dates:
        current_bars: dict[str, pd.Series] = {}
        history: dict[str, pd.DataFrame] = {}
        for sym, df in bars.items():
            mask = df["timestamp"] <= date
            hist = df[mask]
            if hist.empty:
                continue
            bar = hist.iloc[-1]
            if bar["timestamp"] != date:
                continue
            current_bars[sym] = bar
            history[sym] = hist

        for sym, bar in current_bars.items():
            if sym not in history or len(history[sym]) < config.lookback_days:
                continue
            close = float(bar["close"])
            low_bar = float(bar["low"])

            sma_val = compute_sma(history[sym], config.sma_filter_period) if config.sma_filter_period > 0 else 0.0
            sma_blocks = config.sma_filter_period > 0 and close < sma_val

            ref_high = compute_ref_price(history[sym], config.ref_price_method, config.lookback_days)
            buy_target = ref_high * (1 - config.dip_pct)
            dist = (close - buy_target) / ref_high if ref_high > 0 else 999
            proximity_blocks = dist > config.proximity_pct

            strict_fill = low_bar <= buy_target
            proximity_bps = dist * 10000

            momentum_blocks = False
            if config.momentum_period > 0 and len(history[sym]) > config.momentum_period:
                past_close = float(history[sym].iloc[-(config.momentum_period + 1)]["close"])
                if past_close > 0:
                    mom_ret = (close - past_close) / past_close
                    momentum_blocks = mom_ret < config.momentum_min

            is_candidate = not sma_blocks and not proximity_blocks and not momentum_blocks

            rows.append({
                "date": date,
                "symbol": sym,
                "close": close,
                "low": low_bar,
                "ref_high": ref_high,
                "buy_target": buy_target,
                "sma_value": sma_val,
                "sma_blocks": sma_blocks,
                "proximity_bps": proximity_bps,
                "proximity_blocks": proximity_blocks,
                "strict_fill_possible": strict_fill,
                "momentum_blocks": momentum_blocks,
                "is_candidate": is_candidate,
                "would_fill_realistic": is_candidate and strict_fill,
            })

    return pd.DataFrame(rows)


def print_audit_summary(audit_df: pd.DataFrame, config: WorkStealConfig):
    print(f"\n{'='*70}")
    print("SIM-vs-LIVE PARITY AUDIT")
    print(f"{'='*70}")

    total = len(audit_df)
    n_sma_blocked = int(audit_df["sma_blocks"].sum())
    n_prox_blocked = int(audit_df["proximity_blocks"].sum())
    n_mom_blocked = int(audit_df["momentum_blocks"].sum())
    n_candidates = int(audit_df["is_candidate"].sum())
    n_strict_fill = int(audit_df["would_fill_realistic"].sum())

    print(f"\nTotal symbol-day evaluations: {total}")
    print(f"  Blocked by SMA filter:     {n_sma_blocked:>6d} ({n_sma_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked by proximity:      {n_prox_blocked:>6d} ({n_prox_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked by momentum:       {n_mom_blocked:>6d} ({n_mom_blocked/max(total,1)*100:.1f}%)")
    print(f"  Pass all filters:          {n_candidates:>6d} ({n_candidates/max(total,1)*100:.1f}%)")
    print(f"  Would fill (strict):       {n_strict_fill:>6d} ({n_strict_fill/max(total,1)*100:.1f}%)")

    if n_candidates > 0:
        fill_rate = n_strict_fill / n_candidates * 100
        print(f"\n  Fill rate (strict/candidate): {fill_rate:.1f}%")

    cand_df = audit_df[audit_df["is_candidate"]]
    if not cand_df.empty:
        print(f"\n{'='*70}")
        print("PER-SYMBOL CANDIDATE BREAKDOWN")
        print(f"{'='*70}")
        print(f"{'Symbol':<12s} {'Candidates':>10s} {'StrictFill':>10s} {'FillRate':>8s} {'AvgProxBps':>10s}")
        print("-" * 52)
        sym_stats = cand_df.groupby("symbol").agg(
            n_cand=("is_candidate", "sum"),
            n_fill=("would_fill_realistic", "sum"),
            avg_prox=("proximity_bps", "mean"),
        ).sort_values("n_cand", ascending=False)
        for sym, row in sym_stats.iterrows():
            fr = row["n_fill"] / max(row["n_cand"], 1) * 100
            print(f"{sym:<12s} {int(row['n_cand']):>10d} {int(row['n_fill']):>10d} {fr:>7.1f}% {row['avg_prox']:>10.1f}")

    date_df = audit_df.groupby("date").agg(
        total=("symbol", "count"),
        sma_blocked=("sma_blocks", "sum"),
        prox_blocked=("proximity_blocks", "sum"),
        candidates=("is_candidate", "sum"),
        fills=("would_fill_realistic", "sum"),
    )
    active_dates = date_df[date_df["candidates"] > 0]
    if not active_dates.empty:
        print(f"\n{'='*70}")
        print(f"DATES WITH CANDIDATES ({len(active_dates)} of {len(date_df)} days)")
        print(f"{'='*70}")
        print(f"{'Date':<12s} {'Syms':>5s} {'SMAblk':>6s} {'ProxBlk':>7s} {'Cands':>5s} {'Fills':>5s}")
        print("-" * 42)
        for dt, row in active_dates.iterrows():
            ds = str(dt)[:10]
            print(f"{ds:<12s} {int(row['total']):>5d} {int(row['sma_blocked']):>6d} "
                  f"{int(row['prox_blocked']):>7d} {int(row['candidates']):>5d} {int(row['fills']):>5d}")


def run_comparison(
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
):
    print(f"\n{'='*70}")
    print("BACKTEST COMPARISON: default vs realistic fill")
    print(f"{'='*70}")

    config_default = replace(config, realistic_fill=False, daily_checkpoint_only=False)
    config_realistic = replace(config, realistic_fill=True, daily_checkpoint_only=True)

    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_d, trades_d, met_d = run_worksteal_backtest(bars_copy, config_default, start_date, end_date)
    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_r, trades_r, met_r = run_worksteal_backtest(bars_copy, config_realistic, start_date, end_date)

    buys_d = [t for t in trades_d if t.side == "buy"]
    buys_r = [t for t in trades_r if t.side == "buy"]

    print(f"\n{'Metric':<25s} {'Default':>12s} {'Realistic':>12s}")
    print("-" * 51)
    for key in ["total_return_pct", "sortino", "max_drawdown_pct", "win_rate"]:
        vd = met_d.get(key, 0)
        vr = met_r.get(key, 0)
        print(f"{key:<25s} {vd:>12.2f} {vr:>12.2f}")
    print(f"{'entries':<25s} {len(buys_d):>12d} {len(buys_r):>12d}")
    print(f"{'candidates_generated':<25s} {met_d.get('candidates_generated',0):>12d} {met_r.get('candidates_generated',0):>12d}")
    print(f"{'fill_rate':<25s} {met_d.get('fill_rate',0):>12.1%} {met_r.get('fill_rate',0):>12.1%}")


def main():
    parser = argparse.ArgumentParser(description="Sim-vs-live parity audit")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--dip-pct", type=float, default=0.20)
    parser.add_argument("--proximity-pct", type=float, default=0.02)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--sma-filter", type=int, default=20)
    parser.add_argument("--trailing-stop", type=float, default=0.03)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--config-file", default=None)
    args = parser.parse_args()

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading data for {len(symbols)} symbols from {args.data_dir}")

    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols with data")
    if not all_bars:
        print("ERROR: No data loaded")
        return 1

    if not args.start and not args.end:
        latest = max(df["timestamp"].max() for df in all_bars.values())
        args.end = str(latest.date())
        args.start = str((latest - pd.Timedelta(days=args.days)).date())
        print(f"Auto date range: {args.start} to {args.end}")

    config = WorkStealConfig(
        dip_pct=args.dip_pct,
        proximity_pct=args.proximity_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
        max_hold_days=args.max_hold_days,
        lookback_days=args.lookback,
        sma_filter_period=args.sma_filter,
        trailing_stop_pct=args.trailing_stop,
        maker_fee=args.fee,
        ref_price_method="high",
    )

    audit_df = audit_entries(
        {k: v.copy() for k, v in all_bars.items()},
        config,
        start_date=args.start,
        end_date=args.end,
    )
    print_audit_summary(audit_df, config)

    run_comparison(all_bars, config, args.start, args.end)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
