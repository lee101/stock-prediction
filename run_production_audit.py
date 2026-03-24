#!/usr/bin/env python3
"""Production audit for binance_worksteal strategy.

Validates simulator matches production by running:
1. Entry filter breakdown (SMA, momentum, risk-off, proximity)
2. Multi-window backtests (3 x 60-day non-overlapping)
3. Parameter sensitivity analysis (dip_pct, proximity_pct, sma_filter_period)
4. Random baseline comparison
"""
from __future__ import annotations

import argparse
import io
import itertools
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig,
    load_daily_bars,
    run_worksteal_backtest,
)
from binance_worksteal.sim_vs_live_audit import audit_entries
from binance_worksteal.sweep import build_windows
from binance_worksteal.backtest import FULL_UNIVERSE


def _clamp_sortino(val, limit=999.0):
    return max(-limit, min(limit, val))


PRODUCTION_CONFIG = WorkStealConfig(
    dip_pct=0.20,
    proximity_pct=0.02,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    max_positions=5,
    max_hold_days=14,
    lookback_days=20,
    sma_filter_period=20,
    sma_check_method="pre_dip",
    trailing_stop_pct=0.03,
    maker_fee=0.001,
    ref_price_method="high",
    initial_cash=10_000.0,
)


def run_single_backtest(all_bars, config, start_date, end_date):
    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    try:
        eq, trades, metrics = run_worksteal_backtest(
            bars_copy, config, start_date=start_date, end_date=end_date,
        )
    except Exception as e:
        return None, None, {"error": str(e)}
    return eq, trades, metrics


def run_filter_breakdown(all_bars, config, start_date, end_date, out):
    out.write(f"\n{'='*70}\n")
    out.write("SECTION 1: ENTRY FILTER BREAKDOWN\n")
    out.write(f"{'='*70}\n")
    out.write(f"Period: {start_date} to {end_date}\n")
    out.write(f"Config: dip={config.dip_pct:.0%} prox={config.proximity_pct:.1%} "
              f"sma={config.sma_filter_period} method={config.sma_check_method}\n\n")

    audit_df = audit_entries(
        {k: v.copy() for k, v in all_bars.items()},
        config,
        start_date=start_date,
        end_date=end_date,
    )

    if audit_df.empty:
        out.write("No data in audit period.\n")
        return audit_df

    total = len(audit_df)
    n_sma = int(audit_df["sma_blocks"].sum())
    n_prox = int(audit_df["proximity_blocks"].sum())
    n_mom = int(audit_df["momentum_blocks"].sum())
    n_cand = int(audit_df["is_candidate"].sum())
    n_fill = int(audit_df["would_fill_realistic"].sum())

    out.write(f"Total symbol-day evaluations: {total}\n")
    out.write(f"  Blocked by SMA filter:     {n_sma:>6d} ({n_sma/max(total,1)*100:.1f}%)\n")
    out.write(f"  Blocked by proximity:      {n_prox:>6d} ({n_prox/max(total,1)*100:.1f}%)\n")
    out.write(f"  Blocked by momentum:       {n_mom:>6d} ({n_mom/max(total,1)*100:.1f}%)\n")
    out.write(f"  Pass all filters:          {n_cand:>6d} ({n_cand/max(total,1)*100:.1f}%)\n")
    out.write(f"  Would fill (strict):       {n_fill:>6d} ({n_fill/max(total,1)*100:.1f}%)\n")

    if n_cand > 0:
        fill_rate = n_fill / n_cand * 100
        out.write(f"\n  Fill rate (strict/candidate): {fill_rate:.1f}%\n")

    n_days = audit_df["date"].nunique()
    out.write(f"\n  Expected fills per 30 days: {n_fill / max(n_days,1) * 30:.1f}\n")

    cand_df = audit_df[audit_df["is_candidate"]]
    if not cand_df.empty:
        out.write(f"\nPer-symbol candidate counts:\n")
        out.write(f"{'Symbol':<12s} {'Candidates':>10s} {'Fills':>10s} {'FillRate':>8s}\n")
        out.write("-" * 42 + "\n")
        sym_stats = cand_df.groupby("symbol").agg(
            n_cand=("is_candidate", "sum"),
            n_fill=("would_fill_realistic", "sum"),
        ).sort_values("n_cand", ascending=False)
        for sym, row in sym_stats.iterrows():
            fr = row["n_fill"] / max(row["n_cand"], 1) * 100
            out.write(f"{sym:<12s} {int(row['n_cand']):>10d} {int(row['n_fill']):>10d} {fr:>7.1f}%\n")

    return audit_df


def run_multi_window_backtest(all_bars, config, windows, out):
    out.write(f"\n{'='*70}\n")
    out.write("SECTION 2: MULTI-WINDOW BACKTEST (Production Config)\n")
    out.write(f"{'='*70}\n")
    out.write(f"Config: dip={config.dip_pct:.0%} tp={config.profit_target_pct:.0%} "
              f"sl={config.stop_loss_pct:.0%} trail={config.trailing_stop_pct:.0%} "
              f"maxpos={config.max_positions} maxhold={config.max_hold_days}d "
              f"sma={config.sma_filter_period} prox={config.proximity_pct:.1%}\n\n")

    all_results = []
    for i, (start, end) in enumerate(windows):
        eq, trades, metrics = run_single_backtest(all_bars, config, start, end)
        if metrics.get("error"):
            out.write(f"  W{i} ({start} to {end}): ERROR {metrics['error']}\n")
            continue

        buys = [t for t in (trades or []) if t.side in ("buy", "short")]
        exits = [t for t in (trades or []) if t.side in ("sell", "cover")]
        n_days = metrics.get("n_days", 0)

        sortino = _clamp_sortino(metrics.get("sortino", 0))
        out.write(f"  W{i} ({start} to {end}):\n")
        out.write(f"    Return:    {metrics.get('total_return_pct',0):>7.2f}%\n")
        out.write(f"    Sortino:   {sortino:>7.2f}\n")
        out.write(f"    Sharpe:    {metrics.get('sharpe',0):>7.2f}\n")
        out.write(f"    Max DD:    {metrics.get('max_drawdown_pct',0):>7.2f}%\n")
        out.write(f"    Win Rate:  {metrics.get('win_rate',0):>7.1f}%\n")
        out.write(f"    Entries:   {len(buys):>7d}\n")
        out.write(f"    Exits:     {len(exits):>7d}\n")
        out.write(f"    Days:      {n_days:>7d}\n")
        if n_days > 0:
            out.write(f"    Trades/30d:{len(buys)/max(n_days,1)*30:>7.1f}\n")
        out.write("\n")

        all_results.append({
            "window": f"W{i}",
            "start": start,
            "end": end,
            "return_pct": metrics.get("total_return_pct", 0),
            "sortino": sortino,
            "sharpe": metrics.get("sharpe", 0),
            "max_dd_pct": metrics.get("max_drawdown_pct", 0),
            "win_rate": metrics.get("win_rate", 0),
            "n_entries": len(buys),
            "n_exits": len(exits),
            "n_days": n_days,
        })

    if all_results:
        sorts = [r["sortino"] for r in all_results]
        rets = [r["return_pct"] for r in all_results]
        dds = [r["max_dd_pct"] for r in all_results]
        entries = [r["n_entries"] for r in all_results]
        out.write(f"  AGGREGATE:\n")
        out.write(f"    Mean Sortino:  {np.mean(sorts):>7.2f}  (min={np.min(sorts):.2f} max={np.max(sorts):.2f})\n")
        out.write(f"    Mean Return:   {np.mean(rets):>7.2f}%\n")
        out.write(f"    Worst DD:      {np.min(dds):>7.2f}%\n")
        out.write(f"    Mean Entries:  {np.mean(entries):>7.1f}\n")
        out.write(f"    Total Entries: {np.sum(entries):>7.0f}\n")

    return all_results


def _sweep_param(all_bars, base_config, start_date, end_date, param_name, values, out, fmt_value):
    out.write(f"\n--- {param_name} sweep ---\n")
    out.write(f"{'value':>8s} {'Return%':>8s} {'Sortino':>8s} {'MaxDD%':>8s} {'WinRate':>8s} {'Entries':>8s} {'Trades/30d':>10s}\n")
    out.write("-" * 62 + "\n")
    results = []
    for val in values:
        cfg = replace(base_config, **{param_name: val})
        _, trades, m = run_single_backtest(all_bars, cfg, start_date, end_date)
        if not m or m.get("error"):
            continue
        buys = [t for t in (trades or []) if t.side in ("buy", "short")]
        n_days = m.get("n_days", 1)
        tpd = len(buys) / max(n_days, 1) * 30
        sortino = _clamp_sortino(m.get("sortino", 0))
        out.write(f"{fmt_value(val):>8s} {m.get('total_return_pct',0):>8.2f} {sortino:>8.2f} "
                  f"{m.get('max_drawdown_pct',0):>8.2f} {m.get('win_rate',0):>8.1f} "
                  f"{len(buys):>8d} {tpd:>10.1f}\n")
        results.append({"param": param_name, "value": val, **m, "sortino": sortino, "n_entries": len(buys), "trades_per_30d": tpd})
    return results


def run_sensitivity_analysis(all_bars, base_config, start_date, end_date, out):
    out.write(f"\n{'='*70}\n")
    out.write("SECTION 3: PARAMETER SENSITIVITY ANALYSIS\n")
    out.write(f"{'='*70}\n")
    out.write(f"Period: {start_date} to {end_date}\n")

    dip_values = [0.10, 0.15, 0.20, 0.25]
    prox_values = [0.02, 0.03, 0.05]
    sma_values = [0, 10, 20]

    results = []
    results += _sweep_param(all_bars, base_config, start_date, end_date,
                            "dip_pct", dip_values, out, lambda v: f"{v:.0%}")
    results += _sweep_param(all_bars, base_config, start_date, end_date,
                            "proximity_pct", prox_values, out, lambda v: f"{v:.1%}")
    results += _sweep_param(all_bars, base_config, start_date, end_date,
                            "sma_filter_period", sma_values, out, lambda v: f"{v:d}")

    # 3d: combined grid (dip x proximity x sma) - top combos
    out.write(f"\n--- Combined grid: dip x proximity x sma (top 10 by Sortino) ---\n")
    grid_results = []
    for dip, prox, sma in itertools.product(dip_values, prox_values, sma_values):
        cfg = replace(base_config, dip_pct=dip, proximity_pct=prox, sma_filter_period=sma)
        _, trades, m = run_single_backtest(all_bars, cfg, start_date, end_date)
        if not m or m.get("error"):
            continue
        buys = [t for t in (trades or []) if t.side in ("buy", "short")]
        n_days = m.get("n_days", 1)
        tpd = len(buys) / max(n_days, 1) * 30
        grid_results.append({
            "dip": dip, "prox": prox, "sma": sma,
            "return_pct": m.get("total_return_pct", 0),
            "sortino": _clamp_sortino(m.get("sortino", 0)),
            "max_dd_pct": m.get("max_drawdown_pct", 0),
            "win_rate": m.get("win_rate", 0),
            "n_entries": len(buys),
            "trades_per_30d": tpd,
        })

    if grid_results:
        grid_results.sort(key=lambda x: x["sortino"], reverse=True)
        out.write(f"{'dip':>5s} {'prox':>5s} {'sma':>4s} {'Ret%':>7s} {'Sort':>7s} {'DD%':>7s} {'WR':>6s} {'Ent':>5s} {'T/30d':>6s}\n")
        out.write("-" * 56 + "\n")
        for r in grid_results[:10]:
            out.write(f"{r['dip']:>5.0%} {r['prox']:>5.1%} {r['sma']:>4d} "
                      f"{r['return_pct']:>7.2f} {r['sortino']:>7.2f} {r['max_dd_pct']:>7.2f} "
                      f"{r['win_rate']:>6.1f} {r['n_entries']:>5d} {r['trades_per_30d']:>6.1f}\n")

        out.write(f"\nWorst 5 by Sortino:\n")
        for r in grid_results[-5:]:
            out.write(f"{r['dip']:>5.0%} {r['prox']:>5.1%} {r['sma']:>4d} "
                      f"{r['return_pct']:>7.2f} {r['sortino']:>7.2f} {r['max_dd_pct']:>7.2f} "
                      f"{r['win_rate']:>6.1f} {r['n_entries']:>5d} {r['trades_per_30d']:>6.1f}\n")

    return results, grid_results


def run_random_baseline(all_bars, config, start_date, end_date, out, n_trials=20):
    out.write(f"\n{'='*70}\n")
    out.write("SECTION 4: RANDOM BASELINE COMPARISON\n")
    out.write(f"{'='*70}\n")
    out.write(f"Comparing production config vs {n_trials} random configs.\n\n")

    _, _, prod_metrics = run_single_backtest(all_bars, config, start_date, end_date)

    rng = np.random.RandomState(42)
    random_sortinos = []
    random_returns = []

    for trial in range(n_trials):
        rand_cfg = replace(
            config,
            dip_pct=rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
            proximity_pct=rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
            profit_target_pct=rng.choice([0.03, 0.05, 0.10, 0.15, 0.20]),
            stop_loss_pct=rng.choice([0.05, 0.08, 0.10, 0.15]),
            max_positions=int(rng.choice([3, 5, 7, 10])),
            max_hold_days=int(rng.choice([7, 14, 21, 30])),
            sma_filter_period=int(rng.choice([0, 10, 20, 30])),
            trailing_stop_pct=rng.choice([0.0, 0.02, 0.03, 0.05]),
        )
        _, _, m = run_single_backtest(all_bars, rand_cfg, start_date, end_date)
        if m and not m.get("error"):
            random_sortinos.append(_clamp_sortino(m.get("sortino", 0)))
            random_returns.append(m.get("total_return_pct", 0))

    prod_sort = _clamp_sortino(prod_metrics.get("sortino", 0)) if prod_metrics else 0
    prod_ret = prod_metrics.get("total_return_pct", 0) if prod_metrics else 0

    out.write(f"Production config:\n")
    out.write(f"  Sortino: {prod_sort:.2f}\n")
    out.write(f"  Return:  {prod_ret:.2f}%\n\n")

    if random_sortinos:
        out.write(f"Random baseline ({len(random_sortinos)} trials):\n")
        out.write(f"  Sortino: mean={np.mean(random_sortinos):.2f} "
                  f"median={np.median(random_sortinos):.2f} "
                  f"max={np.max(random_sortinos):.2f} "
                  f"min={np.min(random_sortinos):.2f}\n")
        out.write(f"  Return:  mean={np.mean(random_returns):.2f}% "
                  f"median={np.median(random_returns):.2f}% "
                  f"max={np.max(random_returns):.2f}% "
                  f"min={np.min(random_returns):.2f}%\n")
        pct_rank = sum(1 for s in random_sortinos if s < prod_sort) / len(random_sortinos) * 100
        out.write(f"\n  Production Sortino percentile vs random: {pct_rank:.0f}%\n")
        if pct_rank >= 75:
            out.write("  -> Production config OUTPERFORMS most random configs.\n")
        elif pct_rank >= 50:
            out.write("  -> Production config is ABOVE MEDIAN random.\n")
        else:
            out.write("  -> WARNING: Production config UNDERPERFORMS random baseline.\n")

    return prod_metrics, random_sortinos


def generate_recommendations(audit_df, window_results, sensitivity_results, grid_results, out):
    out.write(f"\n{'='*70}\n")
    out.write("SECTION 5: RECOMMENDATIONS\n")
    out.write(f"{'='*70}\n\n")

    if audit_df is not None and not audit_df.empty:
        total = len(audit_df)
        n_fill = int(audit_df["would_fill_realistic"].sum())
        n_days = audit_df["date"].nunique()
        fills_per_30d = n_fill / max(n_days, 1) * 30

        if fills_per_30d < 5:
            out.write(f"LOW TRADE FREQUENCY: {fills_per_30d:.1f} fills/30d\n")
            n_prox = int(audit_df["proximity_blocks"].sum())
            n_sma = int(audit_df["sma_blocks"].sum())
            if n_prox > total * 0.5:
                out.write(f"  -> proximity_pct={PRODUCTION_CONFIG.proximity_pct:.1%} blocks {n_prox/total*100:.0f}% of evals. "
                          f"Consider increasing to 0.03 or 0.05.\n")
            if n_sma > total * 0.5:
                out.write(f"  -> SMA filter blocks {n_sma/total*100:.0f}% of evals. "
                          f"Consider reducing sma_filter_period or using sma_check_method='pre_dip'.\n")
            out.write(f"  -> Consider reducing dip_pct from {PRODUCTION_CONFIG.dip_pct:.0%} to 0.15 or 0.10.\n")
        elif fills_per_30d > 30:
            out.write(f"HIGH TRADE FREQUENCY: {fills_per_30d:.1f} fills/30d - risk of overtrading.\n")
        else:
            out.write(f"TRADE FREQUENCY OK: {fills_per_30d:.1f} fills/30d\n")

    if grid_results:
        reliable = [r for r in grid_results if r["n_entries"] >= 10]
        if not reliable:
            reliable = [r for r in grid_results if r["n_entries"] >= 3]
        if reliable:
            best = reliable[0]
            prod_match = any(
                r["dip"] == PRODUCTION_CONFIG.dip_pct
                and r["prox"] == PRODUCTION_CONFIG.proximity_pct
                and r["sma"] == PRODUCTION_CONFIG.sma_filter_period
                for r in reliable[:3]
            )
            if prod_match:
                out.write("\nProduction config is in TOP 3 of grid search (min 10 entries). No changes needed.\n")
            else:
                out.write(f"\nBest reliable grid config (>={max(10, best['n_entries'])} entries): "
                          f"dip={best['dip']:.0%} prox={best['prox']:.1%} sma={best['sma']} "
                          f"(Sort={best['sortino']:.2f} Ret={best['return_pct']:.2f}% DD={best['max_dd_pct']:.2f}%)\n")
                if best["sortino"] > 0:
                    out.write(f"  -> Consider updating production to these params.\n")

    if window_results:
        neg_windows = [r for r in window_results if r["sortino"] < 0]
        if neg_windows:
            out.write(f"\nWARNING: {len(neg_windows)}/{len(window_results)} windows have negative Sortino.\n")
            for r in neg_windows:
                out.write(f"  {r['window']}: Sort={r['sortino']:.2f} Ret={r['return_pct']:.2f}%\n")
        else:
            out.write(f"\nAll {len(window_results)} windows profitable. Strategy is robust.\n")


def main():
    parser = argparse.ArgumentParser(description="Production audit for binance_worksteal")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--random-trials", type=int, default=20)
    args = parser.parse_args()

    if args.output is None:
        tag = datetime.now().strftime("%Y%m%d")
        args.output = f"binance_worksteal/production_audit_{tag}.txt"

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading {len(symbols)} symbols from {args.data_dir}")
    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols")

    if not all_bars:
        print("ERROR: No data loaded. Check --data-dir path.")
        return 1

    latest = max(df["timestamp"].max() for df in all_bars.values())
    recent_end = str(latest.date())
    recent_start = str((latest - pd.Timedelta(days=args.recent_days)).date())
    print(f"Data range up to: {recent_end}")
    print(f"Recent audit period: {recent_start} to {recent_end}")

    windows = build_windows(all_bars, args.window_days, args.n_windows)
    print(f"Windows: {windows}")

    config = PRODUCTION_CONFIG

    out_buf = io.StringIO()
    tee = _TeeWriter(sys.stdout, out_buf)

    tee.write(f"BINANCE WORKSTEAL PRODUCTION AUDIT\n")
    tee.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.write(f"Symbols: {len(all_bars)}\n")
    tee.write(f"Data through: {recent_end}\n")

    t0 = time.time()

    audit_df = run_filter_breakdown(all_bars, config, recent_start, recent_end, tee)

    window_results = run_multi_window_backtest(all_bars, config, windows, tee)

    sensitivity_end = recent_end
    sensitivity_start = str((latest - pd.Timedelta(days=90)).date())
    sensitivity_results, grid_results = run_sensitivity_analysis(
        all_bars, config, sensitivity_start, sensitivity_end, tee,
    )

    run_random_baseline(all_bars, config, sensitivity_start, sensitivity_end, tee, args.random_trials)

    generate_recommendations(audit_df, window_results, sensitivity_results, grid_results, tee)

    elapsed = time.time() - t0
    tee.write(f"\n{'='*70}\n")
    tee.write(f"Audit completed in {elapsed:.1f}s\n")
    tee.write(f"{'='*70}\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out_buf.getvalue())
    print(f"\nResults saved to {args.output}")

    return 0


class _TeeWriter:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)
            if hasattr(w, "flush"):
                w.flush()


if __name__ == "__main__":
    sys.exit(main() or 0)
