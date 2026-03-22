#!/usr/bin/env python3
"""Expanded hyperparameter sweep for work-stealing strategy.

Focused grid exploring dip thresholds, leverage, position counts,
trailing stops, profit targets, and stop losses. Uses C simulator
batch mode when available for ~10x speedup.

Multi-window evaluation with safety score ranking.
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig, load_daily_bars, run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SWEEP_GRID = {
    "dip_pct": [0.08, 0.10, 0.12, 0.15, 0.20],
    "max_leverage": [1.0, 2.0, 3.0, 5.0],
    "max_positions": [5, 7, 10, 15],
    "trailing_stop_pct": [0.02, 0.03, 0.05],
    "profit_target_pct": [0.10, 0.15, 0.20],
    "stop_loss_pct": [0.08, 0.10, 0.15],
}


def generate_grid(grid=None, max_trials=None, seed=42):
    grid = grid or SWEEP_GRID
    keys = list(grid.keys())
    values = list(grid.values())
    all_combos = list(itertools.product(*values))
    random.seed(seed)
    if max_trials and len(all_combos) > max_trials:
        combos = random.sample(all_combos, max_trials)
    else:
        combos = all_combos
    return keys, combos


def combo_to_config(keys, combo, cash=10000.0):
    params = dict(zip(keys, combo))
    return WorkStealConfig(
        initial_cash=cash,
        sma_filter_period=20,
        lookback_days=20,
        max_hold_days=14,
        proximity_pct=0.03,
        ref_price_method="high",
        **params,
    )


def compute_safety_score(mean_sortino, max_drawdown_pct):
    dd_abs = max(abs(max_drawdown_pct), 0.01)
    return mean_sortino / dd_abs


def build_windows(all_bars, window_days=60, n_windows=3):
    latest = max(df["timestamp"].max() for df in all_bars.values())
    windows = []
    for i in range(n_windows):
        end = latest - pd.Timedelta(days=i * window_days)
        start = end - pd.Timedelta(days=window_days)
        windows.append((str(start.date()), str(end.date())))
    return windows


def _try_load_csim_batch():
    try:
        from binance_worksteal.csim.fast_worksteal import run_worksteal_batch_fast
        return run_worksteal_batch_fast
    except Exception:
        return None


def eval_config_single_window_python(all_bars, config, start_date, end_date):
    try:
        equity_df, trades, metrics = run_worksteal_backtest(
            all_bars, config, start_date=start_date, end_date=end_date,
        )
    except Exception:
        return None
    if not metrics:
        return None
    metrics["n_trades"] = metrics.get("n_trades", len([t for t in trades if t.side in ("sell", "cover")]))
    exits = [t for t in trades if t.side in ("sell", "cover")]
    if exits:
        hold_days = []
        for t in exits:
            buys = [b for b in trades if b.symbol == t.symbol and b.side in ("buy", "short") and b.timestamp <= t.timestamp]
            if buys:
                entry = buys[-1]
                hold_days.append((t.timestamp - entry.timestamp).days)
        metrics["avg_hold_days"] = float(np.mean(hold_days)) if hold_days else 0.0
    else:
        metrics["avg_hold_days"] = 0.0
    return metrics


def eval_config_multi_window_python(all_bars, config, windows):
    window_metrics = []
    for start, end in windows:
        m = eval_config_single_window_python(all_bars, config, start, end)
        if m is None:
            return None
        window_metrics.append(m)
    return _aggregate_window_metrics(window_metrics)


def _aggregate_window_metrics(window_metrics):
    sortinos = [m.get("sortino", 0) for m in window_metrics]
    returns = [m.get("total_return_pct", 0) for m in window_metrics]
    drawdowns = [m.get("max_drawdown_pct", 0) for m in window_metrics]
    n_trades_list = [m.get("n_trades", 0) for m in window_metrics]
    win_rates = [m.get("win_rate", 0) for m in window_metrics]
    avg_holds = [m.get("avg_hold_days", 0) for m in window_metrics]

    mean_sortino = float(np.mean(sortinos))
    worst_dd = float(np.min(drawdowns))
    safety = compute_safety_score(mean_sortino, worst_dd)

    combined = {
        "mean_sortino": mean_sortino,
        "min_sortino": float(np.min(sortinos)),
        "mean_return_pct": float(np.mean(returns)),
        "max_drawdown_pct": worst_dd,
        "mean_win_rate": float(np.mean(win_rates)),
        "total_n_trades": int(np.sum(n_trades_list)),
        "mean_n_trades": float(np.mean(n_trades_list)),
        "avg_hold_days": float(np.mean(avg_holds)),
        "safety_score": safety,
        "n_windows": len(window_metrics),
    }
    for i, m in enumerate(window_metrics):
        combined[f"w{i}_sortino"] = m.get("sortino", 0)
        combined[f"w{i}_return_pct"] = m.get("total_return_pct", 0)
        combined[f"w{i}_drawdown_pct"] = m.get("max_drawdown_pct", 0)
        combined[f"w{i}_n_trades"] = m.get("n_trades", 0)
    return combined


def _eval_batch_csim_window(batch_fn, all_bars, configs, start, end):
    results = batch_fn(all_bars, configs, start_date=start, end_date=end)
    out = []
    for r in results:
        if not r or (r.get("total_trades", 0) == 0 and r.get("total_return", 0) == 0):
            out.append(None)
        else:
            r["n_trades"] = r.get("total_trades", 0)
            r["avg_hold_days"] = 0.0
            out.append(r)
    return out


def eval_batch_multi_window_csim(batch_fn, all_bars, configs, windows):
    all_window_results = []
    for start, end in windows:
        wr = _eval_batch_csim_window(batch_fn, all_bars, configs, start, end)
        all_window_results.append(wr)

    combined = []
    for ci in range(len(configs)):
        wms = []
        failed = False
        for wi in range(len(windows)):
            m = all_window_results[wi][ci]
            if m is None:
                failed = True
                break
            wms.append(m)
        if failed:
            combined.append(None)
        else:
            combined.append(_aggregate_window_metrics(wms))
    return combined


# For multiprocessing fallback
_mp_all_bars = None
_mp_windows = None


def _init_mp_worker(all_bars, windows):
    global _mp_all_bars, _mp_windows
    _mp_all_bars = all_bars
    _mp_windows = windows


def _mp_eval_config(args):
    keys, combo, cash = args
    config = combo_to_config(keys, combo, cash)
    return eval_config_multi_window_python(_mp_all_bars, config, _mp_windows)


def get_per_symbol_pnl(all_bars, config, windows):
    sym_pnl = {}
    for start, end in windows:
        try:
            _, trades, _ = run_worksteal_backtest(
                all_bars, config, start_date=start, end_date=end,
            )
        except Exception:
            continue
        exits = [t for t in trades if t.side in ("sell", "cover")]
        for t in exits:
            sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
    return sym_pnl


def run_sweep(
    all_bars, windows, output_csv,
    max_trials=None, cash=10000.0, n_workers=None,
):
    batch_fn = _try_load_csim_batch()
    use_csim = batch_fn is not None

    keys, combos = generate_grid(max_trials=max_trials)
    total = len(combos)

    print(f"Sweep: {total} configs (grid total: {5*4*4*3*3*3})")
    print(f"Windows: {len(windows)}")
    for i, (s, e) in enumerate(windows):
        print(f"  W{i}: {s} to {e}")
    print(f"Simulator: {'C batch' if use_csim else 'Python'}")

    t0 = time.time()
    results = []

    if use_csim:
        BATCH_SIZE = 256
        configs_all = [combo_to_config(keys, c, cash) for c in combos]
        params_all = [dict(zip(keys, c)) for c in combos]

        done = 0
        for bi in range(0, total, BATCH_SIZE):
            batch_configs = configs_all[bi:bi + BATCH_SIZE]
            batch_params = params_all[bi:bi + BATCH_SIZE]
            batch_results = eval_batch_multi_window_csim(batch_fn, all_bars, batch_configs, windows)

            for pi, multi in enumerate(batch_results):
                if multi is None:
                    continue
                row = {**batch_params[pi], **multi}
                results.append(row)

            done += len(batch_configs)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total} {rate:.1f}/s] {len(results)} valid", flush=True)
    else:
        n_workers = n_workers or max(1, cpu_count() - 1)
        if n_workers > 1 and total > 10:
            print(f"  Using {n_workers} workers")
            args_list = [(keys, c, cash) for c in combos]
            chunksize = max(1, total // (n_workers * 4))
            with Pool(n_workers, initializer=_init_mp_worker, initargs=(all_bars, windows)) as pool:
                iterator = pool.imap(_mp_eval_config, args_list, chunksize=chunksize)
                if tqdm:
                    iterator = tqdm(iterator, total=total, desc="Sweep")
                for ci, multi in enumerate(iterator):
                    if multi is None:
                        continue
                    params = dict(zip(keys, combos[ci]))
                    row = {**params, **multi}
                    results.append(row)
        else:
            iterator = range(total)
            if tqdm:
                iterator = tqdm(iterator, desc="Sweep")
            for ci in iterator:
                config = combo_to_config(keys, combos[ci], cash)
                multi = eval_config_multi_window_python(all_bars, config, windows)
                if multi is None:
                    continue
                params = dict(zip(keys, combos[ci]))
                row = {**params, **multi}
                results.append(row)

                if (ci + 1) % 50 == 0 and not tqdm:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / elapsed if elapsed > 0 else 0
                    print(f"  [{ci+1}/{total} {rate:.1f}/s] {len(results)} valid", flush=True)

    elapsed = time.time() - t0

    if not results:
        print("No valid results")
        return results

    df = pd.DataFrame(results)
    df = df.sort_values("safety_score", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results)} results to {output_csv}")
    print(f"Total time: {elapsed:.1f}s ({len(results)/max(elapsed,0.001):.1f} configs/s)")

    print(f"\nTop 20 by safety_score:")
    for rank, (_, r) in enumerate(df.head(20).iterrows(), 1):
        print(f"  #{rank:2d} safety={r['safety_score']:7.2f} "
              f"sort={r['mean_sortino']:6.2f} ret={r['mean_return_pct']:7.2f}% "
              f"dd={r['max_drawdown_pct']:7.2f}% wr={r['mean_win_rate']:5.1f}% "
              f"tr={r['total_n_trades']:4.0f} | "
              f"dip={r['dip_pct']:.0%} tp={r['profit_target_pct']:.0%} "
              f"sl={r['stop_loss_pct']:.0%} trail={r['trailing_stop_pct']:.0%} "
              f"pos={r['max_positions']:.0f} lev={r['max_leverage']:.0f}x")

    # Per-symbol PnL for top 20
    if not use_csim:
        print(f"\nPer-symbol PnL breakdown (top 20):")
        for rank, (_, r) in enumerate(df.head(20).iterrows(), 1):
            combo = tuple(r[k] for k in keys)
            config = combo_to_config(keys, combo, cash)
            sym_pnl = get_per_symbol_pnl(all_bars, config, windows)
            if sym_pnl:
                top_syms = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
                bot_syms = sorted(sym_pnl.items(), key=lambda x: x[1])[:3]
                top_str = " ".join(f"{s}:${p:.0f}" for s, p in top_syms)
                bot_str = " ".join(f"{s}:${p:.0f}" for s, p in bot_syms)
                print(f"  #{rank:2d} best=[{top_str}] worst=[{bot_str}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Expanded hyperparameter sweep for work-stealing strategy")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.output is None:
        tag = datetime.now().strftime("%Y%m%d")
        args.output = f"binance_worksteal/sweep_expanded_{tag}.csv"

    symbols = args.symbols or FULL_UNIVERSE
    print(f"Loading {len(symbols)} symbols from {args.data_dir}")
    all_bars = load_daily_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols")

    if not all_bars:
        print("ERROR: No data")
        return 1

    windows = build_windows(all_bars, window_days=args.days, n_windows=args.windows)

    run_sweep(
        all_bars, windows, args.output,
        max_trials=args.max_trials, cash=args.cash, n_workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
