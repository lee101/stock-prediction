#!/usr/bin/env python3
"""Per-symbol contribution evaluator for work-stealing strategy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.strategy import (
    WorkStealConfig, load_daily_bars, run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE


PRODUCTION_CONFIG = WorkStealConfig(
    dip_pct=0.20,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    sma_filter_period=20,
    trailing_stop_pct=0.03,
    max_positions=5,
    max_hold_days=14,
)


_csim_fn = None
_csim_checked = False

def _get_csim_fn():
    global _csim_fn, _csim_checked
    if not _csim_checked:
        _csim_checked = True
        try:
            from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast
            _csim_fn = run_worksteal_backtest_fast
        except Exception:
            _csim_fn = None
    return _csim_fn


def _run_backtest(
    all_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_csim: bool = True,
) -> dict:
    fast_fn = _get_csim_fn() if use_csim else None
    if fast_fn is not None:
        try:
            return fast_fn(all_bars, config, start_date=start_date, end_date=end_date)
        except Exception:
            pass
    equity_df, trades, metrics = run_worksteal_backtest(
        all_bars, config, start_date=start_date, end_date=end_date,
    )
    return metrics


def _filter_bars(all_bars: Dict[str, pd.DataFrame], symbols: List[str]) -> Dict[str, pd.DataFrame]:
    return {s: all_bars[s] for s in symbols if s in all_bars}


def _compute_avg_hold_days(trades) -> float:
    entries = {}
    hold_days = []
    for t in trades:
        if t.side in ("buy", "short"):
            entries[t.symbol] = t.timestamp
        elif t.side in ("sell", "cover"):
            if t.symbol in entries:
                delta = (t.timestamp - entries[t.symbol]).days
                hold_days.append(max(1, delta))
                del entries[t.symbol]
    return float(np.mean(hold_days)) if hold_days else 0.0


def compute_rolling_windows(
    all_bars: Dict[str, pd.DataFrame],
    window_days: int = 60,
    n_windows: int = 3,
) -> List[Tuple[str, str]]:
    all_dates = set()
    for df in all_bars.values():
        ts = pd.to_datetime(df["timestamp"], utc=True)
        all_dates.update(ts.tolist())
    all_dates = sorted(all_dates)
    if not all_dates:
        return []

    latest = all_dates[-1]
    windows = []
    for i in range(n_windows):
        end = latest - pd.Timedelta(days=i * window_days)
        start = end - pd.Timedelta(days=window_days)
        if start < all_dates[0]:
            break
        windows.append((str(start.date()), str(end.date())))
    return windows


def evaluate_standalone(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    config: WorkStealConfig,
    windows: List[Tuple[str, str]],
    use_csim: bool = True,  # unused, always uses Python sim for trade details
) -> Dict[str, dict]:
    single_config = replace(config, max_positions=1)
    results = {}
    for sym in symbols:
        if sym not in all_bars:
            continue
        window_metrics = []
        all_hold_days = []
        for start, end in windows:
            _eq, trades, m = run_worksteal_backtest(
                {sym: all_bars[sym].copy()}, single_config,
                start_date=start, end_date=end,
            )
            window_metrics.append(m)
            all_hold_days.append(_compute_avg_hold_days(trades))
        avg_return = float(np.mean([m.get("total_return_pct", 0.0) for m in window_metrics]))
        avg_sortino = float(np.mean([m.get("sortino", 0.0) for m in window_metrics]))
        avg_trades = float(np.mean([m.get("n_trades", m.get("total_trades", 0)) for m in window_metrics]))
        avg_hold = float(np.mean([h for h in all_hold_days if h > 0])) if any(h > 0 for h in all_hold_days) else 0.0
        results[sym] = {
            "standalone_return": avg_return,
            "standalone_sortino": avg_sortino,
            "n_trades": avg_trades,
            "avg_hold_days": avg_hold,
            "per_window": window_metrics,
        }
    return results


def evaluate_leave_one_out(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    config: WorkStealConfig,
    windows: List[Tuple[str, str]],
    full_sortinos: List[float],
    use_csim: bool = True,
) -> Dict[str, dict]:
    results = {}
    for sym in symbols:
        if sym not in all_bars:
            continue
        remaining = [s for s in symbols if s != sym and s in all_bars]
        if not remaining:
            continue
        subset_bars = _filter_bars(all_bars, remaining)
        window_marginals = []
        for wi, (start, end) in enumerate(windows):
            m = _run_backtest(subset_bars, config, start_date=start, end_date=end, use_csim=use_csim)
            without_sortino = m.get("sortino", 0.0)
            marginal = full_sortinos[wi] - without_sortino
            window_marginals.append(marginal)
        results[sym] = {
            "marginal_contribution": float(np.mean(window_marginals)),
            "per_window_marginal": window_marginals,
        }
    return results


def evaluate_full_universe(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    config: WorkStealConfig,
    windows: List[Tuple[str, str]],
    use_csim: bool = True,
) -> Tuple[List[float], List[dict]]:
    full_bars = _filter_bars(all_bars, symbols)
    sortinos = []
    metrics_list = []
    for start, end in windows:
        m = _run_backtest(full_bars, config, start_date=start, end_date=end, use_csim=use_csim)
        sortinos.append(m.get("sortino", 0.0))
        metrics_list.append(m)
    return sortinos, metrics_list


def format_results_table(
    symbols: List[str],
    standalone: Dict[str, dict],
    leave_one_out: Dict[str, dict],
) -> str:
    rows = []
    for sym in symbols:
        sa = standalone.get(sym, {})
        loo = leave_one_out.get(sym, {})
        rows.append({
            "symbol": sym,
            "standalone_return": sa.get("standalone_return", 0.0),
            "standalone_sortino": sa.get("standalone_sortino", 0.0),
            "marginal_contribution": loo.get("marginal_contribution", 0.0),
            "n_trades": sa.get("n_trades", 0.0),
            "avg_hold_days": sa.get("avg_hold_days", 0.0),
        })
    rows.sort(key=lambda r: r["marginal_contribution"], reverse=True)

    header = f"{'Symbol':<12} {'StandRet%':>10} {'StandSort':>10} {'Marginal':>10} {'Trades':>8} {'AvgHold':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['symbol']:<12} {r['standalone_return']:>10.2f} {r['standalone_sortino']:>10.2f} "
            f"{r['marginal_contribution']:>10.3f} {r['n_trades']:>8.1f} {r['avg_hold_days']:>8.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_evaluation(
    data_dir: str,
    symbols: List[str],
    config: WorkStealConfig,
    window_days: int = 60,
    n_windows: int = 3,
    candidate_symbols: Optional[List[str]] = None,
    use_csim: bool = True,
) -> Tuple[str, List[dict]]:
    all_symbols = list(symbols)
    if candidate_symbols:
        for s in candidate_symbols:
            if s not in all_symbols:
                all_symbols.append(s)

    all_bars = load_daily_bars(data_dir, all_symbols)
    print(f"Loaded {len(all_bars)}/{len(all_symbols)} symbols")

    if not all_bars:
        return "ERROR: No data loaded", []

    windows = compute_rolling_windows(all_bars, window_days=window_days, n_windows=n_windows)
    if not windows:
        return "ERROR: Not enough data for rolling windows", []
    print(f"Windows: {windows}")

    eval_symbols = [s for s in all_symbols if s in all_bars]
    base_symbols = [s for s in symbols if s in all_bars]

    print(f"Running full universe backtest ({len(base_symbols)} symbols)...")
    full_sortinos, full_metrics = evaluate_full_universe(
        all_bars, base_symbols, config, windows, use_csim=use_csim,
    )
    avg_full_sortino = float(np.mean(full_sortinos))
    avg_full_return = float(np.mean([m.get("total_return_pct", 0.0) for m in full_metrics]))
    print(f"Full universe: avg return={avg_full_return:.2f}%, avg sortino={avg_full_sortino:.2f}")

    print(f"Running standalone evaluation for {len(eval_symbols)} symbols...")
    standalone = evaluate_standalone(all_bars, eval_symbols, config, windows, use_csim=use_csim)

    print(f"Running leave-one-out for {len(base_symbols)} symbols...")
    loo = evaluate_leave_one_out(all_bars, base_symbols, config, windows, full_sortinos, use_csim=use_csim)

    if candidate_symbols:
        for cs in candidate_symbols:
            if cs not in all_bars or cs in symbols:
                continue
            expanded = base_symbols + [cs]
            expanded_bars = _filter_bars(all_bars, expanded)
            marginals = []
            for wi, (start, end) in enumerate(windows):
                m = _run_backtest(expanded_bars, config, start_date=start, end_date=end, use_csim=use_csim)
                marginals.append(m.get("sortino", 0.0) - full_sortinos[wi])
            loo[cs] = {
                "marginal_contribution": float(np.mean(marginals)),
                "per_window_marginal": marginals,
            }

    table = format_results_table(eval_symbols, standalone, loo)

    result_rows = []
    for sym in eval_symbols:
        sa = standalone.get(sym, {})
        lo = loo.get(sym, {})
        result_rows.append({
            "symbol": sym,
            "standalone_return": sa.get("standalone_return", 0.0),
            "standalone_sortino": sa.get("standalone_sortino", 0.0),
            "marginal_contribution": lo.get("marginal_contribution", 0.0),
            "n_trades": sa.get("n_trades", 0.0),
            "avg_hold_days": sa.get("avg_hold_days", 0.0),
        })

    summary = [
        f"\nFull Universe ({len(base_symbols)} symbols):",
        f"  Avg Return: {avg_full_return:.2f}%",
        f"  Avg Sortino: {avg_full_sortino:.2f}",
        f"  Windows: {len(windows)}x{window_days}d",
        "",
        table,
    ]
    if candidate_symbols:
        cands = [r for r in result_rows if r["symbol"] in candidate_symbols]
        if cands:
            summary.append("\nCandidate symbols (positive marginal = improves universe):")
            for c in sorted(cands, key=lambda x: x["marginal_contribution"], reverse=True):
                sign = "+" if c["marginal_contribution"] >= 0 else ""
                summary.append(
                    f"  {c['symbol']:<12} marginal={sign}{c['marginal_contribution']:.3f} "
                    f"standalone_sort={c['standalone_sortino']:.2f}"
                )

    return "\n".join(summary), result_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-symbol contribution to work-stealing strategy")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--candidate-symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--dip-pct", type=float, default=0.20)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--sma-filter", type=int, default=20)
    parser.add_argument("--trailing-stop", type=float, default=0.03)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--no-csim", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or FULL_UNIVERSE
    config = WorkStealConfig(
        dip_pct=args.dip_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        sma_filter_period=args.sma_filter,
        trailing_stop_pct=args.trailing_stop,
        max_positions=args.max_positions,
        max_hold_days=args.max_hold_days,
    )

    output, rows = run_evaluation(
        data_dir=args.data_dir,
        symbols=symbols,
        config=config,
        window_days=args.days,
        n_windows=args.windows,
        candidate_symbols=args.candidate_symbols,
        use_csim=not args.no_csim,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
