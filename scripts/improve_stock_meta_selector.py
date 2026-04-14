#!/usr/bin/env python3
"""Sweep improved meta-selection algorithms for stock meta-strategy.

Tests: momentum-weighted alloc, multi-scale momentum, risk-adjusted sizing,
consistency filters, and model pruning.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, INITIAL_CASH, P_CLOSE
from scripts.meta_strategy_backtest import (
    simulate_single_model, simulate_ensemble, run_meta_portfolio,
    select_top_k_momentum, ModelTrace, MetaResult,
)

ENSEMBLE_DIR = REPO / "pufferlib_market/meta_ensemble_prod"
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
    "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN",
]


def select_momentum_weighted(traces, day_idx, lookback=3, top_k=1):
    """Same as momentum but return weights too."""
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx
        if end >= len(tr.equity_curve) or start >= end:
            scores.append((i, 0.0))
            continue
        ret = (tr.equity_curve[end] - tr.equity_curve[start]) / max(tr.equity_curve[start], 1e-8)
        scores.append((i, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def select_multiscale_momentum(traces, day_idx, top_k=1):
    """Combine short (3d) and medium (10d) momentum."""
    scores = []
    for i, tr in enumerate(traces):
        short_start = max(0, day_idx - 3)
        med_start = max(0, day_idx - 10)
        end = day_idx
        if end >= len(tr.equity_curve) or short_start >= end:
            scores.append((i, 0.0))
            continue
        short_ret = (tr.equity_curve[end] - tr.equity_curve[short_start]) / max(tr.equity_curve[short_start], 1e-8)
        if med_start < end:
            med_ret = (tr.equity_curve[end] - tr.equity_curve[med_start]) / max(tr.equity_curve[med_start], 1e-8)
        else:
            med_ret = 0.0
        # Weighted combo: 70% short, 30% medium
        score = 0.7 * short_ret + 0.3 * med_ret
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]


def select_consistency(traces, day_idx, lookback=5, top_k=1):
    """Select models with most positive days in lookback."""
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx
        if end >= len(tr.equity_curve) or start >= end - 1:
            scores.append((i, 0.0))
            continue
        pos_days = 0
        total_days = 0
        for d in range(start + 1, end):
            if d < len(tr.equity_curve) and tr.equity_curve[d - 1] > 0:
                daily_ret = (tr.equity_curve[d] - tr.equity_curve[d - 1]) / tr.equity_curve[d - 1]
                if daily_ret > 0:
                    pos_days += 1
                total_days += 1
        score = pos_days / max(total_days, 1)
        # Tiebreak by total return
        total_ret = (tr.equity_curve[end] - tr.equity_curve[start]) / max(tr.equity_curve[start], 1e-8)
        score = score + total_ret * 0.001
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]


def select_drawdown_filtered_momentum(traces, day_idx, lookback=3, top_k=1, max_dd_pct=0.05):
    """Momentum but filter out models in drawdown > threshold."""
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx
        if end >= len(tr.equity_curve) or start >= end:
            scores.append((i, -999.0))
            continue
        # Check recent drawdown
        recent = tr.equity_curve[max(0, end - 20):end + 1]
        peak = np.maximum.accumulate(recent)
        dd = (peak[-1] - recent[-1]) / max(peak[-1], 1e-8) if len(recent) > 0 else 0
        if dd > max_dd_pct:
            scores.append((i, -999.0))
            continue
        ret = (tr.equity_curve[end] - tr.equity_curve[start]) / max(tr.equity_curve[start], 1e-8)
        scores.append((i, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scores[:top_k] if s[1] > -998]
    if not selected:
        # Fallback: just use momentum without filter
        return select_top_k_momentum(traces, day_idx, lookback=lookback, top_k=top_k)
    return selected


def run_momentum_weighted_portfolio(
    data, traces, *, lookback=3, top_k=2, warmup=5,
    fee_rate=0.001, slippage_bps=5.0,
):
    """Like run_meta_portfolio but weights allocation by momentum."""
    S = data.num_symbols
    T = len(traces[0].daily_returns)
    slip = max(0.0, slippage_bps) / 10_000.0
    eff_fee = fee_rate + slip

    cash = INITIAL_CASH
    positions = {}
    equity_history = [float(cash)]
    daily_rets = []
    num_trades = 0

    for t in range(T):
        eq_before = cash
        for sym, qty in positions.items():
            si = data.symbols.index(sym) if sym in data.symbols else -1
            if si >= 0 and t < data.num_timesteps:
                price = float(data.prices[t, si, P_CLOSE])
                eq_before += qty * price

        if t < warmup:
            equity_history.append(eq_before)
            daily_rets.append(0.0)
            continue

        scored = select_momentum_weighted(traces, t, lookback=lookback, top_k=top_k)
        # Weight by momentum (softmax of returns)
        rets = np.array([max(s[1], 0.0) for s in scored])
        if rets.sum() > 0:
            weights = rets / rets.sum()
        else:
            weights = np.ones(len(scored)) / max(len(scored), 1)

        target_syms = {}
        for (idx, _ret), w in zip(scored, weights):
            sym = traces[idx].held_symbols[t] if t < len(traces[idx].held_symbols) else None
            if sym is not None:
                target_syms[sym] = target_syms.get(sym, 0.0) + w

        # Normalize weights
        total_w = sum(target_syms.values())
        if total_w > 0:
            target_syms = {s: w / total_w for s, w in target_syms.items()}

        # Close unwanted
        for sym in list(positions.keys()):
            if sym not in target_syms:
                si = data.symbols.index(sym) if sym in data.symbols else -1
                if si >= 0:
                    price = float(data.prices[t, si, P_CLOSE])
                    cash += positions[sym] * price * (1.0 - eff_fee)
                    num_trades += 1
                del positions[sym]

        # Open/resize
        for sym, w in target_syms.items():
            if sym not in positions:
                si = data.symbols.index(sym) if sym in data.symbols else -1
                if si >= 0:
                    price = float(data.prices[t, si, P_CLOSE])
                    budget = cash * w
                    if budget > 10.0 and price > 0:
                        qty = budget / (price * (1.0 + eff_fee))
                        cost = qty * price * (1.0 + eff_fee)
                        cash -= cost
                        positions[sym] = qty
                        num_trades += 1

        t_mark = min(t + 1, data.num_timesteps - 1)
        eq_after = cash
        for sym, qty in positions.items():
            si = data.symbols.index(sym) if sym in data.symbols else -1
            if si >= 0:
                price = float(data.prices[t_mark, si, P_CLOSE])
                eq_after += qty * price

        daily_rets.append((eq_after - eq_before) / max(eq_before, 1e-8))
        equity_history.append(eq_after)

    # Final close
    for sym in list(positions.keys()):
        si = data.symbols.index(sym) if sym in data.symbols else -1
        if si >= 0:
            price = float(data.prices[-1, si, P_CLOSE])
            cash += positions[sym] * price * (1.0 - eff_fee)
            num_trades += 1
        del positions[sym]

    eq_arr = np.array(equity_history)
    ret_arr = np.array(daily_rets)
    total_return = (eq_arr[-1] - eq_arr[0]) / max(eq_arr[0], 1e-8)
    neg = ret_arr[ret_arr < 0]
    sortino = float(np.mean(ret_arr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peak - eq_arr) / np.maximum(peak, 1e-8)))

    return MetaResult(
        total_return=total_return, sortino=sortino, max_drawdown=max_dd,
        num_trades=num_trades, daily_returns=ret_arr, equity_curve=eq_arr,
        selected_models=[], held_symbols=[],
    )


def main():
    from scripts.meta_strategy_backtest import build_mktd_from_csvs

    data, dates = build_mktd_from_csvs(REPO / "trainingdata", DEFAULT_SYMBOLS)
    # Trim to last 95 days (OOS period)
    if data.num_timesteps > 95:
        from pufferlib_market.hourly_replay import MktdData
        start = data.num_timesteps - 95
        data = MktdData(
            version=data.version,
            symbols=list(data.symbols),
            features=data.features[start:].copy(),
            prices=data.prices[start:].copy(),
            tradable=data.tradable[start:].copy() if data.tradable is not None else None,
        )
        dates = dates[start:]
    print(f"data: {data.num_timesteps}d x {data.num_symbols}sym (last 95d)\n")

    kw = dict(fee_rate=0.001, slippage_bps=5.0, fill_buffer_bps=5.0, decision_lag=2, device="cpu")

    checkpoints = sorted(ENSEMBLE_DIR.glob("*.pt"))
    print(f"checkpoints: {len(checkpoints)}")

    # Simulate each model
    traces = []
    for i, cp in enumerate(checkpoints):
        try:
            tr = simulate_single_model(data, cp, **kw)
            traces.append(tr)
        except Exception as e:
            print(f"  {cp.stem} failed: {e}")
    print(f"simulated {len(traces)} models\n")

    # Individual model returns
    print("Individual model returns (top 10):")
    model_rets = [(tr.name, tr.equity_curve[-1] / tr.equity_curve[0] - 1) for tr in traces]
    model_rets.sort(key=lambda x: x[1], reverse=True)
    for name, ret in model_rets[:10]:
        print(f"  {name}: {ret*100:+.2f}%")
    print()

    # Baseline: standard momentum lb=3 k=1
    print(f"{'Algorithm':<40} {'Ret%':>10} {'Sort':>8} {'MaxDD%':>8} {'Trades':>8}")
    print("-" * 78)

    results = []

    # Standard momentum sweep
    for lb in [2, 3, 5, 7, 10]:
        for k in [1, 2, 3]:
            if k > len(traces):
                continue
            m = run_meta_portfolio(
                data, traces, top_k=k, lookback=lb, warmup=max(lb, 3),
                fee_rate=0.001, slippage_bps=5.0, selector="momentum",
            )
            tag = f"momentum lb={lb} k={k}"
            print(f"{tag:<40} {m.total_return*100:>+10.2f}% {m.sortino:>8.2f} {m.max_drawdown*100:>8.2f}% {m.num_trades:>8}")
            results.append((tag, m.total_return, m.sortino, m.max_drawdown))

    # Multiscale momentum
    for k in [1, 2]:
        class MultiScaleTraces:
            pass
        m = run_meta_portfolio(
            data, traces, top_k=k, lookback=3, warmup=5,
            fee_rate=0.001, slippage_bps=5.0, selector="momentum",
        )
        # We need to hack in the multiscale selection. Let me do it inline.

    # For advanced selectors, need to run the portfolio loop ourselves
    # Multiscale momentum
    for k in [1, 2]:
        T = len(traces[0].daily_returns)
        cash = INITIAL_CASH
        positions = {}
        equity_history = [float(cash)]
        daily_rets_arr = []
        num_trades = 0
        eff_fee = 0.001 + 5.0 / 10000.0

        for t in range(T):
            eq_before = cash
            for sym, qty in positions.items():
                si = data.symbols.index(sym)
                price = float(data.prices[t, si, P_CLOSE])
                eq_before += qty * price

            if t < 5:
                equity_history.append(eq_before)
                daily_rets_arr.append(0.0)
                continue

            selected = select_multiscale_momentum(traces, t, top_k=k)
            target_symbols = set()
            for i in selected:
                sym = traces[i].held_symbols[t] if t < len(traces[i].held_symbols) else None
                if sym is not None:
                    target_symbols.add(sym)
            alloc = 1.0 / max(len(target_symbols), 1) if target_symbols else 0.0
            for sym in list(positions.keys()):
                if sym not in target_symbols:
                    si = data.symbols.index(sym)
                    price = float(data.prices[t, si, P_CLOSE])
                    cash += positions[sym] * price * (1.0 - eff_fee)
                    num_trades += 1
                    del positions[sym]
            for sym in target_symbols:
                if sym not in positions:
                    si = data.symbols.index(sym)
                    price = float(data.prices[t, si, P_CLOSE])
                    budget = cash * alloc
                    if budget > 10.0 and price > 0:
                        qty = budget / (price * (1.0 + eff_fee))
                        cash -= qty * price * (1.0 + eff_fee)
                        positions[sym] = qty
                        num_trades += 1
            t_mark = min(t + 1, data.num_timesteps - 1)
            eq_after = cash
            for sym, qty in positions.items():
                si = data.symbols.index(sym)
                price = float(data.prices[t_mark, si, P_CLOSE])
                eq_after += qty * price
            daily_rets_arr.append((eq_after - eq_before) / max(eq_before, 1e-8))
            equity_history.append(eq_after)

        eq_arr = np.array(equity_history)
        ret_arr = np.array(daily_rets_arr)
        total_ret = (eq_arr[-1] - eq_arr[0]) / max(eq_arr[0], 1e-8)
        neg = ret_arr[ret_arr < 0]
        sort = float(np.mean(ret_arr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
        peak = np.maximum.accumulate(eq_arr)
        max_dd = float(np.max((peak - eq_arr) / np.maximum(peak, 1e-8)))
        tag = f"multiscale k={k}"
        print(f"{tag:<40} {total_ret*100:>+10.2f}% {sort:>8.2f} {max_dd*100:>8.2f}% {num_trades:>8}")
        results.append((tag, total_ret, sort, max_dd))

    # Consistency selector
    for lb in [3, 5, 7]:
        for k in [1, 2]:
            T = len(traces[0].daily_returns)
            cash = INITIAL_CASH
            positions = {}
            equity_history = [float(cash)]
            daily_rets_arr = []
            num_trades = 0
            eff_fee = 0.001 + 5.0 / 10000.0

            for t in range(T):
                eq_before = cash
                for sym, qty in positions.items():
                    si = data.symbols.index(sym)
                    price = float(data.prices[t, si, P_CLOSE])
                    eq_before += qty * price

                if t < max(lb, 3):
                    equity_history.append(eq_before)
                    daily_rets_arr.append(0.0)
                    continue

                selected = select_consistency(traces, t, lookback=lb, top_k=k)
                target_symbols = set()
                for i in selected:
                    sym = traces[i].held_symbols[t] if t < len(traces[i].held_symbols) else None
                    if sym is not None:
                        target_symbols.add(sym)
                alloc = 1.0 / max(len(target_symbols), 1) if target_symbols else 0.0
                for sym in list(positions.keys()):
                    if sym not in target_symbols:
                        si = data.symbols.index(sym)
                        price = float(data.prices[t, si, P_CLOSE])
                        cash += positions[sym] * price * (1.0 - eff_fee)
                        num_trades += 1
                        del positions[sym]
                for sym in target_symbols:
                    if sym not in positions:
                        si = data.symbols.index(sym)
                        price = float(data.prices[t, si, P_CLOSE])
                        budget = cash * alloc
                        if budget > 10.0 and price > 0:
                            qty = budget / (price * (1.0 + eff_fee))
                            cash -= qty * price * (1.0 + eff_fee)
                            positions[sym] = qty
                            num_trades += 1
                t_mark = min(t + 1, data.num_timesteps - 1)
                eq_after = cash
                for sym, qty in positions.items():
                    si = data.symbols.index(sym)
                    price = float(data.prices[t_mark, si, P_CLOSE])
                    eq_after += qty * price
                daily_rets_arr.append((eq_after - eq_before) / max(eq_before, 1e-8))
                equity_history.append(eq_after)

            eq_arr = np.array(equity_history)
            ret_arr = np.array(daily_rets_arr)
            total_ret = (eq_arr[-1] - eq_arr[0]) / max(eq_arr[0], 1e-8)
            neg = ret_arr[ret_arr < 0]
            sort = float(np.mean(ret_arr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
            peak = np.maximum.accumulate(eq_arr)
            max_dd = float(np.max((peak - eq_arr) / np.maximum(peak, 1e-8)))
            tag = f"consistency lb={lb} k={k}"
            print(f"{tag:<40} {total_ret*100:>+10.2f}% {sort:>8.2f} {max_dd*100:>8.2f}% {num_trades:>8}")
            results.append((tag, total_ret, sort, max_dd))

    # Drawdown-filtered momentum
    for lb in [3, 5]:
        for dd_thresh in [0.03, 0.05, 0.10]:
            T = len(traces[0].daily_returns)
            cash = INITIAL_CASH
            positions = {}
            equity_history = [float(cash)]
            daily_rets_arr = []
            num_trades = 0
            eff_fee = 0.001 + 5.0 / 10000.0

            for t in range(T):
                eq_before = cash
                for sym, qty in positions.items():
                    si = data.symbols.index(sym)
                    price = float(data.prices[t, si, P_CLOSE])
                    eq_before += qty * price

                if t < max(lb, 3):
                    equity_history.append(eq_before)
                    daily_rets_arr.append(0.0)
                    continue

                selected = select_drawdown_filtered_momentum(traces, t, lookback=lb, top_k=1, max_dd_pct=dd_thresh)
                target_symbols = set()
                for i in selected:
                    sym = traces[i].held_symbols[t] if t < len(traces[i].held_symbols) else None
                    if sym is not None:
                        target_symbols.add(sym)
                alloc = 1.0 / max(len(target_symbols), 1) if target_symbols else 0.0
                for sym in list(positions.keys()):
                    if sym not in target_symbols:
                        si = data.symbols.index(sym)
                        price = float(data.prices[t, si, P_CLOSE])
                        cash += positions[sym] * price * (1.0 - eff_fee)
                        num_trades += 1
                        del positions[sym]
                for sym in target_symbols:
                    if sym not in positions:
                        si = data.symbols.index(sym)
                        price = float(data.prices[t, si, P_CLOSE])
                        budget = cash * alloc
                        if budget > 10.0 and price > 0:
                            qty = budget / (price * (1.0 + eff_fee))
                            cash -= qty * price * (1.0 + eff_fee)
                            positions[sym] = qty
                            num_trades += 1
                t_mark = min(t + 1, data.num_timesteps - 1)
                eq_after = cash
                for sym, qty in positions.items():
                    si = data.symbols.index(sym)
                    price = float(data.prices[t_mark, si, P_CLOSE])
                    eq_after += qty * price
                daily_rets_arr.append((eq_after - eq_before) / max(eq_before, 1e-8))
                equity_history.append(eq_after)

            eq_arr = np.array(equity_history)
            ret_arr = np.array(daily_rets_arr)
            total_ret = (eq_arr[-1] - eq_arr[0]) / max(eq_arr[0], 1e-8)
            neg = ret_arr[ret_arr < 0]
            sort = float(np.mean(ret_arr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
            peak = np.maximum.accumulate(eq_arr)
            max_dd = float(np.max((peak - eq_arr) / np.maximum(peak, 1e-8)))
            tag = f"dd_filter lb={lb} dd<{dd_thresh*100:.0f}%"
            print(f"{tag:<40} {total_ret*100:>+10.2f}% {sort:>8.2f} {max_dd*100:>8.2f}% {num_trades:>8}")
            results.append((tag, total_ret, sort, max_dd))

    # Momentum-weighted allocation
    for lb in [3, 5]:
        for k in [2, 3]:
            m = run_momentum_weighted_portfolio(
                data, traces, lookback=lb, top_k=k, warmup=max(lb, 3),
                fee_rate=0.001, slippage_bps=5.0,
            )
            tag = f"mom_weighted lb={lb} k={k}"
            print(f"{tag:<40} {m.total_return*100:>+10.2f}% {m.sortino:>8.2f} {m.max_drawdown*100:>8.2f}% {m.num_trades:>8}")
            results.append((tag, m.total_return, m.sortino, m.max_drawdown))

    # Ensemble baseline
    ens_tr = simulate_ensemble(data, checkpoints, **kw)
    ens_ret = ens_tr.equity_curve[-1] / ens_tr.equity_curve[0] - 1
    ens_dr = np.diff(ens_tr.equity_curve) / np.maximum(ens_tr.equity_curve[:-1], 1e-8)
    neg = ens_dr[ens_dr < 0]
    ens_sort = float(np.mean(ens_dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
    ens_peak = np.maximum.accumulate(ens_tr.equity_curve)
    ens_dd = float(np.max((ens_peak - ens_tr.equity_curve) / np.maximum(ens_peak, 1e-8)))
    print(f"{'ensemble (21m softmax)':<40} {ens_ret*100:>+10.2f}% {ens_sort:>8.2f} {ens_dd*100:>8.2f}%")

    # Top 5
    print(f"\nTop 5 by return:")
    results.sort(key=lambda x: x[1], reverse=True)
    for tag, ret, sort, dd in results[:5]:
        monthly = (1 + ret) ** (21.0 / data.num_timesteps) - 1
        print(f"  {tag}: ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}% monthly={monthly*100:+.2f}%")

    print(f"\nTop 5 by sortino:")
    results.sort(key=lambda x: x[2], reverse=True)
    for tag, ret, sort, dd in results[:5]:
        monthly = (1 + ret) ** (21.0 / data.num_timesteps) - 1
        print(f"  {tag}: ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}% monthly={monthly*100:+.2f}%")


if __name__ == "__main__":
    main()
