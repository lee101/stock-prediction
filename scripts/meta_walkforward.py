#!/usr/bin/env python3
"""Walk-forward analysis of stock meta-selector with drawdown filter.

Splits the val period into multiple windows and runs the meta-selector
on each to test robustness.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, MktdData, INITIAL_CASH, P_CLOSE
from scripts.meta_strategy_backtest import (
    simulate_single_model, simulate_ensemble, run_meta_portfolio,
    select_top_k_momentum, build_mktd_from_csvs, ModelTrace,
)

ENSEMBLE_DIR = REPO / "pufferlib_market/meta_ensemble_prod"
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
    "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN",
]


def select_dd_filtered_momentum(traces, day_idx, lookback=3, top_k=1, max_dd_pct=0.05):
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx
        if end >= len(tr.equity_curve) or start >= end:
            scores.append((i, -999.0))
            continue
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
        return select_top_k_momentum(traces, day_idx, lookback=lookback, top_k=top_k)
    return selected


def run_dd_filtered_portfolio(data, traces, lookback=3, warmup=5, fee_rate=0.001, slippage_bps=5.0, max_dd_pct=0.05):
    S = data.num_symbols
    T = len(traces[0].daily_returns)
    eff_fee = fee_rate + slippage_bps / 10000.0
    cash = INITIAL_CASH
    positions = {}
    equity_history = [float(cash)]
    daily_rets = []
    num_trades = 0

    for t in range(T):
        eq_before = cash
        for sym, qty in positions.items():
            si = data.symbols.index(sym)
            price = float(data.prices[t, si, P_CLOSE])
            eq_before += qty * price

        if t < warmup:
            equity_history.append(eq_before)
            daily_rets.append(0.0)
            continue

        selected = select_dd_filtered_momentum(traces, t, lookback=lookback, top_k=1, max_dd_pct=max_dd_pct)
        target_symbols = set()
        for i in selected:
            sym = traces[i].held_symbols[t] if t < len(traces[i].held_symbols) else None
            if sym is not None:
                target_symbols.add(sym)

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
                if cash > 10 and price > 0:
                    qty = cash / (price * (1.0 + eff_fee))
                    cash -= qty * price * (1.0 + eff_fee)
                    positions[sym] = qty
                    num_trades += 1

        t_mark = min(t + 1, data.num_timesteps - 1)
        eq_after = cash
        for sym, qty in positions.items():
            si = data.symbols.index(sym)
            price = float(data.prices[t_mark, si, P_CLOSE])
            eq_after += qty * price
        daily_rets.append((eq_after - eq_before) / max(eq_before, 1e-8))
        equity_history.append(eq_after)

    eq_arr = np.array(equity_history)
    ret_arr = np.array(daily_rets)
    total_ret = (eq_arr[-1] - eq_arr[0]) / max(eq_arr[0], 1e-8)
    neg = ret_arr[ret_arr < 0]
    sort = float(np.mean(ret_arr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peak - eq_arr) / np.maximum(peak, 1e-8)))
    return total_ret, sort, max_dd, num_trades


def main():
    data_full, dates = build_mktd_from_csvs(REPO / "trainingdata", DEFAULT_SYMBOLS)
    print(f"Full data: {data_full.num_timesteps}d x {data_full.num_symbols}sym")
    print(f"Dates: {dates[0].date()} to {dates[-1].date()}\n")

    checkpoints = sorted(ENSEMBLE_DIR.glob("*.pt"))
    kw = dict(fee_rate=0.001, slippage_bps=5.0, fill_buffer_bps=5.0, decision_lag=2, device="cpu")

    # Walk-forward: test on rolling 63-day (3-month) windows
    window_size = 63
    step_size = 21  # 1-month steps
    n_windows = (data_full.num_timesteps - window_size) // step_size + 1

    print(f"Walk-forward: {window_size}d windows, {step_size}d steps, {n_windows} windows\n")
    print(f"{'Window':<15} {'Start':>12} {'End':>12} {'ddFilter':>10} {'Momentum':>10} {'Ensemble':>10}")
    print("-" * 72)

    dd_rets = []
    mom_rets = []
    ens_rets = []

    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        if end > data_full.num_timesteps:
            break

        data_w = MktdData(
            version=data_full.version,
            symbols=list(data_full.symbols),
            features=data_full.features[start:end].copy(),
            prices=data_full.prices[start:end].copy(),
            tradable=data_full.tradable[start:end].copy() if data_full.tradable is not None else None,
        )

        # Simulate models on this window
        traces = []
        for cp in checkpoints:
            try:
                tr = simulate_single_model(data_w, cp, **kw)
                traces.append(tr)
            except Exception:
                continue

        if len(traces) < 2:
            continue

        # DD-filtered meta
        dd_ret, dd_sort, dd_dd, _ = run_dd_filtered_portfolio(data_w, traces)
        dd_rets.append(dd_ret)

        # Standard momentum meta
        m = run_meta_portfolio(data_w, traces, top_k=1, lookback=3, warmup=3,
                              fee_rate=0.001, slippage_bps=5.0, selector="momentum")
        mom_rets.append(m.total_return)

        # Ensemble
        try:
            ens_tr = simulate_ensemble(data_w, checkpoints, **kw)
            ens_ret = ens_tr.equity_curve[-1] / ens_tr.equity_curve[0] - 1
            ens_rets.append(ens_ret)
        except Exception:
            ens_rets.append(0.0)

        w_start = dates[start].date() if start < len(dates) else "?"
        w_end = dates[min(end - 1, len(dates) - 1)].date() if end - 1 < len(dates) else "?"
        print(f"W{w:>3}           {w_start!s:>12} {w_end!s:>12} {dd_ret*100:>+10.2f}% {m.total_return*100:>+10.2f}% {ens_rets[-1]*100:>+10.2f}%")

    print(f"\n{'Summary':<15} {'ddFilter':>10} {'Momentum':>10} {'Ensemble':>10}")
    print("-" * 50)
    print(f"{'Mean':>15} {np.mean(dd_rets)*100:>+10.2f}% {np.mean(mom_rets)*100:>+10.2f}% {np.mean(ens_rets)*100:>+10.2f}%")
    print(f"{'Median':>15} {np.median(dd_rets)*100:>+10.2f}% {np.median(mom_rets)*100:>+10.2f}% {np.median(ens_rets)*100:>+10.2f}%")
    print(f"{'Std':>15} {np.std(dd_rets)*100:>10.2f}% {np.std(mom_rets)*100:>10.2f}% {np.std(ens_rets)*100:>10.2f}%")
    print(f"{'% positive':>15} {np.mean([r>0 for r in dd_rets])*100:>10.0f}% {np.mean([r>0 for r in mom_rets])*100:>10.0f}% {np.mean([r>0 for r in ens_rets])*100:>10.0f}%")
    print(f"{'Min':>15} {min(dd_rets)*100:>+10.2f}% {min(mom_rets)*100:>+10.2f}% {min(ens_rets)*100:>+10.2f}%")
    print(f"{'Max':>15} {max(dd_rets)*100:>+10.2f}% {max(mom_rets)*100:>+10.2f}% {max(ens_rets)*100:>+10.2f}%")

    # Annualized
    dd_monthly = np.mean(dd_rets) * (21.0 / window_size)
    mom_monthly = np.mean(mom_rets) * (21.0 / window_size)
    print(f"\nEstimated monthly: ddFilter={dd_monthly*100:+.2f}% momentum={mom_monthly*100:+.2f}%")


if __name__ == "__main__":
    main()
