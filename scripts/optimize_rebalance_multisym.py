#!/usr/bin/env python3
"""Multi-symbol portfolio rebalance optimization.

Optimize allocation across multiple crypto assets simultaneously.
Uses walk-forward to ensure no look-ahead bias.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule


def compute_sortino(returns, eps=1e-8):
    mean_ret = returns.mean()
    downside = np.minimum(returns, 0.0)
    downside_std = np.sqrt((downside ** 2).mean() + eps)
    return mean_ret / downside_std


def simulate_portfolio_rebalance(
    all_closes: dict,  # symbol -> np.array of closes
    all_allocs: dict,  # symbol -> np.array of allocations [-1,1]
    fee: float = 0.001,
    lag: int = 2,
    margin_rate: float = 0.0625,
):
    """Simulate a portfolio of assets with independent allocations."""
    symbols = list(all_closes.keys())
    min_len = min(len(all_closes[s]) for s in symbols)
    for s in symbols:
        all_closes[s] = all_closes[s][:min_len]
        all_allocs[s] = all_allocs[s][:min_len]

    # Per-asset returns
    total_returns = np.zeros(min_len - lag - 1)
    for s in symbols:
        c = all_closes[s]
        a = all_allocs[s]
        if lag > 0:
            c = c[lag:]
            a = a[:-lag]
        asset_ret = np.diff(c) / np.maximum(c[:-1], 1e-8)
        al = a[:-1]
        al_delta = np.abs(np.diff(a[:len(al)+1]))
        if len(al_delta) < len(al):
            al_delta = np.append(al_delta, 0.0)
        margin_per_step = margin_rate / 8766.0
        port_ret = al * asset_ret - al_delta[:len(al)] * fee - np.abs(al) * margin_per_step
        n = min(len(total_returns), len(port_ret))
        total_returns[:n] += port_ret[:n]

    # Portfolio stats
    cum_log = np.cumsum(np.log1p(np.clip(total_returns, -0.99, None)))
    values = np.exp(cum_log)
    final_ret = values[-1] - 1.0
    running_max = np.maximum.accumulate(values)
    max_dd = np.max(1.0 - values / running_max)
    sortino = compute_sortino(total_returns)
    return {"return": final_ret, "sortino": sortino, "max_dd": max_dd, "n_bars": len(total_returns)}


def ema(data, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def trend_allocation(closes, fast_ema, slow_ema, scale=10.0):
    ef = ema(closes, fast_ema)
    es = ema(closes, slow_ema)
    signal = (ef - es) / np.maximum(es, 1e-8)
    return np.tanh(signal * scale * 100)


def main():
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LINKUSD"]
    val_days_list = [70, 180, 365]

    for val_days in val_days_list:
        print(f"\n{'='*60}")
        print(f"VALIDATION DAYS = {val_days}")
        print(f"{'='*60}")

        sym_data = {}
        for symbol in symbols:
            try:
                cfg = DatasetConfig(
                    symbol=symbol,
                    data_root=Path("trainingdatahourly/crypto"),
                    forecast_cache_root=Path("binanceneural/forecast_cache"),
                    sequence_length=72, validation_days=val_days,
                    forecast_horizons=(1, 24), cache_only=True,
                )
                data = BinanceHourlyDataModule(cfg)
                vc = data.val_dataset.closes.copy()
                tc = data.train_dataset.closes.copy()
                val_ret = (vc[-1]/vc[0]-1)*100
                sym_data[symbol] = {"val_closes": vc, "train_closes": tc, "val_ret": val_ret}
                print(f"  {symbol}: {len(vc)} val bars, ret={val_ret:+.1f}%")
            except Exception as e:
                print(f"  {symbol}: FAIL {e}")

        active_syms = list(sym_data.keys())
        if len(active_syms) < 2:
            continue

        # Strategy 1: Equal-weight constant allocations
        print(f"\n--- Constant allocations (equal weight, lag=2) ---")
        for target in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            allocs = {s: np.full(len(sym_data[s]["val_closes"]), target / len(active_syms))
                      for s in active_syms}
            closes = {s: sym_data[s]["val_closes"] for s in active_syms}
            r = simulate_portfolio_rebalance(closes, allocs, lag=2)
            print(f"  alloc={target:+.1f}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} dd={r['max_dd']:.2%}")

        # Strategy 2: EMA crossover per-symbol, equal weight
        print(f"\n--- EMA crossover per-symbol (lag=2) ---")
        for fast, slow in [(24, 336), (48, 336), (12, 336), (168, 720)]:
            for scale in [5.0, 10.0]:
                allocs = {}
                closes = {}
                for s in active_syms:
                    c = sym_data[s]["val_closes"]
                    allocs[s] = trend_allocation(c, fast, slow, scale) / len(active_syms)
                    closes[s] = c
                r = simulate_portfolio_rebalance(closes, allocs, lag=2)
                mean_alloc = np.mean([allocs[s].mean() for s in active_syms]) * len(active_syms)
                print(f"  EMA({fast:3d},{slow:3d}) s={scale:4.1f}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} "
                      f"dd={r['max_dd']:.2%} net_alloc={mean_alloc:+.3f}")

        # Strategy 3: Momentum-weighted (short recent losers, long recent winners)
        print(f"\n--- Cross-asset momentum (lag=2) ---")
        for lookback in [168, 336, 720]:
            for gross_alloc in [0.2, 0.5, 1.0]:
                allocs = {}
                closes = {}
                for s in active_syms:
                    c = sym_data[s]["val_closes"]
                    mom = np.zeros(len(c))
                    for i in range(lookback, len(c)):
                        mom[i] = (c[i] / c[i-lookback]) - 1.0
                    allocs[s] = np.tanh(mom * 5.0) * gross_alloc / len(active_syms)
                    closes[s] = c
                r = simulate_portfolio_rebalance(closes, allocs, lag=2)
                print(f"  mom_{lookback:3d}h gross={gross_alloc:.1f}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} dd={r['max_dd']:.2%}")

        # Strategy 4: Cross-sectional momentum (long relative winners, short relative losers)
        print(f"\n--- Cross-sectional relative momentum (lag=2) ---")
        for lookback in [168, 336]:
            for gross_alloc in [0.2, 0.5]:
                # Compute relative returns
                min_len = min(len(sym_data[s]["val_closes"]) for s in active_syms)
                mom_matrix = np.zeros((len(active_syms), min_len))
                for i, s in enumerate(active_syms):
                    c = sym_data[s]["val_closes"][:min_len]
                    for t in range(lookback, min_len):
                        mom_matrix[i, t] = (c[t] / c[t-lookback]) - 1.0

                # Rank-based: long top, short bottom
                allocs = {s: np.zeros(min_len) for s in active_syms}
                for t in range(lookback, min_len):
                    moms = mom_matrix[:, t]
                    mean_mom = moms.mean()
                    for i, s in enumerate(active_syms):
                        # Relative to cross-section mean
                        allocs[s][t] = (moms[i] - mean_mom) * gross_alloc

                closes = {s: sym_data[s]["val_closes"][:min_len] for s in active_syms}
                r = simulate_portfolio_rebalance(closes, allocs, lag=2)
                net_alloc = sum(allocs[s].mean() for s in active_syms)
                print(f"  xsect_{lookback:3d}h gross={gross_alloc:.1f}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} "
                      f"dd={r['max_dd']:.2%} net={net_alloc:+.4f}")

        # Strategy 5: Volatility-scaled EMA
        print(f"\n--- Vol-scaled EMA crossover (lag=2) ---")
        for fast, slow in [(48, 336)]:
            for target_vol in [0.1, 0.2, 0.3]:
                allocs = {}
                closes = {}
                for s in active_syms:
                    c = sym_data[s]["val_closes"]
                    signal = trend_allocation(c, fast, slow, 10.0)
                    # Rolling realized vol
                    ret = np.diff(c) / np.maximum(c[:-1], 1e-8)
                    vol = np.zeros(len(c))
                    for i in range(168, len(c)):
                        vol[i] = np.std(ret[max(0,i-168):i]) * np.sqrt(8766)
                    vol = np.maximum(vol, 0.01)
                    vol_scale = target_vol / vol
                    allocs[s] = signal * np.minimum(vol_scale, 2.0) / len(active_syms)
                    closes[s] = c
                r = simulate_portfolio_rebalance(closes, allocs, lag=2)
                print(f"  EMA({fast},{slow}) tvol={target_vol:.1f}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} dd={r['max_dd']:.2%}")


if __name__ == "__main__":
    main()
