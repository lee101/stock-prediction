#!/usr/bin/env python3
"""Grid search over EMA crossover parameters for rebalance allocation.

Uses pre-computed price data to find optimal EMA fast/slow periods
and allocation scaling for lag=2 profitability.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule


def ema(data, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def simulate_rebalance_np(closes, alloc, fee=0.001, lag=0, margin_rate=0.0625):
    if lag > 0:
        closes = closes[lag:]
        alloc = alloc[:-lag]
    n = len(closes)
    asset_ret = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
    al = alloc[:-1]
    al_delta = np.abs(np.diff(alloc))
    margin_per_step = margin_rate / 8766.0
    port_ret = al * asset_ret - al_delta * fee - np.abs(al) * margin_per_step
    init_fee = np.abs(alloc[0]) * fee
    port_ret[0] -= init_fee
    cum_log = np.cumsum(np.log1p(np.clip(port_ret, -0.99, None)))
    values = np.exp(cum_log)
    final_ret = values[-1] - 1.0
    running_max = np.maximum.accumulate(values)
    max_dd = np.max(1.0 - values / running_max)
    excess = port_ret
    mean_ret = excess.mean()
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt((downside ** 2).mean() + 1e-10)
    sortino = mean_ret / downside_std
    turnover = np.abs(np.diff(alloc)).mean()
    return {"return": final_ret, "sortino": sortino, "max_dd": max_dd, "turnover": turnover, "mean_alloc": alloc.mean()}


def main():
    dataset_cfg = DatasetConfig(
        symbol="BTCUSD",
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=70,
        forecast_horizons=(1, 24),
        cache_only=True,
    )
    data = BinanceHourlyDataModule(dataset_cfg)
    val_ds = data.val_dataset
    closes = val_ds.closes.copy()
    print(f"Val: {len(closes)} bars, price {closes[0]:.0f} -> {closes[-1]:.0f} ({(closes[-1]/closes[0]-1)*100:+.1f}%)")

    # Also get train closes for out-of-sample check
    train_ds = data.train_dataset
    train_closes = train_ds.closes.copy()
    print(f"Train: {len(train_closes)} bars, price {train_closes[0]:.0f} -> {train_closes[-1]:.0f}")
    print()

    # EMA crossover grid
    fast_periods = [6, 12, 24, 48, 72, 96, 168]
    slow_periods = [48, 96, 168, 336, 504, 720]
    scales = [0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            ema_fast = ema(closes, fast)
            ema_slow = ema(closes, slow)
            signal = (ema_fast - ema_slow) / np.maximum(ema_slow, 1e-8)
            for scale in scales:
                alloc = np.tanh(signal * scale * 100)
                r = simulate_rebalance_np(closes, alloc, fee=0.001, lag=2, margin_rate=0.0625)
                results.append({
                    "fast": fast, "slow": slow, "scale": scale,
                    **r,
                })

    results.sort(key=lambda x: x["sortino"], reverse=True)
    print("=== Top 20 EMA configs at lag=2 ===")
    print(f"{'fast':>5} {'slow':>5} {'scale':>6} {'ret':>8} {'sort':>8} {'dd':>7} {'turn':>7} {'alloc':>7}")
    for r in results[:20]:
        print(f"{r['fast']:5d} {r['slow']:5d} {r['scale']:6.1f} {r['return']:+7.2%} {r['sortino']:+7.3f} "
              f"{r['max_dd']:6.2%} {r['turnover']:6.4f} {r['mean_alloc']:+6.3f}")

    # Test best config across all lags
    best = results[0]
    print(f"\n=== Best config: EMA({best['fast']},{best['slow']}) scale={best['scale']} ===")
    ema_fast = ema(closes, best["fast"])
    ema_slow = ema(closes, best["slow"])
    signal = (ema_fast - ema_slow) / np.maximum(ema_slow, 1e-8)
    alloc = np.tanh(signal * best["scale"] * 100)
    for lag in [0, 1, 2, 3, 4]:
        r = simulate_rebalance_np(closes, alloc, fee=0.001, lag=lag, margin_rate=0.0625)
        print(f"  lag={lag}: ret={r['return']:+.2%} sort={r['sortino']:+.3f} dd={r['max_dd']:.2%} turn={r['turnover']:.4f}")

    # Test on train data (out-of-sample for this grid search)
    print(f"\n=== Best config on TRAIN data (out-of-sample) ===")
    ema_fast_tr = ema(train_closes, best["fast"])
    ema_slow_tr = ema(train_closes, best["slow"])
    signal_tr = (ema_fast_tr - ema_slow_tr) / np.maximum(ema_slow_tr, 1e-8)
    alloc_tr = np.tanh(signal_tr * best["scale"] * 100)
    for lag in [0, 1, 2, 3]:
        r = simulate_rebalance_np(train_closes, alloc_tr, fee=0.001, lag=lag, margin_rate=0.0625)
        print(f"  train lag={lag}: ret={r['return']:+.2%} sort={r['sortino']:+.3f} dd={r['max_dd']:.2%}")

    # Also try constant allocations for baseline
    print(f"\n=== Constant allocation baselines (lag=2) ===")
    for c in [-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0]:
        alloc_const = np.full(len(closes), c)
        r = simulate_rebalance_np(closes, alloc_const, fee=0.001, lag=2, margin_rate=0.0625)
        print(f"  alloc={c:+.1f}: ret={r['return']:+.2%} sort={r['sortino']:+.3f} dd={r['max_dd']:.2%}")

    # Momentum-based allocation (multi-period returns)
    print(f"\n=== Momentum-based allocation (lag=2) ===")
    for lookback in [24, 48, 168, 336, 720]:
        ret_lb = np.zeros(len(closes))
        for i in range(lookback, len(closes)):
            ret_lb[i] = (closes[i] / closes[i - lookback]) - 1.0
        for scale in [1.0, 5.0, 10.0]:
            alloc_mom = np.tanh(ret_lb * scale)
            r = simulate_rebalance_np(closes, alloc_mom, fee=0.001, lag=2, margin_rate=0.0625)
            if abs(r["sortino"]) > 0.1:
                print(f"  mom_{lookback}h scale={scale}: ret={r['return']:+.2%} sort={r['sortino']:+.3f} "
                      f"dd={r['max_dd']:.2%} turn={r['turnover']:.4f}")

    # Combined: EMA crossover + momentum
    print(f"\n=== Combined EMA + Momentum (lag=2) ===")
    for mom_lb in [168, 336, 720]:
        ret_lb = np.zeros(len(closes))
        for i in range(mom_lb, len(closes)):
            ret_lb[i] = (closes[i] / closes[i - mom_lb]) - 1.0
        ema_fast = ema(closes, best["fast"])
        ema_slow = ema(closes, best["slow"])
        ema_signal = (ema_fast - ema_slow) / np.maximum(ema_slow, 1e-8)
        for ema_w in [0.3, 0.5, 0.7]:
            combined = ema_w * np.tanh(ema_signal * best["scale"] * 100) + (1 - ema_w) * np.tanh(ret_lb * 5.0)
            alloc_comb = np.tanh(combined)
            r = simulate_rebalance_np(closes, alloc_comb, fee=0.001, lag=2, margin_rate=0.0625)
            print(f"  ema_w={ema_w} mom={mom_lb}h: ret={r['return']:+.2%} sort={r['sortino']:+.3f} "
                  f"dd={r['max_dd']:.2%} turn={r['turnover']:.4f}")


if __name__ == "__main__":
    main()
