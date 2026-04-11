#!/usr/bin/env python3
"""Test volatility-targeting rebalance strategies.

Instead of predicting direction, size positions based on realized volatility.
This is prediction-free and works at any decision lag.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule


def compute_sortino(returns, eps=1e-8):
    mean_ret = returns.mean()
    downside = np.minimum(returns, 0.0)
    downside_std = np.sqrt((downside ** 2).mean() + eps)
    return mean_ret / downside_std


def sim_rebalance(closes, alloc, fee=0.001, lag=0, margin_rate=0.0625):
    if lag > 0:
        closes = closes[lag:]
        alloc = alloc[:-lag]
    n = len(closes)
    asset_ret = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
    al = alloc[:-1]
    al_delta = np.abs(np.diff(alloc))
    margin_per_step = margin_rate / 8766.0
    port_ret = al * asset_ret[:len(al)]
    tc = al_delta[:len(al)] * fee
    mc = np.abs(al) * margin_per_step
    port_ret = port_ret - tc[:len(port_ret)] - mc
    cum_log = np.cumsum(np.log1p(np.clip(port_ret, -0.99, None)))
    values = np.exp(cum_log)
    final_ret = values[-1] - 1.0
    running_max = np.maximum.accumulate(values)
    max_dd = np.max(1.0 - values / running_max)
    sortino = compute_sortino(port_ret)
    turnover = np.abs(np.diff(alloc[:len(al)+1])).mean()
    return {"return": final_ret, "sortino": sortino, "max_dd": max_dd, "turnover": turnover,
            "mean_alloc": al.mean(), "trades_per_day": turnover * 24}


def ema(data, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def rolling_vol(closes, window=168):
    ret = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
    vol = np.zeros(len(closes))
    for i in range(window, len(closes)):
        vol[i] = np.std(ret[i-window:i]) * np.sqrt(8766)
    vol[:window] = vol[window]
    return np.maximum(vol, 0.01)


def main():
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LINKUSD"]

    for symbol in symbols:
        try:
            cfg = DatasetConfig(
                symbol=symbol,
                data_root=Path("trainingdatahourly/crypto"),
                forecast_cache_root=Path("binanceneural/forecast_cache"),
                sequence_length=72, validation_days=180,
                forecast_horizons=(1, 24), cache_only=True,
            )
            data = BinanceHourlyDataModule(cfg)
        except Exception as e:
            print(f"{symbol}: FAIL {e}")
            continue

        val_c = data.val_dataset.closes.copy()
        train_c = data.train_dataset.closes.copy()
        # Use a longer test period
        full_c = np.concatenate([train_c, val_c[72:]])
        val_ret = (val_c[-1]/val_c[0]-1)*100
        train_ret = (train_c[-1]/train_c[0]-1)*100

        print(f"\n{'='*60}")
        print(f"{symbol}: val={len(val_c)} bars ({val_ret:+.1f}%), train={len(train_c)} bars ({train_ret:+.1f}%)")
        print(f"{'='*60}")

        # Test on both val and train periods
        for period_name, closes in [("val_180d", val_c), ("train", train_c)]:
            print(f"\n  --- {period_name} ({len(closes)} bars) ---")

            # Baseline: constant allocations
            for ca in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                alloc = np.full(len(closes), ca)
                r = sim_rebalance(closes, alloc, lag=2)
                print(f"  const={ca:+.1f}: ret={r['return']:+7.2%} sort={r['sortino']:+7.4f} dd={r['max_dd']:6.2%}")

            # Vol-targeting: alloc = target_vol / realized_vol
            print()
            vol = rolling_vol(closes, 168)
            for target_vol in [0.1, 0.2, 0.3, 0.5]:
                for direction in [1.0, -1.0]:
                    alloc = direction * np.clip(target_vol / vol, 0, 2.0)
                    r = sim_rebalance(closes, alloc, lag=2)
                    d_str = "long" if direction > 0 else "short"
                    print(f"  vol_target={target_vol:.1f} {d_str:5s}: ret={r['return']:+7.2%} sort={r['sortino']:+7.4f} "
                          f"dd={r['max_dd']:6.2%} turn={r['turnover']:.4f}")

            # Trend + vol target: EMA crossover direction, vol-scaled size
            print()
            for fast, slow in [(48, 336), (168, 720)]:
                ef = ema(closes, fast)
                es = ema(closes, slow)
                trend = np.sign(ef - es)
                for target_vol in [0.2, 0.3]:
                    alloc = trend * np.clip(target_vol / vol, 0, 2.0)
                    r = sim_rebalance(closes, alloc, lag=2)
                    print(f"  trend_EMA({fast:3d},{slow:3d}) tvol={target_vol:.1f}: ret={r['return']:+7.2%} sort={r['sortino']:+7.4f} "
                          f"dd={r['max_dd']:6.2%} turn={r['turnover']:.4f}")

            # Drawdown-responsive: reduce alloc when in drawdown
            print()
            for base_alloc in [0.5, 1.0]:
                running_max = np.maximum.accumulate(closes)
                drawdown = 1.0 - closes / running_max
                for dd_threshold in [0.05, 0.10, 0.15]:
                    alloc = np.where(drawdown > dd_threshold, 0.0, base_alloc)
                    r = sim_rebalance(closes, alloc, lag=2)
                    print(f"  dd_cut={dd_threshold:.0%} base={base_alloc:.1f}: ret={r['return']:+7.2%} sort={r['sortino']:+7.4f} "
                          f"dd={r['max_dd']:6.2%} turn={r['turnover']:.4f}")

            # Combined: trend + vol + drawdown
            print()
            for fast, slow in [(48, 336)]:
                ef = ema(closes, fast)
                es = ema(closes, slow)
                trend = np.sign(ef - es)
                running_max = np.maximum.accumulate(closes)
                drawdown = 1.0 - closes / running_max
                for target_vol in [0.2, 0.3]:
                    for dd_cut in [0.10, 0.15]:
                        alloc = trend * np.clip(target_vol / vol, 0, 2.0)
                        alloc = np.where(drawdown > dd_cut, alloc * 0.2, alloc)
                        r = sim_rebalance(closes, alloc, lag=2)
                        print(f"  combined EMA({fast},{slow}) tvol={target_vol:.1f} ddcut={dd_cut:.0%}: "
                              f"ret={r['return']:+7.2%} sort={r['sortino']:+7.4f} dd={r['max_dd']:6.2%}")


if __name__ == "__main__":
    main()
