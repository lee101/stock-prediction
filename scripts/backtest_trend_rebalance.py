#!/usr/bin/env python3
"""Walk-forward backtest of trend-following rebalance strategy.

Tests across multiple rolling windows to measure true out-of-sample performance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from src.rebalance_trend_overlay import TrendOverlay, TrendOverlayConfig


def sim_rebalance(closes, alloc, fee=0.001, lag=2, margin_rate=0.0625):
    if lag > 0:
        closes = closes[lag:]
        alloc = alloc[:-lag]
    asset_ret = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
    al = alloc[:-1]
    al_delta = np.abs(np.diff(alloc))
    margin_per_step = margin_rate / 8766.0
    port_ret = al * asset_ret - al_delta[:len(al)] * fee - np.abs(al) * margin_per_step
    port_ret[0] -= np.abs(alloc[0]) * fee
    cum_log = np.cumsum(np.log1p(np.clip(port_ret, -0.99, None)))
    values = np.exp(cum_log)
    final_ret = values[-1] - 1.0
    running_max = np.maximum.accumulate(values)
    max_dd = np.max(1.0 - values / running_max)
    mean_ret = port_ret.mean()
    downside = np.minimum(port_ret, 0.0)
    downside_std = np.sqrt((downside ** 2).mean() + 1e-10)
    sortino = mean_ret / downside_std
    turnover = np.abs(np.diff(alloc[:len(al)+1])).mean()
    monthly_ret = (1 + final_ret) ** (30 * 24 / max(len(port_ret), 1)) - 1
    return {
        "return": final_ret, "sortino": sortino, "max_dd": max_dd,
        "turnover": turnover, "monthly_ret": monthly_ret,
        "mean_alloc": al.mean(), "n_bars": len(port_ret),
    }


def main():
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LINKUSD"]
    configs_to_test = [
        ("trend_48_336_v02", TrendOverlayConfig(fast_ema=48, slow_ema=336, target_vol=0.2)),
        ("trend_48_336_v03", TrendOverlayConfig(fast_ema=48, slow_ema=336, target_vol=0.3)),
        ("trend_168_720_v02", TrendOverlayConfig(fast_ema=168, slow_ema=720, target_vol=0.2)),
        ("trend_48_336_v02_dd10", TrendOverlayConfig(fast_ema=48, slow_ema=336, target_vol=0.2, dd_cut=0.10)),
        ("trend_48_336_v03_dd10", TrendOverlayConfig(fast_ema=48, slow_ema=336, target_vol=0.3, dd_cut=0.10)),
        ("constant_short_02", None),  # baseline: constant -0.2
    ]

    # Walk-forward: test on 90-day windows, sliding by 30 days
    window_days = 90
    step_days = 30
    window_bars = window_days * 24
    step_bars = step_days * 24

    for symbol in symbols:
        try:
            cfg = DatasetConfig(
                symbol=symbol,
                data_root=Path("trainingdatahourly/crypto"),
                forecast_cache_root=Path("binanceneural/forecast_cache"),
                sequence_length=72, validation_days=0,
                forecast_horizons=(1, 24), cache_only=True,
            )
            data = BinanceHourlyDataModule(cfg)
        except Exception as e:
            print(f"{symbol}: FAIL {e}")
            continue

        closes = data.train_dataset.closes.copy()
        total_days = len(closes) / 24
        print(f"\n{'='*70}")
        print(f"{symbol}: {len(closes)} bars ({total_days:.0f} days), price {closes[0]:.0f} -> {closes[-1]:.0f}")
        print(f"{'='*70}")

        # Need at least slow_ema bars of warmup before first test window
        warmup_bars = 720 + window_bars
        if len(closes) < warmup_bars + window_bars:
            print(f"  Not enough data (need {warmup_bars + window_bars} bars)")
            continue

        for config_name, overlay_cfg in configs_to_test:
            if overlay_cfg is None:
                # Constant short baseline
                overlay = None
            else:
                overlay = TrendOverlay(overlay_cfg)

            window_results = []
            start = warmup_bars
            while start + window_bars <= len(closes):
                window_closes = closes[start:start + window_bars]
                if overlay is not None:
                    # Use full history up to window start for EMA warmup
                    full_closes = closes[:start + window_bars]
                    alloc = overlay.compute_allocation(full_closes)
                    alloc = alloc[start:]  # only the test window
                else:
                    alloc = np.full(window_bars, -0.2)

                r = sim_rebalance(window_closes, alloc, lag=2)
                window_results.append(r)
                start += step_bars

            if not window_results:
                continue

            rets = [r["return"] for r in window_results]
            sorts = [r["sortino"] for r in window_results]
            dds = [r["max_dd"] for r in window_results]
            monthly = [r["monthly_ret"] for r in window_results]
            n_positive = sum(1 for r in rets if r > 0)
            n_total = len(rets)

            print(f"\n  {config_name} ({n_total} windows of {window_days}d):")
            print(f"    ret:  med={np.median(rets):+.2%} mean={np.mean(rets):+.2%} std={np.std(rets):.2%}")
            print(f"    sort: med={np.median(sorts):+.4f} mean={np.mean(sorts):+.4f}")
            print(f"    dd:   med={np.median(dds):.2%} max={np.max(dds):.2%}")
            print(f"    monthly: med={np.median(monthly):+.2%} mean={np.mean(monthly):+.2%}")
            print(f"    win_rate: {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")

            # Show per-window detail
            if n_total <= 20:
                for i, r in enumerate(window_results):
                    t0 = warmup_bars + i * step_bars
                    print(f"    [{i:2d}] bars {t0:6d}-{t0+window_bars:6d}: "
                          f"ret={r['return']:+.2%} sort={r['sortino']:+.4f} dd={r['max_dd']:.2%} "
                          f"alloc={r['mean_alloc']:+.3f}")


if __name__ == "__main__":
    main()
