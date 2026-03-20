#!/usr/bin/env python3
"""Parity tests: compare Python strategy.py vs C fast_worksteal.py results."""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from binance_worksteal.strategy import WorkStealConfig, run_worksteal_backtest, compute_metrics
from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast, run_worksteal_batch_fast


def make_synthetic_bars(n_symbols=5, n_days=120, seed=42):
    rng = np.random.RandomState(seed)
    all_bars = {}
    base_date = pd.Timestamp("2025-01-01", tz="UTC")
    for s in range(n_symbols):
        sym = f"SYM{s}USD"
        price = 100.0 + s * 50
        rows = []
        for d in range(n_days):
            ret = rng.normal(0, 0.03)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.01)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.01)))
            rows.append({
                "timestamp": base_date + pd.Timedelta(days=d),
                "open": o, "high": h, "low": l, "close": c,
                "volume": rng.uniform(1e6, 1e8),
            })
            price = c
        all_bars[sym] = pd.DataFrame(rows)
    return all_bars


def make_dip_scenario(n_days=60, seed=123):
    """Create data that guarantees dip entries and exits."""
    rng = np.random.RandomState(seed)
    all_bars = {}
    base_date = pd.Timestamp("2025-01-01", tz="UTC")

    for s_idx, (sym, start_price) in enumerate([
        ("AAUSD", 100.0), ("BBUSD", 200.0), ("CCUSD", 50.0),
    ]):
        rows = []
        price = start_price
        for d in range(n_days):
            if d < 20:
                ret = rng.uniform(0.005, 0.02)
            elif d == 20:
                ret = -0.15
            elif d < 30:
                ret = rng.uniform(-0.01, 0.02)
            elif d == 35:
                ret = -0.12
            else:
                ret = rng.normal(0.002, 0.015)

            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.005)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.005)))
            rows.append({
                "timestamp": base_date + pd.Timedelta(days=d),
                "open": o, "high": h, "low": l, "close": c,
                "volume": rng.uniform(1e6, 1e8),
            })
            price = c
        all_bars[sym] = pd.DataFrame(rows)
    return all_bars


def compare_metrics(py_metrics, c_metrics, config_label="", tol=0.02):
    keys = ["total_return", "sortino", "max_drawdown", "final_equity", "win_rate"]
    ok = True
    for k in keys:
        py_val = py_metrics.get(k, 0) or 0
        c_val = c_metrics.get(k, 0) or 0
        if k == "win_rate":
            threshold = 15.0
        elif k in ("sortino", "sharpe"):
            threshold = max(abs(py_val) * 0.15, 0.5)
        elif k == "final_equity":
            threshold = max(abs(py_val) * 0.02, 10.0)
        else:
            threshold = max(abs(py_val) * tol, 0.005)

        diff = abs(py_val - c_val)
        match = diff <= threshold
        status = "OK" if match else "FAIL"
        if not match:
            ok = False
        print(f"  {config_label} {k:20s}: py={py_val:12.6f} c={c_val:12.6f} diff={diff:.6f} [{status}]")
    return ok


def test_basic_parity():
    print("=== Test 1: Basic parity (random walk, 5 symbols, 120 days) ===")
    all_bars = make_synthetic_bars(n_symbols=5, n_days=120, seed=42)
    config = WorkStealConfig(
        dip_pct=0.10, proximity_pct=0.01, profit_target_pct=0.05,
        stop_loss_pct=0.08, max_positions=3, max_hold_days=14,
        lookback_days=20, maker_fee=0.001, initial_cash=10000.0,
        trailing_stop_pct=0.0, max_leverage=1.0, enable_shorts=False,
        sma_filter_period=0, max_drawdown_exit=0.0,
        reentry_cooldown_days=1, max_position_pct=0.25,
    )

    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_df, trades, py_metrics = run_worksteal_backtest(bars_copy, config)

    c_metrics = run_worksteal_backtest_fast(
        {k: v.copy() for k, v in all_bars.items()}, config
    )

    print(f"  Python: ret={py_metrics.get('total_return_pct',0):.4f}% sort={py_metrics.get('sortino',0):.4f} "
          f"trades={len(trades)} dd={py_metrics.get('max_drawdown_pct',0):.4f}%")
    print(f"  C:      ret={c_metrics.get('total_return_pct',0):.4f}% sort={c_metrics.get('sortino',0):.4f} "
          f"trades={c_metrics.get('total_trades',0)} dd={c_metrics.get('max_drawdown_pct',0):.4f}%")

    return compare_metrics(py_metrics, c_metrics, "basic")


def test_dip_scenario():
    print("\n=== Test 2: Dip scenario (forced dips, 3 symbols) ===")
    all_bars = make_dip_scenario()
    config = WorkStealConfig(
        dip_pct=0.10, proximity_pct=0.02, profit_target_pct=0.07,
        stop_loss_pct=0.10, max_positions=2, max_hold_days=14,
        lookback_days=20, maker_fee=0.001, initial_cash=10000.0,
        trailing_stop_pct=0.0, max_leverage=1.0, enable_shorts=False,
        sma_filter_period=0, max_drawdown_exit=0.0,
        reentry_cooldown_days=1, max_position_pct=0.30,
    )

    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_df, trades, py_metrics = run_worksteal_backtest(bars_copy, config)

    c_metrics = run_worksteal_backtest_fast(
        {k: v.copy() for k, v in all_bars.items()}, config
    )

    print(f"  Python: ret={py_metrics.get('total_return_pct',0):.4f}% sort={py_metrics.get('sortino',0):.4f} "
          f"trades={len(trades)} dd={py_metrics.get('max_drawdown_pct',0):.4f}%")
    print(f"  C:      ret={c_metrics.get('total_return_pct',0):.4f}% sort={c_metrics.get('sortino',0):.4f} "
          f"trades={c_metrics.get('total_trades',0)} dd={c_metrics.get('max_drawdown_pct',0):.4f}%")

    return compare_metrics(py_metrics, c_metrics, "dip")


def test_sma_filter():
    print("\n=== Test 3: SMA filter enabled ===")
    all_bars = make_synthetic_bars(n_symbols=4, n_days=100, seed=99)
    config = WorkStealConfig(
        dip_pct=0.08, proximity_pct=0.01, profit_target_pct=0.05,
        stop_loss_pct=0.06, max_positions=3, max_hold_days=10,
        lookback_days=15, maker_fee=0.001, initial_cash=10000.0,
        trailing_stop_pct=0.0, max_leverage=1.0, enable_shorts=False,
        sma_filter_period=20, max_drawdown_exit=0.0,
        reentry_cooldown_days=1, max_position_pct=0.25,
    )

    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_df, trades, py_metrics = run_worksteal_backtest(bars_copy, config)

    c_metrics = run_worksteal_backtest_fast(
        {k: v.copy() for k, v in all_bars.items()}, config
    )

    print(f"  Python: ret={py_metrics.get('total_return_pct',0):.4f}% sort={py_metrics.get('sortino',0):.4f} "
          f"trades={len(trades)} dd={py_metrics.get('max_drawdown_pct',0):.4f}%")
    print(f"  C:      ret={c_metrics.get('total_return_pct',0):.4f}% sort={c_metrics.get('sortino',0):.4f} "
          f"trades={c_metrics.get('total_trades',0)} dd={c_metrics.get('max_drawdown_pct',0):.4f}%")

    return compare_metrics(py_metrics, c_metrics, "sma")


def test_trailing_stop():
    print("\n=== Test 4: Trailing stop enabled ===")
    all_bars = make_dip_scenario(n_days=80, seed=555)
    config = WorkStealConfig(
        dip_pct=0.10, proximity_pct=0.02, profit_target_pct=0.10,
        stop_loss_pct=0.08, max_positions=3, max_hold_days=21,
        lookback_days=20, maker_fee=0.001, initial_cash=10000.0,
        trailing_stop_pct=0.03, max_leverage=1.0, enable_shorts=False,
        sma_filter_period=0, max_drawdown_exit=0.0,
        reentry_cooldown_days=1, max_position_pct=0.30,
    )

    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_df, trades, py_metrics = run_worksteal_backtest(bars_copy, config)

    c_metrics = run_worksteal_backtest_fast(
        {k: v.copy() for k, v in all_bars.items()}, config
    )

    print(f"  Python: ret={py_metrics.get('total_return_pct',0):.4f}% sort={py_metrics.get('sortino',0):.4f} "
          f"trades={len(trades)} dd={py_metrics.get('max_drawdown_pct',0):.4f}%")
    print(f"  C:      ret={c_metrics.get('total_return_pct',0):.4f}% sort={c_metrics.get('sortino',0):.4f} "
          f"trades={c_metrics.get('total_trades',0)} dd={c_metrics.get('max_drawdown_pct',0):.4f}%")

    return compare_metrics(py_metrics, c_metrics, "trail")


def test_leverage():
    print("\n=== Test 5: Leverage 2x ===")
    all_bars = make_dip_scenario(n_days=80, seed=777)
    config = WorkStealConfig(
        dip_pct=0.10, proximity_pct=0.02, profit_target_pct=0.08,
        stop_loss_pct=0.06, max_positions=3, max_hold_days=14,
        lookback_days=20, maker_fee=0.001, initial_cash=10000.0,
        trailing_stop_pct=0.0, max_leverage=2.0, enable_shorts=False,
        sma_filter_period=0, max_drawdown_exit=0.0,
        reentry_cooldown_days=1, max_position_pct=0.25,
    )

    bars_copy = {k: v.copy() for k, v in all_bars.items()}
    eq_df, trades, py_metrics = run_worksteal_backtest(bars_copy, config)

    c_metrics = run_worksteal_backtest_fast(
        {k: v.copy() for k, v in all_bars.items()}, config
    )

    print(f"  Python: ret={py_metrics.get('total_return_pct',0):.4f}% sort={py_metrics.get('sortino',0):.4f} "
          f"trades={len(trades)} dd={py_metrics.get('max_drawdown_pct',0):.4f}%")
    print(f"  C:      ret={c_metrics.get('total_return_pct',0):.4f}% sort={c_metrics.get('sortino',0):.4f} "
          f"trades={c_metrics.get('total_trades',0)} dd={c_metrics.get('max_drawdown_pct',0):.4f}%")

    return compare_metrics(py_metrics, c_metrics, "lev")


def test_batch():
    print("\n=== Test 6: Batch mode consistency ===")
    all_bars = make_synthetic_bars(n_symbols=3, n_days=80, seed=101)
    configs = [
        WorkStealConfig(dip_pct=0.08, profit_target_pct=0.05, stop_loss_pct=0.06,
                        max_positions=2, lookback_days=15, initial_cash=10000.0,
                        max_position_pct=0.25, reentry_cooldown_days=1),
        WorkStealConfig(dip_pct=0.12, profit_target_pct=0.08, stop_loss_pct=0.10,
                        max_positions=3, lookback_days=20, initial_cash=10000.0,
                        max_position_pct=0.30, reentry_cooldown_days=1),
        WorkStealConfig(dip_pct=0.15, profit_target_pct=0.10, stop_loss_pct=0.12,
                        max_positions=2, lookback_days=25, initial_cash=10000.0,
                        trailing_stop_pct=0.03, max_position_pct=0.25, reentry_cooldown_days=1),
    ]

    batch_results = run_worksteal_batch_fast(
        {k: v.copy() for k, v in all_bars.items()}, configs
    )

    single_results = []
    for cfg in configs:
        r = run_worksteal_backtest_fast(
            {k: v.copy() for k, v in all_bars.items()}, cfg
        )
        single_results.append(r)

    ok = True
    for i, (br, sr) in enumerate(zip(batch_results, single_results)):
        for k in ["total_return", "sortino", "max_drawdown", "final_equity"]:
            diff = abs((br.get(k, 0) or 0) - (sr.get(k, 0) or 0))
            if diff > 1e-10:
                print(f"  FAIL config[{i}] {k}: batch={br[k]:.8f} single={sr[k]:.8f}")
                ok = False
    if ok:
        print("  batch vs single: all match [OK]")
    return ok


def test_speed():
    print("\n=== Test 7: Speed benchmark ===")
    all_bars = make_synthetic_bars(n_symbols=10, n_days=200, seed=42)
    config = WorkStealConfig(
        dip_pct=0.10, proximity_pct=0.01, profit_target_pct=0.05,
        stop_loss_pct=0.08, max_positions=5, max_hold_days=14,
        lookback_days=20, maker_fee=0.001, initial_cash=10000.0,
        max_position_pct=0.25, reentry_cooldown_days=1,
    )

    n_runs = 100

    t0 = time.time()
    for _ in range(n_runs):
        run_worksteal_backtest_fast(
            {k: v.copy() for k, v in all_bars.items()}, config
        )
    c_time = time.time() - t0

    t0 = time.time()
    for _ in range(n_runs):
        run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config
        )
    py_time = time.time() - t0

    speedup = py_time / c_time if c_time > 0 else float("inf")
    print(f"  Python: {py_time:.2f}s for {n_runs} runs ({py_time/n_runs*1000:.1f}ms/run)")
    print(f"  C:      {c_time:.2f}s for {n_runs} runs ({c_time/n_runs*1000:.1f}ms/run)")
    print(f"  Speedup: {speedup:.1f}x {'[OK]' if speedup >= 2 else '[SLOW]'}")

    t0 = time.time()
    configs = [config] * 500
    run_worksteal_batch_fast(
        {k: v.copy() for k, v in all_bars.items()}, configs
    )
    batch_time = time.time() - t0
    print(f"  Batch 500 configs: {batch_time:.2f}s ({batch_time/500*1000:.1f}ms/config)")

    return speedup >= 2


def main():
    results = []
    results.append(("basic_parity", test_basic_parity()))
    results.append(("dip_scenario", test_dip_scenario()))
    results.append(("sma_filter", test_sma_filter()))
    results.append(("trailing_stop", test_trailing_stop()))
    results.append(("leverage", test_leverage()))
    results.append(("batch", test_batch()))
    results.append(("speed", test_speed()))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:20s} [{status}]")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
