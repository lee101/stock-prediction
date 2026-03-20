#!/usr/bin/env python3
"""Benchmark: Python vs C single, serial C batch vs parallel C batch."""
from __future__ import annotations
import sys, time, ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from binance_worksteal.strategy import WorkStealConfig, run_worksteal_backtest
from binance_worksteal.csim.fast_worksteal import (
    run_worksteal_backtest_fast, run_worksteal_batch_fast, get_num_threads,
    _prepare_data, _config_to_c, _to_ptr_d, _to_ptr_i, _lib,
    CWorkStealConfig, CSimResult,
)


def make_synthetic_bars(n_symbols=30, n_days=1500, seed=42):
    rng = np.random.RandomState(seed)
    all_bars = {}
    base_date = pd.Timestamp("2022-01-01", tz="UTC")
    for s in range(n_symbols):
        sym = f"SYM{s:02d}USD"
        price = 50.0 + s * 20
        rows = []
        for d in range(n_days):
            ret = rng.normal(0.0002, 0.03)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.01)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.01)))
            rows.append({
                "timestamp": base_date + pd.Timedelta(days=d),
                "open": o, "high": h, "low": l, "close": c,
                "volume": rng.uniform(1e6, 1e8),
            })
            price = max(c, 0.01)
        all_bars[sym] = pd.DataFrame(rows)
    return all_bars


def make_config(dip=0.10, tp=0.05, sl=0.08, max_pos=5, lookback=20,
                trailing=0.0, sma=0, max_hold=14):
    return WorkStealConfig(
        dip_pct=dip, proximity_pct=0.02, profit_target_pct=tp,
        stop_loss_pct=sl, max_positions=max_pos, max_hold_days=max_hold,
        lookback_days=lookback, maker_fee=0.001, initial_cash=10000.0,
        trailing_stop_pct=trailing, max_leverage=1.0, enable_shorts=False,
        sma_filter_period=sma, max_drawdown_exit=0.0,
        reentry_cooldown_days=1, max_position_pct=0.25,
    )


def make_random_configs(n_configs=500, seed=99):
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n_configs):
        configs.append(make_config(
            dip=rng.uniform(0.05, 0.25),
            tp=rng.uniform(0.03, 0.15),
            sl=rng.uniform(0.03, 0.15),
            max_pos=rng.randint(2, 8),
            lookback=rng.randint(10, 40),
            trailing=rng.choice([0.0, 0.02, 0.03, 0.05]),
            sma=rng.choice([0, 10, 20, 30]),
            max_hold=rng.randint(7, 30),
        ))
    return configs


def bench_python_vs_c_small():
    bars = make_synthetic_bars(n_symbols=10, n_days=200, seed=42)
    cfg = make_config()
    n_runs = 50

    print(f"\n--- Python vs C: 10 symbols, 200 days, {n_runs} runs ---")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_worksteal_backtest_fast({k: v.copy() for k, v in bars.items()}, cfg)
    c_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, cfg)
    py_time = time.perf_counter() - t0

    speedup = py_time / c_time if c_time > 0 else float("inf")
    print(f"  Python: {py_time:.2f}s ({py_time/n_runs*1000:.1f}ms/run)")
    print(f"  C:      {c_time:.2f}s ({c_time/n_runs*1000:.1f}ms/run)")
    print(f"  Speedup: {speedup:.1f}x")
    return speedup


def bench_c_sim_only(large_bars):
    cfg = make_config()
    n_runs = 200

    print(f"\n--- C sim kernel only: 30 symbols, 1500 days, {n_runs} runs ---")

    timestamps, valid, opens, highs_arr, lows_arr, closes_arr, fee_rates, symbols = \
        _prepare_data(large_bars, cfg)
    n_bars = len(timestamps)
    n_sym = len(symbols)
    c_cfg = _config_to_c(cfg)
    result = CSimResult()
    eq = np.zeros(n_bars, dtype=np.float64)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _lib.worksteal_simulate(
            _to_ptr_d(timestamps), _to_ptr_i(valid),
            _to_ptr_d(opens), _to_ptr_d(highs_arr),
            _to_ptr_d(lows_arr), _to_ptr_d(closes_arr),
            _to_ptr_d(fee_rates), n_bars, n_sym,
            ctypes.byref(c_cfg), ctypes.byref(result), _to_ptr_d(eq),
        )
    elapsed = time.perf_counter() - t0
    per_run_us = elapsed / n_runs * 1e6

    print(f"  {n_runs} runs: {elapsed:.3f}s ({per_run_us:.0f}us/run)")
    print(f"  Result: ret={result.total_return*100:.2f}% trades={result.total_trades}")
    return per_run_us


def bench_batch_serial_vs_parallel(large_bars, configs):
    n_configs = len(configs)
    print(f"\n--- Batch: {n_configs} configs, 30 symbols, 1500 days ---")

    timestamps, valid, opens, highs_arr, lows_arr, closes_arr, fee_rates, symbols = \
        _prepare_data(large_bars, configs[0])
    n_bars = len(timestamps)
    n_sym = len(symbols)
    nc = n_configs

    c_cfgs = (CWorkStealConfig * nc)()
    for i, cfg in enumerate(configs):
        c = _config_to_c(cfg)
        ctypes.memmove(ctypes.byref(c_cfgs[i]), ctypes.byref(c), ctypes.sizeof(CWorkStealConfig))

    serial_results = (CSimResult * nc)()
    eq = np.zeros(n_bars, dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(nc):
        _lib.worksteal_simulate(
            _to_ptr_d(timestamps), _to_ptr_i(valid),
            _to_ptr_d(opens), _to_ptr_d(highs_arr),
            _to_ptr_d(lows_arr), _to_ptr_d(closes_arr),
            _to_ptr_d(fee_rates), n_bars, n_sym,
            ctypes.byref(c_cfgs[i]), ctypes.byref(serial_results[i]), _to_ptr_d(eq),
        )
    serial_time = time.perf_counter() - t0

    batch_results = (CSimResult * nc)()
    t0 = time.perf_counter()
    _lib.worksteal_simulate_batch(
        _to_ptr_d(timestamps), _to_ptr_i(valid),
        _to_ptr_d(opens), _to_ptr_d(highs_arr),
        _to_ptr_d(lows_arr), _to_ptr_d(closes_arr),
        _to_ptr_d(fee_rates), n_bars, n_sym,
        c_cfgs, batch_results, nc,
    )
    batch_time = time.perf_counter() - t0

    batch_speedup = serial_time / batch_time if batch_time > 0 else float("inf")
    threads = get_num_threads()

    print(f"  OpenMP threads: {threads}")
    print(f"  Serial C:   {serial_time:.3f}s ({serial_time/nc*1000:.2f}ms/config)")
    print(f"  Batch C:    {batch_time:.3f}s ({batch_time/nc*1000:.3f}ms/config)")
    print(f"  Batch speedup: {batch_speedup:.1f}x over serial C")

    mismatches = 0
    for i in range(nc):
        sr, br = serial_results[i], batch_results[i]
        for attr in ["total_return", "sortino", "final_equity"]:
            sv = getattr(sr, attr)
            bv = getattr(br, attr)
            if abs(sv - bv) > 1e-6:
                if mismatches < 3:
                    print(f"  MISMATCH config[{i}] {attr}: serial={sv:.8f} batch={bv:.8f}")
                mismatches += 1
    if mismatches == 0:
        print(f"  Parity: all {nc} configs match [OK]")
    else:
        print(f"  Parity: {mismatches} mismatches [WARN]")

    return serial_time, batch_time, batch_speedup


def bench_end_to_end_sweep(large_bars, configs):
    n_configs = len(configs)
    print(f"\n--- E2E sweep: {n_configs} configs via batch wrapper ---")

    t0 = time.perf_counter()
    run_worksteal_batch_fast(large_bars, configs)
    total_time = time.perf_counter() - t0

    print(f"  Total (data prep + parallel sim): {total_time:.3f}s")
    print(f"  Throughput: {n_configs/total_time:.0f} configs/sec")
    print(f"  Per config: {total_time/n_configs*1000:.2f}ms")

    return total_time


def main():
    print("=" * 60)
    print("WORKSTEAL C SIMULATOR BENCHMARK")
    print("=" * 60)
    print(f"OpenMP threads: {get_num_threads()}")

    py_vs_c = bench_python_vs_c_small()

    print("\nGenerating large dataset (30 symbols, 1500 days)...")
    large_bars = make_synthetic_bars(n_symbols=30, n_days=1500, seed=42)
    configs = make_random_configs(n_configs=500, seed=99)

    kernel_us = bench_c_sim_only(large_bars)
    serial_t, batch_t, batch_sp = bench_batch_serial_vs_parallel(large_bars, configs)
    e2e_time = bench_end_to_end_sweep(large_bars, configs)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Python vs C (incl data prep):     {py_vs_c:.1f}x")
    print(f"  C kernel per run:                 {kernel_us:.0f}us")
    print(f"  Serial vs Parallel batch (500):   {batch_sp:.1f}x")
    print(f"  E2E sweep 500 configs:            {e2e_time:.2f}s ({500/e2e_time:.0f} cfg/s)")
    print(f"  OpenMP threads:                   {get_num_threads()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
