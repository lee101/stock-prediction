#!/usr/bin/env python3
"""
Quick test of different speedup strategies without full backtest.

Tests pure optimization speed using synthetic data.
"""
import os
import time
import torch
import numpy as np
from tabulate import tabulate


def generate_synthetic_data(n=200, device='cuda'):
    """Generate synthetic market data"""
    close_actual = torch.randn(n, device=device) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n, device=device)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n, device=device)) * 0.01
    high_pred = torch.randn(n, device=device) * 0.01 + 0.005
    low_pred = torch.randn(n, device=device) * 0.01 - 0.005
    positions = torch.sign(torch.randn(n, device=device))
    return close_actual, high_actual, low_actual, high_pred, low_pred, positions


def test_sequential_normal():
    """Baseline: Normal optimization (maxfun=500)"""
    os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'
    os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'

    from src.optimization_utils import optimize_entry_exit_multipliers
    import importlib
    import src.optimization_utils
    importlib.reload(src.optimization_utils)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = generate_synthetic_data(device=device)

    times = []
    for _ in range(5):
        start = time.time()
        result = optimize_entry_exit_multipliers(*data, trading_fee=0.001)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'strategy': 'Sequential (maxfun=500)',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'result': result
    }


def test_sequential_fast():
    """Fast optimization (maxfun=100)"""
    os.environ['MARKETSIM_FAST_OPTIMIZE'] = '1'
    os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'

    import importlib
    import src.optimization_utils
    importlib.reload(src.optimization_utils)
    from src.optimization_utils import optimize_entry_exit_multipliers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = generate_synthetic_data(device=device)

    times = []
    for _ in range(5):
        start = time.time()
        result = optimize_entry_exit_multipliers(*data, trading_fee=0.001)
        elapsed = time.time() - start
        times.append(elapsed)

    os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

    return {
        'strategy': 'Sequential (maxfun=100)',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'result': result
    }


def test_threadpool_fast(n_sims=20, workers=4):
    """ThreadPool with fast optimization"""
    os.environ['MARKETSIM_FAST_OPTIMIZE'] = '1'

    import importlib
    import src.optimization_utils
    importlib.reload(src.optimization_utils)
    from src.optimization_utils import optimize_entry_exit_multipliers
    from concurrent.futures import ThreadPoolExecutor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def optimize_one(_):
        data = generate_synthetic_data(device=device)
        return optimize_entry_exit_multipliers(*data, trading_fee=0.001)

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(optimize_one, range(n_sims)))
    elapsed = time.time() - start

    os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

    return {
        'strategy': f'ThreadPool {workers}w + Fast',
        'mean_time': elapsed / n_sims,
        'std_time': 0.0,
        'total_time': elapsed,
        'n_sims': n_sims,
        'result': results[0]
    }


def test_gpu_batch(n_sims=20, batch_size=8):
    """GPU batch optimization"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        return {
            'strategy': f'GPU Batch (bs={batch_size})',
            'mean_time': 0.0,
            'note': 'GPU not available'
        }

    try:
        from src.optimization_utils_gpu_batch import optimize_batch_entry_exit

        # Generate multiple simulations
        all_data = {
            'close': [],
            'high_actual': [],
            'low_actual': [],
            'high_pred': [],
            'low_pred': [],
            'positions': []
        }

        for _ in range(n_sims):
            data = generate_synthetic_data(device=device)
            all_data['close'].append(data[0])
            all_data['high_actual'].append(data[1])
            all_data['low_actual'].append(data[2])
            all_data['high_pred'].append(data[3])
            all_data['low_pred'].append(data[4])
            all_data['positions'].append(data[5])

        start = time.time()
        results = optimize_batch_entry_exit(
            all_data['close'],
            all_data['positions'],
            all_data['high_actual'],
            all_data['high_pred'],
            all_data['low_actual'],
            all_data['low_pred'],
            trading_fee=0.001,
            batch_size=batch_size,
            maxfun=100,
            device=device
        )
        elapsed = time.time() - start

        return {
            'strategy': f'GPU Batch (bs={batch_size})',
            'mean_time': elapsed / n_sims,
            'std_time': 0.0,
            'total_time': elapsed,
            'n_sims': n_sims,
            'result': results[0]
        }
    except Exception as e:
        return {
            'strategy': f'GPU Batch (bs={batch_size})',
            'mean_time': 0.0,
            'note': f'Failed: {e}'
        }


def main():
    print("="*80)
    print("QUICK SPEEDUP TEST - Optimization Performance")
    print("="*80)

    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name(0)})"
    print(f"\nDevice: {device_info}\n")

    results = []

    # 1. Baseline
    print("[1/6] Testing sequential (maxfun=500)...")
    r1 = test_sequential_normal()
    results.append(r1)
    print(f"  ✓ {r1['mean_time']*1000:.1f}ms ± {r1['std_time']*1000:.1f}ms")

    baseline_time = r1['mean_time']

    # 2. Fast mode
    print("[2/6] Testing sequential (maxfun=100)...")
    r2 = test_sequential_fast()
    results.append(r2)
    speedup = baseline_time / r2['mean_time']
    print(f"  ✓ {r2['mean_time']*1000:.1f}ms ± {r2['std_time']*1000:.1f}ms ({speedup:.1f}x speedup)")

    # 3. ThreadPool 4 workers
    print("[3/6] Testing ThreadPool (4 workers, fast mode, 20 sims)...")
    r3 = test_threadpool_fast(n_sims=20, workers=4)
    results.append(r3)
    speedup = baseline_time / r3['mean_time']
    print(f"  ✓ {r3['mean_time']*1000:.1f}ms per sim, {r3['total_time']:.2f}s total ({speedup:.1f}x speedup)")

    # 4. ThreadPool 8 workers
    print("[4/6] Testing ThreadPool (8 workers, fast mode, 20 sims)...")
    r4 = test_threadpool_fast(n_sims=20, workers=8)
    results.append(r4)
    speedup = baseline_time / r4['mean_time']
    print(f"  ✓ {r4['mean_time']*1000:.1f}ms per sim, {r4['total_time']:.2f}s total ({speedup:.1f}x speedup)")

    # 5. GPU Batch (batch_size=8)
    print("[5/6] Testing GPU batch (batch_size=8, 20 sims)...")
    r5 = test_gpu_batch(n_sims=20, batch_size=8)
    results.append(r5)
    if 'note' in r5:
        print(f"  ⊘ {r5['note']}")
    else:
        speedup = baseline_time / r5['mean_time']
        print(f"  ✓ {r5['mean_time']*1000:.1f}ms per sim, {r5['total_time']:.2f}s total ({speedup:.1f}x speedup)")

    # 6. GPU Batch (batch_size=16)
    print("[6/6] Testing GPU batch (batch_size=16, 20 sims)...")
    r6 = test_gpu_batch(n_sims=20, batch_size=16)
    results.append(r6)
    if 'note' in r6:
        print(f"  ⊘ {r6['note']}")
    else:
        speedup = baseline_time / r6['mean_time']
        print(f"  ✓ {r6['mean_time']*1000:.1f}ms per sim, {r6['total_time']:.2f}s total ({speedup:.1f}x speedup)")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    table_data = []
    for r in results:
        if 'note' in r:
            table_data.append([r['strategy'], '-', '-', r['note']])
        else:
            speedup = baseline_time / r['mean_time'] if r['mean_time'] > 0 else 0
            per_sim = f"{r['mean_time']*1000:.1f}ms"
            speedup_str = f"{speedup:.1f}x"
            total = f"{r.get('total_time', r['mean_time']):.2f}s" if 'total_time' in r else '-'
            table_data.append([r['strategy'], per_sim, speedup_str, total])

    headers = ['Strategy', 'Per Optimization', 'Speedup', 'Total (20 sims)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_speedup = max([baseline_time / r['mean_time'] for r in results if 'mean_time' in r and r['mean_time'] > 0])
    best_strategy = [r for r in results if 'mean_time' in r and r['mean_time'] > 0 and baseline_time / r['mean_time'] == best_speedup][0]

    print(f"\nBest strategy: {best_strategy['strategy']}")
    print(f"Speedup: {best_speedup:.1f}x")
    print(f"\nFor 50 simulations:")
    print(f"  Baseline: {baseline_time * 50:.1f}s ({baseline_time * 50 / 60:.1f} min)")
    print(f"  Optimized: {best_strategy['mean_time'] * 50:.1f}s ({best_strategy['mean_time'] * 50 / 60:.1f} min)")
    print(f"  Time saved: {(baseline_time - best_strategy['mean_time']) * 50:.1f}s")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
