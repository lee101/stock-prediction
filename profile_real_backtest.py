#!/usr/bin/env python3
"""
Profile real backtest to identify actual bottlenecks.
Shows where time is spent in actual run_single_simulation calls.
"""

import time
import cProfile
import pstats
import io
from contextlib import contextmanager

# Monkey-patch to track timing
_timings = {}

@contextmanager
def timer(name):
    """Context manager to track time spent in sections"""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name not in _timings:
        _timings[name] = []
    _timings[name].append(elapsed)


def instrument_backtest():
    """Add timing instrumentation to backtest functions"""
    import backtest_test3_inline as bt

    # Wrap the forecasting functions
    original_compute_toto = bt._compute_toto_forecast if hasattr(bt, '_compute_toto_forecast') else None
    original_compute_kronos = bt._compute_kronos_forecast if hasattr(bt, '_compute_kronos_forecast') else None
    original_optimize_entry = bt.optimize_entry_exit_multipliers
    original_optimize_always = bt.optimize_always_on_multipliers

    if original_compute_toto:
        def timed_compute_toto(*args, **kwargs):
            with timer('toto_forecast'):
                return original_compute_toto(*args, **kwargs)
        bt._compute_toto_forecast = timed_compute_toto

    if original_compute_kronos:
        def timed_compute_kronos(*args, **kwargs):
            with timer('kronos_forecast'):
                return original_compute_kronos(*args, **kwargs)
        bt._compute_kronos_forecast = timed_compute_kronos

    def timed_optimize_entry(*args, **kwargs):
        with timer('optimize_entry_exit'):
            return original_optimize_entry(*args, **kwargs)
    bt.optimize_entry_exit_multipliers = timed_optimize_entry

    def timed_optimize_always(*args, **kwargs):
        with timer('optimize_always_on'):
            return original_optimize_always(*args, **kwargs)
    bt.optimize_always_on_multipliers = timed_optimize_always


def profile_backtest(symbol="ETHUSD", num_sims=10):
    """Profile a real backtest run"""
    print("="*80)
    print(f"PROFILING REAL BACKTEST: {symbol} ({num_sims} simulations)")
    print("="*80)
    print()

    # Clear timings
    global _timings
    _timings = {}

    # Instrument
    instrument_backtest()

    from backtest_test3_inline import backtest_forecasts

    # Run with profiling
    print("Running backtest with instrumentation...")
    start_total = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        results = backtest_forecasts(symbol, num_simulations=num_sims)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results = None

    profiler.disable()
    total_time = time.time() - start_total

    print(f"\nTotal time: {total_time:.1f}s")
    print()

    # Analyze timing breakdown
    if _timings:
        print("="*80)
        print("TIMING BREAKDOWN")
        print("="*80)

        total_tracked = sum(sum(times) for times in _timings.values())

        print(f"{'Component':<30} {'Total (s)':<12} {'Calls':<10} {'Avg (ms)':<12} {'% of total':<12}")
        print("-"*80)

        for name in sorted(_timings.keys()):
            times = _timings[name]
            total = sum(times)
            count = len(times)
            avg = (total / count * 1000) if count else 0
            pct = (total / total_time * 100) if total_time else 0

            print(f"{name:<30} {total:<12.3f} {count:<10} {avg:<12.3f} {pct:<12.1f}%")

        print("-"*80)
        print(f"{'Tracked time':<30} {total_tracked:<12.3f}")
        print(f"{'Untracked/overhead':<30} {total_time - total_tracked:<12.3f} {(total_time-total_tracked)/total_time*100:<12.1f}%")

    # Top expensive functions
    print("\n" + "="*80)
    print("TOP 20 EXPENSIVE FUNCTIONS (by cumulative time)")
    print("="*80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)

    output = s.getvalue()
    for line in output.split('\n')[:25]:
        print(line)

    # Analyze what's being called repeatedly
    print("\n" + "="*80)
    print("MOST CALLED FUNCTIONS")
    print("="*80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('calls')
    ps.print_stats(15)

    output = s.getvalue()
    for line in output.split('\n')[:20]:
        print(line)

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if _timings:
        # Check if model is being called repeatedly
        model_calls = _timings.get('toto_forecast', []) + _timings.get('kronos_forecast', [])
        opt_calls = _timings.get('optimize_entry_exit', []) + _timings.get('optimize_always_on', [])

        if model_calls:
            print(f"Model inference: {len(model_calls)} calls, {sum(model_calls):.1f}s total")
            print(f"  Per simulation: {len(model_calls)/num_sims:.1f} calls")
            print(f"  Per call: {sum(model_calls)/len(model_calls)*1000:.1f}ms")

        if opt_calls:
            print(f"Optimization: {len(opt_calls)} calls, {sum(opt_calls):.1f}s total")
            print(f"  Per simulation: {len(opt_calls)/num_sims:.1f} calls")
            print(f"  Per call: {sum(opt_calls)/len(opt_calls)*1000:.1f}ms")

    print("\n🔍 Caching opportunities:")
    print("  - Model predictions called once per (simulation, key) = 4x per sim")
    print("  - Walk-forward: data overlaps between simulations")
    print("  - Could cache predictions for overlapping time periods")
    print("  - Optimization is fast (now using DIRECT)")

    return results


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSD"
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    profile_backtest(symbol, num_sims)
