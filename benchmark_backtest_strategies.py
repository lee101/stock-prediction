#!/usr/bin/env python3
"""
Benchmark different strategies for speeding up backtest optimization.

Tests:
1. Sequential baseline
2. ThreadPoolExecutor (CPU-bound work sharing GPU)
3. ProcessPoolExecutor (true parallelism)
4. DIRECT_fast optimization mode
5. GPU batch optimization
6. Combined approaches

Usage:
    python benchmark_backtest_strategies.py [--symbol ETHUSD] [--num-sims 20]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
import os

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    strategy: str
    total_time: float
    per_sim_time: float
    speedup: float
    num_sims: int
    quality_metric: float = 0.0
    notes: str = ""


class BacktestBenchmark:
    def __init__(self, symbol: str, num_sims: int):
        self.symbol = symbol
        self.num_sims = num_sims
        self.results = []

    def run_all(self):
        """Run all benchmark strategies"""
        print(f"\n{'='*80}")
        print(f"Benchmarking backtest strategies for {self.symbol}")
        print(f"Number of simulations: {self.num_sims}")
        print(f"{'='*80}\n")

        # 1. Sequential baseline
        self.benchmark_sequential()

        # 2. ThreadPool (shares GPU models)
        for workers in [2, 4, 8]:
            self.benchmark_threadpool(workers)

        # 3. ProcessPool (true parallelism, GPU per process)
        if torch.cuda.is_available():
            for workers in [2, 4]:
                self.benchmark_processpool(workers)

        # 4. Fast optimization mode
        self.benchmark_fast_optimize()

        # 5. Batch GPU optimization
        if torch.cuda.is_available():
            self.benchmark_gpu_batch()

        # 6. Combined: ThreadPool + Fast mode
        self.benchmark_combined_threadpool_fast(workers=4)

        # 7. Combined: ProcessPool + Fast mode
        if torch.cuda.is_available():
            self.benchmark_combined_processpool_fast(workers=4)

        self.print_summary()

    def benchmark_sequential(self):
        """Baseline: sequential execution"""
        print(f"\n[1/7] Sequential baseline...")

        os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'
        from backtest_test3_inline import backtest_forecasts

        start = time.time()
        try:
            results = backtest_forecasts(self.symbol, num_simulations=self.num_sims)
            elapsed = time.time() - start
            quality = self._extract_quality(results)

            result = BenchmarkResult(
                strategy="Sequential",
                total_time=elapsed,
                per_sim_time=elapsed / self.num_sims,
                speedup=1.0,
                num_sims=self.num_sims,
                quality_metric=quality,
                notes="Baseline"
            )
            self.results.append(result)
            print(f"  ✓ Completed in {elapsed:.2f}s ({elapsed/self.num_sims:.3f}s per sim)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    def benchmark_threadpool(self, workers: int):
        """ThreadPoolExecutor - shares GPU models"""
        print(f"\n[2/7] ThreadPool (workers={workers})...")

        os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

        # Import fresh to get parallel version
        import importlib
        import backtest_test3_inline_parallel
        importlib.reload(backtest_test3_inline_parallel)

        start = time.time()
        try:
            results = backtest_test3_inline_parallel.backtest_forecasts_parallel(
                self.symbol, num_simulations=self.num_sims, max_workers=workers
            )
            elapsed = time.time() - start
            quality = self._extract_quality(results)

            baseline_time = self.results[0].total_time if self.results else elapsed
            result = BenchmarkResult(
                strategy=f"ThreadPool-{workers}w",
                total_time=elapsed,
                per_sim_time=elapsed / self.num_sims,
                speedup=baseline_time / elapsed if elapsed > 0 else 0,
                num_sims=self.num_sims,
                quality_metric=quality,
                notes="Shares GPU"
            )
            self.results.append(result)
            print(f"  ✓ Completed in {elapsed:.2f}s ({result.speedup:.2f}x speedup)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    def benchmark_processpool(self, workers: int):
        """ProcessPoolExecutor - separate GPU per process"""
        print(f"\n[3/7] ProcessPool (workers={workers})...")

        os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

        start = time.time()
        try:
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=workers) as executor:
                sims_per_worker = self.num_sims // workers
                remainder = self.num_sims % workers

                futures = []
                for i in range(workers):
                    n = sims_per_worker + (1 if i < remainder else 0)
                    future = executor.submit(self._run_backtest_worker, self.symbol, n)
                    futures.append(future)

                results_list = [f.result() for f in as_completed(futures)]

            elapsed = time.time() - start
            quality = np.mean([self._extract_quality(r) for r in results_list])

            baseline_time = self.results[0].total_time if self.results else elapsed
            result = BenchmarkResult(
                strategy=f"ProcessPool-{workers}w",
                total_time=elapsed,
                per_sim_time=elapsed / self.num_sims,
                speedup=baseline_time / elapsed if elapsed > 0 else 0,
                num_sims=self.num_sims,
                quality_metric=quality,
                notes="True parallel, GPU per process"
            )
            self.results.append(result)
            print(f"  ✓ Completed in {elapsed:.2f}s ({result.speedup:.2f}x speedup)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    def benchmark_fast_optimize(self):
        """Fast optimization mode (DIRECT maxfun=100)"""
        print(f"\n[4/7] Fast optimization mode...")

        os.environ['MARKETSIM_FAST_OPTIMIZE'] = '1'

        # Reload to pick up env var
        import importlib
        import backtest_test3_inline
        importlib.reload(backtest_test3_inline)

        start = time.time()
        try:
            results = backtest_test3_inline.backtest_forecasts(
                self.symbol, num_simulations=self.num_sims
            )
            elapsed = time.time() - start
            quality = self._extract_quality(results)

            baseline_time = self.results[0].total_time if self.results else elapsed
            result = BenchmarkResult(
                strategy="FastOptimize",
                total_time=elapsed,
                per_sim_time=elapsed / self.num_sims,
                speedup=baseline_time / elapsed if elapsed > 0 else 0,
                num_sims=self.num_sims,
                quality_metric=quality,
                notes="DIRECT maxfun=100 (6x opt speedup)"
            )
            self.results.append(result)
            print(f"  ✓ Completed in {elapsed:.2f}s ({result.speedup:.2f}x speedup)")
            print(f"     Quality loss: {(1 - quality/self.results[0].quality_metric)*100:.1f}%")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        finally:
            os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

    def benchmark_gpu_batch(self):
        """Batch GPU optimization (vectorized across simulations)"""
        print(f"\n[5/7] GPU batch optimization...")

        print(f"  ⊘ Not implemented yet - requires vectorized optimization")
        # This would require rewriting optimization to handle batched tensors
        # e.g., optimize all simulations simultaneously on GPU

    def benchmark_combined_threadpool_fast(self, workers: int):
        """ThreadPool + Fast optimization"""
        print(f"\n[6/7] Combined: ThreadPool + Fast optimization...")

        os.environ['MARKETSIM_FAST_OPTIMIZE'] = '1'

        import importlib
        import backtest_test3_inline_parallel
        importlib.reload(backtest_test3_inline_parallel)

        start = time.time()
        try:
            results = backtest_test3_inline_parallel.backtest_forecasts_parallel(
                self.symbol, num_simulations=self.num_sims, max_workers=workers
            )
            elapsed = time.time() - start
            quality = self._extract_quality(results)

            baseline_time = self.results[0].total_time if self.results else elapsed
            result = BenchmarkResult(
                strategy=f"Thread{workers}w+Fast",
                total_time=elapsed,
                per_sim_time=elapsed / self.num_sims,
                speedup=baseline_time / elapsed if elapsed > 0 else 0,
                num_sims=self.num_sims,
                quality_metric=quality,
                notes="Best balance"
            )
            self.results.append(result)
            print(f"  ✓ Completed in {elapsed:.2f}s ({result.speedup:.2f}x speedup)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        finally:
            os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

    def benchmark_combined_processpool_fast(self, workers: int):
        """ProcessPool + Fast optimization"""
        print(f"\n[7/7] Combined: ProcessPool + Fast optimization...")

        os.environ['MARKETSIM_FAST_OPTIMIZE'] = '1'

        start = time.time()
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                sims_per_worker = self.num_sims // workers
                remainder = self.num_sims % workers

                futures = []
                for i in range(workers):
                    n = sims_per_worker + (1 if i < remainder else 0)
                    future = executor.submit(self._run_backtest_worker, self.symbol, n)
                    futures.append(future)

                results_list = [f.result() for f in as_completed(futures)]

            elapsed = time.time() - start
            quality = np.mean([self._extract_quality(r) for r in results_list])

            baseline_time = self.results[0].total_time if self.results else elapsed
            result = BenchmarkResult(
                strategy=f"Proc{workers}w+Fast",
                total_time=elapsed,
                per_sim_time=elapsed / self.num_sims,
                speedup=baseline_time / elapsed if elapsed > 0 else 0,
                num_sims=self.num_sims,
                quality_metric=quality,
                notes="Max parallelism + fast opt"
            )
            self.results.append(result)
            print(f"  ✓ Completed in {elapsed:.2f}s ({result.speedup:.2f}x speedup)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        finally:
            os.environ['MARKETSIM_FAST_OPTIMIZE'] = '0'

    @staticmethod
    def _run_backtest_worker(symbol: str, num_sims: int):
        """Worker function for ProcessPool"""
        import backtest_test3_inline
        return backtest_test3_inline.backtest_forecasts(symbol, num_simulations=num_sims)

    @staticmethod
    def _extract_quality(results):
        """Extract quality metric from results dataframe"""
        try:
            if hasattr(results, 'empty') and not results.empty:
                if 'maxdiff_return' in results.columns:
                    return float(results['maxdiff_return'].mean())
                elif 'simple_strategy_return' in results.columns:
                    return float(results['simple_strategy_return'].mean())
            return 0.0
        except:
            return 0.0

    def print_summary(self):
        """Print formatted summary table"""
        print(f"\n{'='*100}")
        print(f"BENCHMARK SUMMARY: {self.symbol} ({self.num_sims} simulations)")
        print(f"{'='*100}")
        print(f"{'Strategy':<25} {'Time (s)':<12} {'Per Sim (s)':<12} {'Speedup':<10} {'Quality':<10} {'Notes':<20}")
        print(f"{'-'*100}")

        for r in self.results:
            print(f"{r.strategy:<25} {r.total_time:>10.2f}s  {r.per_sim_time:>10.3f}s  {r.speedup:>8.2f}x  {r.quality_metric:>8.4f}  {r.notes:<20}")

        if self.results:
            best = max(self.results, key=lambda x: x.speedup)
            print(f"\n{'='*100}")
            print(f"WINNER: {best.strategy} - {best.speedup:.2f}x speedup ({best.total_time:.2f}s total)")
            print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark backtest optimization strategies")
    parser.add_argument('--symbol', default='ETHUSD', help='Symbol to backtest')
    parser.add_argument('--num-sims', type=int, default=20, help='Number of simulations')
    parser.add_argument('--gpu-info', action='store_true', help='Show GPU info')

    args = parser.parse_args()

    if args.gpu_info:
        print(f"\nGPU Info:")
        print(f"  Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Count: {torch.cuda.device_count()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    benchmark = BacktestBenchmark(args.symbol, args.num_sims)
    benchmark.run_all()


if __name__ == '__main__':
    main()
