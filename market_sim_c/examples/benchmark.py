#!/usr/bin/env python3
"""
Performance benchmarks for market simulation

Compares C/CUDA implementation against pure Python
"""

import time
import numpy as np
import torch
from typing import Dict

from market_sim_c.python.market_sim import make_env, MarketSimConfig


def benchmark_env_steps(env, num_steps: int = 10000):
    """Benchmark environment step performance"""
    obs, _ = env.reset()

    start_time = time.time()

    for _ in range(num_steps):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            obs, _ = env.reset()

    elapsed = time.time() - start_time
    steps_per_second = num_steps / elapsed

    return steps_per_second


def benchmark_vectorized(num_agents: int = 64, num_steps: int = 1000):
    """Benchmark vectorized environment performance"""
    config = MarketSimConfig(
        num_assets=10,
        num_agents=num_agents,
        max_steps=1000,
    )

    env = make_env(config, use_python=True)  # Will use C when compiled

    start_time = time.time()

    obs, _ = env.reset()

    for _ in range(num_steps):
        actions = np.random.randint(0, env.action_space.n, size=num_agents)
        obs, rewards, dones, truncated, info = env.step(actions)

    elapsed = time.time() - start_time
    total_steps = num_steps * num_agents
    steps_per_second = total_steps / elapsed

    return steps_per_second


def benchmark_cuda_kernels():
    """Benchmark CUDA kernel performance"""
    try:
        import market_sim_cuda

        device = torch.device("cuda")

        # Test portfolio values computation
        num_agents = 1024
        num_assets = 100

        cash = torch.rand(num_agents, device=device)
        positions = torch.rand(num_agents, num_assets, device=device)
        prices = torch.rand(num_assets, device=device)

        # Warmup
        for _ in range(10):
            market_sim_cuda.compute_portfolio_values(cash, positions, prices)

        torch.cuda.synchronize()
        start_time = time.time()

        iterations = 1000
        for _ in range(iterations):
            market_sim_cuda.compute_portfolio_values(cash, positions, prices)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        ops_per_second = iterations / elapsed

        return ops_per_second

    except ImportError:
        return None


def run_all_benchmarks():
    """Run comprehensive benchmark suite"""
    print("=" * 70)
    print("MARKET SIMULATION PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Single agent benchmark
    print("\n1. Single Agent Performance")
    print("-" * 70)

    config = MarketSimConfig(num_assets=10, num_agents=1)
    env = make_env(config, use_python=True)

    python_fps = benchmark_env_steps(env, num_steps=10000)
    print(f"Python Implementation: {python_fps:,.0f} steps/sec")

    # Try C implementation if available
    try:
        env_c = make_env(config, use_python=False)
        c_fps = benchmark_env_steps(env_c, num_steps=10000)
        print(f"C Implementation:      {c_fps:,.0f} steps/sec")
        print(f"Speedup:               {c_fps / python_fps:.2f}x")
    except Exception as e:
        print(f"C Implementation:      Not available ({e})")

    # Vectorized benchmarks
    print("\n2. Vectorized Environment Performance")
    print("-" * 70)

    for num_agents in [1, 8, 64, 256]:
        fps = benchmark_vectorized(num_agents=num_agents, num_steps=1000)
        print(f"{num_agents:3d} agents: {fps:10,.0f} total steps/sec "
              f"({fps/num_agents:8,.0f} per agent)")

    # CUDA kernel benchmarks
    print("\n3. CUDA Kernel Performance")
    print("-" * 70)

    if torch.cuda.is_available():
        kernel_fps = benchmark_cuda_kernels()

        if kernel_fps is not None:
            print(f"Portfolio valuation: {kernel_fps:,.0f} ops/sec")
            print(f"  (1024 agents × 100 assets)")
        else:
            print("CUDA kernels not compiled")
    else:
        print("CUDA not available")

    # Memory usage
    print("\n4. Memory Usage")
    print("-" * 70)

    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024

    print(f"Current process: {mem_mb:.1f} MB")

    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU allocated:   {gpu_mb:.1f} MB")

    # System info
    print("\n5. System Information")
    print("-" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"CPU cores:       {psutil.cpu_count()}")
    print(f"RAM:             {psutil.virtual_memory().total / 1024**3:.1f} GB")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
