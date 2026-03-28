"""
bench_env.py — Benchmark the C trading environment steps/sec.

Usage (from repo root):
    source .venv313/bin/activate
    cd pufferlib_market && python setup.py build_ext --inplace && cd ..
    python pufferlib_market/bench_env.py
"""

from __future__ import annotations

import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _write_mktd_bin(path: Path, num_symbols: int = 12, num_timesteps: int = 5000) -> None:
    """Write a synthetic MKTD v2 binary for benchmarking."""
    magic = b"MKTD"
    version = 2
    features_per_sym = 16
    price_features = 5
    padding = b"\x00" * 40
    header = struct.pack(
        "<4sIIIII40s",
        magic,
        version,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        padding,
    )
    rng = np.random.default_rng(42)

    # Symbol name table
    sym_table = b""
    for i in range(num_symbols):
        name = f"SYM{i:02d}".encode()
        sym_table += name + b"\x00" * (16 - len(name))

    # Features: random floats in [-1, 1]
    features = rng.random((num_timesteps, num_symbols, features_per_sym), dtype=np.float32).astype(np.float32) * 2 - 1

    # Prices: random walk, all positive — vectorised over time axis
    steps = rng.standard_normal((num_timesteps, num_symbols)).astype(np.float32) * 0.5
    base_prices = np.cumsum(steps, axis=0) + 100.0
    base_prices = np.maximum(base_prices, 1.0)
    spread = np.abs(rng.standard_normal((num_timesteps, num_symbols)).astype(np.float32)) * 0.3
    prices = np.empty((num_timesteps, num_symbols, price_features), dtype=np.float32)
    prices[:, :, 0] = base_prices               # open
    prices[:, :, 1] = base_prices + spread       # high
    prices[:, :, 2] = np.maximum(base_prices - spread, 0.1)  # low
    prices[:, :, 3] = base_prices               # close
    prices[:, :, 4] = 1000.0                    # volume

    # Tradable mask: all tradable
    tradable = np.ones((num_timesteps, num_symbols), dtype=np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))
        f.write(tradable.tobytes(order="C"))


def run_benchmark(
    data_path: Path,
    num_envs: int = 128,
    target_steps: int = 1_000_000,
    max_episode_steps: int = 720,
) -> float:
    """Run num_envs parallel envs for target_steps total steps. Returns steps/sec."""
    import pufferlib_market.binding as binding

    num_symbols = 12
    obs_size = num_symbols * 16 + 5 + num_symbols  # 221 for 12 symbols

    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf,
        act_buf,
        rew_buf,
        term_buf,
        trunc_buf,
        num_envs,
        42,                  # seed
        max_steps=max_episode_steps,
        fee_rate=0.001,
        max_leverage=1.0,
        periods_per_year=8760.0,
        reward_scale=10.0,
        reward_clip=5.0,
        cash_penalty=0.01,
    )
    binding.vec_reset(vec_handle, 42)

    # Warm up: 10k steps (not measured)
    warmup = 10_000 // num_envs
    rng = np.random.default_rng(0)
    for _ in range(warmup):
        act_buf[:] = rng.integers(0, 1 + 2 * num_symbols, size=num_envs, dtype=np.int32)
        binding.vec_step(vec_handle)

    # Timed run
    steps_per_iter = num_envs
    iters = max(1, target_steps // steps_per_iter)
    t0 = time.perf_counter()
    for _ in range(iters):
        act_buf[:] = rng.integers(0, 1 + 2 * num_symbols, size=num_envs, dtype=np.int32)
        binding.vec_step(vec_handle)
    elapsed = time.perf_counter() - t0

    total_steps = iters * steps_per_iter
    binding.vec_close(vec_handle)

    return total_steps / elapsed


def main() -> None:
    print("pufferlib_market C env benchmark")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "bench_data.bin"
        print("Generating synthetic MKTD data (12 symbols, 5000 timesteps)...")
        _write_mktd_bin(data_path, num_symbols=12, num_timesteps=5000)

        # Must call shared() to load data before any env init
        import pufferlib_market.binding as binding
        binding.shared(data_path=str(data_path))

        print(f"\nRunning benchmark: 128 parallel envs x ~1M total steps")
        sps = run_benchmark(data_path, num_envs=128, target_steps=1_000_000)
        print(f"\nResult: {sps:,.0f} steps/sec")
        print(f"         ({sps / 1e6:.2f}M steps/sec)")


if __name__ == "__main__":
    main()
