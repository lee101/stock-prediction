"""Benchmark gpu_trading_env SPS vs batch size.

The current CUDA kernel is single-instrument, one thread per parallel env.
Goal: measure peak SPS on the 5090 across B in {1k, 4k, 16k, 64k, 256k, 1M}.
"""
from __future__ import annotations

import time
import torch

import gpu_trading_env


def bench_single_instrument(B: int, steps: int = 20000, T_synth: int = 8192) -> dict:
    env = gpu_trading_env.make(B=B, T_synth=T_synth)
    action = torch.zeros(B, 4, device="cuda", dtype=torch.float32)
    close = env.ohlc[:, 3]
    mid = close.mean()
    action[:, 0] = mid * 1.001
    action[:, 1] = mid * 0.999
    action[:, 2] = 0.5
    action[:, 3] = 0.5

    for _ in range(200):
        env.step(action)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    total = steps * B
    return {"B": B, "steps": steps, "total_env_steps": total, "sec": dt, "sps": total / dt}


def main() -> None:
    print("gpu_trading_env single-instrument bench (5090)")
    print("=" * 60)
    print(f"{'B':>8} {'steps':>8} {'total':>12} {'sec':>8} {'SPS':>16}")
    for B in (1024, 4096, 16384, 65536, 262144, 1_048_576):
        try:
            r = bench_single_instrument(B, steps=max(200, 4_000_000 // B))
            print(f"{r['B']:>8d} {r['steps']:>8d} {r['total_env_steps']:>12,d} "
                  f"{r['sec']:>8.3f} {r['sps']:>16,.0f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>8d} OOM")
            break
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
