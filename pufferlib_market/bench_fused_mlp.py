"""Benchmark: fused_mlp_relu Triton kernel vs nn.Sequential for RL policy.

Benchmarks the shapes typical in PPO training:
  - batch_size = 128 steps (standard rollout slice)
  - batch_size = 256 steps (mixed batch)
  - hidden_size = 1024 (RL policy standard)
  - hidden_size = 512 / 256 (smaller configs)

Prints speedup ratio and notes the compute capability of the current GPU.

Usage:
    python pufferlib_market/bench_fused_mlp.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pufferlib_market.kernels.fused_mlp import fused_mlp_relu, HAS_TRITON

WARMUP_ITERS = 50
BENCH_ITERS  = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_weights(in_dim, hidden, out_dim, device, dtype=torch.bfloat16):
    W1 = torch.randn(hidden, in_dim,  device=device, dtype=dtype)
    b1 = torch.zeros(hidden,          device=device, dtype=dtype)
    W2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.zeros(out_dim,         device=device, dtype=dtype)
    return W1, b1, W2, b2


def _bench_sequential(x, W1, b1, W2, b2):
    fc1 = nn.Linear(W1.shape[1], W1.shape[0], bias=True, device=x.device, dtype=torch.bfloat16)
    fc2 = nn.Linear(W2.shape[1], W2.shape[0], bias=True, device=x.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fc1.weight.copy_(W1); fc1.bias.copy_(b1)
        fc2.weight.copy_(W2); fc2.bias.copy_(b2)
    model = nn.Sequential(fc1, nn.ReLU(), fc2).eval()

    for _ in range(WARMUP_ITERS):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = model(x)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_fused(x, W1, b1, W2, b2):
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = fused_mlp_relu(x, W1, b1, W2, b2)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            _ = fused_mlp_relu(x, W1, b1, W2, b2)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _run_config(label, batch_size, in_dim, hidden, out_dim):
    x = torch.randn(batch_size, in_dim, device=DEVICE, dtype=torch.bfloat16)
    W1, b1, W2, b2 = _build_weights(in_dim, hidden, out_dim, DEVICE)

    t_seq = _bench_sequential(x, W1, b1, W2, b2)
    fps_seq = BENCH_ITERS * batch_size / t_seq
    print(f"  [{label}] nn.Sequential : {t_seq*1000:7.1f} ms  |  {fps_seq:>12,.0f} samples/s")

    if HAS_TRITON and DEVICE.type == "cuda":
        t_fused = _bench_fused(x, W1, b1, W2, b2)
        fps_fused = BENCH_ITERS * batch_size / t_fused
        speedup = fps_fused / fps_seq
        print(f"  [{label}] fused Triton  : {t_fused*1000:7.1f} ms  |  {fps_fused:>12,.0f} samples/s  |  {speedup:.2f}x speedup")
    else:
        print(f"  [{label}] Triton not available or not on CUDA — skipping fused bench.")
    print()


def main():
    if DEVICE.type == "cuda":
        major, minor = torch.cuda.get_device_capability(DEVICE)
        name = torch.cuda.get_device_name(DEVICE)
        print(f"GPU : {name} (CC {major}.{minor})")
    else:
        print("Device: CPU (no CUDA)")
    print(f"HAS_TRITON : {HAS_TRITON}")
    print(f"Warmup={WARMUP_ITERS} iters, Bench={BENCH_ITERS} iters")
    print()

    # in_dim = 221 = 12*17+5 (crypto12 obs) or 215 = stocks12 daily obs
    IN_DIM = 221

    print("=== hidden=1024 (RL standard) ===")
    _run_config("B=128, h=1024",  128, IN_DIM, 1024, 1024)
    _run_config("B=256, h=1024",  256, IN_DIM, 1024, 1024)
    _run_config("B=4096, h=1024", 4096, IN_DIM, 1024, 1024)

    print("=== hidden=512 ===")
    _run_config("B=128, h=512",  128, IN_DIM, 512, 512)
    _run_config("B=256, h=512",  256, IN_DIM, 512, 512)

    print("=== hidden=256 ===")
    _run_config("B=128, h=256",  128, IN_DIM, 256, 256)
    _run_config("B=256, h=256",  256, IN_DIM, 256, 256)


if __name__ == "__main__":
    main()
