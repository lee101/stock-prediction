"""Benchmark: fused_mlp_relu Triton kernel vs nn.Sequential reference.

Usage:
    python pufferlib_market/kernels/bench_mlp.py
"""

import time
import torch
import torch.nn as nn

from pufferlib_market.kernels.fused_mlp import fused_mlp_relu, HAS_TRITON

# ─── Config ───────────────────────────────────────────────────────────────────

BATCH_SIZE  = 128
IN_DIM      = 221        # Typical obs size for crypto12 (12*17+5)
HIDDEN_SIZE = 1024
OUT_DIM     = 1024
WARMUP_ITERS = 50
BENCH_ITERS  = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_weights(in_dim, hidden, out_dim, device, dtype=torch.bfloat16):
    W1 = torch.randn(hidden,  in_dim, device=device, dtype=dtype)
    b1 = torch.zeros(hidden,          device=device, dtype=dtype)
    W2 = torch.randn(out_dim, hidden,  device=device, dtype=dtype)
    b2 = torch.zeros(out_dim,          device=device, dtype=dtype)
    return W1, b1, W2, b2


def bench_sequential(x, W1, b1, W2, b2, n_iters):
    """Reference: nn.Sequential with two Linear + ReLU layers."""
    fc1 = nn.Linear(W1.shape[1], W1.shape[0], bias=True, device=x.device, dtype=torch.bfloat16)
    fc2 = nn.Linear(W2.shape[1], W2.shape[0], bias=True, device=x.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fc1.weight.copy_(W1)
        fc1.bias.copy_(b1)
        fc2.weight.copy_(W2)
        fc2.bias.copy_(b2)

    model = nn.Sequential(fc1, nn.ReLU(), fc2).eval()

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model(x)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_fused(x, W1, b1, W2, b2, n_iters):
    """Fused Triton kernel."""
    # Warmup (triggers autotuning)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = fused_mlp_relu(x, W1, b1, W2, b2)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = fused_mlp_relu(x, W1, b1, W2, b2)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print(f"Device: {DEVICE}")
    print(f"HAS_TRITON: {HAS_TRITON}")
    print(f"Batch={BATCH_SIZE}, in_dim={IN_DIM}, hidden={HIDDEN_SIZE}, out_dim={OUT_DIM}")
    print(f"Warmup={WARMUP_ITERS}, bench iters={BENCH_ITERS}")
    print()

    dtype = torch.bfloat16
    x = torch.randn(BATCH_SIZE, IN_DIM, device=DEVICE, dtype=dtype)
    W1, b1, W2, b2 = build_weights(IN_DIM, HIDDEN_SIZE, OUT_DIM, DEVICE, dtype)

    # Sequential reference
    t_seq = bench_sequential(x, W1, b1, W2, b2, BENCH_ITERS)
    fps_seq = BENCH_ITERS * BATCH_SIZE / t_seq

    print(f"nn.Sequential (ref):  {t_seq*1000:.1f} ms total  |  {fps_seq:,.0f} samples/sec")

    if HAS_TRITON and DEVICE.type == "cuda":
        t_fused = bench_fused(x, W1, b1, W2, b2, BENCH_ITERS)
        fps_fused = BENCH_ITERS * BATCH_SIZE / t_fused
        speedup = fps_fused / fps_seq
        print(f"fused_mlp_relu (Triton): {t_fused*1000:.1f} ms total  |  {fps_fused:,.0f} samples/sec  |  {speedup:.2f}x speedup")
    else:
        print("Triton not available or not on CUDA — skipping fused kernel bench.")

    # Also benchmark larger batch (4096) — closer to PPO training
    print()
    print("--- Large batch: 4096 ---")
    x_big = torch.randn(4096, IN_DIM, device=DEVICE, dtype=dtype)
    t_seq_big = bench_sequential(x_big, W1, b1, W2, b2, BENCH_ITERS)
    fps_seq_big = BENCH_ITERS * 4096 / t_seq_big
    print(f"nn.Sequential (ref):  {t_seq_big*1000:.1f} ms total  |  {fps_seq_big:,.0f} samples/sec")

    if HAS_TRITON and DEVICE.type == "cuda":
        t_fused_big = bench_fused(x_big, W1, b1, W2, b2, BENCH_ITERS)
        fps_fused_big = BENCH_ITERS * 4096 / t_fused_big
        speedup_big = fps_fused_big / fps_seq_big
        print(f"fused_mlp_relu (Triton): {t_fused_big*1000:.1f} ms total  |  {fps_fused_big:,.0f} samples/sec  |  {speedup_big:.2f}x speedup")


if __name__ == "__main__":
    main()
