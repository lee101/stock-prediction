"""Benchmark: Triton multi_query_attention vs F.scaled_dot_product_attention.

Run:
    source .venv313/bin/activate
    python binanceneural/kernels/bench_attention.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from binanceneural.kernels.attention import HAS_TRITON, multi_query_attention


def _make_inputs(batch: int, heads: int, seq: int, head_dim: int, device: str, dtype: torch.dtype):
    Q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, 1, seq, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, 1, seq, head_dim, device=device, dtype=dtype)
    return Q, K, V


def _bench(fn, warmup: int = 10, iters: int = 200) -> float:
    """Return median latency in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def bench_sdpa(Q, K, V, causal: bool = False):
    """F.scaled_dot_product_attention path (needs K/V expanded to match Q heads)."""
    n_rep = Q.shape[1]  # K has 1 head; expand to match Q
    K_exp = K.expand(-1, n_rep, -1, -1)
    V_exp = V.expand(-1, n_rep, -1, -1)
    return F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=causal)


def bench_triton(Q, K, V, causal: bool = False):
    return multi_query_attention(Q, K, V, causal=causal)


def run_benchmark(
    batch: int = 64,
    heads: int = 8,
    seq: int = 48,
    head_dim: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    causal: bool = True,
):
    print(
        f"\n{'='*60}\n"
        f"Config: batch={batch}, heads={heads}, seq={seq}, head_dim={head_dim}, "
        f"dtype={dtype}, causal={causal}\n"
        f"{'='*60}"
    )

    Q, K, V = _make_inputs(batch, heads, seq, head_dim, device, dtype)

    sdpa_ms = _bench(lambda: bench_sdpa(Q, K, V, causal))
    print(f"  SDPA (torch):    {sdpa_ms:.3f} ms")

    if HAS_TRITON:
        triton_ms = _bench(lambda: bench_triton(Q, K, V, causal))
        print(f"  Triton kernel:   {triton_ms:.3f} ms")
        speedup = sdpa_ms / triton_ms if triton_ms > 0 else float("inf")
        print(f"  Speedup:         {speedup:.2f}x")
    else:
        print("  Triton not available — skipping Triton benchmark")

    # Correctness check
    with torch.no_grad():
        ref = bench_sdpa(Q, K, V, causal).float()
        if HAS_TRITON:
            got = bench_triton(Q, K, V, causal).float()
            max_err = (ref - got).abs().max().item()
            print(f"  Max abs error vs SDPA: {max_err:.6f}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark")
        return

    # Primary config matching the model's training shape
    run_benchmark(batch=64, heads=8, seq=48, head_dim=64, dtype=torch.bfloat16, causal=True)

    # Additional configs
    run_benchmark(batch=64, heads=8, seq=48, head_dim=64, dtype=torch.bfloat16, causal=False)
    run_benchmark(batch=32, heads=8, seq=64, head_dim=64, dtype=torch.bfloat16, causal=True)
    run_benchmark(batch=1,  heads=8, seq=48, head_dim=64, dtype=torch.bfloat16, causal=True)


if __name__ == "__main__":
    main()
