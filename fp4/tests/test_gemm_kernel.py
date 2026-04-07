"""Tests for the CUTLASS NVFP4 GEMM torch extension.

Skips cleanly when not on a Blackwell GPU (sm120/sm121) or when CUDA is
unavailable. On Blackwell, asserts:
  (a) numerical agreement with a torch.matmul reference within
      block-quantization tolerance, and
  (b) achieved TFLOPS strictly greater than torch BF16 matmul on the
      same shape.
"""
from __future__ import annotations

import time

import pytest
import torch

import importlib
gemm_mod = importlib.import_module("fp4.kernels.gemm")


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 12


pytestmark = pytest.mark.skipif(
    not _is_blackwell(),
    reason=f"NVFP4 GEMM requires Blackwell sm>=12.0; have "
           f"{torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'no CUDA'}",
)


def _bench(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def test_extension_loads():
    ext = gemm_mod._try_load()
    if ext is None:
        pytest.skip(f"extension didn't load: {gemm_mod.skip_reason()}")
    assert hasattr(ext, "nvfp4_matmul_bf16")
    assert getattr(ext, "HAVE_BLACKWELL", False)


def test_numerical_match():
    ext = gemm_mod._try_load()
    if ext is None:
        pytest.skip(f"extension unavailable: {gemm_mod.skip_reason()}")
    torch.manual_seed(0)
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.5

    out = gemm_mod.nvfp4_matmul_bf16(a, b)
    ref = torch.matmul(a.float(), b.float()).to(torch.bfloat16)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    # Block-quantization tolerance: NVFP4 has 4 bits per element + per-16
    # block scale, so per-element relative error is O(0.1). Compare via
    # relative Frobenius norm.
    rel = (out.float() - ref.float()).norm() / (ref.float().norm() + 1e-6)
    assert rel < 0.20, f"NVFP4 GEMM relative error too large: {rel.item():.4f}"


def test_tflops_beats_bf16():
    ext = gemm_mod._try_load()
    if ext is None:
        pytest.skip(f"extension unavailable: {gemm_mod.skip_reason()}")
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    bf16_sec = _bench(lambda: torch.matmul(a, b))
    nvfp4_sec = _bench(lambda: gemm_mod.nvfp4_matmul_bf16(a, b))

    flop = 2.0 * M * N * K
    bf16_tflops  = flop / bf16_sec  / 1e12
    nvfp4_tflops = flop / nvfp4_sec / 1e12
    print(f"\nBF16 : {bf16_tflops:8.1f} TFLOPS")
    print(f"NVFP4: {nvfp4_tflops:8.1f} TFLOPS")
    assert nvfp4_tflops > bf16_tflops, (
        f"NVFP4 ({nvfp4_tflops:.1f}) did not beat BF16 ({bf16_tflops:.1f})"
    )
