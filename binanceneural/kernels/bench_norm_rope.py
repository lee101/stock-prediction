"""Benchmark Triton RoPE and RMS norm kernels vs PyTorch reference.

Usage:
    python binanceneural/kernels/bench_norm_rope.py

Prints wall-clock speedup for each kernel at typical model shapes.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

# ---- check availability ---------------------------------------------------
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from binanceneural.kernels.rope import apply_rope, apply_rope_fused, HAS_TRITON as _ROPE_TRITON
    from binanceneural.kernels.norm import rms_norm, fused_rms_norm_qkv
except ImportError as e:
    print(f"Could not import kernels: {e}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _warmup_and_time(fn, n_warmup: int = 5, n_bench: int = 50) -> float:
    """Return median wall-clock time in ms for fn()."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]  # median


def _fmt(name: str, ref_ms: float, triton_ms: float) -> str:
    speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
    return f"  {name:<40s}  ref={ref_ms:.3f}ms  triton={triton_ms:.3f}ms  speedup={speedup:.2f}x"


# ---------------------------------------------------------------------------
# RoPE benchmark
# ---------------------------------------------------------------------------

def bench_rope(device: torch.device) -> None:
    print("\n=== RoPE (interleaved convention) ===")
    # Typical: batch=4, seq=64, heads=8, head_dim=64
    for B, T, Hq, Hk, D in [
        (4, 48, 8, 4, 64),
        (8, 64, 8, 4, 64),
        (1, 128, 16, 8, 64),
    ]:
        half_dim = D // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D)).to(device)
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]  # (1, T, 1, half_dim)
        sin = freqs.sin()[None, :, None, :]

        q = torch.randn(B, T, Hq, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T, Hk, D, device=device, dtype=torch.bfloat16)

        # PyTorch reference — cos/sin broadcast over (B, T, H, half_dim)
        def ref():
            def _rot(x):
                xr = x[..., ::2]; xi = x[..., 1::2]
                # cos shape: (1, T, 1, half_dim) broadcasts over B and H
                or_ = xr * cos - xi * sin; oi = xr * sin + xi * cos
                return torch.stack([or_, oi], dim=-1).flatten(-2)
            return _rot(q), _rot(k)

        ref_ms = _warmup_and_time(ref)

        # Triton apply_rope
        def triton_fn():
            return apply_rope(q, k, cos, sin)

        if _ROPE_TRITON:
            triton_ms = _warmup_and_time(triton_fn)
        else:
            triton_ms = float("nan")

        # Triton apply_rope_fused (no pre-computed cos/sin)
        def triton_fused():
            return apply_rope_fused(q, k, inv_freq)

        if _ROPE_TRITON:
            fused_ms = _warmup_and_time(triton_fused)
        else:
            fused_ms = float("nan")

        tag = f"rope B={B} T={T} Hq={Hq} Hk={Hk} D={D}"
        print(_fmt(f"{tag} [apply_rope]", ref_ms, triton_ms))
        print(_fmt(f"{tag} [apply_rope_fused]", ref_ms, fused_ms))


# ---------------------------------------------------------------------------
# RMS norm benchmark
# ---------------------------------------------------------------------------

def bench_rms_norm(device: torch.device) -> None:
    print("\n=== RMS Norm (unweighted) ===")
    for rows, N in [
        (64, 256),
        (192, 256),
        (384, 512),
        (768, 512),
    ]:
        x = torch.randn(rows, N, device=device, dtype=torch.bfloat16)

        def ref_fn():
            if hasattr(F, "rms_norm"):
                return F.rms_norm(x, (N,), eps=1e-5)
            v = x.float().pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(v + 1e-5)).to(x.dtype)

        def triton_fn():
            return rms_norm(x, weight=None, eps=1e-5)

        ref_ms = _warmup_and_time(ref_fn)
        triton_ms = _warmup_and_time(triton_fn) if HAS_TRITON else float("nan")

        print(_fmt(f"rms_norm rows={rows} N={N}", ref_ms, triton_ms))


# ---------------------------------------------------------------------------
# Fused RMS norm + QKV benchmark
# ---------------------------------------------------------------------------

def bench_fused_qkv(device: torch.device) -> None:
    print("\n=== Fused RMS Norm + QKV Projection ===")
    for B, T, N, Dq, Dk, Dv in [
        (4, 48, 256, 256, 128, 128),   # nano model hidden=256, 8Q/4K heads × 32
        (8, 64, 256, 256, 128, 128),
        (4, 48, 512, 512, 256, 256),
    ]:
        x = torch.randn(B * T, N, device=device, dtype=torch.bfloat16)
        wq = torch.randn(Dq, N, device=device, dtype=torch.bfloat16)
        wk = torch.randn(Dk, N, device=device, dtype=torch.bfloat16)
        wv = torch.randn(Dv, N, device=device, dtype=torch.bfloat16)

        def ref_fn():
            v = x.float().pow(2).mean(-1, keepdim=True)
            normed = (x * torch.rsqrt(v + 1e-5)).to(x.dtype)
            q = F.linear(normed, wq)
            k = F.linear(normed, wk)
            v = F.linear(normed, wv)
            return q, k, v

        def triton_fn():
            return fused_rms_norm_qkv(x, None, wq, wk, wv, eps=1e-5)

        ref_ms = _warmup_and_time(ref_fn)
        triton_ms = _warmup_and_time(triton_fn) if HAS_TRITON else float("nan")

        tag = f"fused_qkv rows={B*T} N={N} Dq={Dq}"
        print(_fmt(tag, ref_ms, triton_ms))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — cannot benchmark Triton kernels.")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"Torch: {torch.__version__}")

    if not HAS_TRITON:
        print("WARNING: Triton not installed — all triton timings will be NaN.")

    bench_rope(device)
    bench_rms_norm(device)
    bench_fused_qkv(device)

    print("\nDone.")


if __name__ == "__main__":
    main()
