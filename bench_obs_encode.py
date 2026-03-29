"""Benchmark: fused_obs_norm_linear_relu vs baseline (normalize + F.linear + F.relu).

Tests on the stocks12 observation size (OBS=209), hidden=1024, B=64 and B=256.
Reports: microseconds per forward pass, peak memory allocated (delta vs baseline).

Run from repo root:
    source .venv313/bin/activate
    PYTHONPATH=. python pufferlib_market/bench_obs_encode.py
"""

import torch
import torch.nn.functional as F
from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu, HAS_TRITON


def baseline_obs_encode(obs, mean, std, weight, bias, eps=1e-5):
    """Reference three-step path: normalize -> linear -> relu.

    Computes in FP32 then casts output to match weight dtype (BF16 if weight is BF16).
    This matches the semantics of fused_obs_norm_linear_relu's fallback path.
    """
    obs_norm = (obs.float() - mean.float()) / (std.float() + eps)
    h = F.linear(obs_norm, weight.float(), bias.float())
    return F.relu(h).to(weight.dtype)


def bench_one(fn, warmup=50, iters=200):
    """Time a zero-argument callable after warmup; return mean microseconds."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / iters * 1000  # microseconds per call


def measure_memory(fn):
    """Return peak allocated memory delta in KB when running fn once."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    after = torch.cuda.max_memory_allocated()
    return (after - before) / 1024  # KB


def run_config(B, OBS, H):
    device = torch.device("cuda")
    obs    = torch.randn(B, OBS, device=device, dtype=torch.float32)
    mean   = torch.randn(OBS, device=device, dtype=torch.float32)
    std    = torch.abs(torch.randn(OBS, device=device)) + 0.1
    weight = torch.randn(H, OBS, device=device, dtype=torch.bfloat16)
    bias   = torch.randn(H, device=device, dtype=torch.float32)

    print(f"\n{'='*60}")
    print(f"  Config: B={B}, OBS={OBS}, H={H}")
    print(f"{'='*60}")

    if not HAS_TRITON:
        print("  [WARN] Triton not available -- skipping correctness check and benchmark.")
        return

    # Correctness check: both kernel and baseline accumulate in FP32 then cast to BF16.
    # BF16 has ~7 mantissa bits; quantization spacing at large values (O(100)) is ~0.5.
    # Tolerance of 1.0 is appropriate for any correct FP32-accumulated kernel that outputs
    # BF16, since two independently-quantized BF16 values can differ by up to 1.0 LSB.
    ref = baseline_obs_encode(obs, mean, std, weight, bias)
    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    err = (out.float() - ref.float()).abs().max().item()
    ok  = err <= 1.0
    print(f"  Correctness: max_abs_err={err:.6f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  [ERROR] Correctness check FAILED -- kernel output too far from reference.")
        return

    # Timing
    t_baseline = bench_one(lambda: baseline_obs_encode(obs, mean, std, weight, bias))
    t_fused    = bench_one(lambda: fused_obs_norm_linear_relu(obs, mean, std, weight, bias))
    speedup = t_baseline / t_fused if t_fused > 0 else float("inf")

    print(f"  Baseline (normalize + F.linear + F.relu):  {t_baseline:7.2f} us/call")
    print(f"  Fused Triton kernel:                       {t_fused:7.2f} us/call")
    print(f"  Speedup:                                   {speedup:.2f}x")

    # Memory: fused skips the large (B, OBS) obs_norm intermediate
    mem_baseline = measure_memory(lambda: baseline_obs_encode(obs, mean, std, weight, bias))
    mem_fused    = measure_memory(lambda: fused_obs_norm_linear_relu(obs, mean, std, weight, bias))
    obs_norm_kb  = B * OBS * 4 / 1024  # float32 bytes -> KB

    print(f"  Peak alloc delta baseline: {mem_baseline:.1f} KB")
    print(f"  Peak alloc delta fused:    {mem_fused:.1f} KB")
    print(f"  Expected obs_norm tensor:  {obs_norm_kb:.1f} KB  (eliminated by fusion)")


def main():
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available -- cannot benchmark Triton kernels.")
        return

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name}  (CC {major}.{minor})")

    # stocks12 obs size, hidden=1024
    run_config(B=64,  OBS=209, H=1024)
    run_config(B=256, OBS=209, H=1024)

    # Larger OBS to stress the K-loop
    run_config(B=64,  OBS=256, H=1024)
    run_config(B=256, OBS=512, H=512)


if __name__ == "__main__":
    main()
