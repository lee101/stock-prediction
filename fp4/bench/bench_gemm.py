"""Microbench: NVFP4 vs BF16 vs FP8 GEMM TFLOPS on Blackwell.

Skips formats whose hardware/kernels are missing with a clear message rather
than crashing.  Designed to be runnable standalone:

    python fp4/bench/bench_gemm.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Callable

import torch

SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (2048, 8192, 2048),
]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_gemm(fn: Callable[[], torch.Tensor], iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


def _tflops(m: int, n: int, k: int, sec: float) -> float:
    return (2.0 * m * n * k) / sec / 1e12


def _bf16_gemm(m: int, n: int, k: int) -> float:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    sec = _time_gemm(lambda: torch.matmul(a, b))
    return _tflops(m, n, k, sec)


def _fp8_gemm(m: int, n: int, k: int) -> float | None:
    if not hasattr(torch, "_scaled_mm"):
        return None
    try:
        e4m3 = torch.float8_e4m3fn  # type: ignore[attr-defined]
    except AttributeError:
        return None
    try:
        a = torch.randn(m, k, device="cuda").to(e4m3)
        b = torch.randn(n, k, device="cuda").to(e4m3).t()
        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")

        def _run() -> torch.Tensor:
            return torch._scaled_mm(  # type: ignore[attr-defined]
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
            )

        sec = _time_gemm(_run)
        return _tflops(m, n, k, sec)
    except Exception as exc:  # pragma: no cover - hardware path
        print(f"  fp8 skipped: {type(exc).__name__}: {exc}")
        return None


def _nvfp4_gemm(m: int, n: int, k: int) -> float | None:
    try:
        import importlib
        fp4_gemm = importlib.import_module("fp4.fp4.kernels.gemm")
    except Exception:
        try:
            import importlib
            fp4_gemm = importlib.import_module("fp4.kernels.gemm")
        except Exception as exc:
            print(f"  nvfp4 skipped: import failed: {exc}")
            return None
    if not fp4_gemm.have_cutlass():
        print(f"  nvfp4 skipped: {fp4_gemm.skip_reason()}")
        return None
    try:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        # warmup + time real CUTLASS NVFP4 kernel directly
        sec = _time_gemm(lambda: fp4_gemm.nvfp4_matmul_bf16(a, b))
        return _tflops(m, n, k, sec)
    except Exception as exc:  # pragma: no cover
        print(f"  nvfp4 skipped: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping GEMM bench.")
        return 0

    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}  sm={dev.major}.{dev.minor}  mem={dev.total_memory/1e9:.1f}GB")
    print(f"torch={torch.__version__}")
    print()
    header = f"{'shape':<22}{'BF16 TFLOPs':>14}{'FP8 TFLOPs':>14}{'NVFP4 TFLOPs':>16}{'speedup':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    for (m, n, k) in SHAPES:
        bf = _bf16_gemm(m, n, k)
        f8 = _fp8_gemm(m, n, k)
        f4 = _nvfp4_gemm(m, n, k)
        f8_s = f"{f8:>14.1f}" if f8 is not None else f"{'skip':>14}"
        f4_s = f"{f4:>16.1f}" if f4 is not None else f"{'skip':>16}"
        sp = (f4 / bf) if (f4 is not None and bf > 0) else None
        sp_s = f"{sp:>11.2f}x" if sp is not None else f"{'-':>12}"
        print(f"{str((m,n,k)):<22}{bf:>14.1f}{f8_s}{f4_s}{sp_s}")
        rows.append({"shape": [m, n, k], "bf16_tflops": bf,
                     "fp8_tflops": f8, "nvfp4_tflops": f4,
                     "speedup_vs_bf16": sp})

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.date.today().isoformat()
    out_path = out_dir / f"bench_gemm_{stamp}.json"
    payload = {
        "gpu": dev.name,
        "sm": f"{dev.major}.{dev.minor}",
        "torch": torch.__version__,
        "iters": args.iters,
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
