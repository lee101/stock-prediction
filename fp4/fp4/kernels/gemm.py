"""GEMM backend dispatcher for fp4.

On first call (and only on Blackwell + CUDA), lazily compiles the CUTLASS
NVFP4xNVFP4 -> BF16 GEMM extension via torch.utils.cpp_extension.load and
caches the resulting module.  On any failure (non-Blackwell, no CUDA, compile
error, etc.) we fall back to a plain torch.matmul on the dequantized
emulation tensors and log a one-line reason on first use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

_KERNEL_DIR = Path(__file__).resolve().parent
_CUTLASS_INC = (_KERNEL_DIR / ".." / ".." / ".." / "external" / "flash-attention"
                / "csrc" / "cutlass" / "include").resolve()
_CUTLASS_TOOLS_INC = (_KERNEL_DIR / ".." / ".." / ".." / "external" / "flash-attention"
                      / "csrc" / "cutlass" / "tools" / "util" / "include").resolve()

_ext: Optional[object] = None
_attempted: bool = False
_skip_reason: Optional[str] = None


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 12  # SM120 / SM121 (RTX 50 series, B200, etc.)


def _try_load() -> Optional[object]:
    global _ext, _attempted, _skip_reason
    if _attempted:
        return _ext
    _attempted = True

    if not torch.cuda.is_available():
        _skip_reason = "no CUDA"
        print(f"[fp4.kernels.gemm] CUTLASS NVFP4 extension not loaded: {_skip_reason}",
              file=sys.stderr)
        return None
    if not _is_blackwell():
        cap = torch.cuda.get_device_capability(0)
        _skip_reason = f"non-Blackwell GPU (sm_{cap[0]}{cap[1]})"
        print(f"[fp4.kernels.gemm] CUTLASS NVFP4 extension not loaded: {_skip_reason}",
              file=sys.stderr)
        return None
    if not _CUTLASS_INC.exists():
        _skip_reason = f"CUTLASS headers not found at {_CUTLASS_INC}"
        print(f"[fp4.kernels.gemm] CUTLASS NVFP4 extension not loaded: {_skip_reason}",
              file=sys.stderr)
        return None

    try:
        # CUDA_HOME must be set before importing torch.utils.cpp_extension so
        # its module-level _find_cuda_home() picks up the right toolkit.
        cur = os.environ.get("CUDA_HOME", "")
        if not cur or not os.path.exists(os.path.join(cur, "bin", "nvcc")):
            for cand in ("/usr/local/cuda", "/usr/local/cuda-13", "/usr/local/cuda-12.9",
                         "/usr/local/cuda-12.8", "/usr/local/cuda-12"):
                if os.path.exists(os.path.join(cand, "bin", "nvcc")):
                    os.environ["CUDA_HOME"] = cand
                    os.environ["CUDA_PATH"] = cand
                    os.environ["PATH"] = os.path.join(cand, "bin") + os.pathsep + os.environ.get("PATH", "")
                    break
        from torch.utils.cpp_extension import load
        import torch.utils.cpp_extension as _cppx
        # Force-override torch's cached cuda_home if it was resolved before our env fix.
        good = os.environ.get("CUDA_HOME")
        if good and getattr(_cppx, "CUDA_HOME", None) != good:
            _cppx.CUDA_HOME = good
        cap = torch.cuda.get_device_capability(0)
        arch = f"{cap[0]}{cap[1]}"  # "120" for RTX 5090
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}+PTX")
        # /tmp may be sandboxed (compiler-spawned subprocess temp files vanish
        # before the assembler reads them). Force compiler temp files onto a
        # writable on-disk location under the repo-local tmp/ directory.
        from fp4.paths import ensure_tmp
        alt_tmp = str(ensure_tmp() / "cuda_build")
        os.makedirs(alt_tmp, exist_ok=True)
        os.environ["TMPDIR"] = alt_tmp
        _ext = load(
            name="fp4_nvfp4_gemm_ext",
            sources=[str(_KERNEL_DIR / "gemm_ext.cpp"),
                     str(_KERNEL_DIR / "gemm_kernel.cu")],
            extra_include_paths=[str(_CUTLASS_INC), str(_CUTLASS_TOOLS_INC)],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                f"-gencode=arch=compute_{arch}a,code=sm_{arch}a",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
        return _ext
    except Exception as exc:
        _skip_reason = f"compile failed: {type(exc).__name__}: {exc}"
        print(f"[fp4.kernels.gemm] CUTLASS NVFP4 extension not loaded: {_skip_reason}",
              file=sys.stderr)
        _ext = None
        return None


def have_cutlass() -> bool:
    return _try_load() is not None


def skip_reason() -> Optional[str]:
    _try_load()
    return _skip_reason


def nvfp4_matmul_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """NVFP4xNVFP4 -> BF16 GEMM. Inputs are bf16; quantized internally.

    Falls back to torch.matmul(a.bf16, b.bf16) on any failure.
    """
    ext = _try_load()
    if ext is not None and a.is_cuda and b.is_cuda:
        try:
            return ext.nvfp4_matmul_bf16(a.contiguous(), b.contiguous())
        except Exception as exc:
            global _skip_reason
            _skip_reason = f"runtime error: {type(exc).__name__}: {exc}"
            print(f"[fp4.kernels.gemm] CUTLASS path failed at runtime, falling back: {_skip_reason}",
                  file=sys.stderr)
    return torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16))


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generic GEMM dispatcher. On Blackwell uses NVFP4; else torch.matmul."""
    if a.is_cuda and b.is_cuda and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
        ext = _try_load()
        if ext is not None:
            try:
                return ext.nvfp4_matmul_bf16(a.contiguous(), b.contiguous())
            except Exception:
                pass
    return torch.matmul(a, b)
