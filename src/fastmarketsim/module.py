from __future__ import annotations

import logging
import platform
import threading
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME, load as load_extension_module

_LOCK = threading.Lock()
_EXTENSION: Any | None = None


def _extra_cflags() -> list[str]:
    flags = ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"]
    if platform.system() != "Windows":
        flags.append("-fopenmp")
    return flags


def _extra_ldflags() -> list[str]:
    if platform.system() == "Windows":
        return []
    return ["-fopenmp"]


def _extra_cuda_cflags() -> list[str]:
    return ["-O3", "--use_fast_math"]


def load_extension(*, verbose: bool = False) -> Any:
    """Compile (if necessary) and load the C++ market simulator bindings."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION

        repo_root = Path(__file__).resolve().parents[2]
        cpp_root = repo_root / "cppsimulator"
        sources = [
            cpp_root / "src" / "market_sim.cpp",
            cpp_root / "src" / "forecast.cpp",
            cpp_root / "bindings" / "market_sim_py.cpp",
        ]
        build_dir = cpp_root / "build_py"
        build_dir.mkdir(parents=True, exist_ok=True)

        has_cuda = bool(torch.version.cuda) and CUDA_HOME is not None and torch.cuda.is_available()
        if bool(torch.version.cuda) and torch.cuda.is_available() and CUDA_HOME is None:
            logging.warning(
                "fastmarketsim: CUDA toolkit not detected (set CUDA_HOME) – building CPU-only extension."
            )

        _EXTENSION = load_extension_module(
            name="market_sim_ext",
            sources=[str(src) for src in sources],
            extra_cflags=_extra_cflags(),
            extra_ldflags=_extra_ldflags(),
            extra_cuda_cflags=_extra_cuda_cflags() if has_cuda else [],
            extra_include_paths=[str(cpp_root / "include")],
            build_directory=str(build_dir),
            with_cuda=has_cuda,
            verbose=verbose,
        )
        setattr(_EXTENSION, "_fastmarketsim_has_cuda", has_cuda)
        return _EXTENSION
