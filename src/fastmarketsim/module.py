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


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[1], here.parents[2], Path.cwd().resolve()]
    for root in candidates:
        cpp_root = root / "cppsimulator"
        if (cpp_root / "src" / "market_sim.cpp").exists() and (
            cpp_root / "bindings" / "market_sim_py.cpp"
        ).exists():
            return root
    raise FileNotFoundError(
        "Unable to locate cppsimulator sources relative to fastmarketsim/module.py"
    )


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


def _has_usable_cuda_toolkit() -> bool:
    if not (bool(torch.version.cuda) and CUDA_HOME is not None and torch.cuda.is_available()):
        return False

    cuda_root = Path(CUDA_HOME)
    lib_candidates = [
        cuda_root / "lib64" / "libcudart.so",
        cuda_root / "lib64" / "libcudart_static.a",
        cuda_root / "lib" / "x64" / "cudart.lib",
    ]
    return any(candidate.exists() for candidate in lib_candidates)


def load_extension(*, verbose: bool = False) -> Any:
    """Compile (if necessary) and load the C++ market simulator bindings."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION

        repo_root = _resolve_repo_root()
        cpp_root = repo_root / "cppsimulator"
        sources = [
            cpp_root / "src" / "market_sim.cpp",
            cpp_root / "src" / "forecast.cpp",
            cpp_root / "bindings" / "market_sim_py.cpp",
        ]
        build_dir = cpp_root / "build_py"
        build_dir.mkdir(parents=True, exist_ok=True)

        has_cuda = _has_usable_cuda_toolkit()
        if bool(torch.version.cuda) and torch.cuda.is_available() and CUDA_HOME is None:
            logging.warning(
                "fastmarketsim: CUDA toolkit not detected (set CUDA_HOME) – building CPU-only extension."
            )
        elif bool(torch.version.cuda) and torch.cuda.is_available() and not has_cuda:
            logging.warning(
                "fastmarketsim: CUDA runtime detected but libcudart is unavailable under %s; building CPU-only extension.",
                CUDA_HOME,
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
