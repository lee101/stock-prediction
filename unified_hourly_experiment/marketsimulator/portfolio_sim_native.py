"""Native C++ backend loader for portfolio simulator."""

from __future__ import annotations

import logging
import os
import platform
import threading
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load as load_extension_module

_LOCK = threading.Lock()
_EXTENSION: Any | None = None
_LOAD_ERROR: str | None = None


def _extra_cflags() -> list[str]:
    flags = ["-O3", "-std=c++17"]
    if platform.system() != "Windows":
        flags.append("-fopenmp")
    return flags


def _extra_ldflags() -> list[str]:
    if platform.system() == "Windows":
        return []
    return ["-fopenmp"]


def load_portfolio_native_extension(*, verbose: bool = False) -> Any | None:
    """Compile/load the native portfolio simulator extension.

    Returns `None` when disabled or unavailable.
    """

    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if os.environ.get("PORTFOLIO_SIM_DISABLE_NATIVE", "").strip().lower() in {"1", "true", "yes"}:
        return None

    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _LOAD_ERROR is not None:
            return None
        try:
            root = Path(__file__).resolve().parent
            src = root / "native" / "portfolio_sim_ext.cpp"
            build_dir = root / "native" / "build_py"
            build_dir.mkdir(parents=True, exist_ok=True)

            _EXTENSION = load_extension_module(
                name="portfolio_sim_native_ext",
                sources=[str(src)],
                extra_cflags=_extra_cflags(),
                extra_ldflags=_extra_ldflags(),
                build_directory=str(build_dir),
                verbose=verbose,
            )
            return _EXTENSION
        except Exception as exc:  # pragma: no cover - platform/compiler dependent
            _LOAD_ERROR = str(exc)
            logging.warning("portfolio native backend unavailable: %s", exc)
            return None


def portfolio_native_load_error() -> str | None:
    return _LOAD_ERROR


__all__ = ["load_portfolio_native_extension", "portfolio_native_load_error"]
