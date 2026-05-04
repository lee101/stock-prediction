"""Compatibility loader for the in-repo ``gpu_trading_env`` package.

The importable package lives under ``gpu_trading_env/python`` for packaging,
but repo-root test runs can otherwise import this outer directory as an empty
namespace package. Execute the real package initializer in this module so
``import gpu_trading_env`` behaves the same with or without installation.
"""
from __future__ import annotations

from pathlib import Path


_OUTER_DIR = Path(__file__).resolve().parent
_IMPL_DIR = _OUTER_DIR / "python" / "gpu_trading_env"
_IMPL_INIT = _IMPL_DIR / "__init__.py"

__file__ = str(_IMPL_INIT)
__path__ = [str(_IMPL_DIR), str(_OUTER_DIR)]  # type: ignore[name-defined]

exec(compile(_IMPL_INIT.read_text(encoding="utf-8"), str(_IMPL_INIT), "exec"))
