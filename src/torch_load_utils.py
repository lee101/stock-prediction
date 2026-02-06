from __future__ import annotations

import inspect
import sys
import types
from typing import Any


def _install_pathlib_local_shim() -> None:
    """Ensure `pathlib._local` exists for older Python versions.

    Python 3.13+ may pickle Path objects from `pathlib._local`. Loading those
    pickles in Python 3.12 and earlier can fail because `pathlib._local` does
    not exist there.
    """

    if "pathlib._local" in sys.modules:
        return

    try:  # pragma: no cover - present on some newer Pythons
        import importlib

        importlib.import_module("pathlib._local")
        return
    except Exception:
        pass

    import pathlib

    module = types.ModuleType("pathlib._local")
    # Provide the common classes used by pickled Path objects.
    for attr in (
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    ):
        if hasattr(pathlib, attr):
            setattr(module, attr, getattr(pathlib, attr))
    sys.modules["pathlib._local"] = module


def torch_load_compat(*args: Any, **kwargs: Any) -> Any:
    """Like `torch.load`, but with small compatibility shims installed."""

    _install_pathlib_local_shim()

    import torch

    # Older torch versions don't accept `weights_only`.
    if "weights_only" in kwargs:
        try:
            sig = inspect.signature(torch.load)
        except (TypeError, ValueError):  # pragma: no cover - weird wrappers
            sig = None
        if sig is not None and "weights_only" not in sig.parameters:
            kwargs.pop("weights_only", None)

    return torch.load(*args, **kwargs)


__all__ = ["torch_load_compat"]

