"""Compatibility helpers for ``tblib`` pickling support.

fal's isolate runtime expects ``tblib.pickling_support`` to expose
``unpickle_exception_with_attrs`` when deserialising exceptions. Older
tblib releases (<=3.1) do not ship the helper which results in a failed
unpickling step when the worker streams back an exception payload.

Import this module (or call :func:`ensure_tblib_pickling_support`) ahead
of any fal worker initialisation to guarantee the helpers are present.
"""

from __future__ import annotations

from typing import Any, Optional


_PATCH_FLAG = "_fal_tblib_patch_applied"


def _install_unpickle_shim(pickling_support: Any) -> None:
    """Inject ``unpickle_exception_with_attrs`` for tblib<=3.1."""

    def unpickle_exception_with_attrs(
        func: Any,
        attrs: dict[str, Any],
        cause: Optional[BaseException],
        tb: Any,
        context: Optional[BaseException],
        suppress_context: bool,
        notes: Optional[Any],
    ) -> BaseException:
        inst = func.__new__(func)
        for key, value in attrs.items():
            setattr(inst, key, value)
        inst.__cause__ = cause
        inst.__traceback__ = tb
        inst.__context__ = context
        inst.__suppress_context__ = suppress_context
        if notes is not None:
            inst.__notes__ = notes
        return inst

    pickling_support.unpickle_exception_with_attrs = unpickle_exception_with_attrs


def ensure_tblib_pickling_support() -> None:
    """Make sure ``tblib`` exposes the helpers fal's isolate expects."""
    try:
        from tblib import pickling_support  # type: ignore
    except Exception:
        return

    if getattr(pickling_support, _PATCH_FLAG, False):
        return

    if not hasattr(pickling_support, "unpickle_exception_with_attrs"):
        _install_unpickle_shim(pickling_support)

    install = getattr(pickling_support, "install", None)
    if callable(install):
        install()

    setattr(pickling_support, _PATCH_FLAG, True)


# Apply patch eagerly on import so datastore modules only need to import.
ensure_tblib_pickling_support()
