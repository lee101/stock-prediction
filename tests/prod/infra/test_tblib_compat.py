from __future__ import annotations

import importlib
import sys
import types


def test_ensure_tblib_pickling_support_injects_shim() -> None:
    original_modules = {
        "tblib": sys.modules.pop("tblib", None),
        "tblib.pickling_support": sys.modules.pop("tblib.pickling_support", None),
        "src.tblib_compat": sys.modules.pop("src.tblib_compat", None),
    }

    try:
        pickling_support = types.ModuleType("tblib.pickling_support")
        install_calls = {"count": 0}

        def install() -> None:
            install_calls["count"] += 1

        pickling_support.install = install  # type: ignore[attr-defined]

        tblib_module = types.ModuleType("tblib")
        tblib_module.pickling_support = pickling_support  # type: ignore[attr-defined]

        sys.modules["tblib"] = tblib_module
        sys.modules["tblib.pickling_support"] = pickling_support

        compat = importlib.import_module("src.tblib_compat")
        importlib.reload(compat)

        DummyError = type("DummyError", (Exception,), {})
        exc = pickling_support.unpickle_exception_with_attrs(  # type: ignore[attr-defined]
            DummyError,
            {"detail": "boom"},
            None,
            None,
            None,
            False,
            ("note",),
        )

        assert isinstance(exc, DummyError)
        assert exc.detail == "boom"
        assert getattr(exc, "__notes__", ()) == ("note",)
        assert install_calls["count"] == 1
        assert getattr(pickling_support, "_fal_tblib_patch_applied", False)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if original_modules["src.tblib_compat"] is not None:
            importlib.reload(original_modules["src.tblib_compat"])
