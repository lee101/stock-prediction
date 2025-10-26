from __future__ import annotations

import importlib
import inspect
import sys


def test_toto_wrapper_import_without_torch(monkeypatch):
    original_torch = sys.modules.get("torch")
    original_numpy = sys.modules.get("numpy")

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "src.models.toto_wrapper", raising=False)

    original_import_module = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    module = importlib.import_module("src.models.toto_wrapper")

    assert module.torch is None

    amp_param = inspect.signature(module.TotoPipeline.__init__).parameters["amp_dtype"]
    assert amp_param.default is None

    restore_kwargs = {}
    if original_torch is not None:
        restore_kwargs["torch_module"] = original_torch
    if original_numpy is not None:
        restore_kwargs["numpy_module"] = original_numpy

    if restore_kwargs:
        # Restore the original heavy modules so later tests are unaffected.
        module.setup_toto_wrapper_imports(**restore_kwargs)
