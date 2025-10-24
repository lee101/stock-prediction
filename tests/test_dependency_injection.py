from __future__ import annotations

import types

import pytest

from src import dependency_injection as di


def _fake_module(name: str) -> object:
    return types.SimpleNamespace(__name__=name)


@pytest.fixture(autouse=True)
def _reset_and_stub_runtime_imports(monkeypatch: pytest.MonkeyPatch):
    di._reset_for_tests()
    monkeypatch.setattr(di, "setup_src_imports", lambda *args, **kwargs: None)
    yield
    di._reset_for_tests()


def test_setup_imports_injects_modules_and_notifies_observers():
    torch_mod = _fake_module("torch")
    numpy_mod = _fake_module("numpy")
    pandas_mod = _fake_module("pandas")
    extra_mod = _fake_module("scipy")

    observed = []

    def observer(module: object) -> None:
        observed.append(module)

    di.register_observer("torch", observer)

    di.setup_imports(torch=torch_mod, numpy=numpy_mod, pandas=pandas_mod, scipy=extra_mod)

    modules = di.injected_modules()
    assert modules["torch"] is torch_mod
    assert modules["numpy"] is numpy_mod
    assert modules["pandas"] is pandas_mod
    assert modules["scipy"] is extra_mod
    assert observed == [torch_mod]


def test_register_observer_immediately_receives_existing_module():
    torch_mod = _fake_module("torch-existing")
    di.setup_imports(torch=torch_mod)

    observed = []
    di.register_observer("torch", observed.append)

    assert observed == [torch_mod]


def test_resolve_torch_imports_and_notifies(monkeypatch: pytest.MonkeyPatch):
    imported = _fake_module("torch-imported")
    import_calls: list[str] = []

    def fake_import(name: str) -> object:
        import_calls.append(name)
        return imported

    monkeypatch.setattr(di, "import_module", fake_import)

    observed = []
    di.register_observer("torch", observed.append)

    result = di.resolve_torch()

    assert result is imported
    assert di.injected_modules()["torch"] is imported
    assert import_calls == ["torch"]
    assert observed == [imported]
