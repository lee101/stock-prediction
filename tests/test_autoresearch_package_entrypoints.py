from __future__ import annotations

import runpy
import types

import pytest


@pytest.mark.parametrize(
    ("package_name", "exit_code"),
    [
        ("autoresearch_stock", 11),
        ("autoresearch_binance", 22),
        ("autoresearch_crypto", 33),
    ],
)
def test_autoresearch_package_entrypoint_delegates_to_train_main(
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
    exit_code: int,
) -> None:
    fake_train = types.ModuleType(f"{package_name}.train")
    calls: list[str] = []

    def _fake_main() -> int:
        calls.append(package_name)
        return exit_code

    fake_train.main = _fake_main
    monkeypatch.setitem(__import__("sys").modules, f"{package_name}.train", fake_train)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(package_name, run_name="__main__")

    assert calls == [package_name]
    assert exc_info.value.code == exit_code
