from __future__ import annotations

import types

import pytest

from tests import chronos_compile_test_utils as utils


def test_reset_torch_compile_state_prefers_modern_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    modern_calls: list[str] = []
    legacy_calls: list[str] = []

    monkeypatch.setattr(
        utils.torch,
        "compiler",
        types.SimpleNamespace(reset=lambda: modern_calls.append("modern")),
        raising=False,
    )
    monkeypatch.setattr(
        utils.torch,
        "_dynamo",
        types.SimpleNamespace(reset=lambda: legacy_calls.append("legacy")),
        raising=False,
    )

    utils.reset_torch_compile_state()

    assert modern_calls == ["modern"]
    assert legacy_calls == []


def test_reset_torch_compile_state_falls_back_to_legacy_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    modern_calls: list[str] = []
    legacy_calls: list[str] = []

    def _boom() -> None:
        modern_calls.append("modern")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        utils.torch,
        "compiler",
        types.SimpleNamespace(reset=_boom),
        raising=False,
    )
    monkeypatch.setattr(
        utils.torch,
        "_dynamo",
        types.SimpleNamespace(reset=lambda: legacy_calls.append("legacy")),
        raising=False,
    )

    utils.reset_torch_compile_state()

    assert modern_calls == ["modern"]
    assert legacy_calls == ["legacy"]


def test_clear_cuda_memory_if_available_is_noop_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_cache_calls: list[str] = []

    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils.torch.cuda, "empty_cache", lambda: empty_cache_calls.append("cache"))

    utils.clear_cuda_memory_if_available()

    assert empty_cache_calls == []


def test_is_transient_nonfinite_forecast_error_matches_expected_runtime_error() -> None:
    assert utils.is_transient_nonfinite_forecast_error(
        RuntimeError("Chronos2 produced non-finite forecasts for TEST.")
    )
    assert not utils.is_transient_nonfinite_forecast_error(RuntimeError("some other runtime failure"))
    assert not utils.is_transient_nonfinite_forecast_error(ValueError("Chronos2 produced non-finite forecasts"))
