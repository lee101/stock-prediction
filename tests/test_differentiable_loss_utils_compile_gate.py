from __future__ import annotations

import differentiable_loss_utils as dlu
import torch


def test_blackwell_disables_objective_compile(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_FORCE_COMPILE", raising=False)
    monkeypatch.delenv("TORCH_NO_COMPILE", raising=False)
    monkeypatch.setattr(dlu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dlu.torch.cuda, "get_device_capability", lambda _index=0: (12, 0))
    assert dlu._compile_enabled() is False


def test_force_compile_overrides_blackwell_guard(monkeypatch) -> None:
    monkeypatch.setenv("TORCH_FORCE_COMPILE", "1")
    monkeypatch.delenv("TORCH_NO_COMPILE", raising=False)
    monkeypatch.setattr(dlu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dlu.torch.cuda, "get_device_capability", lambda _index=0: (12, 0))
    assert dlu._compile_enabled() is True


def test_compile_disabled_without_cuda(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_FORCE_COMPILE", raising=False)
    monkeypatch.delenv("TORCH_NO_COMPILE", raising=False)
    monkeypatch.setattr(dlu.torch.cuda, "is_available", lambda: False)
    assert dlu._compile_enabled() is False


def test_maybe_compile_falls_back_to_eager_after_compile_failure(monkeypatch) -> None:
    monkeypatch.setattr(dlu, "_COMPILE_ENABLED", True)

    def eager_fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    call_count = {"compiled": 0}

    def fake_compile(fn, **_kwargs):
        def _compiled(*args, **kwargs):
            call_count["compiled"] += 1
            raise UnboundLocalError("synthetic dynamo failure")

        return _compiled

    monkeypatch.setattr(dlu.torch, "compile", fake_compile)
    wrapped = dlu._maybe_compile(eager_fn, mode="reduce-overhead")
    x = torch.tensor([1.0])

    first = wrapped(x)
    second = wrapped(x)

    assert torch.equal(first, torch.tensor([2.0]))
    assert torch.equal(second, torch.tensor([2.0]))
    assert call_count["compiled"] == 1
