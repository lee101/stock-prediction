from __future__ import annotations

import differentiable_loss_utils as dlu


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
