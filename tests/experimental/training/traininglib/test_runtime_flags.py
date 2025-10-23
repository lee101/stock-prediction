from typing import List

import pytest
import torch
import torch.nn.functional as F

from traininglib import runtime_flags


class _DummyContext:
    def __init__(self, calls: List[dict], should_raise: bool, **kwargs):
        self._calls = calls
        self._kwargs = kwargs
        self._should_raise = should_raise

    def __enter__(self):
        self._calls.append(self._kwargs)
        if self._should_raise:
            raise RuntimeError("failed to set fast kernels")
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_enable_fast_kernels_cpu_only(monkeypatch):
    calls: List[dict] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends.cuda,
        "sdp_kernel",
        lambda **kwargs: _DummyContext(calls, should_raise=False, **kwargs),
    )

    with runtime_flags.enable_fast_kernels():
        pass

    assert calls == []


def test_enable_fast_kernels_prefers_mem_efficient_without_flash(monkeypatch):
    calls: List[dict] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(
        torch.backends.cuda,
        "is_flash_attention_available",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "sdp_kernel",
        lambda **kwargs: _DummyContext(calls, should_raise=False, **kwargs),
    )

    with runtime_flags.enable_fast_kernels():
        pass

    assert len(calls) == 1
    assert calls[0]["enable_flash"] is False
    assert calls[0]["enable_mem_efficient"] is True
    assert calls[0]["enable_math"] is True


def test_enable_fast_kernels_falls_back_on_failure(monkeypatch):
    calls: List[dict] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        torch.backends.cuda,
        "is_flash_attention_available",
        lambda: True,
        raising=False,
    )

    def _factory(**kwargs):
        should_raise = kwargs["enable_flash"] or kwargs["enable_mem_efficient"]
        return _DummyContext(calls, should_raise=should_raise, **kwargs)

    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", _factory)

    with runtime_flags.enable_fast_kernels():
        pass

    assert len(calls) == 2
    assert calls[0]["enable_flash"] is True
    assert calls[0]["enable_mem_efficient"] is True
    assert calls[1] == {
        "enable_flash": False,
        "enable_math": True,
        "enable_mem_efficient": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for flash-attn patch")
def test_sdpa_patch_uses_flash_attn(monkeypatch):

    calls: List[torch.Tensor] = []

    def fake_flash(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        **_: object,
    ) -> torch.Tensor:
        calls.append(q)
        return q.clone()

    monkeypatch.setattr(runtime_flags, "_flash_attn_func", fake_flash)
    monkeypatch.setattr(runtime_flags, "_sage_attn", None)

    q = torch.randn(2, 8, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 8, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 8, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)

    with runtime_flags._sdpa_kernel_patch():
        out = F.scaled_dot_product_attention(q, k, v)
        (out.sum()).backward()

    assert len(calls) == 1
    assert out.shape == q.shape
    assert q.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for sageattention patch")
def test_sdpa_patch_skips_sage_when_dropout(monkeypatch):

    monkeypatch.setattr(runtime_flags, "_flash_attn_func", None)

    invoked = {"sage": False}

    def fake_sage(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        sm_scale: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        invoked["sage"] = True
        return torch.zeros_like(q)

    monkeypatch.setattr(runtime_flags, "_sage_attn", fake_sage)

    q = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float16)
    k = q.clone()
    v = q.clone()

    torch.manual_seed(0)
    reference = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

    with runtime_flags._sdpa_kernel_patch():
        torch.manual_seed(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

    assert not invoked["sage"]
    assert torch.allclose(out, reference, atol=1e-4, rtol=1e-3)
