from __future__ import annotations

import types

from src.torch_backend import configure_tf32_backends, maybe_set_float32_precision


class _BackendNS(types.SimpleNamespace):
    pass


def test_configure_tf32_prefers_new_api(monkeypatch):
    matmul = types.SimpleNamespace(allow_tf32=False, fp32_precision="ieee")
    conv = types.SimpleNamespace(fp32_precision="ieee")
    cuda = types.SimpleNamespace(matmul=matmul)
    cudnn = types.SimpleNamespace(allow_tf32=False, conv=conv)
    torch_module = types.SimpleNamespace(backends=_BackendNS(cuda=cuda, cudnn=cudnn))

    state = configure_tf32_backends(torch_module)

    assert state == {"new_api": False, "legacy_api": True}
    assert matmul.allow_tf32 is True
    assert cudnn.allow_tf32 is True
    assert matmul.fp32_precision == "ieee"
    assert conv.fp32_precision == "ieee"


def test_configure_tf32_uses_legacy_when_new_missing():
    matmul = types.SimpleNamespace(allow_tf32=False)
    cudnn = types.SimpleNamespace(allow_tf32=False)
    cuda = types.SimpleNamespace(matmul=matmul)
    backends = _BackendNS(cuda=cuda, cudnn=cudnn)
    torch_module = types.SimpleNamespace(backends=backends)

    state = configure_tf32_backends(torch_module)

    assert state == {"new_api": False, "legacy_api": True}
    assert matmul.allow_tf32 is True
    assert cudnn.allow_tf32 is True


def test_configure_tf32_uses_new_api_when_legacy_missing():
    matmul = types.SimpleNamespace(fp32_precision="ieee")
    conv = types.SimpleNamespace(fp32_precision="ieee")
    cuda = types.SimpleNamespace(matmul=matmul)
    cudnn = types.SimpleNamespace(conv=conv)
    torch_module = types.SimpleNamespace(backends=_BackendNS(cuda=cuda, cudnn=cudnn))

    state = configure_tf32_backends(torch_module)

    assert state == {"new_api": True, "legacy_api": False}
    assert matmul.fp32_precision == "tf32"
    assert conv.fp32_precision == "tf32"


def test_configure_tf32_can_disable_legacy_api():
    matmul = types.SimpleNamespace(allow_tf32=True)
    cudnn = types.SimpleNamespace(allow_tf32=True)
    cuda = types.SimpleNamespace(matmul=matmul)
    backends = _BackendNS(cuda=cuda, cudnn=cudnn)
    torch_module = types.SimpleNamespace(backends=backends)

    state = configure_tf32_backends(torch_module, enabled=False)

    assert state == {"new_api": False, "legacy_api": True}
    assert matmul.allow_tf32 is False
    assert cudnn.allow_tf32 is False


def test_configure_tf32_can_disable_new_api():
    matmul = types.SimpleNamespace(fp32_precision="tf32")
    conv = types.SimpleNamespace(fp32_precision="tf32")
    cuda = types.SimpleNamespace(matmul=matmul)
    cudnn = types.SimpleNamespace(conv=conv)
    torch_module = types.SimpleNamespace(backends=_BackendNS(cuda=cuda, cudnn=cudnn))

    state = configure_tf32_backends(torch_module, enabled=False)

    assert state == {"new_api": True, "legacy_api": False}
    assert matmul.fp32_precision == "ieee"
    assert conv.fp32_precision == "ieee"


def test_maybe_set_float32_precision_uses_legacy_setter_when_needed():
    calls: list[str] = []
    cuda = types.SimpleNamespace(is_available=lambda: True)
    backends = _BackendNS(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()))
    torch_module = types.SimpleNamespace(
        backends=backends,
        cuda=cuda,
        set_float32_matmul_precision=lambda mode: calls.append(mode),
    )

    maybe_set_float32_precision(torch_module, "high")

    assert calls == ["high"]


def test_maybe_set_float32_precision_skips_when_new_api_available():
    calls: list[str] = []
    cuda = types.SimpleNamespace(is_available=lambda: True)
    backends = _BackendNS(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(fp32_precision="ieee")))
    torch_module = types.SimpleNamespace(
        backends=backends,
        cuda=cuda,
        set_float32_matmul_precision=lambda mode: calls.append(mode),
    )

    maybe_set_float32_precision(torch_module, "high")

    assert calls == []


def test_maybe_set_float32_precision_skips_when_cuda_unavailable():
    calls: list[str] = []
    cuda = types.SimpleNamespace(is_available=lambda: False)
    backends = _BackendNS(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()))
    torch_module = types.SimpleNamespace(
        backends=backends,
        cuda=cuda,
        set_float32_matmul_precision=lambda mode: calls.append(mode),
    )

    maybe_set_float32_precision(torch_module, "high")

    assert calls == []
