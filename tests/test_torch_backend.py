from __future__ import annotations

import types

from src.torch_backend import configure_tf32_backends


class _BackendNS(types.SimpleNamespace):
    pass


def test_configure_tf32_prefers_new_api(monkeypatch):
    matmul = types.SimpleNamespace(fp32_precision="ieee")
    conv = types.SimpleNamespace(fp32_precision="ieee")
    cuda = types.SimpleNamespace(matmul=matmul)
    cudnn = types.SimpleNamespace(conv=conv)
    torch_module = types.SimpleNamespace(backends=_BackendNS(cuda=cuda, cudnn=cudnn))

    state = configure_tf32_backends(torch_module)

    assert state == {"new_api": True, "legacy_api": False}
    assert matmul.fp32_precision == "tf32"
    assert conv.fp32_precision == "tf32"


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
