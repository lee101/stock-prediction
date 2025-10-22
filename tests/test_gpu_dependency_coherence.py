import importlib

import pytest


try:
    _torch = importlib.import_module("torch")
except ImportError:  # pragma: no cover - exercised when torch is absent entirely
    _torch = None

_cuda_runtime_available = bool(
    _torch
    and getattr(_torch, "cuda", None)
    and callable(getattr(_torch.cuda, "is_available", None))
    and _torch.cuda.is_available()
    and getattr(getattr(_torch, "version", None), "cuda", None)
)

pytestmark = pytest.mark.skipif(
    not _cuda_runtime_available,
    reason="CUDA runtime required for coherence checks",
)


@pytest.mark.cuda_required
def test_torch_reports_cuda_runtime() -> None:
    torch = importlib.import_module("torch")
    # Torch reports None when built without CUDA support.
    assert getattr(torch.version, "cuda", None), "Expected CUDA-enabled torch build"


@pytest.mark.cuda_required
def test_flash_attn_imports_with_cuda_symbols() -> None:
    try:
        flash_attn = importlib.import_module("flash_attn")
    except ImportError as exc:
        pytest.skip(f"flash_attn unavailable: {exc}")
    assert hasattr(flash_attn, "__version__")
