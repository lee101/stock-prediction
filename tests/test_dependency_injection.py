from __future__ import annotations

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

from src import dependency_injection as deps


def _make_fake_torch() -> ModuleType:
    module = ModuleType("fake_torch")
    module.__version__ = "0.0.0"
    module.cuda = SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
        empty_cache=lambda: None,
        current_device=lambda: 0,
        get_device_name=lambda idx: f"fake:{idx}",
        amp=SimpleNamespace(autocast=lambda **_: nullcontext()),
    )
    module.backends = SimpleNamespace(
        cuda=SimpleNamespace(
            matmul=SimpleNamespace(allow_tf32=False),
            cudnn=SimpleNamespace(
                allow_tf32=False,
                benchmark=False,
            ),
            enable_flash_sdp=lambda *_: None,
            enable_math_sdp=lambda *_: None,
            enable_mem_efficient_sdp=lambda *_: None,
        )
    )
    module.set_float32_matmul_precision = lambda *_: None
    module.no_grad = lambda: nullcontext()
    module.inference_mode = lambda: nullcontext()
    module.tensor = lambda data, dtype=None: SimpleNamespace(data=data, dtype=dtype)
    module.zeros = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    module.full = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    module.ones_like = lambda tensor, **kwargs: SimpleNamespace(base=tensor, kwargs=kwargs)
    module.zeros_like = lambda tensor, **kwargs: SimpleNamespace(base=tensor, kwargs=kwargs)
    module.float32 = object()
    module.float16 = object()
    module.bfloat16 = object()
    module.bool = object()
    return module


def test_setup_imports_notifies_observers(monkeypatch):
    previous_modules = deps.injected_modules()
    previous_torch = previous_modules.get("torch") or deps.resolve_torch()

    events: list[ModuleType] = []
    deps.register_observer("torch", lambda module: events.append(module))

    assert events and events[-1] is previous_torch

    fake_torch = _make_fake_torch()
    deps.setup_imports(torch_module=fake_torch)

    assert events[-1] is fake_torch
    assert deps.injected_modules().get("torch") is fake_torch

    deps.setup_imports(torch_module=previous_torch)
    assert events[-1] is previous_torch
    assert deps.injected_modules().get("torch") is previous_torch
