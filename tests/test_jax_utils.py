from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

from binanceneural import jax_utils


def test_configure_default_jax_platforms_prefers_cpu_for_hidden_cuda(monkeypatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    jax_utils.clear_jax_device_cache()

    selected = jax_utils.configure_default_jax_platforms()

    assert selected == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_configure_default_jax_platforms_preserves_explicit_choice(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    jax_utils.clear_jax_device_cache()

    selected = jax_utils.configure_default_jax_platforms()

    assert selected is None
    assert os.environ["JAX_PLATFORMS"] == "cuda"


def test_configure_default_jax_platforms_updates_imported_jax_config(monkeypatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    jax_utils.clear_jax_device_cache()

    updates: list[tuple[str, str]] = []
    fake_jax = SimpleNamespace(
        config=SimpleNamespace(update=lambda key, value: updates.append((key, value)))
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    selected = jax_utils.configure_default_jax_platforms()

    assert selected == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert updates == [("jax_platforms", "cpu")]


def test_as_jax_array_uses_cpu_device_when_cuda_hidden(monkeypatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    jax_utils.clear_jax_device_cache()

    array = jax_utils.as_jax_array([1.0, 2.0], dtype=np.float32)

    assert np.asarray(array).tolist() == [1.0, 2.0]
    assert next(iter(array.devices())).platform == "cpu"


def test_as_jax_array_falls_back_to_cpu_on_cuda_backend_init_error(monkeypatch) -> None:
    import jax
    import jax.numpy as jnp

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    jax_utils.clear_jax_device_cache()
    monkeypatch.setattr(jax_utils, "preferred_jax_device", lambda: None)

    captured: dict[str, object] = {}

    def _fail_asarray(_value):
        raise RuntimeError(
            "Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA"
        )

    def _fake_devices(platform: str):
        assert platform == "cpu"
        return ["cpu-device"]

    def _fake_device_put(value, *, device=None):
        captured["device"] = device
        return np.asarray(value)

    monkeypatch.setattr(jnp, "asarray", _fail_asarray)
    monkeypatch.setattr(jax, "devices", _fake_devices)
    monkeypatch.setattr(jax, "device_put", _fake_device_put)

    array = jax_utils.as_jax_array([1.0, 2.0], dtype=np.float32)

    assert captured["device"] == "cpu-device"
    assert np.asarray(array).tolist() == [1.0, 2.0]
