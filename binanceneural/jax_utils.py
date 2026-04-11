from __future__ import annotations

from functools import lru_cache
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import jax


def _is_cuda_backend_init_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "Unable to initialize backend 'cuda'" in text
        or "no supported devices found for platform CUDA" in text
    )


def configure_default_jax_platforms() -> str | None:
    """Prefer CPU JAX in explicitly headless environments.

    Some test and CI runs execute with the CUDA-enabled JAX plugin installed but
    no visible GPU devices. In that situation JAX may try the CUDA backend
    first and fail before falling back. If the caller has explicitly hidden CUDA
    devices, treat that as a request to run on CPU unless JAX_PLATFORMS was
    already set.
    """

    if os.environ.get("JAX_PLATFORMS"):
        return None
    hidden_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if hidden_cuda is not None and not hidden_cuda.strip():
        os.environ["JAX_PLATFORMS"] = "cpu"
        jax_module = sys.modules.get("jax")
        if jax_module is not None:
            config = getattr(jax_module, "config", None)
            if config is not None:
                config.update("jax_platforms", "cpu")
        return "cpu"
    return None


def headless_cpu_jax_requested() -> bool:
    if os.environ.get("CPU_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    hidden_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    return hidden_cuda is not None and not hidden_cuda.strip()


@lru_cache(maxsize=1)
def preferred_jax_device() -> "jax.Device | None":
    if not headless_cpu_jax_requested():
        return None
    configure_default_jax_platforms()
    import jax

    return jax.devices("cpu")[0]


def clear_jax_device_cache() -> None:
    preferred_jax_device.cache_clear()


def as_jax_array(value: Any, *, dtype: Any | None = None) -> "jax.Array":
    configure_default_jax_platforms()
    import jax
    import jax.numpy as jnp

    target_dtype = None if dtype is None else np.dtype(dtype)
    device = preferred_jax_device()

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()

    if isinstance(value, jax.Array):
        array = value if target_dtype is None or value.dtype == target_dtype else value.astype(target_dtype)
        if device is not None:
            return jax.device_put(array, device=device)
        return array

    numpy_value = np.asarray(value, dtype=target_dtype)
    if device is not None:
        return jax.device_put(numpy_value, device=device)
    try:
        return jnp.asarray(numpy_value)
    except RuntimeError as exc:
        if not _is_cuda_backend_init_error(exc):
            raise
        cpu_device = jax.devices("cpu")[0]
        return jax.device_put(numpy_value, device=cpu_device)
