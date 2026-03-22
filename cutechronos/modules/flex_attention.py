"""Alternative attention backends for CuteChronos2.

Provides drop-in replacements for ``unscaled_attention`` using:
- ``sdpa_unscaled_attention``: PyTorch SDPA with ``scale=1.0``
- ``flex_unscaled_attention``: PyTorch FlexAttention with ``scale=1.0``

Both produce the same result as the reference:
    softmax(Q @ K^T + mask) @ V   (no 1/sqrt(d_k) scaling)

SDPA is the recommended backend for Chronos2 because:
- It natively supports additive masks (the mask pattern Chronos2 uses)
- It auto-selects the fastest kernel (FlashAttention2, cuDNN, math fallback)
- It works with torch.compile without special handling

FlexAttention is provided for the mask-free case and for future use with
custom score modifications. For masked attention, FlexAttention's score_mod
requires vmap-compatible tensor indexing which is fragile with general
additive mask tensors; use SDPA instead.
"""

from __future__ import annotations

import time as _time
from typing import Callable

import torch
import torch.nn.functional as F

from cutechronos.modules._fallbacks import unscaled_attention as _eager_fallback

# Try importing FlexAttention at module level (avoid per-call import overhead)
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    _has_flex_attention = True
except ImportError:
    _has_flex_attention = False


def sdpa_unscaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unscaled dot-product attention via PyTorch SDPA.

    Uses ``F.scaled_dot_product_attention`` with ``scale=1.0`` to disable
    the default 1/sqrt(d_k) scaling, matching Chronos2 semantics.

    Args:
        q: (B, H, S, D) query tensor
        k: (B, H, S, D) key tensor
        v: (B, H, S, D) value tensor
        mask: additive attention mask, broadcastable to (B, H, S, S).
              Values are 0.0 (attend) or -inf (masked). Can be None.

    Returns:
        (B, H, S, D) attention output
    """
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)


def flex_unscaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unscaled dot-product attention via PyTorch FlexAttention.

    Uses ``flex_attention`` with ``scale=1.0``. For masked attention, falls
    back to SDPA since FlexAttention's score_mod does not support general
    additive mask tensor indexing under vmap tracing.

    For the mask-free case, uses FlexAttention directly (faster than SDPA
    when compiled since it avoids mask tensor allocation).

    Args:
        q: (B, H, S, D) query tensor
        k: (B, H, S, D) key tensor
        v: (B, H, S, D) value tensor
        mask: additive attention mask, broadcastable to (B, H, S, S), or None.

    Returns:
        (B, H, S, D) attention output
    """
    if mask is not None:
        # FlexAttention score_mod with tensor lookups is fragile under
        # torch.compile vmap tracing. Fall back to SDPA which handles
        # additive masks natively and efficiently.
        return sdpa_unscaled_attention(q, k, v, mask)

    if _has_flex_attention:
        return _flex_attention(q, k, v, scale=1.0)
    return sdpa_unscaled_attention(q, k, v)


def eager_unscaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference unscaled dot-product attention (pure PyTorch, no fused kernel).

    Delegates to ``cutechronos.modules._fallbacks.unscaled_attention``.

    Args:
        q: (B, H, S, D) query tensor
        k: (B, H, S, D) key tensor
        v: (B, H, S, D) value tensor
        mask: additive attention mask, broadcastable to (B, H, S, S), or None.

    Returns:
        (B, H, S, D) attention output
    """
    return _eager_fallback(q, k, v, mask)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Callable] = {
    "sdpa": sdpa_unscaled_attention,
    "flex": flex_unscaled_attention,
    "eager": eager_unscaled_attention,
}

# Try to register Triton backend if available
try:
    from cutechronos.triton_kernels.attention import unscaled_attention as _triton_attn
    _BACKENDS["triton"] = _triton_attn
except (ImportError, ModuleNotFoundError):
    pass


def get_attention_backend(name: str) -> Callable:
    """Return an attention function by name.

    Args:
        name: one of "sdpa", "flex", "triton", "eager"

    Returns:
        Callable with signature (q, k, v, mask=None) -> Tensor

    Raises:
        KeyError: if name is not a registered backend
    """
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise KeyError(f"Unknown attention backend {name!r}. Available: {available}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    """Return names of all available attention backends."""
    return sorted(_BACKENDS.keys())


def benchmark_backends(
    batch_size: int = 4,
    num_heads: int = 12,
    seq_len: int = 130,
    d_kv: int = 64,
    *,
    with_mask: bool = True,
    warmup: int = 10,
    repeats: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """Benchmark all available attention backends and return timing in ms.

    Args:
        batch_size: batch dimension
        num_heads: number of attention heads
        seq_len: sequence length
        d_kv: per-head dimension
        with_mask: whether to include an additive mask
        warmup: warmup iterations
        repeats: timed iterations
        device: device to benchmark on
        dtype: tensor dtype

    Returns:
        Dict mapping backend name to mean time in milliseconds.
    """
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, d_kv, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, d_kv, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, d_kv, device=device, dtype=dtype)

    mask = None
    if with_mask:
        mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=dtype)
        # Simulate some padding: last 10% of tokens masked
        pad_start = max(1, int(seq_len * 0.9))
        mask[:, :, :, pad_start:] = float("-inf")

    results: dict[str, float] = {}

    for name, fn in _BACKENDS.items():
        # Warmup
        for _ in range(warmup):
            fn(q, k, v, mask)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed
        t0 = _time.perf_counter()
        for _ in range(repeats):
            fn(q, k, v, mask)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = _time.perf_counter() - t0

        results[name] = (elapsed / repeats) * 1000.0  # ms

    return results


def get_best_attention_backend(
    *,
    with_mask: bool = True,
    device: str = "cuda",
    seq_len: int = 130,
) -> tuple[str, Callable]:
    """Benchmark backends and return the fastest (name, function) pair.

    Runs a quick benchmark with the given parameters and returns the backend
    with the lowest mean latency.

    Args:
        with_mask: whether to test with an additive mask
        device: device to benchmark on
        seq_len: sequence length for the benchmark

    Returns:
        (backend_name, attention_function) tuple
    """
    timings = benchmark_backends(
        with_mask=with_mask,
        device=device,
        seq_len=seq_len,
        warmup=5,
        repeats=20,
    )

    best_name = min(timings, key=timings.get)  # type: ignore[arg-type]
    return best_name, _BACKENDS[best_name]
