from __future__ import annotations

import contextlib
import math
import warnings
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
except Exception:  # pragma: no cover - optional dependency
    _flash_attn_func = None  # type: ignore[assignment]

try:
    import sageattention

    _sage_attn = sageattention.sageattn
except Exception:  # pragma: no cover - optional dependency
    _sage_attn = None  # type: ignore[assignment]


_FLASH_ATTENTION_DTYPES = {torch.float16, torch.bfloat16}
_SAGE_ATTENTION_DTYPES = {torch.float16, torch.bfloat16}


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _bool_safely(fn: Callable[[], bool]) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


def _flash_sdp_available() -> bool:
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.backends.cuda, "is_flash_attention_available"):
        return _bool_safely(torch.backends.cuda.is_flash_attention_available)

    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    # Flash attention kernels land on Ampere (SM80) or newer.
    return major >= 8


def _mem_efficient_sdp_preferred() -> bool:
    if not torch.cuda.is_available():
        return False

    # Triton-based mem-efficient kernels have been stable since Volta (SM70).
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 7


def _sdpa_preconditions_met(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
) -> bool:
    if attn_mask is not None:
        # Flash/Sage attention only support causal masking currently.
        return False
    if q.device.type != "cuda":
        return False
    if q.dtype not in _FLASH_ATTENTION_DTYPES and (
        _sage_attn is None or q.dtype not in _SAGE_ATTENTION_DTYPES
    ):
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.ndim != 4:
        return False
    if q.size(-1) > 256:
        # FlashAttention v2 kernels currently cap head_dim at 256.
        return False
    if dropout_p > 0.0 and _flash_attn_func is None:
        # SageAttention does not provide a dropout-capable kernel.
        return False
    return True


def _invoke_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
) -> Optional[torch.Tensor]:
    if _flash_attn_func is None or q.dtype not in _FLASH_ATTENTION_DTYPES:
        return None

    try:
        scale = 1.0 / math.sqrt(q.size(-1))
        qkv = (q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
        out = _flash_attn_func(
            qkv[0],
            qkv[1],
            qkv[2],
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
        )
        return out.transpose(1, 2)
    except Exception:
        return None


def _invoke_sage_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> Optional[torch.Tensor]:
    if _sage_attn is None or q.dtype not in _SAGE_ATTENTION_DTYPES:
        return None
    try:
        scale = 1.0 / math.sqrt(q.size(-1))
        return _sage_attn(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=scale,
        )
    except Exception:
        return None


@contextlib.contextmanager
def _sdpa_kernel_patch():
    """
    Temporarily monkey patch PyTorch SDPA to run flash-attn / SageAttention fast kernels.
    """
    if not torch.cuda.is_available():
        yield False
        return

    if _flash_attn_func is None and _sage_attn is None:
        yield False
        return

    original_sdpa = F.scaled_dot_product_attention

    def _patched_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not _sdpa_preconditions_met(q, k, v, attn_mask, dropout_p):
            return original_sdpa(q, k, v, attn_mask, dropout_p, is_causal)

        flash_out = _invoke_flash_attn(q, k, v, dropout_p, is_causal)
        if flash_out is not None:
            return flash_out

        sage_out = _invoke_sage_attn(q, k, v, is_causal)
        if sage_out is not None:
            return sage_out

        return original_sdpa(q, k, v, attn_mask, dropout_p, is_causal)

    F.scaled_dot_product_attention = _patched_sdpa  # type: ignore[assignment]
    try:
        yield True
    finally:
        F.scaled_dot_product_attention = original_sdpa  # type: ignore[assignment]


@contextlib.contextmanager
def enable_fast_kernels():
    """
    Context manager that enables useful CUDA fast paths (TF32 + Flash attention) when available.
    """
    # TF32 on Ampere/Hopper improves throughput without hurting accuracy much.
    # These tweaks must be guarded because CUDA initialisation might fail on CPU-only nodes.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception as exc:
        warnings.warn(f"Unable to enable TF32 fast matmul: {exc}")

    if not torch.cuda.is_available():
        yield
        return

    sdpa_patch_ctx: contextlib.AbstractContextManager = _sdpa_kernel_patch()

    with sdpa_patch_ctx:
        flash_available = _flash_sdp_available()
        mem_efficient_available = _mem_efficient_sdp_preferred()

        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=flash_available,
                enable_math=True,
                enable_mem_efficient=mem_efficient_available,
            ):
                yield
                return
        except Exception as exc:
            warnings.warn(f"Falling back to math-only SDP kernels: {exc}")

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield
