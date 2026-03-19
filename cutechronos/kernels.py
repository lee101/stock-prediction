from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME, load as load_extension_module

_LOCK = threading.Lock()
_EXTENSION: Any | None = None
_EXTENSION_ATTEMPTED = False


def _kernel_root() -> Path:
    return Path(__file__).resolve().parent / "cpp"


def _extra_cflags() -> list[str]:
    flags = ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"]
    if os.name != "nt":
        flags.append("-fopenmp")
    return flags


def _extra_ldflags() -> list[str]:
    if os.name == "nt":
        return []
    return ["-fopenmp"]


def _should_build_extension(build_extension: bool) -> bool:
    if build_extension:
        return True
    env = os.getenv("CUTECHRONOS_BUILD_EXT", "0").strip().lower()
    return env in {"1", "true", "yes", "on"}


def load_cutechronos_extension(
    *, build_extension: bool = True, verbose: bool = False
) -> Any | None:
    """Compile and load the cutechronos C++/CUDA preprocessing kernels.

    The extension is optional by design. If compilation fails or CUDA toolchain is
    unavailable, callers should continue with pure-PyTorch fallback.
    """

    global _EXTENSION
    global _EXTENSION_ATTEMPTED

    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ATTEMPTED and _EXTENSION is None:
        return None
    if not _should_build_extension(build_extension):
        return None

    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ATTEMPTED and _EXTENSION is None:
            return None

        _EXTENSION_ATTEMPTED = True
        root = _kernel_root()
        cpp_source = root / "preprocessing.cpp"
        cuda_source = root / "preprocessing.cu"

        if not cpp_source.exists():
            return None

        has_cuda = bool(
            torch.cuda.is_available()
            and torch.version.cuda
            and CUDA_HOME is not None
        )
        sources = [str(cpp_source)]
        if has_cuda and cuda_source.exists():
            sources.append(str(cuda_source))

        build_dir = root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            _EXTENSION = load_extension_module(
                name="cutechronos_preprocessing",
                sources=sources,
                extra_cflags=_extra_cflags(),
                extra_ldflags=_extra_ldflags(),
                extra_cuda_cflags=["-O3", "--use_fast_math"] if has_cuda else [],
                with_cuda=has_cuda,
                build_directory=str(build_dir),
                verbose=verbose,
            )
            setattr(_EXTENSION, "_cutechronos_has_cuda", has_cuda)
        except Exception as exc:
            print(
                f"[cutechronos] Extension build failed, falling back to PyTorch: {exc}"
            )
            _EXTENSION = None

        return _EXTENSION


def fused_preprocess(
    context: torch.Tensor,
    patch_size: int = 16,
    context_length: int = 512,
    use_arcsinh: bool = False,
    *,
    build_extension: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Chronos2 preprocessing: instance-norm + patch + time_encoding.

    Parameters
    ----------
    context : torch.Tensor
        Raw input of shape (B, L). May contain NaN values.
    patch_size : int
        Size of each patch (default 16).
    context_length : int
        Model's context length (default 512). Also used as time_encoding_scale.
    use_arcsinh : bool
        Apply arcsinh after normalization (default False).
    build_extension : bool
        Whether to attempt building the C++/CUDA extension.

    Returns
    -------
    patched_context : torch.Tensor
        Shape (B, P, 3*patch_size) with [time_enc, normalized, mask] concatenated.
    attention_mask : torch.Tensor
        Shape (B, P), 1.0 if any non-NaN in patch, 0.0 otherwise.
    loc : torch.Tensor
        Shape (B, 1), per-series mean.
    scale : torch.Tensor
        Shape (B, 1), per-series std.
    """
    ext = load_cutechronos_extension(build_extension=build_extension)
    if ext is not None:
        has_cuda = bool(getattr(ext, "_cutechronos_has_cuda", False))
        if context.is_cuda and not has_cuda:
            return _fallback_preprocess(context, patch_size, context_length, use_arcsinh)
        try:
            result = ext.fused_preprocess(context, patch_size, context_length, use_arcsinh)
            return result[0], result[1], result[2], result[3]
        except RuntimeError as exc:
            if "without CUDA support" in str(exc):
                return _fallback_preprocess(
                    context, patch_size, context_length, use_arcsinh
                )
            raise

    return _fallback_preprocess(context, patch_size, context_length, use_arcsinh)


def _fallback_preprocess(
    context: torch.Tensor,
    patch_size: int,
    context_length: int,
    use_arcsinh: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback matching the Chronos2 _prepare_patched_context logic."""
    ctx = context.float()
    B, L = ctx.shape

    # Truncate
    if L > context_length:
        ctx = ctx[:, -context_length:]
        L = context_length

    # NaN mask
    nan_mask = torch.isnan(ctx)
    valid_mask = nan_mask.logical_not().float()

    # Instance normalization
    ctx_zeroed = torch.where(nan_mask, torch.zeros_like(ctx), ctx)
    count = valid_mask.sum(dim=-1, keepdim=True)
    count_safe = torch.where(count > 0, count, torch.ones_like(count))

    loc = ctx_zeroed.sum(dim=-1, keepdim=True) / count_safe
    loc = torch.where(count > 0, loc, torch.zeros_like(loc))

    diff = torch.where(nan_mask, torch.zeros_like(ctx), ctx - loc)
    variance = (diff * diff).sum(dim=-1, keepdim=True) / count_safe
    scale = torch.sqrt(variance)
    scale = torch.where(count > 0, scale, torch.ones_like(scale))
    scale = torch.where(scale == 0.0, torch.full_like(scale, 1e-5), scale)

    normalized = (ctx - loc) / scale
    if use_arcsinh:
        normalized = torch.arcsinh(normalized)
    normalized = torch.where(nan_mask, torch.zeros_like(normalized), normalized)

    # Patching
    L_padded = L
    if L % patch_size != 0:
        pad_len = patch_size - (L % patch_size)
        L_padded = L + pad_len
        pad_zeros = torch.zeros(B, pad_len, device=ctx.device, dtype=ctx.dtype)
        normalized = torch.cat([pad_zeros, normalized], dim=1)
        pad_mask = torch.zeros(B, pad_len, device=ctx.device, dtype=ctx.dtype)
        valid_mask = torch.cat([pad_mask, valid_mask], dim=1)

    num_patches = L_padded // patch_size
    patched = normalized.unfold(1, patch_size, patch_size)  # (B, P, ps)
    patched_mask = valid_mask.unfold(1, patch_size, patch_size)
    patched = torch.where(patched_mask > 0, patched, torch.zeros_like(patched))

    attn_mask = (patched_mask.sum(dim=-1) > 0).float()

    # Time encoding
    final_len = num_patches * patch_size
    time_enc = torch.arange(-final_len, 0, device=ctx.device, dtype=torch.float32)
    time_enc = time_enc.reshape(1, num_patches, patch_size).expand(B, -1, -1) / float(
        context_length
    )

    # Concat
    patched_out = torch.cat([time_enc, patched, patched_mask], dim=2)

    return patched_out, attn_mask, loc, scale
