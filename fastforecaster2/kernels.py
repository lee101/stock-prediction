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
    env = os.getenv("FASTFORECASTER_BUILD_EXT", "0").strip().lower()
    return env in {"1", "true", "yes", "on"}


def load_fastforecaster2_extension(*, build_extension: bool, verbose: bool = False) -> Any | None:
    """Compile and load optional C++/CUDA kernels used by FastForecaster2.

    The extension is optional by design. If compilation fails or CUDA toolchain is
    unavailable, callers should continue with pure-PyTorch kernels.
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
        cpp_source = root / "fast_ops.cpp"
        cuda_source = root / "fast_ops_cuda.cu"

        if not cpp_source.exists():
            return None

        has_cuda = bool(torch.cuda.is_available() and torch.version.cuda and CUDA_HOME is not None)
        sources = [str(cpp_source)]
        if has_cuda and cuda_source.exists():
            sources.append(str(cuda_source))

        build_dir = root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            _EXTENSION = load_extension_module(
                name="fastforecaster2_ext",
                sources=sources,
                extra_cflags=_extra_cflags(),
                extra_ldflags=_extra_ldflags(),
                extra_cuda_cflags=["-O3", "--use_fast_math"] if has_cuda else [],
                with_cuda=has_cuda,
                build_directory=str(build_dir),
                verbose=verbose,
            )
            setattr(_EXTENSION, "_fastforecaster2_has_cuda", has_cuda)
        except Exception as exc:
            print(f"[fastforecaster2] Optional extension build failed, falling back to PyTorch kernels: {exc}")
            _EXTENSION = None

        return _EXTENSION


def _mae_forward(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    use_cpp: bool,
    build_extension: bool,
) -> torch.Tensor:
    if use_cpp:
        extension = load_fastforecaster2_extension(build_extension=build_extension)
        if extension is not None and hasattr(extension, "fast_mae"):
            has_cuda = bool(getattr(extension, "_fastforecaster2_has_cuda", False))
            if prediction.is_cuda and not has_cuda:
                return torch.mean(torch.abs(prediction - target))
            try:
                return extension.fast_mae(prediction, target)
            except RuntimeError as exc:
                if "without CUDA support" in str(exc):
                    return torch.mean(torch.abs(prediction - target))
                raise
    return torch.mean(torch.abs(prediction - target))


def _weighted_mae_forward(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    use_cpp: bool,
    build_extension: bool,
) -> torch.Tensor:
    if use_cpp:
        extension = load_fastforecaster2_extension(build_extension=build_extension)
        if extension is not None and hasattr(extension, "fast_weighted_mae"):
            has_cuda = bool(getattr(extension, "_fastforecaster2_has_cuda", False))
            if prediction.is_cuda and not has_cuda:
                return _weighted_mae_fallback(prediction, target, weights)
            try:
                return extension.fast_weighted_mae(prediction, target, weights)
            except RuntimeError as exc:
                if "without CUDA support" in str(exc):
                    return _weighted_mae_fallback(prediction, target, weights)
                raise

    return _weighted_mae_fallback(prediction, target, weights)


class _FastMaeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, use_cpp: bool, build_extension: bool):  # type: ignore[override]
        ctx.save_for_backward(prediction, target)
        return _mae_forward(
            prediction,
            target,
            use_cpp=bool(use_cpp),
            build_extension=bool(build_extension),
        )

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        prediction, target = ctx.saved_tensors
        scale = grad_output.to(device=prediction.device, dtype=prediction.dtype) / float(max(1, prediction.numel()))
        grad_pred = torch.sign(prediction - target) * scale
        grad_target = -grad_pred
        return grad_pred, grad_target, None, None


class _FastWeightedMaeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, weights, use_cpp: bool, build_extension: bool):  # type: ignore[override]
        ctx.save_for_backward(prediction, target, weights)
        return _weighted_mae_forward(
            prediction,
            target,
            weights,
            use_cpp=bool(use_cpp),
            build_extension=bool(build_extension),
        )

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        prediction, target, weights = ctx.saved_tensors
        grad_scale = grad_output.to(device=prediction.device, dtype=prediction.dtype) / float(max(1, prediction.numel()))

        w = weights.to(device=prediction.device, dtype=prediction.dtype)
        view_shape = [1] * prediction.ndim
        view_shape[-1] = int(w.shape[0])
        w = w.view(*view_shape)

        grad_pred = torch.sign(prediction - target) * w * grad_scale
        grad_target = -grad_pred

        grad_weights = None
        if ctx.needs_input_grad[2]:
            abs_err = torch.abs(prediction - target)
            reduce_dims = tuple(range(prediction.ndim - 1))
            if reduce_dims:
                grad_weights = abs_err.sum(dim=reduce_dims)
            else:
                grad_weights = abs_err
            grad_weights = grad_weights * grad_scale
            grad_weights = grad_weights.to(device=weights.device, dtype=weights.dtype)

        return grad_pred, grad_target, grad_weights, None, None


def mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    use_cpp: bool,
    build_extension: bool,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    # Preserve native PyTorch autograd behavior unless C++ kernels are explicitly requested.
    if not use_cpp:
        return torch.mean(torch.abs(prediction - target))
    if prediction.requires_grad or target.requires_grad:
        return _FastMaeAutograd.apply(prediction, target, use_cpp, build_extension)
    return _mae_forward(prediction, target, use_cpp=use_cpp, build_extension=build_extension)


def weighted_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    use_cpp: bool,
    build_extension: bool,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim < 1:
        raise ValueError("prediction must have at least 1 dimension")
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {tuple(weights.shape)}")
    if prediction.shape[-1] != int(weights.shape[0]):
        raise ValueError(
            f"weights length must match prediction.shape[-1]: {int(weights.shape[0])} vs {prediction.shape[-1]}"
        )
    # Preserve native PyTorch autograd behavior unless C++ kernels are explicitly requested.
    if not use_cpp:
        return _weighted_mae_fallback(prediction, target, weights)
    if prediction.requires_grad or target.requires_grad or weights.requires_grad:
        return _FastWeightedMaeAutograd.apply(prediction, target, weights, use_cpp, build_extension)
    return _weighted_mae_forward(
        prediction,
        target,
        weights,
        use_cpp=use_cpp,
        build_extension=build_extension,
    )


def _weighted_mae_fallback(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(device=prediction.device, dtype=prediction.dtype)
    view_shape = [1] * prediction.ndim
    view_shape[-1] = int(w.shape[0])
    w = w.view(*view_shape)
    return torch.mean(torch.abs(prediction - target) * w)
