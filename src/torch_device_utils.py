"""PyTorch device management utilities.

This module standardizes device selection and tensor creation across
the codebase. Previously duplicated across 30+ files.
"""

from typing import Any

import torch
from numpy.typing import NDArray

from src.backtest_env_utils import cpu_fallback_enabled, read_env_flag


def get_strategy_device(
    force_cpu: bool = False,
    env_flag: str | None = None,
) -> torch.device:
    """Get the appropriate compute device for strategy calculations.

    Args:
        force_cpu: If True, always return CPU device
        env_flag: Optional environment variable name to check for CPU forcing

    Returns:
        torch.device for computation (either 'cuda' or 'cpu')

    Examples:
        >>> device = get_strategy_device()  # Auto-detect
        >>> device = get_strategy_device(force_cpu=True)  # Force CPU
        >>> device = get_strategy_device(env_flag="MAXDIFF_FORCE_CPU")
    """
    if force_cpu:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Check global CPU fallback flag
    if cpu_fallback_enabled():
        return torch.device("cpu")

    # Check specific environment flag if provided
    if env_flag is not None:
        flag_value = read_env_flag([env_flag])
        if flag_value is True:
            return torch.device("cpu")

    return torch.device("cuda")


def to_tensor(
    data: NDArray[Any] | torch.Tensor | list[float] | float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert data to PyTorch tensor on the specified device.

    Args:
        data: Input data (numpy array, tensor, list, or scalar)
        dtype: Target tensor dtype (default: float32)
        device: Target device (default: auto-detect with get_strategy_device)

    Returns:
        PyTorch tensor on the specified device

    Examples:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> tensor = to_tensor(arr)
        >>> tensor = to_tensor(arr, device=torch.device("cpu"))
    """
    if device is None:
        device = get_strategy_device()

    return torch.as_tensor(data, dtype=dtype, device=device)


def require_cuda(
    feature: str,
    symbol: str | None = None,
    allow_fallback: bool = True,
) -> torch.device:
    """Require CUDA for a feature, with optional fallback.

    Args:
        feature: Name of feature requiring CUDA (for error messages)
        symbol: Optional symbol name (for error messages)
        allow_fallback: If True, fall back to CPU; if False, raise error

    Returns:
        torch.device (cuda if available, cpu if fallback allowed)

    Raises:
        RuntimeError: If CUDA not available and fallback not allowed

    Examples:
        >>> device = require_cuda("maxdiff", symbol="AAPL")
        >>> device = require_cuda("training", allow_fallback=False)  # May raise
    """
    if torch.cuda.is_available() and not cpu_fallback_enabled():
        return torch.device("cuda")

    if not allow_fallback:
        symbol_msg = f" for {symbol}" if symbol else ""
        raise RuntimeError(
            f"{feature}{symbol_msg} requires CUDA but CUDA is not available"
        )

    return torch.device("cpu")


def move_to_device(
    tensor: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Move a tensor to the specified device.

    Args:
        tensor: Input tensor
        device: Target device (default: auto-detect)

    Returns:
        Tensor on the target device

    Examples:
        >>> tensor = torch.randn(10)
        >>> tensor = move_to_device(tensor, torch.device("cpu"))
    """
    if device is None:
        device = get_strategy_device()

    return tensor.to(device)


def get_device_name(device: torch.device) -> str:
    """Get a human-readable device name.

    Args:
        device: PyTorch device

    Returns:
        Device name string

    Examples:
        >>> get_device_name(torch.device("cuda"))
        'cuda'
        >>> get_device_name(torch.device("cpu"))
        'cpu'
    """
    if device.type == "cuda":
        if device.index is not None:
            return f"cuda:{device.index}"
        return "cuda"  # type: ignore[unreachable]  # mypy false positive
    return "cpu"


def is_cuda_device(device: torch.device) -> bool:
    """Check if a device is CUDA.

    Args:
        device: PyTorch device

    Returns:
        True if device is CUDA, False otherwise

    Examples:
        >>> is_cuda_device(torch.device("cuda"))
        True
        >>> is_cuda_device(torch.device("cpu"))
        False
    """
    return device.type == "cuda"


def get_optimal_device_for_size(
    num_elements: int,
    threshold: int = 1000,
) -> torch.device:
    """Get optimal device based on data size.

    For small tensors, CPU may be faster due to transfer overhead.
    For large tensors, CUDA is preferred if available.

    Args:
        num_elements: Number of elements in the tensor
        threshold: Minimum elements to prefer CUDA (default: 1000)

    Returns:
        Optimal device for the given size

    Examples:
        >>> device = get_optimal_device_for_size(100)  # May return CPU
        >>> device = get_optimal_device_for_size(100000)  # Likely CUDA if available
    """
    if num_elements < threshold:
        return torch.device("cpu")

    return get_strategy_device()
