from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def taylor_time_encoding(indices: Tensor, order: int = 4, scale: float | Tensor = 32.0) -> Tensor:
    """
    Produce a Taylor-series style positional encoding for temporal indices.

    Args:
        indices: Tensor of shape [...], typically representing step indices.
        order: Number of Taylor coefficients to emit.
        scale: Normalisation constant controlling the spread of the encoding.

    Returns:
        Tensor of shape [..., order] with the n-th column equal to
        (indices / scale) ** n / n!.
    """
    if order <= 0:
        raise ValueError("order must be positive")
    if not torch.is_tensor(indices):
        raise TypeError("indices must be a torch.Tensor")

    indices = indices.to(dtype=torch.float32)
    if torch.is_tensor(scale):
        scale_tensor = scale.to(indices.device, dtype=indices.dtype)
    else:
        scale_tensor = torch.tensor(scale, device=indices.device, dtype=indices.dtype)
    scale_tensor = scale_tensor.clamp_min(1e-6)
    scaled = indices[..., None] / scale_tensor

    coeffs = []
    for n in range(1, order + 1):
        coeffs.append((scaled**n) / math.factorial(n))
    return torch.cat(coeffs, dim=-1)


def _build_haar_kernels(channels: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    norm = 1.0 / math.sqrt(2.0)
    low = torch.tensor([norm, norm], device=device, dtype=dtype)
    high = torch.tensor([norm, -norm], device=device, dtype=dtype)
    low = low.view(1, 1, 2).repeat(channels, 1, 1)
    high = high.view(1, 1, 2).repeat(channels, 1, 1)
    return low, high


def haar_wavelet_pyramid(series: Tensor, levels: int = 1, padding_mode: str = "reflect") -> Tuple[Tensor, List[Tensor]]:
    """
    Build a multi-level Haar wavelet pyramid for a batch of 1D series.

    Args:
        series: Tensor shaped [B, C, T].
        levels: Number of detail levels to generate.
        padding_mode: Passed to F.pad when odd-length series require padding.

    Returns:
        approx: The final low-pass approximation tensor.
        details: List of length `levels` with high-pass detail tensors per level.
    """
    if series.ndim != 3:
        raise ValueError("series must have shape [B, C, T]")
    if levels < 1:
        raise ValueError("levels must be >= 1")

    approx = series
    details: List[Tensor] = []
    low_kernel, high_kernel = _build_haar_kernels(
        series.size(1),
        device=series.device,
        dtype=series.dtype,
    )

    for _ in range(levels):
        if approx.size(-1) < 2:
            raise ValueError("series length too short for requested levels")
        if approx.size(-1) % 2 != 0:
            approx = F.pad(approx, (0, 1), mode=padding_mode)

        low = F.conv1d(approx, low_kernel, stride=2, groups=approx.size(1))
        high = F.conv1d(approx, high_kernel, stride=2, groups=approx.size(1))
        details.append(high)
        approx = low
    return approx, details


def soft_drawdown(log_returns: Tensor, smoothing: float = 10.0) -> Tuple[Tensor, Tensor]:
    """
    Compute a differentiable approximation to cumulative wealth and drawdown.

    Args:
        log_returns: Tensor shaped [..., T] representing log returns over time.
        smoothing: Positive temperature parameter controlling the softness of the running max.

    Returns:
        wealth: Exponentiated cumulative wealth tensor [..., T].
        drawdown: Fractional drawdown tensor [..., T] with values in [0, 1].
    """
    if log_returns.ndim < 1:
        raise ValueError("log_returns must have at least one dimension")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    wealth_log = torch.cumsum(log_returns, dim=-1)
    wealth = wealth_log.exp()

    alpha = torch.tensor(smoothing, dtype=wealth.dtype, device=wealth.device)
    soft_max = wealth_log[..., :1]
    soft_values = [soft_max]
    for t in range(1, wealth_log.size(-1)):
        current = wealth_log[..., t : t + 1]
        stacked = torch.cat([soft_max, current], dim=-1)
        soft_max = torch.logsumexp(alpha * stacked, dim=-1, keepdim=True) / alpha
        soft_values.append(soft_max)

    soft_max = torch.cat(soft_values, dim=-1)

    reference = soft_max.exp()
    drawdown = 1.0 - wealth / reference.clamp_min(1e-12)
    return wealth, drawdown


def risk_budget_mismatch(weights: Tensor, cov: Tensor, target_budget: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Penalise deviation from a desired risk budget in a differentiable fashion.

    Args:
        weights: Portfolio weights tensor [..., A].
        cov: Covariance matrix tensor [A, A].
        target_budget: Target fraction per asset broadcastable to weights.
        eps: Small number to stabilise divisions.

    Returns:
        Scalar tensor representing squared error between realised and target risk budgets.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")

    weights = weights.to(dtype=cov.dtype)
    target_budget = target_budget.to(dtype=cov.dtype)

    marginal = weights @ cov
    port_var = (marginal * weights).sum(dim=-1, keepdim=True).clamp_min(eps)
    risk_contrib = weights * marginal
    risk_frac = risk_contrib / port_var

    target = target_budget / target_budget.sum(dim=-1, keepdim=True).clamp_min(eps)
    return ((risk_frac - target) ** 2).sum(dim=-1).mean()


@dataclass(slots=True)
class TradeMemoryState:
    ema_pnl: Tensor
    cumulative_pnl: Tensor
    steps: Tensor


def trade_memory_update(
    state: TradeMemoryState | None,
    pnl: Tensor,
    ema_decay: float = 0.95,
    clamp_range: Tuple[float, float] = (-5.0, 5.0),
) -> Tuple[TradeMemoryState, Tensor, Tensor]:
    """
    Maintain differentiable trade memory useful for adaptive risk signals.

    Args:
        state: Previous TradeMemoryState or None.
        pnl: Tensor of per-step P&L values.
        ema_decay: Exponential decay coefficient in [0, 1).
        clamp_range: Optional range applied to the cumulative signal to stabilise training.

    Returns:
        new_state: Updated TradeMemoryState.
        regret_signal: Smooth penalty encouraging the policy to recover losses.
        leverage_signal: Squashed signal suitable for scaling exposure.
    """
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not torch.is_tensor(pnl):
        raise TypeError("pnl must be a torch.Tensor")

    pnl = pnl.to(torch.float32)
    device = pnl.device
    dtype = pnl.dtype
    if state is None:
        ema = pnl
        cumulative = pnl
        steps = torch.ones_like(pnl, device=device, dtype=dtype)
    else:
        ema_prev = state.ema_pnl.to(device=device, dtype=dtype)
        cumulative_prev = state.cumulative_pnl.to(device=device, dtype=dtype)
        steps_prev = state.steps.to(device=device, dtype=dtype)
        ema = ema_decay * ema_prev + (1.0 - ema_decay) * pnl
        cumulative = cumulative_prev + pnl
        steps = steps_prev + 1.0

    cumulative_clamped = cumulative.clamp(*clamp_range)
    regret_signal = F.softplus(-cumulative_clamped)
    leverage_signal = torch.tanh(ema)

    new_state = TradeMemoryState(ema, cumulative, steps)
    return new_state, regret_signal, leverage_signal


def augment_market_features(
    features: Tensor,
    returns: Tensor,
    use_taylor: bool,
    taylor_order: int,
    taylor_scale: float,
    use_wavelet: bool,
    wavelet_levels: int,
    padding_mode: str = "reflect",
) -> Tensor:
    """
    Append optional Taylor positional encodings and Haar wavelet detail features.

    Args:
        features: Base feature tensor [T, A, F].
        returns: Forward return tensor [T, A].
        use_taylor: Whether to append Taylor encodings.
        use_wavelet: Whether to append Haar wavelet detail/approximation channels.

    Returns:
        Augmented feature tensor [T, A, F'].
    """
    augmented = features
    T, A, _ = features.shape
    device = features.device
    dtype = features.dtype

    if use_taylor and taylor_order > 0:
        idx = torch.arange(T, device=device, dtype=dtype)
        enc = taylor_time_encoding(idx, order=taylor_order, scale=taylor_scale)
        enc = enc.to(device=device, dtype=dtype).unsqueeze(1).expand(-1, A, -1)
        augmented = torch.cat([augmented, enc], dim=-1)

    if use_wavelet and wavelet_levels > 0:
        series = returns.transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        approx, details = haar_wavelet_pyramid(series, levels=wavelet_levels, padding_mode=padding_mode)
        wavelet_streams = []
        total_levels = len(details)
        for i, detail in enumerate(details):
            scale = 2 ** (i + 1)
            upsampled = detail.repeat_interleave(scale, dim=-1)[..., :T]
            upsampled = upsampled.squeeze(0).transpose(0, 1).unsqueeze(-1)
            wavelet_streams.append(upsampled)
        approx_up = approx.repeat_interleave(2 ** total_levels, dim=-1)[..., :T]
        approx_up = approx_up.squeeze(0).transpose(0, 1).unsqueeze(-1)
        wavelet_streams.append(approx_up)
        if wavelet_streams:
            wavelet_feats = torch.cat(wavelet_streams, dim=-1)
            augmented = torch.cat([augmented, wavelet_feats], dim=-1)

    return augmented
