"""
Differentiable utilities for portfolio gating and shutdown logic.

The helpers here mirror the numpy-based logic used inside ``PortfolioEnv`` but
operate purely on PyTorch tensors so they can be reused in gradient-based
research code (e.g., differentiable trading objectives, policy gradients).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class LossShutdownParams:
    """
    Hyper-parameters controlling the loss shutdown gating.

    Attributes:
        probe_weight: Maximum weight magnitude allowed while an asset/direction is
            cooling down after a loss (acts as the probe trade size).
        cooldown_steps: Number of steps to keep an asset/direction gated after a loss.
        min_position: Absolute weight threshold below which a position is considered inactive.
        return_tolerance: Absolute net return threshold treated as neutral (neither win nor loss).
        penalty_scale: Optional penalty coefficient applied to absolute weights that remain
            in cooldown (mirrors ``PortfolioEnvConfig.loss_shutdown_penalty``).
    """

    probe_weight: float = 0.05
    cooldown_steps: int = 3
    min_position: float = 1e-4
    return_tolerance: float = 1e-5
    penalty_scale: float = 0.0


@dataclass
class LossShutdownState:
    """
    Container for the long/short cooldown counters.

    The tensors are integer-valued but kept as ``torch.Tensor`` so the same state
    object can move across devices and participate in differentiable graphs when
    needed (the counters themselves do not require gradients).
    """

    long_counters: torch.Tensor
    short_counters: Optional[torch.Tensor] = None

    @classmethod
    def zeros(
        cls,
        num_assets: int,
        *,
        allow_short: bool,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.int32,
    ) -> "LossShutdownState":
        long = torch.zeros(num_assets, dtype=dtype, device=device)
        short = torch.zeros(num_assets, dtype=dtype, device=device) if allow_short else None
        return cls(long_counters=long, short_counters=short)

    def clone(self) -> "LossShutdownState":
        return LossShutdownState(
            long_counters=self.long_counters.clone(),
            short_counters=None if self.short_counters is None else self.short_counters.clone(),
        )


def loss_shutdown_adjust(
    proposed_weights: torch.Tensor,
    state: LossShutdownState,
    params: LossShutdownParams,
    *,
    allow_short: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Clamp proposed weights using loss shutdown counters.

    Returns:
        adjusted_weights: Tensor after applying probe caps to active cooldown assets.
        penalty: Scalar penalty proportional to absolute weight in cooldown directions.
        clipped_amount: Total absolute weight magnitude removed due to gating.
    """

    weights = proposed_weights
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        weights = weights.to(torch.float32)

    probe = torch.clamp(torch.tensor(params.probe_weight, dtype=weights.dtype, device=weights.device), 0.0, 1.0)
    min_pos = torch.tensor(params.min_position, dtype=weights.dtype, device=weights.device)

    active_long = state.long_counters.to(dtype=torch.bool)
    adjusted = weights
    if torch.any(active_long):
        positive_mask = weights > min_pos
        exceed_long = active_long & positive_mask & (weights > probe)
        adjusted = torch.where(exceed_long, torch.full_like(weights, probe), adjusted)

    active_short: Optional[torch.Tensor] = None
    if allow_short and state.short_counters is not None:
        active_short = state.short_counters.to(dtype=torch.bool)
        if torch.any(active_short):
            negative_mask = adjusted < -min_pos
            exceed_short = active_short & negative_mask & (adjusted < -probe)
            adjusted = torch.where(exceed_short, -torch.full_like(weights, probe), adjusted)

    clipped = torch.clamp(weights.abs() - adjusted.abs(), min=0.0)
    clipped_amount = clipped.sum()

    penalty_scale = torch.tensor(params.penalty_scale, dtype=weights.dtype, device=weights.device)
    penalty = torch.tensor(0.0, dtype=weights.dtype, device=weights.device)
    if penalty_scale.item() > 0.0:
        penalty_contrib = torch.tensor(0.0, dtype=weights.dtype, device=weights.device)
        if torch.any(active_long):
            penalty_contrib = penalty_contrib + torch.sum(adjusted.abs() * (active_long & (adjusted > min_pos)).to(adjusted.dtype))
        if active_short is not None and torch.any(active_short):
            penalty_contrib = penalty_contrib + torch.sum(adjusted.abs() * (active_short & (adjusted < -min_pos)).to(adjusted.dtype))
        penalty = penalty_scale * penalty_contrib

    return adjusted, penalty, clipped_amount


def update_loss_shutdown_state(
    executed_weights: torch.Tensor,
    net_asset_returns: torch.Tensor,
    state: LossShutdownState,
    params: LossShutdownParams,
    *,
    allow_short: bool,
) -> LossShutdownState:
    """
    Update cooldown counters based on realised net returns.

    Args:
        executed_weights: Allocations that just experienced ``net_asset_returns``.
        net_asset_returns: Net per-asset returns (including costs) for the latest step.
        state: Previous cooldown state.
        params: Shutdown parameters.
        allow_short: Whether short positions are allowed (and thus tracked separately).
    """

    device = executed_weights.device
    dtype = state.long_counters.dtype
    cooldown_value = max(1, int(params.cooldown_steps))
    cooldown_tensor = torch.full_like(state.long_counters, cooldown_value, device=device)
    min_pos = torch.tensor(params.min_position, dtype=executed_weights.dtype, device=device)
    tolerance = torch.tensor(params.return_tolerance, dtype=executed_weights.dtype, device=device)

    long_counters = torch.clamp(state.long_counters - 1, min=0)
    positive_mask = executed_weights > min_pos
    loss_long = positive_mask & (net_asset_returns < -tolerance)
    profit_long = positive_mask & (net_asset_returns > tolerance)
    long_counters = torch.where(loss_long, cooldown_tensor, long_counters)
    long_counters = torch.where(profit_long, torch.zeros_like(long_counters), long_counters)

    short_counters: Optional[torch.Tensor] = None
    if allow_short and state.short_counters is not None:
        short_counters = torch.clamp(state.short_counters - 1, min=0)
        negative_mask = executed_weights < -min_pos
        loss_short = negative_mask & (net_asset_returns < -tolerance)
        profit_short = negative_mask & (net_asset_returns > tolerance)
        short_counters = torch.where(loss_short, torch.full_like(state.short_counters, cooldown_value, device=device), short_counters)
        short_counters = torch.where(profit_short, torch.zeros_like(short_counters), short_counters)

    return LossShutdownState(long_counters=long_counters.to(dtype=dtype), short_counters=None if short_counters is None else short_counters.to(dtype=dtype))


__all__ = [
    "LossShutdownParams",
    "LossShutdownState",
    "loss_shutdown_adjust",
    "update_loss_shutdown_state",
]
