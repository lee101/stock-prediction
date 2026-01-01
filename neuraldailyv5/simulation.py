"""V5 Portfolio simulation with ramp-into-position and Sortino optimization.

Key features:
- Portfolio-based: Track weights instead of individual trades
- Ramp-into-position: Gradual rebalancing throughout the day
- Turnover tracking: Penalize excessive trading
- Sortino ratio: Focus on downside risk
- Multi-day simulation: Day-by-day portfolio evolution
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from neuraldailyv5.config import SimulationConfigV5


@dataclass
class PortfolioState:
    """Current portfolio state."""
    weights: torch.Tensor        # (batch, num_assets) - current weights
    cash_weight: torch.Tensor    # (batch,) - weight in cash
    cumulative_pnl: torch.Tensor  # (batch,) - cumulative PnL
    cumulative_turnover: torch.Tensor  # (batch,) - total turnover
    day: int = 0


@dataclass
class DayResult:
    """Result from simulating one day."""
    pnl: torch.Tensor            # (batch,) - day's PnL
    turnover: torch.Tensor       # (batch,) - rebalancing turnover
    rebalance_cost: torch.Tensor  # (batch,) - fees + slippage
    weights_after: torch.Tensor  # (batch, num_assets) - weights after rebalancing


@dataclass
class SimulationResult:
    """Complete simulation result across all days."""
    # Per-day metrics
    daily_pnl: torch.Tensor        # (batch, num_days)
    daily_turnover: torch.Tensor   # (batch, num_days)
    daily_weights: torch.Tensor    # (batch, num_days, num_assets)

    # Aggregated metrics
    total_pnl: torch.Tensor        # (batch,)
    total_turnover: torch.Tensor   # (batch,)

    # Risk metrics
    sortino_ratio: torch.Tensor    # scalar
    sharpe_ratio: torch.Tensor     # scalar
    max_drawdown: torch.Tensor     # scalar
    mean_return: torch.Tensor      # scalar
    downside_deviation: torch.Tensor  # scalar


def compute_rebalance_cost(
    current_weights: torch.Tensor,
    target_weights: torch.Tensor,
    reference_values: torch.Tensor,
    config: SimulationConfigV5,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cost of rebalancing from current to target weights.

    Args:
        current_weights: (batch, num_assets) - current portfolio weights
        target_weights: (batch, num_assets) - target portfolio weights
        reference_values: (batch, num_assets) - reference prices for each asset
        config: Simulation config
        temperature: Soft rebalancing temperature (0 = hard)

    Returns:
        cost: (batch,) - total rebalancing cost
        turnover: (batch,) - total turnover (sum of abs weight changes)
    """
    # Compute weight changes
    delta_weights = target_weights - current_weights  # (batch, num_assets)

    # Turnover = sum of absolute weight changes / 2 (buys + sells counted once)
    turnover = delta_weights.abs().sum(dim=-1) / 2

    # Trading cost = fees + slippage
    fee_rate = config.maker_fee
    slippage_rate = config.slippage_bps / 10000

    # Cost proportional to turnover
    # Each trade incurs fee + slippage on the traded amount
    trade_cost_rate = fee_rate + slippage_rate

    # If ramping, split cost across periods with market impact
    if config.ramp_periods > 1:
        # Market impact per period
        impact_per_period = config.market_impact_bps / 10000
        total_impact = impact_per_period * config.ramp_periods
        trade_cost_rate = trade_cost_rate + total_impact

    cost = turnover * trade_cost_rate

    return cost, turnover


def apply_rebalance(
    current_weights: torch.Tensor,
    target_weights: torch.Tensor,
    rebalance_threshold: float = 0.02,
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Apply rebalancing with threshold.

    Only rebalance if drift exceeds threshold.

    Args:
        current_weights: (batch, num_assets)
        target_weights: (batch, num_assets)
        rebalance_threshold: Min drift to trigger rebalancing
        temperature: Soft rebalancing (blend current and target)

    Returns:
        new_weights: (batch, num_assets)
    """
    # Compute drift
    drift = (target_weights - current_weights).abs().max(dim=-1, keepdim=True).values

    # Binary mask: rebalance if drift > threshold
    if temperature <= 0:
        rebalance_mask = (drift > rebalance_threshold).float()
    else:
        # Soft threshold
        rebalance_prob = torch.sigmoid((drift - rebalance_threshold) / temperature)
        rebalance_mask = rebalance_prob

    # Blend weights
    new_weights = current_weights * (1 - rebalance_mask) + target_weights * rebalance_mask

    return new_weights


def simulate_day(
    *,
    current_weights: torch.Tensor,   # (batch, num_assets)
    target_weights: torch.Tensor,    # (batch, num_assets)
    day_returns: torch.Tensor,       # (batch, num_assets) - asset returns for this day
    config: SimulationConfigV5,
    temperature: float = 0.0,
) -> DayResult:
    """
    Simulate one trading day with portfolio rebalancing.

    Timeline:
    1. Market opens with current_weights
    2. Throughout day, ramp into target_weights (incur costs)
    3. Market closes, apply day_returns to end-of-day weights
    4. Compute PnL

    Args:
        current_weights: Portfolio weights at start of day
        target_weights: Target weights for this day
        day_returns: Returns for each asset on this day
        config: Simulation config
        temperature: Soft rebalancing temperature

    Returns:
        DayResult with day's metrics
    """
    batch_size = current_weights.size(0)
    device = current_weights.device

    # Compute rebalancing cost
    rebalance_cost, turnover = compute_rebalance_cost(
        current_weights, target_weights,
        torch.ones_like(current_weights),  # Reference values (normalized)
        config,
        temperature,
    )

    # Apply rebalancing
    weights_after_rebalance = apply_rebalance(
        current_weights, target_weights,
        config.rebalance_threshold, temperature,
    )

    # Apply market returns to rebalanced portfolio
    # PnL = sum(weight_i * return_i) - rebalance_cost
    market_pnl = (weights_after_rebalance * day_returns).sum(dim=-1)
    total_pnl = market_pnl - rebalance_cost

    # Update weights based on returns (drift from price changes)
    # weight_new = weight_old * (1 + return) / sum(weight_old * (1 + return))
    weights_with_returns = weights_after_rebalance * (1 + day_returns)
    weights_normalized = weights_with_returns / (weights_with_returns.sum(dim=-1, keepdim=True) + 1e-8)

    return DayResult(
        pnl=total_pnl,
        turnover=turnover,
        rebalance_cost=rebalance_cost,
        weights_after=weights_normalized,
    )


def simulate_portfolio(
    *,
    target_weights_sequence: torch.Tensor,  # (batch, num_days, num_assets)
    daily_returns: torch.Tensor,            # (batch, num_days, num_assets)
    initial_weights: Optional[torch.Tensor] = None,  # (batch, num_assets)
    asset_class: Optional[torch.Tensor] = None,  # (num_assets,) - 0=equity, 1=crypto
    config: SimulationConfigV5,
    temperature: float = 0.0,
) -> SimulationResult:
    """
    Simulate portfolio over multiple days.

    Args:
        target_weights_sequence: Target weights for each day
        daily_returns: Asset returns for each day
        initial_weights: Starting portfolio (default: all cash)
        asset_class: Asset class for leverage limits
        config: Simulation config
        temperature: Soft rebalancing temperature

    Returns:
        SimulationResult with full simulation metrics
    """
    batch_size, num_days, num_assets = target_weights_sequence.shape
    device = target_weights_sequence.device

    # Initialize weights (start with equal weights or all cash)
    if initial_weights is None:
        initial_weights = torch.zeros(batch_size, num_assets, device=device)

    current_weights = initial_weights

    # Storage for per-day metrics
    daily_pnl = []
    daily_turnover = []
    daily_weights = []

    # Simulate each day
    for day in range(num_days):
        target = target_weights_sequence[:, day, :]
        returns = daily_returns[:, day, :]

        # Apply leverage limits based on asset class
        if asset_class is not None:
            equity_mask = (asset_class < 0.5).float()
            crypto_mask = (asset_class >= 0.5).float()

            # Limit total leverage per asset class
            equity_target = target * equity_mask.unsqueeze(0)
            crypto_target = target * crypto_mask.unsqueeze(0)

            equity_sum = equity_target.sum(dim=-1, keepdim=True)
            crypto_sum = crypto_target.sum(dim=-1, keepdim=True)

            # Scale down if exceeds leverage
            equity_scale = torch.clamp(config.equity_max_leverage / (equity_sum + 1e-8), max=1.0)
            crypto_scale = torch.clamp(config.crypto_max_leverage / (crypto_sum + 1e-8), max=1.0)

            target = equity_target * equity_scale + crypto_target * crypto_scale

        result = simulate_day(
            current_weights=current_weights,
            target_weights=target,
            day_returns=returns,
            config=config,
            temperature=temperature,
        )

        daily_pnl.append(result.pnl)
        daily_turnover.append(result.turnover)
        daily_weights.append(result.weights_after)

        current_weights = result.weights_after

    # Stack results
    daily_pnl = torch.stack(daily_pnl, dim=1)       # (batch, num_days)
    daily_turnover = torch.stack(daily_turnover, dim=1)
    daily_weights = torch.stack(daily_weights, dim=1)  # (batch, num_days, num_assets)

    # Aggregate metrics
    total_pnl = daily_pnl.sum(dim=1)
    total_turnover = daily_turnover.sum(dim=1)

    # Risk metrics
    mean_return = daily_pnl.mean()
    std_return = daily_pnl.std() + 1e-8

    # Sharpe ratio (daily, can annualize by * sqrt(252))
    sharpe_ratio = mean_return / std_return

    # Sortino ratio (focus on downside)
    downside_returns = torch.clamp(daily_pnl - config.min_acceptable_return, max=0)
    downside_deviation = (downside_returns.square().mean()).sqrt() + 1e-8
    sortino_ratio = mean_return / downside_deviation

    # Maximum drawdown
    cumulative_pnl = daily_pnl.cumsum(dim=1)
    running_max = cumulative_pnl.cummax(dim=1).values
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    return SimulationResult(
        daily_pnl=daily_pnl,
        daily_turnover=daily_turnover,
        daily_weights=daily_weights,
        total_pnl=total_pnl,
        total_turnover=total_turnover,
        sortino_ratio=sortino_ratio,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        mean_return=mean_return,
        downside_deviation=downside_deviation,
    )


class PortfolioSimulatorV5:
    """
    V5 Portfolio simulator for training.

    Wraps simulation functions with config and provides loss computation.
    """

    def __init__(self, config: SimulationConfigV5, num_assets: int):
        self.config = config
        self.num_assets = num_assets

    def simulate(
        self,
        target_weights: torch.Tensor,    # (batch, num_assets) - single day target
        daily_returns: torch.Tensor,     # (batch, num_days, num_assets)
        initial_weights: Optional[torch.Tensor] = None,
        asset_class: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> SimulationResult:
        """
        Simulate portfolio with single target weight held across all days.

        For training, we predict a single target portfolio and hold it
        for the lookahead period.
        """
        batch_size, num_days, num_assets = daily_returns.shape
        device = daily_returns.device

        # Expand single target to all days
        target_weights_sequence = target_weights.unsqueeze(1).expand(-1, num_days, -1)

        return simulate_portfolio(
            target_weights_sequence=target_weights_sequence,
            daily_returns=daily_returns,
            initial_weights=initial_weights,
            asset_class=asset_class,
            config=self.config,
            temperature=temperature,
        )

    def simulate_sequence(
        self,
        target_weights_sequence: torch.Tensor,  # (batch, num_days, num_assets)
        daily_returns: torch.Tensor,            # (batch, num_days, num_assets)
        initial_weights: Optional[torch.Tensor] = None,
        asset_class: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> SimulationResult:
        """
        Simulate portfolio with day-by-day target weights.

        For inference/advanced training with daily portfolio updates.
        """
        return simulate_portfolio(
            target_weights_sequence=target_weights_sequence,
            daily_returns=daily_returns,
            initial_weights=initial_weights,
            asset_class=asset_class,
            config=self.config,
            temperature=temperature,
        )


def compute_v5_loss(
    result: SimulationResult,
    outputs: Dict[str, torch.Tensor],
    daily_volatility: Optional[torch.Tensor] = None,  # (batch, num_days, num_assets)
    config: SimulationConfigV5 = None,
    *,
    return_weight: float = 1.0,
    sortino_weight: float = 0.2,
    nepa_weight: float = 0.1,
    turnover_penalty: float = 0.05,
    concentration_penalty: float = 0.05,
    volatility_calibration_weight: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute V5 training loss.

    Loss components:
    1. Return loss: Maximize total PnL
    2. Sortino loss: Risk-adjusted returns with downside focus
    3. NEPA loss: Cosine similarity for sequence coherence
    4. Turnover penalty: Discourage excessive rebalancing
    5. Concentration penalty: Encourage diversification
    6. Volatility calibration: Ensure predicted vol matches realized vol

    Args:
        result: SimulationResult from portfolio simulation
        outputs: Model outputs (includes nepa_loss, weights, volatility)
        daily_volatility: Realized volatility for calibration
        config: Simulation config

    Returns:
        total_loss: Scalar loss
        loss_components: Dict of individual loss components
    """
    device = result.total_pnl.device

    # Return loss (negative because we maximize)
    return_loss = -result.total_pnl.mean()

    # Sortino loss (negative because we maximize)
    sortino_loss = -result.sortino_ratio

    # NEPA loss (from model outputs)
    nepa_loss_val = outputs.get("nepa_loss", torch.tensor(0.0, device=device))

    # Turnover penalty
    turnover_loss = result.total_turnover.mean()

    # Concentration penalty (encourage entropy in weights)
    weights = outputs.get("weights", None)
    if weights is not None:
        # Negative entropy = concentration
        # max entropy = uniform distribution
        eps = 1e-8
        weight_entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        # Invert: higher entropy = lower concentration
        concentration_loss = -weight_entropy
    else:
        concentration_loss = torch.tensor(0.0, device=device)

    # Volatility calibration
    vol_calibration_loss = torch.tensor(0.0, device=device)
    if daily_volatility is not None and "volatility" in outputs:
        pred_vol = outputs["volatility"]  # (batch, num_assets)
        # Compare to realized volatility (use std of daily returns)
        realized_vol = daily_volatility.std(dim=1)  # (batch, num_assets)
        vol_calibration_loss = F.mse_loss(pred_vol, realized_vol)

    # Total loss
    total_loss = (
        return_weight * return_loss
        + sortino_weight * sortino_loss
        + nepa_weight * nepa_loss_val
        + turnover_penalty * turnover_loss
        + concentration_penalty * concentration_loss
        + volatility_calibration_weight * vol_calibration_loss
    )

    loss_components = {
        "return_loss": return_loss,
        "sortino_loss": sortino_loss,
        "nepa_loss": nepa_loss_val,
        "turnover_loss": turnover_loss,
        "concentration_loss": concentration_loss,
        "vol_calibration_loss": vol_calibration_loss,
        "total_loss": total_loss,
    }

    return total_loss, loss_components
