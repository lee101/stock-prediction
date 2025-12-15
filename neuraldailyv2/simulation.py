"""Unified trading simulation for V2 - used by BOTH training AND inference.

The key innovation in V2 is a single simulation function that:
- Training: Uses differentiable sigmoid fills with temperature annealing
- Inference: Uses binary fills (temperature=0) to match real execution

Temperature annealing bridges the gap: training starts soft and anneals
toward binary, so by the end of training the model has learned to produce
actions that work well with binary fills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from neuraldailyv2.config import SimulationConfig

_EPS = 1e-8
DAILY_PERIODS_PER_YEAR_CRYPTO = 365
DAILY_PERIODS_PER_YEAR_STOCK = 252


@dataclass
class SimulationResult:
    """Results from unified simulation."""

    pnl: torch.Tensor  # (batch, seq_len) - per-step PnL
    returns: torch.Tensor  # (batch, seq_len) - per-step returns
    portfolio_values: torch.Tensor  # (batch, seq_len) - portfolio value path
    cash: torch.Tensor  # (batch,) - final cash
    inventory: torch.Tensor  # (batch,) - final inventory
    buy_fill_probability: torch.Tensor  # (batch, seq_len)
    sell_fill_probability: torch.Tensor  # (batch, seq_len)
    executed_buys: torch.Tensor  # (batch, seq_len)
    executed_sells: torch.Tensor  # (batch, seq_len)
    inventory_path: torch.Tensor  # (batch, seq_len) - inventory over time


def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
    """Clamp values to avoid division by zero."""
    return torch.clamp(value.abs(), min=_EPS)


def _approx_fill_probability(
    limit_price: torch.Tensor,
    trigger_price: torch.Tensor,
    reference_price: torch.Tensor,
    *,
    is_buy: bool,
    temperature: float,
) -> torch.Tensor:
    """
    Differentiable approximation of fill probability.

    For buys: fills if low <= limit_price
    For sells: fills if high >= limit_price

    Args:
        limit_price: The limit order price
        trigger_price: Low (for buys) or High (for sells)
        reference_price: Close price for scaling
        is_buy: True for buy orders, False for sell orders
        temperature: Controls sigmoid sharpness. 0 = binary, >0 = soft

    Returns:
        Fill probability in [0, 1]
    """
    if temperature <= 0:
        # Binary fill: 100% or 0%
        if is_buy:
            return (trigger_price <= limit_price).float()
        else:
            return (trigger_price >= limit_price).float()

    # Differentiable sigmoid approximation
    scale = torch.clamp(reference_price.abs(), min=1e-4)
    if is_buy:
        # Buy fills if low <= limit_price, so score = (limit_price - low) / scale
        score = (limit_price - trigger_price) / (scale * temperature)
    else:
        # Sell fills if high >= limit_price, so score = (high - limit_price) / scale
        score = (trigger_price - limit_price) / (scale * temperature)

    return torch.sigmoid(score)


def simulate_trades(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    trade_intensity: torch.Tensor,
    asset_class: torch.Tensor,
    config: SimulationConfig,
    temperature: float = 0.0,
    single_step: bool = False,
    buy_trade_intensity: Optional[torch.Tensor] = None,
    sell_trade_intensity: Optional[torch.Tensor] = None,
) -> SimulationResult:
    """
    UNIFIED simulation used by training AND inference.

    This is the core V2 innovation - a single simulation function that:
    - temperature > 0: Uses differentiable sigmoid fills (training)
    - temperature = 0: Uses binary fills (inference, matches real execution)

    Args:
        highs: (batch, seq_len) - bar high prices
        lows: (batch, seq_len) - bar low prices
        closes: (batch, seq_len) - bar close prices
        buy_prices: (batch, seq_len) - limit buy prices from model
        sell_prices: (batch, seq_len) - limit sell prices from model
        trade_intensity: (batch, seq_len) - trade amount [0, max_leverage]
        asset_class: (batch,) - 0.0 for stocks, 1.0 for crypto
        config: Simulation configuration
        temperature: Fill approximation sharpness. 0 = binary, >0 = soft
        single_step: If True, only simulate the last timestep (for inference efficiency)
        buy_trade_intensity: Optional separate intensity for buys
        sell_trade_intensity: Optional separate intensity for sells

    Returns:
        SimulationResult with all simulation outputs
    """
    # Validate shapes
    batch_shape = closes.shape[:-1]
    seq_len = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype

    # Use separate intensities if provided
    buy_intensity = buy_trade_intensity if buy_trade_intensity is not None else trade_intensity
    sell_intensity = sell_trade_intensity if sell_trade_intensity is not None else trade_intensity

    # Determine leverage limits per asset
    # asset_class: 0 = stock (2x leverage), 1 = crypto (1x leverage)
    leverage_limits = torch.where(
        asset_class.unsqueeze(-1) > 0.5,
        torch.tensor(config.crypto_max_leverage, device=device, dtype=dtype),
        torch.tensor(config.equity_max_leverage, device=device, dtype=dtype),
    )
    leverage_limits = leverage_limits.expand_as(closes)  # (batch, seq_len)

    # Initialize state
    cash = torch.full(batch_shape, config.initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, config.initial_inventory, dtype=dtype, device=device)
    prev_value = cash + inventory * closes[..., 0]

    # Accumulators
    pnl_list = []
    return_list = []
    value_list = []
    buy_prob_list = []
    sell_prob_list = []
    exec_buy_list = []
    exec_sell_list = []
    inventory_list = []

    # Determine step range
    start_idx = seq_len - 1 if single_step else 0

    for idx in range(start_idx, seq_len):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        b_price = torch.clamp(buy_prices[..., idx], min=_EPS)
        s_price = torch.clamp(sell_prices[..., idx], min=_EPS)
        step_limit = torch.clamp(leverage_limits[..., idx], min=_EPS)

        # Clamp intensities to leverage limit
        b_intensity = torch.clamp(buy_intensity[..., idx], min=0.0)
        b_intensity = torch.minimum(b_intensity, step_limit)
        s_intensity = torch.clamp(sell_intensity[..., idx], min=0.0)
        s_intensity = torch.minimum(s_intensity, step_limit)

        # Convert intensity to fraction of limit
        b_frac = b_intensity / step_limit
        s_frac = s_intensity / step_limit

        # Compute fill probabilities (differentiable or binary based on temperature)
        buy_prob = _approx_fill_probability(
            b_price, low, close, is_buy=True, temperature=temperature
        )
        sell_prob = _approx_fill_probability(
            s_price, high, close, is_buy=False, temperature=temperature
        )

        # Calculate capacity
        equity = cash + inventory * close
        fee_mult = 1.0 + config.maker_fee

        # Buy capacity: how much can we buy with available cash or leverage
        max_buy_cash = torch.where(
            b_price > 0,
            cash / _safe_denominator(b_price * fee_mult),
            torch.zeros_like(cash),
        )

        # Leveraged buy capacity (for stocks with 2x leverage)
        target_notional = step_limit * torch.clamp(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = torch.where(
            b_price > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * fee_mult),
            torch.zeros_like(cash),
        )

        # Use cash-based for 1x, leverage-based for >1x
        buy_capacity = torch.where(
            step_limit <= 1.0 + 1e-6,
            torch.clamp(max_buy_cash, min=0.0),
            leveraged_capacity,
        )

        # Sell capacity: can only sell what we own
        sell_capacity = torch.clamp(inventory, min=0.0)

        # Calculate executed quantities
        buy_qty = b_frac * buy_capacity
        sell_qty = s_frac * sell_capacity

        # Apply fill probabilities
        executed_buys = buy_qty * buy_prob
        executed_sells = sell_qty * sell_prob

        # Update state
        buy_cost = executed_buys * b_price * fee_mult
        sell_revenue = executed_sells * s_price * (1.0 - config.maker_fee)

        cash = cash - buy_cost + sell_revenue
        inventory = inventory + executed_buys - executed_sells

        # Calculate portfolio value and returns
        portfolio_value = cash + inventory * close
        pnl = portfolio_value - prev_value
        returns = pnl / _safe_denominator(prev_value)
        prev_value = portfolio_value

        # Store results
        pnl_list.append(pnl)
        return_list.append(returns)
        value_list.append(portfolio_value)
        buy_prob_list.append(buy_prob)
        sell_prob_list.append(sell_prob)
        exec_buy_list.append(executed_buys)
        exec_sell_list.append(executed_sells)
        inventory_list.append(inventory.clone())

    # Stack results
    pnl_tensor = torch.stack(pnl_list, dim=-1)
    returns_tensor = torch.stack(return_list, dim=-1)
    values_tensor = torch.stack(value_list, dim=-1)
    buy_prob_tensor = torch.stack(buy_prob_list, dim=-1)
    sell_prob_tensor = torch.stack(sell_prob_list, dim=-1)
    exec_buy_tensor = torch.stack(exec_buy_list, dim=-1)
    exec_sell_tensor = torch.stack(exec_sell_list, dim=-1)
    inventory_tensor = torch.stack(inventory_list, dim=-1)

    return SimulationResult(
        pnl=pnl_tensor,
        returns=returns_tensor,
        portfolio_values=values_tensor,
        cash=cash,
        inventory=inventory,
        buy_fill_probability=buy_prob_tensor,
        sell_fill_probability=sell_prob_tensor,
        executed_buys=exec_buy_tensor,
        executed_sells=exec_sell_tensor,
        inventory_path=inventory_tensor,
    )


def compute_objective(
    returns: torch.Tensor,
    *,
    periods_per_year: float = DAILY_PERIODS_PER_YEAR_STOCK,
    return_weight: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute trading objective: Sortino + weighted return.

    Args:
        returns: (batch, seq_len) - per-step returns
        periods_per_year: Annualization factor
        return_weight: Weight for annual return in objective

    Returns:
        (score, sortino, annual_return) - all shape (batch,)
    """
    mean_return = returns.mean(dim=-1)
    downside = torch.clamp(-returns, min=0.0)
    downside_std = torch.sqrt(torch.mean(downside ** 2, dim=-1) + _EPS)

    sortino = mean_return / _safe_denominator(downside_std)
    sortino = sortino * math.sqrt(float(periods_per_year))

    annual_return = mean_return * periods_per_year
    score = sortino + return_weight * annual_return

    return score, sortino, annual_return


def compute_loss(
    returns: torch.Tensor,
    inventory_path: torch.Tensor,
    asset_class: torch.Tensor,
    config: SimulationConfig,
    *,
    return_weight: float = 0.08,
    exposure_penalty: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute training loss from simulation returns.

    Args:
        returns: (batch, seq_len) - per-step returns
        inventory_path: (batch, seq_len) - inventory over time
        asset_class: (batch,) - 0 for stocks, 1 for crypto
        config: Simulation config
        return_weight: Weight for return in objective
        exposure_penalty: Penalty for high exposure

    Returns:
        (loss, metrics_dict)
    """
    # Determine periods per year based on asset class (use stock for mixed batches)
    periods_per_year = DAILY_PERIODS_PER_YEAR_STOCK

    score, sortino, annual_return = compute_objective(
        returns,
        periods_per_year=periods_per_year,
        return_weight=return_weight,
    )

    # Base loss: negative score (we maximize score)
    loss = -score.mean()

    # Leverage cost for stocks only
    is_stock = asset_class < 0.5
    stock_inventory = inventory_path * is_stock.unsqueeze(-1)
    leveraged_amount = F.relu(stock_inventory - 1.0)
    daily_leverage_rate = config.leverage_fee_rate / DAILY_PERIODS_PER_YEAR_STOCK
    leverage_cost = (leveraged_amount * daily_leverage_rate).mean()
    loss = loss + leverage_cost

    # Optional exposure penalty
    if exposure_penalty > 0:
        avg_inventory = inventory_path.abs().mean()
        loss = loss + exposure_penalty * (avg_inventory ** 2)

    metrics = {
        "loss": loss.item(),
        "score": score.mean().item(),
        "sortino": sortino.mean().item(),
        "annual_return": annual_return.mean().item(),
        "leverage_cost": leverage_cost.item(),
    }

    return loss, metrics
