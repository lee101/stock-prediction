"""V5 Differentiable simulation for hourly trading with position length.

Key features:
- Soft entry/exit fills with temperature annealing
- Position length determines forced exit time (0-24 hours)
- Sortino-based loss function
- Fee-aware profit calculation
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from neuralhourlystocksv5.config import SimulationConfigStocksV5, TrainingConfigStocksV5


@dataclass
class HourlyTradeResult:
    """Result from simulating an hourly trade."""

    entry_filled: torch.Tensor  # (batch,) soft probability of entry
    exit_tp_filled: torch.Tensor  # (batch,) take-profit hit probability
    exit_forced: torch.Tensor  # (batch,) forced exit probability
    pnl: torch.Tensor  # (batch,) profit/loss
    returns: torch.Tensor  # (batch,) return percentage
    actual_hold_hours: torch.Tensor  # (batch,) weighted hold time
    fees: torch.Tensor  # (batch,) total fees paid


def _soft_fill_prob(
    target_price: torch.Tensor,
    bar_price: torch.Tensor,
    reference_price: torch.Tensor,
    is_buy: bool,
    temperature: float = 0.01,
) -> torch.Tensor:
    """
    Compute soft fill probability for limit orders.

    For buy: fills if bar_low <= buy_price
    For sell: fills if bar_high >= sell_price

    Uses sigmoid with temperature for differentiability.
    """
    # Normalize price difference by reference
    price_diff = target_price - bar_price
    normalized_diff = price_diff / (reference_price * temperature + 1e-8)

    if is_buy:
        # Buy fills when bar_low <= buy_price, i.e., buy_price - bar_low >= 0
        # Higher probability when buy_price is above bar_low
        fill_prob = torch.sigmoid(normalized_diff)
    else:
        # Sell fills when bar_high >= sell_price, i.e., bar_high - sell_price >= 0
        # Higher probability when sell_price is below bar_high
        fill_prob = torch.sigmoid(-normalized_diff)

    return fill_prob


def simulate_hourly_trade(
    *,
    # Future hourly data (lookahead for up to 24 hours)
    future_highs: torch.Tensor,  # (batch, 24)
    future_lows: torch.Tensor,  # (batch, 24)
    future_closes: torch.Tensor,  # (batch, 24)
    # Actions from model
    buy_price: torch.Tensor,  # (batch,)
    sell_price: torch.Tensor,  # (batch,)
    position_length: torch.Tensor,  # (batch,) 0-24 soft hours
    position_size: torch.Tensor,  # (batch,) 0-1
    # Config
    reference_price: torch.Tensor,
    config: SimulationConfigStocksV5,
    temperature: float = 0.01,
) -> HourlyTradeResult:
    """
    Simulate a single hourly trade decision with learned position length.

    Flow:
    1. Hour 0: Check if buy fills (low <= buy_price)
    2. Hours 1 to position_length: Check if sell fills (high >= sell_price)
    3. At position_length: Forced exit at close with slippage

    position_length = 0 means skip trade (no entry attempted)

    Args:
        future_highs: Next 24 hours of high prices
        future_lows: Next 24 hours of low prices
        future_closes: Next 24 hours of close prices
        buy_price: Entry limit price
        sell_price: Take-profit price
        position_length: How many hours to hold (0 = skip)
        position_size: Fraction of capital to use
        reference_price: Reference price for normalization
        config: Simulation configuration
        temperature: Temperature for soft fills

    Returns:
        HourlyTradeResult with entry/exit probabilities and P&L
    """
    batch_size = buy_price.size(0)
    max_hours = future_highs.size(1)
    device = buy_price.device

    # Soft skip signal: if position_length < 0.5, mostly skip
    # This creates a soft boundary for the "no trade" decision
    skip_weight = torch.sigmoid((0.5 - position_length) * 10)  # ~1 when length≈0
    trade_weight = 1.0 - skip_weight

    # Entry check on hour 0
    entry_prob = _soft_fill_prob(
        buy_price,
        future_lows[:, 0],
        reference_price,
        is_buy=True,
        temperature=temperature,
    )
    # Scale by trade weight (reduce entry probability if skipping)
    entry_prob = entry_prob * trade_weight

    # Track cumulative "still holding" probability
    # Starts at 1.0, decreases as we hit take-profit
    cumulative_holding = torch.ones(batch_size, device=device)
    weighted_exit_price = torch.zeros(batch_size, device=device)
    tp_probability = torch.zeros(batch_size, device=device)
    forced_exit_prob = torch.zeros(batch_size, device=device)
    weighted_hold_hours = torch.zeros(batch_size, device=device)

    # Round position length for hour-by-hour simulation
    # Use soft floor/ceiling for gradient flow
    position_length_clamped = position_length.clamp(0.5, max_hours)

    for hour in range(1, max_hours):
        hour_high = future_highs[:, hour]
        hour_close = future_closes[:, hour]

        # Soft "still within holding period" check
        # Creates smooth transition around the position_length boundary
        hours_remaining = position_length_clamped - hour
        still_holding_weight = torch.sigmoid(hours_remaining * 5)  # Soft step

        # Take-profit check: sell if high >= sell_price
        tp_prob_now = _soft_fill_prob(
            sell_price,
            hour_high,
            reference_price,
            is_buy=False,
            temperature=temperature,
        )

        # Probability of TP on this specific hour
        # = (probability still holding) * (probability TP fills this hour) * (still in holding period)
        tp_this_hour = cumulative_holding * tp_prob_now * still_holding_weight

        # Accumulate TP probability and weighted exit price
        tp_probability = tp_probability + tp_this_hour
        weighted_exit_price = weighted_exit_price + tp_this_hour * sell_price
        weighted_hold_hours = weighted_hold_hours + tp_this_hour * hour

        # Update cumulative holding (reduce by TP probability)
        cumulative_holding = cumulative_holding * (1 - tp_prob_now * still_holding_weight)

        # Forced exit check: at position_length hour
        # Soft check: is this hour the exit hour?
        exit_hour_proximity = 1.0 - torch.abs(position_length_clamped - hour).clamp(
            0, 1
        )
        forced_exit_this_hour = (
            cumulative_holding * exit_hour_proximity * (1 - still_holding_weight)
        )

        forced_exit_price = hour_close * (1 - config.forced_exit_slippage)
        weighted_exit_price = (
            weighted_exit_price + forced_exit_this_hour * forced_exit_price
        )
        weighted_hold_hours = weighted_hold_hours + forced_exit_this_hour * hour
        forced_exit_prob = forced_exit_prob + forced_exit_this_hour

        # Update cumulative holding
        cumulative_holding = cumulative_holding * (1 - exit_hour_proximity)

    # Handle any remaining position at max hours (edge case)
    final_forced = cumulative_holding
    final_exit_price = future_closes[:, -1] * (1 - config.forced_exit_slippage)
    weighted_exit_price = weighted_exit_price + final_forced * final_exit_price
    weighted_hold_hours = weighted_hold_hours + final_forced * max_hours
    forced_exit_prob = forced_exit_prob + final_forced

    # Ensure probabilities sum correctly
    total_exit_prob = tp_probability + forced_exit_prob
    # Normalize to prevent probability > 1
    total_exit_prob = total_exit_prob.clamp(max=1.0)

    # Compute returns
    # Gross return = (exit_price - entry_price) / entry_price
    effective_exit_price = weighted_exit_price / (
        total_exit_prob.clamp(min=1e-8)
    )  # Weighted average
    gross_return = (effective_exit_price - buy_price) / (buy_price + 1e-8)

    # Fees: entry + exit (2 * maker_fee)
    total_fees = 2 * config.maker_fee

    # Net return = (gross - fees) * entry_probability * position_size
    net_return = (gross_return - total_fees) * entry_prob * position_size

    # Scale hold hours by entry probability
    actual_hold_hours = weighted_hold_hours * entry_prob

    return HourlyTradeResult(
        entry_filled=entry_prob,
        exit_tp_filled=tp_probability.clamp(0, 1),
        exit_forced=forced_exit_prob.clamp(0, 1),
        pnl=net_return * reference_price,  # Absolute P&L
        returns=net_return,
        actual_hold_hours=actual_hold_hours,
        fees=torch.full_like(entry_prob, total_fees) * entry_prob,
    )


def compute_v5_loss(
    result: HourlyTradeResult,
    length_probs: torch.Tensor,
    position_length: torch.Tensor,
    buy_offset: torch.Tensor,
    sell_offset: torch.Tensor,
    config: TrainingConfigStocksV5,
) -> Dict[str, torch.Tensor]:
    """
    V5 loss function combining Sortino, return, and penalties.

    Components:
    1. Sortino ratio (risk-adjusted returns)
    2. Raw returns
    3. Forced exit penalty (discourage hitting max hold time)
    4. No-trade penalty (discourage always skipping)
    5. Spread utilization (encourage using offset range)

    Args:
        result: HourlyTradeResult from simulation
        length_probs: Softmax probabilities for position length
        position_length: Decoded position length values
        buy_offset: Buy price offset from midpoint
        sell_offset: Sell price offset from midpoint
        config: Training configuration

    Returns:
        Dict with loss components and total loss
    """
    # Sortino ratio on batch returns
    returns = result.returns
    mean_return = returns.mean()

    # Downside deviation (only consider negative returns)
    downside = torch.clamp(-returns, min=0.0)
    downside_std = torch.sqrt((downside**2).mean() + 1e-8)

    # Annualized Sortino (hourly data: 24 * 365 hours per year)
    sortino = mean_return / downside_std * math.sqrt(24 * 365)

    # Component losses
    sortino_loss = -config.sortino_weight * sortino
    return_loss = -config.return_weight * mean_return * 24 * 365  # Annualized

    # Forced exit penalty: discourage hitting max hold time
    forced_exit_rate = result.exit_forced.mean()
    forced_penalty = config.forced_exit_penalty * forced_exit_rate

    # No-trade penalty: discourage always predicting position_length=0
    # length_probs[:, 0] is probability of "skip trade"
    no_trade_prob = length_probs[:, 0].mean()
    no_trade_penalty = config.no_trade_penalty * no_trade_prob

    # Spread utilization: encourage using the full offset range
    # Penalize if always using minimum offset (8bps)
    avg_offset = (buy_offset + sell_offset).mean()
    min_offset = config.min_price_offset_pct
    max_offset = config.max_price_offset_pct
    offset_utilization = (avg_offset - min_offset) / (max_offset - min_offset + 1e-8)
    spread_util_loss = -config.spread_utilization * offset_utilization

    # Total loss
    total_loss = (
        sortino_loss
        + return_loss
        + forced_penalty
        + no_trade_penalty
        + spread_util_loss
    )

    return {
        "loss": total_loss,
        "sortino_loss": sortino_loss,
        "return_loss": return_loss,
        "forced_penalty": forced_penalty,
        "no_trade_penalty": no_trade_penalty,
        "spread_util_loss": spread_util_loss,
        # Metrics (not losses)
        "sortino": sortino,
        "mean_return": mean_return,
        "forced_exit_rate": forced_exit_rate,
        "no_trade_rate": no_trade_prob,
        "avg_offset": avg_offset,
        "tp_rate": result.exit_tp_filled.mean(),
        "avg_hold_hours": result.actual_hold_hours.mean(),
        "avg_position_length": position_length.mean(),
        "entry_rate": result.entry_filled.mean(),
    }


def simulate_batch(
    *,
    batch: Dict[str, torch.Tensor],
    actions: Dict[str, torch.Tensor],
    config: SimulationConfigStocksV5,
    temperature: float = 0.01,
) -> HourlyTradeResult:
    """
    Convenience function to simulate a batch of trades.

    Args:
        batch: Dict with future_highs, future_lows, future_closes, current_close
        actions: Dict with buy_price, sell_price, position_length, position_size
        config: Simulation configuration
        temperature: Temperature for soft fills

    Returns:
        HourlyTradeResult
    """
    return simulate_hourly_trade(
        future_highs=batch["future_highs"],
        future_lows=batch["future_lows"],
        future_closes=batch["future_closes"],
        buy_price=actions["buy_price"],
        sell_price=actions["sell_price"],
        position_length=actions["position_length"],
        position_size=actions["position_size"],
        reference_price=batch["current_close"],
        config=config,
        temperature=temperature,
    )
