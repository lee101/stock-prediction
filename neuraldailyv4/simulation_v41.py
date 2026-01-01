"""V4.1 Simulation: Single-trade with aggregated predictions.

KEY FIX: Training simulation now matches inference exactly.
- Aggregate prices FIRST (trimmed mean across windows)
- Simulate ONE trade with aggregated prices
- No more train/inference mismatch

Additional fixes:
- Minimum spread enforcement (2%)
- Position size regularization
- Wider price targets
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from neuraldailyv4.config import SimulationConfigV4


@dataclass
class TradeResult:
    """Result from simulating a single aggregated trade."""
    entry_filled: torch.Tensor      # (batch,) probability of entry fill
    tp_hit: torch.Tensor            # (batch,) probability of TP hit
    forced_exit: torch.Tensor       # (batch,) probability of forced exit
    returns: torch.Tensor           # (batch,) net returns
    actual_hold_days: torch.Tensor  # (batch,) expected hold days
    exit_price: torch.Tensor        # (batch,) expected exit price

    # Aggregated metrics
    sharpe: torch.Tensor            # scalar
    forced_exit_rate: torch.Tensor  # scalar
    mean_return: torch.Tensor       # scalar


def trimmed_mean(values: torch.Tensor, trim_fraction: float = 0.25, dim: int = -1) -> torch.Tensor:
    """Compute trimmed mean along a dimension."""
    n = values.size(dim)
    trim_n = int(n * trim_fraction)

    if trim_n <= 0 or n - 2 * trim_n <= 0:
        return values.mean(dim=dim)

    sorted_vals, _ = torch.sort(values, dim=dim)

    if dim == -1:
        trimmed = sorted_vals[..., trim_n:-trim_n]
    else:
        trimmed = torch.narrow(sorted_vals, dim, trim_n, n - 2 * trim_n)

    return trimmed.mean(dim=dim)


def aggregate_predictions(
    buy_quantiles: torch.Tensor,    # (batch, num_windows, num_quantiles)
    sell_quantiles: torch.Tensor,   # (batch, num_windows, num_quantiles)
    exit_days: torch.Tensor,        # (batch, num_windows)
    position_size: torch.Tensor,    # (batch, num_windows)
    confidence: torch.Tensor,       # (batch, num_windows)
    trim_fraction: float = 0.25,
    min_spread: float = 0.02,       # 2% minimum spread
) -> dict:
    """
    Aggregate predictions across windows using trimmed mean.

    This is now done BEFORE simulation, matching inference exactly.
    """
    # Use median quantile (q50)
    median_idx = buy_quantiles.size(-1) // 2
    buy_prices = buy_quantiles[..., median_idx]   # (batch, num_windows)
    sell_prices = sell_quantiles[..., median_idx]

    # Trimmed mean across windows
    agg_buy = trimmed_mean(buy_prices, trim_fraction, dim=1)    # (batch,)
    agg_sell = trimmed_mean(sell_prices, trim_fraction, dim=1)
    agg_exit_days = trimmed_mean(exit_days, trim_fraction, dim=1)
    agg_position = trimmed_mean(position_size, trim_fraction, dim=1)
    agg_confidence = trimmed_mean(confidence, trim_fraction, dim=1)

    # Enforce minimum spread (key fix for risk/reward)
    min_sell = agg_buy * (1 + min_spread)
    agg_sell = torch.maximum(agg_sell, min_sell)

    return {
        "buy_price": agg_buy,
        "sell_price": agg_sell,
        "exit_days": agg_exit_days,
        "position_size": agg_position,
        "confidence": agg_confidence,
    }


def simulate_single_trade(
    *,
    # Future data
    future_highs: torch.Tensor,    # (batch, lookahead_days)
    future_lows: torch.Tensor,     # (batch, lookahead_days)
    future_closes: torch.Tensor,   # (batch, lookahead_days)
    # Aggregated predictions
    buy_price: torch.Tensor,       # (batch,)
    sell_price: torch.Tensor,      # (batch,)
    exit_days: torch.Tensor,       # (batch,)
    position_size: torch.Tensor,   # (batch,)
    # Config
    reference_price: torch.Tensor, # (batch,)
    config: SimulationConfigV4,
    temperature: float = 0.0,
) -> TradeResult:
    """
    Simulate a single trade with aggregated predictions.

    This is the EXACT same logic as inference backtest.
    Day 0: Check entry (low <= buy_price)
    Days 1-N: Check TP (high >= sell_price)
    Day N: Forced exit at close if no TP
    """
    batch_size = buy_price.size(0)
    lookahead = future_highs.size(1)
    device = buy_price.device

    # Clamp exit_days
    exit_days_int = exit_days.round().long().clamp(1, lookahead - 1)

    # Day 0: Entry check
    day0_low = future_lows[:, 0]
    if temperature <= 0:
        entry_filled = (day0_low <= buy_price).float()
    else:
        # Soft fills for training
        diff = buy_price - day0_low
        normalized = diff / (reference_price * temperature + 1e-8)
        entry_filled = torch.sigmoid(normalized)

    # Initialize tracking
    tp_probability = torch.zeros(batch_size, device=device)
    expected_exit_price = torch.zeros(batch_size, device=device)
    expected_hold_days = torch.zeros(batch_size, device=device)
    cumulative_no_tp = torch.ones(batch_size, device=device)

    # Days 1 to lookahead: Check TP and forced exit
    for day in range(1, lookahead):
        day_high = future_highs[:, day]
        day_close = future_closes[:, day]

        # Are we still in the trade? (before exit_days)
        still_holding = (day <= exit_days_int).float()

        # TP check
        if temperature <= 0:
            tp_today = (day_high >= sell_price).float()
        else:
            diff = day_high - sell_price
            normalized = diff / (reference_price * temperature + 1e-8)
            tp_today = torch.sigmoid(normalized)

        # Probability of hitting TP on this specific day
        tp_this_day = cumulative_no_tp * tp_today * still_holding

        # Accumulate
        tp_probability = tp_probability + tp_this_day
        expected_exit_price = expected_exit_price + tp_this_day * sell_price
        expected_hold_days = expected_hold_days + tp_this_day * day

        # Update cumulative no-TP
        cumulative_no_tp = cumulative_no_tp * (1 - tp_today * still_holding)

        # Forced exit on exit_day
        is_exit_day = (day == exit_days_int).float()
        forced_prob = cumulative_no_tp * is_exit_day
        forced_price = day_close * (1 - config.forced_exit_slippage)

        expected_exit_price = expected_exit_price + forced_prob * forced_price
        expected_hold_days = expected_hold_days + forced_prob * day
        cumulative_no_tp = cumulative_no_tp * (1 - is_exit_day)

    # Any remaining probability uses final close
    if cumulative_no_tp.sum() > 0:
        final_close = future_closes[:, -1] * (1 - config.forced_exit_slippage)
        expected_exit_price = expected_exit_price + cumulative_no_tp * final_close
        expected_hold_days = expected_hold_days + cumulative_no_tp * (lookahead - 1)

    # Clamp TP probability
    tp_probability = tp_probability.clamp(0, 1)
    forced_exit_prob = (1 - tp_probability) * entry_filled

    # Compute returns
    gross_return = (expected_exit_price - buy_price) / (buy_price + 1e-8)
    fees = 2 * config.maker_fee
    net_return = (gross_return - fees) * entry_filled * position_size

    # Metrics
    mean_return = net_return.mean()
    std_return = net_return.std() + 1e-8
    sharpe = mean_return / std_return

    return TradeResult(
        entry_filled=entry_filled,
        tp_hit=tp_probability * entry_filled,
        forced_exit=forced_exit_prob,
        returns=net_return,
        actual_hold_days=expected_hold_days,
        exit_price=expected_exit_price,
        sharpe=sharpe,
        forced_exit_rate=forced_exit_prob.mean(),
        mean_return=mean_return,
    )


def compute_v41_loss(
    result: TradeResult,
    buy_price: torch.Tensor,
    sell_price: torch.Tensor,
    position_size: torch.Tensor,
    reference_price: torch.Tensor,
    *,
    return_weight: float = 1.0,
    sharpe_weight: float = 0.5,
    forced_exit_penalty: float = 0.5,
    position_reg_weight: float = 0.1,
    spread_bonus_weight: float = 0.1,
    target_position: float = 0.3,
) -> torch.Tensor:
    """
    Compute V4.1 loss with proper incentives.

    Key improvements:
    - Higher Sharpe weight (risk-adjusted returns matter more)
    - Higher forced exit penalty (avoid bad exits)
    - Position regularization (prevent overfitting via size growth)
    - Spread bonus (wider spreads = better risk/reward)
    """
    # Return loss (negative = maximize)
    return_loss = -result.mean_return

    # Sharpe loss
    sharpe_loss = -result.sharpe

    # Forced exit penalty (these are the bad trades)
    fe_loss = result.forced_exit_rate

    # Position regularization: penalize deviation from target
    # This prevents the model from increasing position size to overfit
    pos_mean = position_size.mean()
    position_loss = (pos_mean - target_position).square()

    # Spread bonus: reward wider spreads (better risk/reward)
    spread = (sell_price - buy_price) / (reference_price + 1e-8)
    spread_mean = spread.mean()
    # Target 3% spread, penalize tighter spreads
    spread_loss = F.relu(0.03 - spread_mean)

    total_loss = (
        return_weight * return_loss
        + sharpe_weight * sharpe_loss
        + forced_exit_penalty * fe_loss
        + position_reg_weight * position_loss
        + spread_bonus_weight * spread_loss
    )

    return total_loss
