"""V4 Multi-window trade simulation with trimmed mean aggregation.

Key features:
- Simulates trades across multiple future windows
- Aggregates results via trimmed mean for robustness
- Uses median quantiles for fill checking
- Supports confidence-weighted aggregation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from neuraldailyv4.config import SimulationConfigV4


@dataclass
class WindowResult:
    """Result from simulating a single window."""
    entry_filled: torch.Tensor      # (batch,) bool
    tp_hit: torch.Tensor            # (batch,) bool
    forced_exit: torch.Tensor       # (batch,) bool
    returns: torch.Tensor           # (batch,) float
    actual_hold_days: torch.Tensor  # (batch,) float
    confidence: torch.Tensor        # (batch,) float


@dataclass
class MultiWindowResult:
    """Aggregated result from all windows."""
    # Per-window results
    window_returns: torch.Tensor     # (batch, num_windows)
    window_tp_hit: torch.Tensor      # (batch, num_windows)
    window_entry_filled: torch.Tensor  # (batch, num_windows)

    # Aggregated metrics
    aggregated_return: torch.Tensor  # (batch,)
    avg_tp_rate: torch.Tensor        # (batch,)
    avg_hold_days: torch.Tensor      # (batch,)

    # For loss computation
    sharpe: torch.Tensor             # scalar
    forced_exit_rate: torch.Tensor   # scalar
    avg_confidence: torch.Tensor     # scalar


def _soft_fill_prob(
    target_price: torch.Tensor,
    actual_price: torch.Tensor,
    reference_price: torch.Tensor,
    is_buy: bool,
    temperature: float,
) -> torch.Tensor:
    """
    Compute soft fill probability using sigmoid.

    For buys: fills when actual_price <= target_price
    For sells: fills when actual_price >= target_price
    """
    if temperature <= 0:
        # Hard fills
        if is_buy:
            return (actual_price <= target_price).float()
        else:
            return (actual_price >= target_price).float()

    # Soft fills with temperature
    diff = target_price - actual_price
    if not is_buy:
        diff = -diff

    # Normalize by reference price for stability
    normalized_diff = diff / (reference_price * temperature + 1e-8)
    return torch.sigmoid(normalized_diff)


def simulate_window(
    *,
    # Future data for this window
    future_highs: torch.Tensor,    # (batch, window_size)
    future_lows: torch.Tensor,     # (batch, window_size)
    future_closes: torch.Tensor,   # (batch, window_size)
    # Actions for this window (using median quantile)
    buy_price: torch.Tensor,       # (batch,)
    sell_price: torch.Tensor,      # (batch,)
    exit_day: torch.Tensor,        # (batch,) - day within window to exit
    position_size: torch.Tensor,   # (batch,)
    confidence: torch.Tensor,      # (batch,)
    # Config
    reference_price: torch.Tensor, # (batch,)
    config: SimulationConfigV4,
    temperature: float = 0.0,
) -> WindowResult:
    """
    Simulate a single window's trade.

    Args:
        future_highs/lows/closes: Price data for window (day 0 = entry day)
        buy_price: Limit buy price
        sell_price: Take profit price
        exit_day: Day to force exit (1 to window_size)
        position_size: Position size
        confidence: Model confidence
        reference_price: Reference price for normalization
        config: Simulation config
        temperature: Soft fill temperature (0 = hard fills)

    Returns:
        WindowResult with trade outcome
    """
    batch_size = buy_price.size(0)
    window_size = future_highs.size(1)
    device = buy_price.device

    # Day 0: Check entry fill
    day0_low = future_lows[:, 0]
    entry_prob = _soft_fill_prob(buy_price, day0_low, reference_price, is_buy=True, temperature=temperature)

    # Track cumulative TP probability and exit
    cumulative_no_tp = torch.ones(batch_size, device=device)
    exit_day_int = exit_day.round().long().clamp(1, window_size)

    # Weighted exit price (mix of TP and forced)
    weighted_exit_price = torch.zeros(batch_size, device=device)
    tp_probability = torch.zeros(batch_size, device=device)

    for day in range(1, window_size):
        day_high = future_highs[:, day]
        day_close = future_closes[:, day]

        # Check if still in window (before exit_day)
        still_holding = (day < exit_day_int).float()

        # TP check
        tp_prob_today = _soft_fill_prob(
            sell_price, day_high, reference_price,
            is_buy=False, temperature=temperature
        )

        # Probability of TP on this specific day
        tp_this_day = cumulative_no_tp * tp_prob_today * still_holding

        # Accumulate TP probability and weighted price
        tp_probability = tp_probability + tp_this_day
        weighted_exit_price = weighted_exit_price + tp_this_day * sell_price

        # Update cumulative no-TP (for days still holding)
        cumulative_no_tp = cumulative_no_tp * (1 - tp_prob_today * still_holding)

        # Forced exit on exit_day
        is_exit_day = (day == exit_day_int).float()
        forced_exit_today = cumulative_no_tp * is_exit_day * entry_prob
        forced_exit_price = day_close * (1 - config.forced_exit_slippage)

        weighted_exit_price = weighted_exit_price + forced_exit_today * forced_exit_price
        cumulative_no_tp = cumulative_no_tp * (1 - is_exit_day)

    # Handle case where exit_day is at end of window
    final_forced = cumulative_no_tp * entry_prob
    final_close = future_closes[:, -1] * (1 - config.forced_exit_slippage)
    weighted_exit_price = weighted_exit_price + final_forced * final_close
    tp_probability = tp_probability.clamp(0, 1)

    # Compute returns
    gross_return = (weighted_exit_price - buy_price) / (buy_price + 1e-8)
    fees = 2 * config.maker_fee  # Entry + exit fees
    net_return = (gross_return - fees) * entry_prob * position_size

    # Actual hold days (weighted by exit probabilities)
    actual_hold_days = exit_day * entry_prob

    return WindowResult(
        entry_filled=entry_prob,
        tp_hit=tp_probability,
        forced_exit=(1 - tp_probability) * entry_prob,
        returns=net_return,
        actual_hold_days=actual_hold_days,
        confidence=confidence,
    )


def trimmed_mean(
    values: torch.Tensor,
    trim_fraction: float = 0.25,
    weights: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute trimmed mean along a dimension.

    Args:
        values: Input tensor
        trim_fraction: Fraction to trim from each end (0.25 = trim top/bottom 25%)
        weights: Optional weights for weighted trimmed mean
        dim: Dimension to compute along

    Returns:
        Trimmed mean tensor
    """
    n = values.size(dim)
    trim_n = int(n * trim_fraction)

    if trim_n <= 0 or n - 2 * trim_n <= 0:
        # No trimming possible, return regular mean
        if weights is not None:
            return (values * weights).sum(dim=dim) / (weights.sum(dim=dim) + 1e-8)
        return values.mean(dim=dim)

    # Sort values
    sorted_vals, sorted_idx = torch.sort(values, dim=dim)

    # Trim
    if dim == -1:
        trimmed = sorted_vals[..., trim_n:-trim_n]
        if weights is not None:
            # Gather weights in sorted order
            sorted_weights = torch.gather(weights, dim, sorted_idx)
            trimmed_weights = sorted_weights[..., trim_n:-trim_n]
            return (trimmed * trimmed_weights).sum(dim=dim) / (trimmed_weights.sum(dim=dim) + 1e-8)
    else:
        trimmed = torch.narrow(sorted_vals, dim, trim_n, n - 2 * trim_n)
        if weights is not None:
            sorted_weights = torch.gather(weights, dim, sorted_idx)
            trimmed_weights = torch.narrow(sorted_weights, dim, trim_n, n - 2 * trim_n)
            return (trimmed * trimmed_weights).sum(dim=dim) / (trimmed_weights.sum(dim=dim) + 1e-8)

    return trimmed.mean(dim=dim)


def simulate_multi_window(
    *,
    # Future data for all windows
    future_highs: torch.Tensor,     # (batch, lookahead_days)
    future_lows: torch.Tensor,      # (batch, lookahead_days)
    future_closes: torch.Tensor,    # (batch, lookahead_days)
    # Decoded actions
    buy_quantiles: torch.Tensor,    # (batch, num_windows, num_quantiles)
    sell_quantiles: torch.Tensor,   # (batch, num_windows, num_quantiles)
    position_size: torch.Tensor,    # (batch, num_windows)
    confidence: torch.Tensor,       # (batch, num_windows)
    exit_days: torch.Tensor,        # (batch, num_windows) - absolute days
    # Config
    reference_price: torch.Tensor,  # (batch,)
    config: SimulationConfigV4,
    temperature: float = 0.0,
) -> MultiWindowResult:
    """
    Simulate trades across all windows and aggregate.

    Uses median quantile for actual trading decisions.
    Aggregates results via trimmed mean.
    """
    batch_size = buy_quantiles.size(0)
    num_windows = config.num_windows
    window_size = config.window_size
    device = buy_quantiles.device

    # Use median quantile for trading
    median_idx = buy_quantiles.size(-1) // 2
    buy_prices = buy_quantiles[..., median_idx]   # (batch, num_windows)
    sell_prices = sell_quantiles[..., median_idx]

    # Simulate each window
    window_results = []
    for w in range(num_windows):
        # Extract window's future data
        start_day = w * window_size
        end_day = min((w + 1) * window_size, future_highs.size(1))

        if start_day >= future_highs.size(1):
            # No data for this window
            dummy = WindowResult(
                entry_filled=torch.zeros(batch_size, device=device),
                tp_hit=torch.zeros(batch_size, device=device),
                forced_exit=torch.zeros(batch_size, device=device),
                returns=torch.zeros(batch_size, device=device),
                actual_hold_days=torch.zeros(batch_size, device=device),
                confidence=confidence[:, w],
            )
            window_results.append(dummy)
            continue

        window_highs = future_highs[:, start_day:end_day]
        window_lows = future_lows[:, start_day:end_day]
        window_closes = future_closes[:, start_day:end_day]

        # Pad if needed
        actual_size = window_highs.size(1)
        if actual_size < window_size:
            pad_size = window_size - actual_size
            window_highs = F.pad(window_highs, (0, pad_size), value=window_highs[:, -1:].mean())
            window_lows = F.pad(window_lows, (0, pad_size), value=window_lows[:, -1:].mean())
            window_closes = F.pad(window_closes, (0, pad_size), value=window_closes[:, -1:].mean())

        # Exit day within window (1 to window_size)
        exit_day_in_window = exit_days[:, w] - (w * window_size)
        exit_day_in_window = exit_day_in_window.clamp(1, window_size)

        result = simulate_window(
            future_highs=window_highs,
            future_lows=window_lows,
            future_closes=window_closes,
            buy_price=buy_prices[:, w],
            sell_price=sell_prices[:, w],
            exit_day=exit_day_in_window,
            position_size=position_size[:, w],
            confidence=confidence[:, w],
            reference_price=reference_price,
            config=config,
            temperature=temperature,
        )
        window_results.append(result)

    # Stack results
    window_returns = torch.stack([r.returns for r in window_results], dim=1)  # (batch, num_windows)
    window_tp_hit = torch.stack([r.tp_hit for r in window_results], dim=1)
    window_entry_filled = torch.stack([r.entry_filled for r in window_results], dim=1)
    window_confidence = torch.stack([r.confidence for r in window_results], dim=1)
    window_hold_days = torch.stack([r.actual_hold_days for r in window_results], dim=1)
    window_forced_exit = torch.stack([r.forced_exit for r in window_results], dim=1)

    # Aggregate via trimmed mean
    if config.use_confidence_weighting:
        weights = window_confidence
    else:
        weights = None

    aggregated_return = trimmed_mean(window_returns, config.trim_fraction, weights)
    avg_tp_rate = trimmed_mean(window_tp_hit, config.trim_fraction, weights)
    avg_hold_days = trimmed_mean(window_hold_days, config.trim_fraction, weights)

    # Compute Sharpe ratio
    mean_return = aggregated_return.mean()
    std_return = aggregated_return.std() + 1e-8
    sharpe = mean_return / std_return

    # Forced exit rate
    forced_exit_rate = window_forced_exit.mean()

    return MultiWindowResult(
        window_returns=window_returns,
        window_tp_hit=window_tp_hit,
        window_entry_filled=window_entry_filled,
        aggregated_return=aggregated_return,
        avg_tp_rate=avg_tp_rate,
        avg_hold_days=avg_hold_days,
        sharpe=sharpe,
        forced_exit_rate=forced_exit_rate,
        avg_confidence=window_confidence.mean(),
    )


def compute_v4_loss(
    result: MultiWindowResult,
    buy_quantiles: torch.Tensor,
    sell_quantiles: torch.Tensor,
    future_lows: torch.Tensor,
    future_highs: torch.Tensor,
    quantile_levels: tuple,
    config: SimulationConfigV4,
    *,
    return_weight: float = 1.0,
    sharpe_weight: float = 0.1,
    forced_exit_penalty: float = 0.1,
    quantile_calibration_weight: float = 0.05,
    position_regularization: float = 0.01,
    quantile_ordering_weight: float = 0.0,
    exit_days_penalty_weight: float = 0.0,
    exit_days: Optional[torch.Tensor] = None,
    position_size: Optional[torch.Tensor] = None,
    utilization_loss_weight: float = 0.0,
    utilization_target: float = 0.5,
) -> torch.Tensor:
    """
    Compute V4 loss with multi-window aggregation.

    Loss components:
    1. Return loss: Maximize aggregated returns
    2. Sharpe loss: Risk-adjusted returns
    3. Forced exit penalty: Discourage deadline exits
    4. Quantile calibration: Ensure uncertainty is calibrated
    5. Position regularization: Prevent extreme positions
    6. Quantile ordering: Enforce q_low < q_mid < q_high (V7+)
    7. Exit days penalty: Prefer shorter holds (V7+)
    """
    # Return loss (negative because we maximize)
    return_loss = -result.aggregated_return.mean()

    # Sharpe loss
    sharpe_loss = -result.sharpe

    # Forced exit penalty
    fe_loss = result.forced_exit_rate

    # Quantile calibration loss
    # Check if actual prices fall within quantile ranges as expected
    quantile_loss = _quantile_calibration_loss(
        buy_quantiles, sell_quantiles,
        future_lows, future_highs,
        quantile_levels
    )

    # Position regularization (encourage larger positions, target 50%)
    # Bug fix: Use actual position_size instead of avg_confidence
    if position_size is not None:
        pos_mean = position_size.mean()
    else:
        # Fallback to avg_confidence if position_size not provided
        pos_mean = result.avg_confidence
    position_loss = (pos_mean - 0.5).square()

    # Portfolio utilization penalty (encourage using portfolio capacity)
    utilization_loss = torch.tensor(0.0, device=buy_quantiles.device)
    if utilization_loss_weight > 0 and position_size is not None:
        # Sum of position sizes per sample (batch, num_windows) -> (batch,)
        daily_utilization = position_size.sum(dim=-1) if position_size.dim() > 1 else position_size
        avg_utilization = daily_utilization.mean()
        # Penalize deviation from target utilization
        utilization_loss = (avg_utilization - utilization_target).square()

    # Quantile ordering loss (enforce monotonicity: q10 < q50 < q90)
    ordering_loss = torch.tensor(0.0, device=buy_quantiles.device)
    if quantile_ordering_weight > 0 and buy_quantiles.size(-1) >= 2:
        ordering_loss = _quantile_ordering_loss(buy_quantiles, sell_quantiles)

    # Exit days penalty (prefer shorter holds, optimal around 2 days)
    exit_loss = torch.tensor(0.0, device=buy_quantiles.device)
    if exit_days_penalty_weight > 0 and exit_days is not None:
        # Penalize exit_days > 2 (our empirically optimal hold period)
        optimal_exit = 2.0
        exit_deviation = F.relu(exit_days.mean() - optimal_exit)
        exit_loss = exit_deviation

    # Total loss
    total_loss = (
        return_weight * return_loss
        + sharpe_weight * sharpe_loss
        + forced_exit_penalty * fe_loss
        + quantile_calibration_weight * quantile_loss
        + position_regularization * position_loss
        + quantile_ordering_weight * ordering_loss
        + exit_days_penalty_weight * exit_loss
        + utilization_loss_weight * utilization_loss
    )

    return total_loss


def _quantile_ordering_loss(
    buy_quantiles: torch.Tensor,
    sell_quantiles: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce monotonicity: lower quantiles should have lower values.

    For buy_quantiles: q10 < q50 < q90 (lower bound estimates)
    For sell_quantiles: q10 < q50 < q90 (upper bound estimates)

    Returns margin violation loss.
    """
    # buy_quantiles: (batch, num_windows, num_quantiles) where quantiles are [q25, q50, q75]
    # For proper ordering: q25 <= q50 <= q75

    violations = []
    num_q = buy_quantiles.size(-1)

    for i in range(num_q - 1):
        # buy: q[i] should be <= q[i+1]
        buy_violation = F.relu(buy_quantiles[..., i] - buy_quantiles[..., i + 1])
        violations.append(buy_violation.mean())

        # sell: q[i] should be <= q[i+1]
        sell_violation = F.relu(sell_quantiles[..., i] - sell_quantiles[..., i + 1])
        violations.append(sell_violation.mean())

    return sum(violations) / len(violations) if violations else torch.tensor(0.0)


def _quantile_calibration_loss(
    buy_quantiles: torch.Tensor,
    sell_quantiles: torch.Tensor,
    future_lows: torch.Tensor,
    future_highs: torch.Tensor,
    quantile_levels: tuple,
) -> torch.Tensor:
    """
    Compute quantile calibration loss.

    For each quantile level q, approximately q fraction of actual values
    should fall below the predicted quantile.
    """
    losses = []

    # Use first day low/high for calibration check
    actual_low = future_lows[:, 0]   # (batch,)
    actual_high = future_highs[:, 0]

    for i, q in enumerate(quantile_levels):
        # Buy quantiles: fraction of lows below prediction should be ~q
        buy_q = buy_quantiles[:, 0, i]  # First window, i-th quantile
        fraction_below = (actual_low < buy_q).float().mean()
        losses.append((fraction_below - q).abs())

        # Sell quantiles: fraction of highs below prediction should be ~(1-q)
        # (because sell_q is an upper bound)
        sell_q = sell_quantiles[:, 0, i]
        fraction_below_sell = (actual_high < sell_q).float().mean()
        losses.append((fraction_below_sell - (1 - q)).abs())

    return sum(losses) / len(losses) if losses else torch.tensor(0.0)
