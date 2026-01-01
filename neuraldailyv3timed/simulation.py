"""V3 Timed: Trade Episode Simulation with explicit exit timing.

Key differences from V2:
- Simulates individual "trade episodes" instead of continuous inventory
- Each episode: entry at buy_price, exit at sell_price OR exit_days deadline
- Model learns WHEN to exit, not just at what price
- Training matches inference: same exit logic in simulation and live trading

Trade Episode Flow:
1. Day 0: Check if entry fills (low <= buy_price)
2. Days 1 to exit_days: Check if take profit fills (high >= sell_price)
3. Day exit_days: If no TP, force close at market (close price with slippage)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from neuraldailyv3timed.config import SimulationConfig

_EPS = 1e-8
DAILY_PERIODS_PER_YEAR_CRYPTO = 365
DAILY_PERIODS_PER_YEAR_STOCK = 252


@dataclass
class TradeEpisodeResult:
    """Results from trade episode simulation.

    Each prediction is a complete trade: entry -> hold -> exit.
    """

    # Per-trade metrics (batch,)
    net_pnl: torch.Tensor          # Net P&L after fees
    gross_pnl: torch.Tensor        # Gross P&L before fees
    total_fees: torch.Tensor       # Total fees paid
    returns: torch.Tensor          # Return on investment
    entry_filled: torch.Tensor     # Whether entry order filled (bool)
    tp_hit: torch.Tensor           # Whether take profit was hit (bool)
    sl_hit: torch.Tensor           # Whether stop loss was hit (bool)
    forced_exit: torch.Tensor      # Whether forced exit at deadline (bool)
    actual_hold_days: torch.Tensor # Actual days held
    entry_price: torch.Tensor      # Actual entry price
    exit_price: torch.Tensor       # Actual exit price

    # Aggregated metrics
    avg_return: torch.Tensor       # Mean return across batch
    sharpe: torch.Tensor           # Sharpe ratio (simplified)
    tp_rate: torch.Tensor          # Fraction that hit take profit
    sl_rate: torch.Tensor          # Fraction that hit stop loss
    forced_exit_rate: torch.Tensor # Fraction that hit deadline
    avg_hold_days: torch.Tensor    # Average hold duration


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


def simulate_trade_episode(
    *,
    # Future OHLC data: shape (batch, max_hold_days + 1)
    # Day 0 = entry day, Days 1..max_hold_days = hold period
    future_highs: torch.Tensor,
    future_lows: torch.Tensor,
    future_closes: torch.Tensor,

    # Model predictions: shape (batch,)
    buy_price: torch.Tensor,       # Entry limit price
    sell_price: torch.Tensor,      # Take profit price
    exit_days: torch.Tensor,       # Max hold duration (1-10)
    trade_amount: torch.Tensor,    # Position size (fraction of equity)

    # Configuration
    config: SimulationConfig,
    temperature: float = 0.0,
    stop_loss_pct: float = -0.05,  # Stop loss as % from entry (e.g., -0.05 = -5%)
) -> TradeEpisodeResult:
    """
    Simulate a complete trade episode: entry -> hold -> exit.

    This is the core V3 innovation: each model prediction is a complete trade
    with explicit exit timing. The model learns:
    - Entry price (when to buy)
    - Take profit price (when to sell for profit)
    - Exit days (maximum hold duration, force exit if TP not hit)

    Trade Episode Flow:
    1. Day 0: Check if entry fills (low <= buy_price)
    2. Days 1 to ceil(exit_days): Check if TP fills (high >= sell_price)
    3. Day ceil(exit_days): If no TP, force close at market with slippage

    Args:
        future_highs: (batch, max_days+1) - bar highs for days 0 to max_hold_days
        future_lows: (batch, max_days+1) - bar lows
        future_closes: (batch, max_days+1) - bar closes
        buy_price: (batch,) - entry limit price
        sell_price: (batch,) - take profit price
        exit_days: (batch,) - max hold duration (continuous 1-10, will be rounded)
        trade_amount: (batch,) - position size as fraction of equity
        config: Simulation configuration
        temperature: Fill approximation sharpness. 0 = binary, >0 = soft

    Returns:
        TradeEpisodeResult with per-trade and aggregated metrics
    """
    batch_size = buy_price.shape[0]
    max_hold_days = config.max_hold_days
    device = buy_price.device
    dtype = buy_price.dtype

    # Ensure we have enough future data
    assert future_highs.shape[1] >= max_hold_days + 1, \
        f"Need {max_hold_days + 1} days of future data, got {future_highs.shape[1]}"

    # Clamp exit_days to valid range
    exit_days = torch.clamp(exit_days, min=float(config.min_hold_days), max=float(max_hold_days))

    # Reference price for scaling fill probability
    reference_price = future_closes[:, 0]

    # =========== Day 0: Entry ===========
    day0_low = future_lows[:, 0]
    day0_close = future_closes[:, 0]

    # Entry fill probability (differentiable during training)
    entry_prob = _approx_fill_probability(
        buy_price, day0_low, reference_price,
        is_buy=True, temperature=temperature
    )

    # For actual entry price: use limit price if filled, otherwise close
    # (In reality, no fill means no trade, but we need differentiability)
    entry_filled = entry_prob > 0.5 if temperature <= 0 else entry_prob
    actual_entry_price = buy_price  # Fill at limit price

    # Calculate stop-loss price (e.g., entry * 0.95 for -5% stop)
    stop_loss_price = buy_price * (1.0 + stop_loss_pct) if stop_loss_pct < 0 else None

    # =========== Days 1 to max_hold: Check Take Profit and Stop Loss ===========
    # Track if TP or SL has been hit and on which day
    tp_hit = torch.zeros(batch_size, dtype=dtype, device=device)
    sl_hit = torch.zeros(batch_size, dtype=dtype, device=device)
    actual_exit_price = torch.zeros(batch_size, dtype=dtype, device=device)
    actual_hold_days = exit_days.clone()  # Default to exit_days if forced exit

    for day in range(1, max_hold_days + 1):
        # Only check days within the exit window
        # Use soft comparison for differentiability during training
        if temperature > 0:
            # Soft: day_active = sigmoid((exit_days - day) / temp) * (1 - tp_hit - sl_hit)
            day_active = torch.sigmoid((exit_days - day + 0.5) / max(temperature * 5, 0.5))
            day_active = day_active * (1.0 - tp_hit) * (1.0 - sl_hit) * entry_filled
        else:
            # Hard: day_active = (day <= exit_days) & not tp_hit & not sl_hit & entry_filled
            day_active = ((day <= exit_days.ceil()) & (tp_hit < 0.5) & (sl_hit < 0.5) & (entry_filled > 0.5)).float()

        day_high = future_highs[:, day]
        day_low = future_lows[:, day]
        day_close = future_closes[:, day]

        # =========== Check Stop Loss First (happens earlier in day) ===========
        if stop_loss_price is not None:
            sl_prob = _approx_fill_probability(
                stop_loss_price, day_low, reference_price,
                is_buy=True, temperature=temperature  # SL fills when low <= stop_loss_price
            )
            new_sl = day_active * sl_prob
            sl_hit = sl_hit + new_sl * (1.0 - sl_hit)  # Soft OR

            # Update exit price and hold days when SL is hit
            just_hit_sl = new_sl * (1.0 - (sl_hit - new_sl).clamp(0, 1))
            actual_exit_price = actual_exit_price + just_hit_sl * stop_loss_price
            actual_hold_days = actual_hold_days * (1.0 - just_hit_sl) + just_hit_sl * day

            # Update day_active to exclude SL hits for TP check
            day_active = day_active * (1.0 - sl_hit)

        # =========== Check Take Profit ===========
        tp_prob = _approx_fill_probability(
            sell_price, day_high, reference_price,
            is_buy=False, temperature=temperature
        )

        # Update TP status
        new_tp = day_active * tp_prob
        tp_hit = tp_hit + new_tp * (1.0 - tp_hit)  # Soft OR

        # Update exit price and hold days when TP is hit
        # Only update if this is the first TP hit
        just_hit_tp = new_tp * (1.0 - (tp_hit - new_tp).clamp(0, 1))
        actual_exit_price = actual_exit_price + just_hit_tp * sell_price
        actual_hold_days = actual_hold_days * (1.0 - just_hit_tp) + just_hit_tp * day

    # =========== Forced Exit at Deadline ===========
    # For positions that didn't hit TP or SL, exit at market on exit_days
    forced_exit = entry_filled * (1.0 - tp_hit) * (1.0 - sl_hit)

    # Get close price on exit day (with slippage)
    # Use soft indexing for differentiability
    if temperature > 0:
        # Soft: weighted average of closes around exit_days
        exit_day_weights = torch.zeros(batch_size, max_hold_days + 1, dtype=dtype, device=device)
        for day in range(1, max_hold_days + 1):
            weight = torch.exp(-((exit_days - day) ** 2) / max(temperature * 2, 0.1))
            exit_day_weights[:, day] = weight
        exit_day_weights = exit_day_weights / (exit_day_weights.sum(dim=1, keepdim=True) + _EPS)
        forced_exit_price = (future_closes * exit_day_weights).sum(dim=1)
    else:
        # Hard: get close at ceil(exit_days)
        exit_day_idx = exit_days.ceil().long().clamp(1, max_hold_days)
        forced_exit_price = future_closes.gather(1, exit_day_idx.unsqueeze(1)).squeeze(1)

    # Apply slippage to forced exits
    forced_exit_price = forced_exit_price * (1.0 - config.forced_exit_slippage)

    # Combine TP and forced exit prices
    actual_exit_price = actual_exit_price + forced_exit * forced_exit_price

    # =========== Calculate P&L ===========
    # Gross P&L = (exit - entry) * position_size
    price_diff = actual_exit_price - actual_entry_price
    gross_pnl = price_diff * trade_amount * entry_filled

    # Fees: maker fee on entry and exit
    entry_fee = actual_entry_price * trade_amount * config.maker_fee * entry_filled
    exit_fee = actual_exit_price * trade_amount * config.maker_fee * entry_filled
    total_fees = entry_fee + exit_fee

    # Net P&L
    net_pnl = gross_pnl - total_fees

    # Return on investment (relative to entry cost)
    entry_cost = actual_entry_price * trade_amount * entry_filled
    returns = net_pnl / _safe_denominator(entry_cost)

    # =========== Aggregated Metrics ===========
    # Only count trades that actually filled
    n_filled = entry_filled.sum().clamp(min=1.0)

    avg_return = (returns * entry_filled).sum() / n_filled
    return_std = torch.sqrt(((returns - avg_return) ** 2 * entry_filled).sum() / n_filled + _EPS)
    sharpe = avg_return / _safe_denominator(return_std)

    tp_rate = (tp_hit * entry_filled).sum() / n_filled
    sl_rate = (sl_hit * entry_filled).sum() / n_filled
    forced_exit_rate = (forced_exit * entry_filled).sum() / n_filled
    avg_hold_days = (actual_hold_days * entry_filled).sum() / n_filled

    return TradeEpisodeResult(
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        returns=returns,
        entry_filled=entry_filled,
        tp_hit=tp_hit,
        sl_hit=sl_hit,
        forced_exit=forced_exit,
        actual_hold_days=actual_hold_days,
        entry_price=actual_entry_price,
        exit_price=actual_exit_price,
        avg_return=avg_return,
        sharpe=sharpe,
        tp_rate=tp_rate,
        sl_rate=sl_rate,
        forced_exit_rate=forced_exit_rate,
        avg_hold_days=avg_hold_days,
    )


def compute_episode_loss(
    result: TradeEpisodeResult,
    exit_days: torch.Tensor,
    asset_class: torch.Tensor,
    config: SimulationConfig,
    *,
    return_weight: float = 0.08,
    forced_exit_penalty: float = 0.1,
    risk_penalty: float = 0.05,
    hold_time_penalty: float = 0.02,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute training loss from trade episode results.

    Loss components:
    1. Negative return (maximize returns)
    2. Forced exit penalty (encourage hitting TP)
    3. Risk penalty (penalize return variance)
    4. Hold time penalty (small penalty for holding too long)

    Args:
        result: TradeEpisodeResult from simulate_trade_episode
        exit_days: (batch,) - predicted exit days
        asset_class: (batch,) - 0 for stocks, 1 for crypto
        config: Simulation config
        return_weight: Weight for return in objective
        forced_exit_penalty: Penalty for trades that hit deadline
        risk_penalty: Penalty for return variance
        hold_time_penalty: Penalty for long hold times

    Returns:
        (loss, metrics_dict)
    """
    # Base loss: negative average return
    return_loss = -result.avg_return

    # Sharpe-based loss component
    sharpe_loss = -result.sharpe * 0.1

    # Forced exit penalty: discourage trades that don't hit TP
    fe_penalty = forced_exit_penalty * result.forced_exit_rate

    # Risk penalty: penalize high variance
    entry_filled = result.entry_filled
    n_filled = entry_filled.sum().clamp(min=1.0)
    return_var = ((result.returns - result.avg_return) ** 2 * entry_filled).sum() / n_filled
    risk_loss = risk_penalty * return_var

    # Hold time penalty: small penalty for holding longer
    # Normalized by max hold days
    avg_normalized_hold = result.avg_hold_days / config.max_hold_days
    hold_loss = hold_time_penalty * avg_normalized_hold

    # Total loss
    loss = return_loss + sharpe_loss + fe_penalty + risk_loss + hold_loss

    # Metrics for logging
    metrics = {
        "loss": loss.item(),
        "return_loss": return_loss.item(),
        "sharpe_loss": sharpe_loss.item(),
        "forced_exit_penalty": fe_penalty.item(),
        "risk_loss": risk_loss.item(),
        "hold_loss": hold_loss.item(),
        "avg_return": result.avg_return.item(),
        "sharpe": result.sharpe.item(),
        "tp_rate": result.tp_rate.item(),
        "sl_rate": result.sl_rate.item(),
        "forced_exit_rate": result.forced_exit_rate.item(),
        "avg_hold_days": result.avg_hold_days.item(),
        "avg_net_pnl": (result.net_pnl * entry_filled).sum().item() / max(n_filled.item(), 1),
    }

    return loss, metrics


# ============================================================
# Legacy compatibility: Keep the old V2 simulation for reference
# ============================================================

@dataclass
class SimulationResult:
    """Results from V2-style continuous simulation (legacy)."""

    pnl: torch.Tensor
    returns: torch.Tensor
    portfolio_values: torch.Tensor
    cash: torch.Tensor
    inventory: torch.Tensor
    buy_fill_probability: torch.Tensor
    sell_fill_probability: torch.Tensor
    executed_buys: torch.Tensor
    executed_sells: torch.Tensor
    inventory_path: torch.Tensor


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
    Legacy V2-style continuous simulation.

    Kept for compatibility but V3 should use simulate_trade_episode.
    """
    batch_shape = closes.shape[:-1]
    seq_len = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype

    buy_intensity = buy_trade_intensity if buy_trade_intensity is not None else trade_intensity
    sell_intensity = sell_trade_intensity if sell_trade_intensity is not None else trade_intensity

    leverage_limits = torch.where(
        asset_class.unsqueeze(-1) > 0.5,
        torch.tensor(config.crypto_max_leverage, device=device, dtype=dtype),
        torch.tensor(config.equity_max_leverage, device=device, dtype=dtype),
    )
    leverage_limits = leverage_limits.expand_as(closes)

    cash = torch.full(batch_shape, config.initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, config.initial_inventory, dtype=dtype, device=device)
    prev_value = cash + inventory * closes[..., 0]

    pnl_list, return_list, value_list = [], [], []
    buy_prob_list, sell_prob_list = [], []
    exec_buy_list, exec_sell_list = [], []
    inventory_list = []

    start_idx = seq_len - 1 if single_step else 0

    for idx in range(start_idx, seq_len):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        b_price = torch.clamp(buy_prices[..., idx], min=_EPS)
        s_price = torch.clamp(sell_prices[..., idx], min=_EPS)
        step_limit = torch.clamp(leverage_limits[..., idx], min=_EPS)

        b_intensity = torch.clamp(buy_intensity[..., idx], min=0.0)
        b_intensity = torch.minimum(b_intensity, step_limit)
        s_intensity = torch.clamp(sell_intensity[..., idx], min=0.0)
        s_intensity = torch.minimum(s_intensity, step_limit)

        b_frac = b_intensity / step_limit
        s_frac = s_intensity / step_limit

        buy_prob = _approx_fill_probability(b_price, low, close, is_buy=True, temperature=temperature)
        sell_prob = _approx_fill_probability(s_price, high, close, is_buy=False, temperature=temperature)

        equity = cash + inventory * close
        fee_mult = 1.0 + config.maker_fee

        max_buy_cash = torch.where(
            b_price > 0,
            cash / _safe_denominator(b_price * fee_mult),
            torch.zeros_like(cash),
        )

        target_notional = step_limit * torch.clamp(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = torch.where(
            b_price > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * fee_mult),
            torch.zeros_like(cash),
        )

        buy_capacity = torch.where(
            step_limit <= 1.0 + 1e-6,
            torch.clamp(max_buy_cash, min=0.0),
            leveraged_capacity,
        )
        sell_capacity = torch.clamp(inventory, min=0.0)

        buy_qty = b_frac * buy_capacity
        sell_qty = s_frac * sell_capacity
        executed_buys = buy_qty * buy_prob
        executed_sells = sell_qty * sell_prob

        buy_cost = executed_buys * b_price * fee_mult
        sell_revenue = executed_sells * s_price * (1.0 - config.maker_fee)

        cash = cash - buy_cost + sell_revenue
        inventory = inventory + executed_buys - executed_sells

        portfolio_value = cash + inventory * close
        pnl = portfolio_value - prev_value
        returns = pnl / _safe_denominator(prev_value)
        prev_value = portfolio_value

        pnl_list.append(pnl)
        return_list.append(returns)
        value_list.append(portfolio_value)
        buy_prob_list.append(buy_prob)
        sell_prob_list.append(sell_prob)
        exec_buy_list.append(executed_buys)
        exec_sell_list.append(executed_sells)
        inventory_list.append(inventory.clone())

    return SimulationResult(
        pnl=torch.stack(pnl_list, dim=-1),
        returns=torch.stack(return_list, dim=-1),
        portfolio_values=torch.stack(value_list, dim=-1),
        cash=cash,
        inventory=inventory,
        buy_fill_probability=torch.stack(buy_prob_list, dim=-1),
        sell_fill_probability=torch.stack(sell_prob_list, dim=-1),
        executed_buys=torch.stack(exec_buy_list, dim=-1),
        executed_sells=torch.stack(exec_sell_list, dim=-1),
        inventory_path=torch.stack(inventory_list, dim=-1),
    )


def compute_objective(
    returns: torch.Tensor,
    *,
    periods_per_year: float = DAILY_PERIODS_PER_YEAR_STOCK,
    return_weight: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trading objective: Sortino + weighted return (legacy)."""
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
    """Compute training loss from simulation returns (legacy V2)."""
    periods_per_year = DAILY_PERIODS_PER_YEAR_STOCK

    score, sortino, annual_return = compute_objective(
        returns, periods_per_year=periods_per_year, return_weight=return_weight
    )

    loss = -score.mean()

    is_stock = asset_class < 0.5
    stock_inventory = inventory_path * is_stock.unsqueeze(-1)
    leveraged_amount = F.relu(stock_inventory - 1.0)
    daily_leverage_rate = config.leverage_fee_rate / DAILY_PERIODS_PER_YEAR_STOCK
    leverage_cost = (leveraged_amount * daily_leverage_rate).mean()
    loss = loss + leverage_cost

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
