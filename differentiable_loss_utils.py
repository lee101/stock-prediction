"""Differentiable trading PnL utilities for hourly crypto training.

The helpers in this module mirror ``loss_utils.py`` but add support for
variable buy/sell amounts, maker-fee aware fills, and 1x leverage constraints
while remaining fully differentiable for neural training. All calculations are
vectorised across an arbitrary batch dimension with the final axis representing
time steps (hours).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from src.fixtures import all_crypto_symbols

DEFAULT_MAKER_FEE_RATE = 0.001  # 10 bps maker fee (Alpaca crypto current rate)
HOURLY_PERIODS_PER_YEAR = 24 * 365  # 8760 hours
DAILY_PERIODS_PER_YEAR_CRYPTO = 365  # Crypto trades 24/7
DAILY_PERIODS_PER_YEAR_STOCK = 252   # ~252 trading days/year for stocks
_EPS = 1e-8

def _compile_enabled() -> bool:
    if os.environ.get("TORCH_FORCE_COMPILE", ""):
        return True
    if os.environ.get("TORCH_NO_COMPILE", ""):
        return False
    if not hasattr(torch, "compile"):
        return False
    try:
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(0)
            # Blackwell / sm120 currently trips Triton/Inductor on these tiny
            # reduction kernels, so prefer the stable eager path there.
            if int(major) >= 12:
                return False
    except Exception:
        pass
    return True


_COMPILE_ENABLED = _compile_enabled()


def _maybe_compile(fn=None, **kwargs):
    """Conditionally apply torch.compile; respects TORCH_NO_COMPILE env var."""
    if fn is None:
        return lambda f: _maybe_compile(f, **kwargs)
    if _COMPILE_ENABLED and hasattr(torch, "compile"):
        return torch.compile(fn, **kwargs)
    return fn


def get_periods_per_year(frequency: str = "hourly", symbol: str = "") -> float:
    """Get annualization factor based on data frequency and symbol type.

    Args:
        frequency: "hourly" or "daily"
        symbol: Symbol name (e.g., "BTCUSD", "AAPL")

    Returns:
        Number of periods per year for annualization
    """
    if frequency == "hourly":
        return HOURLY_PERIODS_PER_YEAR
    elif frequency == "daily":
        is_crypto = symbol.upper() in all_crypto_symbols
        return DAILY_PERIODS_PER_YEAR_CRYPTO if is_crypto else DAILY_PERIODS_PER_YEAR_STOCK
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def _as_tensor(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):  # pragma: no cover - trivial branch
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _check_shapes(reference: torch.Tensor, *others: torch.Tensor) -> None:
    for tensor in others:
        if tensor.shape != reference.shape:
            raise ValueError("All tensors must share the same shape for simulation")


def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value, min=_EPS)


def approx_buy_fill_probability(
    buy_price: torch.Tensor,
    low_price: torch.Tensor,
    close_price: torch.Tensor,
    *,
    temperature: float = 5e-4,
) -> torch.Tensor:
    """Smooth approximation of whether a buy limit order hits within the bar.

    Args:
        buy_price: Limit price suggested by the policy
        low_price: Observed bar low
        close_price: Observed close for scaling
        temperature: Controls the steepness of the fill sigmoid; interpreted as a
            fraction of the close price (default 5 bps)
    """

    _check_shapes(buy_price, low_price, close_price)
    scale = torch.clamp(close_price.abs(), min=1e-4)
    temp = _as_tensor(temperature, close_price)
    score = (buy_price - low_price) / _safe_denominator(scale * temp)
    return torch.sigmoid(score)


def approx_sell_fill_probability(
    sell_price: torch.Tensor,
    high_price: torch.Tensor,
    close_price: torch.Tensor,
    *,
    temperature: float = 5e-4,
) -> torch.Tensor:
    """Smooth approximation of whether a sell limit order hits within the bar."""

    _check_shapes(sell_price, high_price, close_price)
    scale = torch.clamp(close_price.abs(), min=1e-4)
    temp = _as_tensor(temperature, close_price)
    score = (high_price - sell_price) / _safe_denominator(scale * temp)
    return torch.sigmoid(score)


@dataclass
class HourlySimulationResult:
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


def simulate_hourly_trades(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    opens: torch.Tensor | None = None,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    trade_intensity: torch.Tensor,
    buy_trade_intensity: torch.Tensor | None = None,
    sell_trade_intensity: torch.Tensor | None = None,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    temperature: float = 5e-4,
    max_leverage: float | torch.Tensor = 1.0,
    can_short: bool | float | torch.Tensor = False,
    can_long: bool | float | torch.Tensor = True,
    decision_lag_bars: int = 0,
    market_order_entry: bool = False,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
) -> HourlySimulationResult:
    """Simulate hourly fills with differentiable maker execution logic.

    All tensors must share shape ``(..., T)`` where ``T`` is the number of
    hours. ``max_leverage`` can be a scalar or tensor broadcastable to the same
    shape and controls the per-asset leverage cap. When greater than 1 it
    allows borrowing up to the specified multiple of equity; when less than 1
    it throttles position sizes.
    """

    _check_shapes(highs, lows, closes, buy_prices, sell_prices, trade_intensity)
    if closes.ndim == 0:
        raise ValueError("Input tensors must include a time dimension")

    buy_intensity = buy_trade_intensity if buy_trade_intensity is not None else trade_intensity
    sell_intensity = sell_trade_intensity if sell_trade_intensity is not None else trade_intensity
    _check_shapes(highs, buy_intensity, sell_intensity)

    if opens is not None:
        _check_shapes(highs, opens)

    original_steps = closes.shape[-1]
    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        if opens is not None:
            opens = opens[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
        trade_intensity = trade_intensity[..., :-lag]
        buy_intensity = buy_intensity[..., :-lag]
        sell_intensity = sell_intensity[..., :-lag]
        if torch.is_tensor(max_leverage) and max_leverage.ndim > 0 and max_leverage.shape[-1] == original_steps:
            max_leverage = max_leverage[..., lag:]

    margin_cost_per_step = margin_annual_rate / HOURLY_PERIODS_PER_YEAR

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    temperature_tensor = torch.as_tensor(temperature, dtype=dtype, device=device)
    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)
    max_leverage_tensor = _as_tensor(max_leverage, closes).expand(closes.shape)
    can_short_tensor = _as_tensor(can_short, closes).expand(batch_shape)
    can_long_tensor = _as_tensor(can_long, closes).expand(batch_shape)
    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    prev_value = cash + inventory * closes[..., 0]

    fee_buy = 1.0 + fee
    fee_sell = 1.0 - fee

    pnl_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    returns_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    values_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    buy_prob_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    sell_prob_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_buy_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_sell_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    inventory_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)

    for idx in range(steps):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        b_price = torch.clamp(buy_prices[..., idx], min=_EPS)
        s_price = torch.clamp(sell_prices[..., idx], min=_EPS)
        if market_order_entry and opens is not None:
            b_price = torch.clamp(opens[..., idx], min=_EPS)
        step_limit = torch.clamp(max_leverage_tensor[..., idx], min=_EPS)
        b_intensity = torch.minimum(torch.clamp(buy_intensity[..., idx], min=0.0), step_limit)
        s_intensity = torch.minimum(torch.clamp(sell_intensity[..., idx], min=0.0), step_limit)
        b_frac_limit = b_intensity / torch.clamp(step_limit, min=_EPS)
        s_frac_limit = s_intensity / torch.clamp(step_limit, min=_EPS)

        if market_order_entry:
            buy_prob = torch.ones_like(b_price)
        else:
            buy_threshold = b_price * (1.0 - fill_buffer_pct) if fill_buffer_pct > 0 else b_price
            buy_prob = approx_buy_fill_probability(buy_threshold, low, close, temperature=float(temperature_tensor))
        sell_threshold = s_price * (1.0 + fill_buffer_pct) if fill_buffer_pct > 0 else s_price
        sell_prob = approx_sell_fill_probability(sell_threshold, high, close, temperature=float(temperature_tensor))

        equity = cash + inventory * close
        # Allow borrowing up to ``step_limit`` * equity; translated to units.
        max_buy_cash = torch.where(
            b_price > 0,
            cash / _safe_denominator(b_price * fee_buy),
            0.0,
        )

        target_notional = step_limit * torch.clamp(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = torch.where(
            b_price > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * fee_buy),
            0.0,
        )

        buy_capacity = torch.where(step_limit <= 1.0 + 1e-6, torch.clamp(max_buy_cash, min=0.0), leveraged_capacity)
        buy_qty = b_frac_limit * buy_capacity
        if can_long_tensor.numel() > 0:
            cover_only_cap = torch.clamp(-inventory, min=0.0)
            buy_qty = torch.where(
                can_long_tensor > 0.5,
                buy_qty,
                torch.minimum(buy_qty, cover_only_cap),
            )

        long_to_close = torch.clamp(inventory, min=0.0)
        max_short_qty = torch.where(
            s_price > 0,
            (step_limit * torch.clamp(equity, min=_EPS)) / _safe_denominator(s_price * fee_buy),
            0.0,
        )
        current_short_qty = torch.clamp(-inventory, min=0.0)
        short_open_cap = torch.clamp(max_short_qty - current_short_qty, min=0.0)
        sell_capacity = long_to_close + torch.where(can_short_tensor > 0.5, short_open_cap, 0.0)
        sell_qty = s_frac_limit * sell_capacity

        executed_buys = buy_qty * buy_prob
        executed_sells = sell_qty * sell_prob

        cash = cash - executed_buys * b_price * fee_buy + executed_sells * s_price * fee_sell
        inventory = inventory + executed_buys - executed_sells

        if margin_cost_per_step > 0:
            pos_value = torch.abs(inventory * close)
            eq = cash + inventory * close
            margin_used = torch.clamp(pos_value - torch.clamp(eq, min=0.0), min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inventory * close

        prior_value = prev_value
        pnl = portfolio_value - prior_value
        returns = pnl / _safe_denominator(prior_value)
        prev_value = portfolio_value

        pnl_out[..., idx] = pnl
        returns_out[..., idx] = returns
        values_out[..., idx] = portfolio_value
        buy_prob_out[..., idx] = buy_prob
        sell_prob_out[..., idx] = sell_prob
        exec_buy_out[..., idx] = executed_buys
        exec_sell_out[..., idx] = executed_sells
        inventory_out[..., idx] = inventory

    return HourlySimulationResult(
        pnl=pnl_out,
        returns=returns_out,
        portfolio_values=values_out,
        cash=cash,
        inventory=inventory,
        buy_fill_probability=buy_prob_out,
        sell_fill_probability=sell_prob_out,
        executed_buys=exec_buy_out,
        executed_sells=exec_sell_out,
        inventory_path=inventory_out,
    )


def simulate_hourly_trades_binary(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    opens: torch.Tensor | None = None,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    trade_intensity: torch.Tensor,
    buy_trade_intensity: torch.Tensor | None = None,
    sell_trade_intensity: torch.Tensor | None = None,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    max_leverage: float | torch.Tensor = 1.0,
    can_short: bool | float | torch.Tensor = False,
    can_long: bool | float | torch.Tensor = True,
    decision_lag_bars: int = 0,
    market_order_entry: bool = False,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
) -> HourlySimulationResult:
    """Binary fill simulation (100% or 0%) for realistic backtesting.

    Unlike simulate_hourly_trades which uses probabilistic fills,
    this uses binary all-or-nothing fills when price touches limit.
    Used for validation metrics to track real-world performance.
    """
    _check_shapes(highs, lows, closes, buy_prices, sell_prices, trade_intensity)
    if closes.ndim == 0:
        raise ValueError("Input tensors must include a time dimension")

    buy_intensity = buy_trade_intensity if buy_trade_intensity is not None else trade_intensity
    sell_intensity = sell_trade_intensity if sell_trade_intensity is not None else trade_intensity
    _check_shapes(highs, buy_intensity, sell_intensity)

    if opens is not None:
        _check_shapes(highs, opens)

    original_steps = closes.shape[-1]
    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        if opens is not None:
            opens = opens[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
        trade_intensity = trade_intensity[..., :-lag]
        buy_intensity = buy_intensity[..., :-lag]
        sell_intensity = sell_intensity[..., :-lag]
        if torch.is_tensor(max_leverage) and max_leverage.ndim > 0 and max_leverage.shape[-1] == original_steps:
            max_leverage = max_leverage[..., lag:]

    margin_cost_per_step = margin_annual_rate / HOURLY_PERIODS_PER_YEAR

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)
    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    max_leverage_tensor = _as_tensor(max_leverage, closes).expand(closes.shape)
    can_short_tensor = _as_tensor(can_short, closes).expand(batch_shape)
    can_long_tensor = _as_tensor(can_long, closes).expand(batch_shape)
    prev_value = cash + inventory * closes[..., 0]

    fee_buy = 1.0 + fee
    fee_sell = 1.0 - fee

    pnl_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    returns_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    values_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    buy_fill_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    sell_fill_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_buy_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_sell_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    inventory_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)

    for idx in range(steps):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        b_price = torch.clamp(buy_prices[..., idx], min=_EPS)
        s_price = torch.clamp(sell_prices[..., idx], min=_EPS)
        if market_order_entry and opens is not None:
            b_price = torch.clamp(opens[..., idx], min=_EPS)
        step_limit = torch.clamp(max_leverage_tensor[..., idx], min=_EPS)
        b_intensity = torch.minimum(torch.clamp(buy_intensity[..., idx], min=0.0), step_limit)
        s_intensity = torch.minimum(torch.clamp(sell_intensity[..., idx], min=0.0), step_limit)
        b_frac_limit = b_intensity / torch.clamp(step_limit, min=_EPS)
        s_frac_limit = s_intensity / torch.clamp(step_limit, min=_EPS)

        if market_order_entry:
            buy_fill = b_intensity > 0
        else:
            buy_threshold = b_price * (1.0 - fill_buffer_pct) if fill_buffer_pct > 0 else b_price
            buy_fill = (low <= buy_threshold) & (b_intensity > 0)
        sell_threshold = s_price * (1.0 + fill_buffer_pct) if fill_buffer_pct > 0 else s_price
        sell_fill = (high >= sell_threshold) & (s_intensity > 0)

        equity = cash + inventory * close
        max_buy_cash = torch.where(
            b_price > 0,
            cash / _safe_denominator(b_price * fee_buy),
            0.0,
        )
        target_notional = step_limit * torch.clamp(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = torch.where(
            b_price > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * fee_buy),
            0.0,
        )
        buy_capacity = torch.where(step_limit <= 1.0 + 1e-6, torch.clamp(max_buy_cash, min=0.0), leveraged_capacity)

        buy_qty = torch.where(buy_fill, b_frac_limit * buy_capacity, 0.0)
        cover_only_cap = torch.clamp(-inventory, min=0.0)
        buy_qty = torch.where(
            can_long_tensor > 0.5,
            buy_qty,
            torch.minimum(buy_qty, cover_only_cap),
        )

        long_to_close = torch.clamp(inventory, min=0.0)
        max_short_qty = torch.where(
            s_price > 0,
            (step_limit * torch.clamp(equity, min=_EPS)) / _safe_denominator(s_price * fee_buy),
            0.0,
        )
        current_short_qty = torch.clamp(-inventory, min=0.0)
        short_open_cap = torch.clamp(max_short_qty - current_short_qty, min=0.0)
        sell_capacity = long_to_close + torch.where(can_short_tensor > 0.5, short_open_cap, 0.0)
        sell_qty = torch.where(sell_fill, s_frac_limit * sell_capacity, 0.0)

        executed_buys = buy_qty
        executed_sells = sell_qty

        cash = cash - executed_buys * b_price * fee_buy + executed_sells * s_price * fee_sell
        inventory = inventory + executed_buys - executed_sells

        if margin_cost_per_step > 0:
            pos_value = torch.abs(inventory * close)
            eq = cash + inventory * close
            margin_used = torch.clamp(pos_value - torch.clamp(eq, min=0.0), min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inventory * close

        prior_value = prev_value
        pnl = portfolio_value - prior_value
        returns = pnl / _safe_denominator(prior_value)
        prev_value = portfolio_value

        pnl_out[..., idx] = pnl
        returns_out[..., idx] = returns
        values_out[..., idx] = portfolio_value
        buy_fill_out[..., idx] = buy_fill.float()
        sell_fill_out[..., idx] = sell_fill.float()
        exec_buy_out[..., idx] = executed_buys
        exec_sell_out[..., idx] = executed_sells
        inventory_out[..., idx] = inventory

    return HourlySimulationResult(
        pnl=pnl_out,
        returns=returns_out,
        portfolio_values=values_out,
        cash=cash,
        inventory=inventory,
        buy_fill_probability=buy_fill_out,
        sell_fill_probability=sell_fill_out,
        executed_buys=exec_buy_out,
        executed_sells=exec_sell_out,
        inventory_path=inventory_out,
    )


def _has_smoothness(penalty: float | torch.Tensor) -> bool:
    if isinstance(penalty, (int, float)):
        return penalty != 0.0
    return bool(penalty.any())


def _apply_smoothness(loss: torch.Tensor, hourly_returns: torch.Tensor, penalty: float | torch.Tensor) -> torch.Tensor:
    if not _has_smoothness(penalty) or hourly_returns.shape[-1] <= 1:
        return loss
    weight = _as_tensor(penalty, hourly_returns)
    diffs = hourly_returns[..., 1:] - hourly_returns[..., :-1]
    return loss + (weight * diffs.abs().mean(dim=-1)).mean()


@_maybe_compile(mode="reduce-overhead")
def _sortino_core(hourly_returns: torch.Tensor, periods: torch.Tensor, return_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused sortino core: mean -> downside_std -> sortino -> score."""
    mean_return = hourly_returns.mean(dim=-1)
    downside_sq = torch.clamp(-hourly_returns, min=0.0).square()
    downside_std = (downside_sq.mean(dim=-1) + _EPS).sqrt()
    sortino = mean_return / downside_std.clamp(min=_EPS)
    sortino = sortino * periods.clamp(min=_EPS).sqrt()
    annual_return = mean_return * periods
    score = sortino + return_weight * annual_return
    return score, sortino, annual_return


def compute_hourly_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    smoothness_penalty: float | torch.Tensor = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (score, sortino, annual_return) for hourly return series."""
    if hourly_returns.ndim == 0:
        raise ValueError("hourly_returns must have at least one dimension")
    periods = _as_tensor(periods_per_year, hourly_returns)
    score, sortino, annual_return = _sortino_core(hourly_returns, periods, return_weight)
    if _has_smoothness(smoothness_penalty) and hourly_returns.shape[-1] > 1:
        returns_diff = hourly_returns[..., 1:] - hourly_returns[..., :-1]
        score = score - returns_diff.std(dim=-1) * smoothness_penalty
    return score, sortino, annual_return


def combined_sortino_pnl_loss(
    hourly_returns: torch.Tensor,
    *,
    target_sign: float = 1.0,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    smoothness_penalty: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Loss function that encourages joint Sortino + return maximisation."""

    score, _, _ = compute_hourly_objective(
        hourly_returns,
        periods_per_year=periods_per_year,
        return_weight=return_weight,
    )
    loss = -target_sign * score.mean()
    return _apply_smoothness(loss, hourly_returns, smoothness_penalty)


@_maybe_compile(mode="reduce-overhead")
def compute_sharpe_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharpe ratio objective: penalizes ALL volatility, not just downside."""
    mean_return = hourly_returns.mean(dim=-1)
    total_std = torch.sqrt(torch.mean(hourly_returns**2, dim=-1) - mean_return**2 + _EPS)
    sharpe = mean_return / _safe_denominator(total_std)
    periods = _as_tensor(periods_per_year, mean_return)
    sharpe = sharpe * torch.sqrt(torch.clamp(periods, min=_EPS))
    annual_return = mean_return * periods
    score = sharpe + return_weight * annual_return
    return score, sharpe, annual_return


@_maybe_compile(mode="reduce-overhead")
def compute_calmar_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calmar ratio objective: annual_return / max_drawdown."""
    mean_return = hourly_returns.mean(dim=-1)
    periods = _as_tensor(periods_per_year, mean_return)
    annual_return = mean_return * periods
    cumulative = torch.cumsum(hourly_returns, dim=-1)
    running_max = torch.cummax(cumulative, dim=-1).values
    drawdowns = running_max - cumulative
    max_dd = drawdowns.max(dim=-1).values + _EPS
    calmar = annual_return / max_dd
    score = calmar + return_weight * annual_return
    return score, calmar, annual_return


def compute_log_wealth_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kelly-criterion inspired: maximize E[log(1+r)]. Naturally risk-averse."""
    log_returns = torch.log1p(hourly_returns.clamp(min=-0.99))
    mean_log_return = log_returns.mean(dim=-1)
    periods = _as_tensor(periods_per_year, mean_log_return)
    annual_log_return = mean_log_return * periods
    mean_return = hourly_returns.mean(dim=-1)
    annual_return = mean_return * periods
    score = annual_log_return + return_weight * annual_return
    return score, annual_log_return, annual_return


def compute_sortino_dd_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    dd_penalty: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sortino + drawdown penalty: penalizes max drawdown directly."""
    score, sortino, annual_return = compute_hourly_objective(
        hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
    )
    cumulative = torch.cumsum(hourly_returns, dim=-1)
    running_max = torch.cummax(cumulative, dim=-1).values
    drawdowns = running_max - cumulative
    max_dd = drawdowns.max(dim=-1).values
    score = score - dd_penalty * max_dd
    return score, sortino, annual_return


def compute_multiwindow_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    dd_penalty: float = 0.0,
    window_fractions: Iterable[float] = (0.33, 0.5, 0.75, 1.0),
    aggregation: str = "minimax",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-window robustness objective with shared prefix sums.

    Computes sortino-based score on multiple trailing sub-windows sharing
    a single cumsum/cummax pass. Returns (score, sortino_full, annual_full).
    """
    T = hourly_returns.shape[-1]
    fracs = sorted(set(window_fractions))
    ppy = _as_tensor(periods_per_year, hourly_returns)

    full_cumsum = None
    if dd_penalty > 0:
        full_cumsum = torch.cumsum(hourly_returns, dim=-1)

    window_scores = []
    sortino_full = None
    annual_full = None
    for frac in fracs:
        w = max(int(T * frac), 2)
        sub = hourly_returns[..., -w:]
        sc, s_ratio, a_ret = _sortino_core(sub, ppy, return_weight)
        if frac >= 1.0 or w >= T:
            sortino_full = s_ratio
            annual_full = a_ret
        if dd_penalty > 0:
            start_idx = T - w
            cum_sub = full_cumsum[..., start_idx:]
            if start_idx > 0:
                cum_sub = cum_sub - full_cumsum[..., start_idx - 1:start_idx]
            rmax = torch.cummax(cum_sub, dim=-1).values
            max_dd = (rmax - cum_sub).max(dim=-1).values
            sc = sc - dd_penalty * max_dd
        window_scores.append(sc)

    if sortino_full is None:
        _, sortino_full, annual_full = _sortino_core(hourly_returns, ppy, return_weight)

    stacked = torch.stack(window_scores, dim=0)
    if aggregation == "minimax":
        score = stacked.min(dim=0).values
    elif aggregation == "mean":
        score = stacked.mean(dim=0)
    elif aggregation == "softmin":
        score = -torch.logsumexp(-stacked * 5.0, dim=0) / 5.0
    else:
        score = stacked.min(dim=0).values

    return score, sortino_full, annual_full


def compute_loss_by_type(
    hourly_returns: torch.Tensor,
    loss_type: str = "sortino",
    *,
    target_sign: float = 1.0,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    smoothness_penalty: float | torch.Tensor = 0.0,
    dd_penalty: float = 1.0,
    multiwindow_fractions: str = "0.33,0.5,0.75,1.0",
    multiwindow_aggregation: str = "minimax",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unified loss dispatcher. Returns (loss, score, ratio_metric, annual_return)."""
    if loss_type == "sharpe":
        score, ratio, annual_return = compute_sharpe_objective(
            hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
        )
    elif loss_type == "calmar":
        score, ratio, annual_return = compute_calmar_objective(
            hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
        )
    elif loss_type == "log_wealth":
        score, ratio, annual_return = compute_log_wealth_objective(
            hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
        )
    elif loss_type == "sortino_dd":
        score, ratio, annual_return = compute_sortino_dd_objective(
            hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
            dd_penalty=dd_penalty,
        )
    elif loss_type in ("multiwindow", "multiwindow_dd"):
        fracs = [float(x) for x in multiwindow_fractions.split(",") if x.strip()]
        ddp = dd_penalty if loss_type == "multiwindow_dd" else 0.0
        score, ratio, annual_return = compute_multiwindow_objective(
            hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
            dd_penalty=ddp, window_fractions=fracs, aggregation=multiwindow_aggregation,
        )
    else:
        score, ratio, annual_return = compute_hourly_objective(
            hourly_returns, periods_per_year=periods_per_year, return_weight=return_weight,
        )

    loss = -target_sign * score.mean()
    loss = _apply_smoothness(loss, hourly_returns, smoothness_penalty)
    return loss, score, ratio, annual_return
