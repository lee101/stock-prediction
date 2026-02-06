"""Differentiable trading PnL utilities for hourly crypto training.

The helpers in this module mirror ``loss_utils.py`` but add support for
variable buy/sell amounts, maker-fee aware fills, and 1x leverage constraints
while remaining fully differentiable for neural training. All calculations are
vectorised across an arbitrary batch dimension with the final axis representing
time steps (hours).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from src.fixtures import all_crypto_symbols

DEFAULT_MAKER_FEE_RATE = 0.0008  # 8 bps maker fee
HOURLY_PERIODS_PER_YEAR = 24 * 365  # 8760 hours
DAILY_PERIODS_PER_YEAR_CRYPTO = 365  # Crypto trades 24/7
DAILY_PERIODS_PER_YEAR_STOCK = 252   # ~252 trading days/year for stocks
_EPS = 1e-8


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

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    temperature_tensor = torch.as_tensor(temperature, dtype=dtype, device=device)
    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)
    max_leverage_tensor = torch.broadcast_to(_as_tensor(max_leverage, closes), closes.shape)
    can_short_tensor = torch.broadcast_to(_as_tensor(can_short, closes), batch_shape)
    can_long_tensor = torch.broadcast_to(_as_tensor(can_long, closes), batch_shape)
    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    prev_value = cash + inventory * closes[..., 0]

    pnl_list = []
    return_list = []
    value_list = []
    buy_prob_list = []
    sell_prob_list = []
    exec_buy_list = []
    exec_sell_list = []
    inventory_history = []

    for idx in range(steps):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        b_price = torch.clamp(buy_prices[..., idx], min=_EPS)
        s_price = torch.clamp(sell_prices[..., idx], min=_EPS)
        step_limit = torch.clamp(max_leverage_tensor[..., idx], min=_EPS)
        b_intensity = torch.minimum(torch.clamp(buy_intensity[..., idx], min=0.0), step_limit)
        s_intensity = torch.minimum(torch.clamp(sell_intensity[..., idx], min=0.0), step_limit)
        b_frac_limit = b_intensity / torch.clamp(step_limit, min=_EPS)
        s_frac_limit = s_intensity / torch.clamp(step_limit, min=_EPS)

        buy_prob = approx_buy_fill_probability(b_price, low, close, temperature=float(temperature_tensor))
        sell_prob = approx_sell_fill_probability(s_price, high, close, temperature=float(temperature_tensor))

        equity = cash + inventory * close
        # Allow borrowing up to ``step_limit`` * equity; translated to units.
        max_buy_cash = torch.where(
            b_price > 0,
            cash / _safe_denominator(b_price * (1.0 + fee)),
            torch.zeros_like(cash),
        )

        target_notional = step_limit * torch.clamp(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = torch.where(
            b_price > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * (1.0 + fee)),
            torch.zeros_like(cash),
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

        # Long-close capacity is always allowed; short-open capacity depends on can_short.
        long_to_close = torch.clamp(inventory, min=0.0)
        max_short_qty = torch.where(
            s_price > 0,
            (step_limit * torch.clamp(equity, min=_EPS)) / _safe_denominator(s_price * (1.0 + fee)),
            torch.zeros_like(cash),
        )
        current_short_qty = torch.clamp(-inventory, min=0.0)
        short_open_cap = torch.clamp(max_short_qty - current_short_qty, min=0.0)
        sell_capacity = long_to_close + torch.where(can_short_tensor > 0.5, short_open_cap, 0.0)
        sell_qty = s_frac_limit * sell_capacity

        executed_buys = buy_qty * buy_prob
        executed_sells = sell_qty * sell_prob

        cash = cash - executed_buys * b_price * (1.0 + fee) + executed_sells * s_price * (1.0 - fee)
        inventory = inventory + executed_buys - executed_sells
        portfolio_value = cash + inventory * close

        prior_value = prev_value
        pnl = portfolio_value - prior_value
        returns = pnl / _safe_denominator(prior_value)
        prev_value = portfolio_value

        pnl_list.append(pnl)
        return_list.append(returns)
        value_list.append(portfolio_value)
        buy_prob_list.append(buy_prob)
        sell_prob_list.append(sell_prob)
        exec_buy_list.append(executed_buys)
        exec_sell_list.append(executed_sells)
        inventory_history.append(inventory.clone())

    pnl_tensor = torch.stack(pnl_list, dim=-1)
    returns_tensor = torch.stack(return_list, dim=-1)
    values_tensor = torch.stack(value_list, dim=-1)
    buy_prob_tensor = torch.stack(buy_prob_list, dim=-1)
    sell_prob_tensor = torch.stack(sell_prob_list, dim=-1)
    exec_buy_tensor = torch.stack(exec_buy_list, dim=-1)
    exec_sell_tensor = torch.stack(exec_sell_list, dim=-1)
    inventory_path_tensor = torch.stack(inventory_history, dim=-1)

    return HourlySimulationResult(
        pnl=pnl_tensor,
        returns=returns_tensor,
        portfolio_values=values_tensor,
        cash=cash,
        inventory=inventory,
        buy_fill_probability=buy_prob_tensor,
        sell_fill_probability=sell_prob_tensor,
        executed_buys=exec_buy_tensor,
        executed_sells=exec_sell_tensor,
        inventory_path=inventory_path_tensor,
    )


def simulate_hourly_trades_binary(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
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

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)
    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    max_leverage_tensor = torch.broadcast_to(_as_tensor(max_leverage, closes), closes.shape)
    can_short_tensor = torch.broadcast_to(_as_tensor(can_short, closes), batch_shape)
    can_long_tensor = torch.broadcast_to(_as_tensor(can_long, closes), batch_shape)
    prev_value = cash + inventory * closes[..., 0]

    pnl_list = []
    return_list = []
    value_list = []
    buy_fill_list = []
    sell_fill_list = []
    exec_buy_list = []
    exec_sell_list = []
    inventory_history = []

    for idx in range(steps):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        b_price = torch.clamp(buy_prices[..., idx], min=_EPS)
        s_price = torch.clamp(sell_prices[..., idx], min=_EPS)
        step_limit = torch.clamp(max_leverage_tensor[..., idx], min=_EPS)
        b_intensity = torch.minimum(torch.clamp(buy_intensity[..., idx], min=0.0), step_limit)
        s_intensity = torch.minimum(torch.clamp(sell_intensity[..., idx], min=0.0), step_limit)
        b_frac_limit = b_intensity / torch.clamp(step_limit, min=_EPS)
        s_frac_limit = s_intensity / torch.clamp(step_limit, min=_EPS)

        # Binary fills: 100% if price touches limit, 0% otherwise
        buy_fill = (low <= b_price) & (b_intensity > 0)
        sell_fill = (high >= s_price) & (s_intensity > 0)

        equity = cash + inventory * close
        max_buy_cash = torch.where(
            b_price > 0,
            cash / _safe_denominator(b_price * (1.0 + fee)),
            torch.zeros_like(cash),
        )
        target_notional = step_limit * torch.clamp(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = torch.where(
            b_price > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * (1.0 + fee)),
            torch.zeros_like(cash),
        )
        buy_capacity = torch.where(step_limit <= 1.0 + 1e-6, torch.clamp(max_buy_cash, min=0.0), leveraged_capacity)

        buy_qty = torch.where(buy_fill, b_frac_limit * buy_capacity, torch.zeros_like(cash))
        cover_only_cap = torch.clamp(-inventory, min=0.0)
        buy_qty = torch.where(
            can_long_tensor > 0.5,
            buy_qty,
            torch.minimum(buy_qty, cover_only_cap),
        )

        long_to_close = torch.clamp(inventory, min=0.0)
        max_short_qty = torch.where(
            s_price > 0,
            (step_limit * torch.clamp(equity, min=_EPS)) / _safe_denominator(s_price * (1.0 + fee)),
            torch.zeros_like(cash),
        )
        current_short_qty = torch.clamp(-inventory, min=0.0)
        short_open_cap = torch.clamp(max_short_qty - current_short_qty, min=0.0)
        sell_capacity = long_to_close + torch.where(can_short_tensor > 0.5, short_open_cap, 0.0)
        sell_qty = torch.where(sell_fill, s_frac_limit * sell_capacity, torch.zeros_like(cash))

        executed_buys = buy_qty
        executed_sells = sell_qty

        cash = cash - executed_buys * b_price * (1.0 + fee) + executed_sells * s_price * (1.0 - fee)
        inventory = inventory + executed_buys - executed_sells
        portfolio_value = cash + inventory * close

        prior_value = prev_value
        pnl = portfolio_value - prior_value
        returns = pnl / _safe_denominator(prior_value)
        prev_value = portfolio_value

        pnl_list.append(pnl)
        return_list.append(returns)
        value_list.append(portfolio_value)
        buy_fill_list.append(buy_fill.float())
        sell_fill_list.append(sell_fill.float())
        exec_buy_list.append(executed_buys)
        exec_sell_list.append(executed_sells)
        inventory_history.append(inventory.clone())

    pnl_tensor = torch.stack(pnl_list, dim=-1)
    returns_tensor = torch.stack(return_list, dim=-1)
    values_tensor = torch.stack(value_list, dim=-1)
    buy_fill_tensor = torch.stack(buy_fill_list, dim=-1)
    sell_fill_tensor = torch.stack(sell_fill_list, dim=-1)
    exec_buy_tensor = torch.stack(exec_buy_list, dim=-1)
    exec_sell_tensor = torch.stack(exec_sell_list, dim=-1)
    inventory_path_tensor = torch.stack(inventory_history, dim=-1)

    return HourlySimulationResult(
        pnl=pnl_tensor,
        returns=returns_tensor,
        portfolio_values=values_tensor,
        cash=cash,
        inventory=inventory,
        buy_fill_probability=buy_fill_tensor,
        sell_fill_probability=sell_fill_tensor,
        executed_buys=exec_buy_tensor,
        executed_sells=exec_sell_tensor,
        inventory_path=inventory_path_tensor,
    )


def compute_hourly_objective(
    hourly_returns: torch.Tensor,
    *,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (score, sortino, annual_return) for hourly return series."""

    if hourly_returns.ndim == 0:
        raise ValueError("hourly_returns must have at least one dimension")
    mean_return = hourly_returns.mean(dim=-1)
    downside = torch.clamp(-hourly_returns, min=0.0)
    downside_std = torch.sqrt(torch.mean(downside**2, dim=-1) + _EPS)
    sortino = mean_return / _safe_denominator(downside_std)
    periods = _as_tensor(periods_per_year, mean_return)
    sortino = sortino * torch.sqrt(torch.clamp(periods, min=_EPS))
    annual_return = mean_return * periods
    score = sortino + return_weight * annual_return
    return score, sortino, annual_return


def combined_sortino_pnl_loss(
    hourly_returns: torch.Tensor,
    *,
    target_sign: float = 1.0,
    periods_per_year: float | torch.Tensor = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
) -> torch.Tensor:
    """Loss function that encourages joint Sortino + return maximisation."""

    score, _, _ = compute_hourly_objective(
        hourly_returns,
        periods_per_year=periods_per_year,
        return_weight=return_weight,
    )
    return -target_sign * score.mean()
