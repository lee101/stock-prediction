"""Vectorized differentiable trading simulation.

Drop-in replacement for simulate_hourly_trades() with:
- Pre-computed fill probabilities (vectorized sigmoid)
- Pre-allocated output tensors (no list append + stack)
- Minimized per-step ops (only state-dependent ops in loop)
- Optional torch.compile for fused kernels
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple

from differentiable_loss_utils import (
    HourlySimulationResult,
    HOURLY_PERIODS_PER_YEAR,
    _EPS,
    _as_tensor,
    _check_shapes,
    _safe_denominator,
)

def _precompute_fill_probs(
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    lows: torch.Tensor,
    highs: torch.Tensor,
    closes: torch.Tensor,
    temperature: float,
    fill_buffer_pct: float,
    market_order_entry: bool,
    opens: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized fill probability computation across all timesteps."""
    scale = torch.clamp(closes.abs(), min=1e-4)
    temp = temperature

    b_prices = torch.clamp(buy_prices, min=_EPS)
    s_prices = torch.clamp(sell_prices, min=_EPS)

    if market_order_entry and opens is not None:
        b_prices = torch.clamp(opens, min=_EPS)

    if market_order_entry:
        buy_probs = torch.ones_like(b_prices)
    else:
        buy_threshold = b_prices * (1.0 - fill_buffer_pct) if fill_buffer_pct > 0 else b_prices
        buy_scores = (buy_threshold - lows) / (scale * temp + _EPS)
        buy_probs = torch.sigmoid(buy_scores)

    sell_threshold = s_prices * (1.0 + fill_buffer_pct) if fill_buffer_pct > 0 else s_prices
    sell_scores = (highs - sell_threshold) / (scale * temp + _EPS)
    sell_probs = torch.sigmoid(sell_scores)

    return b_prices, s_prices, buy_probs, sell_probs


def simulate_hourly_trades_fast(
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
    maker_fee: float = 0.0008,
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
    """Vectorized simulate_hourly_trades - same interface, faster execution."""

    _check_shapes(highs, lows, closes, buy_prices, sell_prices, trade_intensity)
    if closes.ndim == 0:
        raise ValueError("Input tensors must include a time dimension")

    buy_intensity = buy_trade_intensity if buy_trade_intensity is not None else trade_intensity
    sell_intensity = sell_trade_intensity if sell_trade_intensity is not None else trade_intensity
    _check_shapes(highs, buy_intensity, sell_intensity)
    if opens is not None:
        _check_shapes(highs, opens)

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
        if torch.is_tensor(max_leverage) and max_leverage.ndim > 0:
            max_leverage = max_leverage[..., lag:]

    margin_cost_per_step = margin_annual_rate / HOURLY_PERIODS_PER_YEAR
    has_margin = margin_cost_per_step > 0

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype
    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)

    max_lev = torch.broadcast_to(_as_tensor(max_leverage, closes), closes.shape)
    can_short_t = torch.broadcast_to(_as_tensor(can_short, closes), batch_shape)
    can_long_t = torch.broadcast_to(_as_tensor(can_long, closes), batch_shape)

    # Pre-compute fill probabilities and clamped prices (vectorized)
    b_prices, s_prices, buy_probs, sell_probs = _precompute_fill_probs(
        buy_prices, sell_prices, lows, highs, closes,
        temperature, fill_buffer_pct, market_order_entry, opens,
    )

    # Pre-compute intensities and frac limits (vectorized)
    b_int = torch.minimum(torch.clamp(buy_intensity, min=0.0), max_lev)
    s_int = torch.minimum(torch.clamp(sell_intensity, min=0.0), max_lev)
    b_frac = b_int / torch.clamp(max_lev, min=_EPS)
    s_frac = s_int / torch.clamp(max_lev, min=_EPS)

    # Pre-allocate output tensors
    pnl_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    returns_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    values_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    buy_prob_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    sell_prob_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_buy_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_sell_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    inventory_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)

    # Init state
    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    prev_value = cash + inventory * closes[..., 0]

    fee_buy = 1.0 + fee
    fee_sell = 1.0 - fee
    has_can_long = can_long_t.numel() > 0

    for idx in range(steps):
        close = closes[..., idx]
        bp = b_prices[..., idx]
        sp = s_prices[..., idx]
        sl = max_lev[..., idx]
        bf = b_frac[..., idx]
        sf = s_frac[..., idx]

        equity = cash + inventory * close
        equity_pos = torch.clamp(equity, min=_EPS)

        # Buy capacity
        max_buy_cash = torch.where(
            bp > 0, cash / (bp * fee_buy + _EPS), torch.zeros_like(cash),
        )
        target_notional = sl * equity_pos
        current_notional = inventory * bp
        leveraged_cap = torch.where(
            bp > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / (bp * fee_buy + _EPS),
            torch.zeros_like(cash),
        )
        buy_cap = torch.where(sl <= 1.0 + 1e-6, torch.clamp(max_buy_cash, min=0.0), leveraged_cap)
        buy_qty = bf * buy_cap

        if has_can_long:
            cover_cap = torch.clamp(-inventory, min=0.0)
            buy_qty = torch.where(can_long_t > 0.5, buy_qty, torch.minimum(buy_qty, cover_cap))

        # Sell capacity
        long_to_close = torch.clamp(inventory, min=0.0)
        max_short = torch.where(
            sp > 0, target_notional / (sp * fee_buy + _EPS), torch.zeros_like(cash),
        )
        short_open = torch.clamp(max_short - torch.clamp(-inventory, min=0.0), min=0.0)
        sell_cap = long_to_close + torch.where(can_short_t > 0.5, short_open, torch.zeros_like(cash))
        sell_qty = sf * sell_cap

        # Execute
        exec_buys = buy_qty * buy_probs[..., idx]
        exec_sells = sell_qty * sell_probs[..., idx]

        cash = cash - exec_buys * bp * fee_buy + exec_sells * sp * fee_sell
        inventory = inventory + exec_buys - exec_sells

        if has_margin:
            pos_val = torch.abs(inventory * close)
            eq = cash + inventory * close
            margin_used = torch.clamp(pos_val - torch.clamp(eq, min=0.0), min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inventory * close
        pnl = portfolio_value - prev_value
        ret = pnl / torch.clamp(prev_value, min=_EPS)
        prev_value = portfolio_value

        pnl_out[..., idx] = pnl
        returns_out[..., idx] = ret
        values_out[..., idx] = portfolio_value
        buy_prob_out[..., idx] = buy_probs[..., idx]
        sell_prob_out[..., idx] = sell_probs[..., idx]
        exec_buy_out[..., idx] = exec_buys
        exec_sell_out[..., idx] = exec_sells
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


# Compiled version - use reduce-overhead to avoid CUDA graph capture hangs
try:
    simulate_hourly_trades_compiled = torch.compile(
        simulate_hourly_trades_fast,
        mode="reduce-overhead",
        fullgraph=False,
    )
except Exception:
    simulate_hourly_trades_compiled = simulate_hourly_trades_fast
