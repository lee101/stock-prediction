"""Compiled sim+loss fusion for binanceneural training.

Replaces the Triton-forward + Python-backward sim with a single torch.compile'd
path that fuses BOTH forward and backward, eliminating ~1800 kernel launches in
the backward pass.

The key insight: the current approach uses a Triton autograd.Function whose
backward re-runs the entire Python sim to build an autograd graph. This creates
72 steps * ~25 ops = 1800 intermediate tensors and kernel launches. By compiling
the entire sim+loss as one unit, the compiler fuses these into a much smaller
number of fused kernels.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_EPS = 1e-8


@dataclass
class CompiledSimTrajectory:
    returns: torch.Tensor
    portfolio_values: torch.Tensor
    cash_path: torch.Tensor
    inventory_path: torch.Tensor
    executed_buys: torch.Tensor
    executed_sells: torch.Tensor
    buy_fill_probability: torch.Tensor
    sell_fill_probability: torch.Tensor
    closes: torch.Tensor
    highs: torch.Tensor
    lows: torch.Tensor
    buy_prices: torch.Tensor
    sell_prices: torch.Tensor
    loss: torch.Tensor
    score: torch.Tensor
    sortino: torch.Tensor
    annual_return: torch.Tensor


def _sortino_from_returns(
    returns: torch.Tensor,
    *,
    periods_per_year: float,
    return_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_return = returns.mean(dim=-1)
    downside_sq = (-returns).clamp(min=0.0).square()
    downside_std = (downside_sq.mean(dim=-1) + _EPS).sqrt()
    periods = torch.as_tensor(periods_per_year, dtype=returns.dtype, device=returns.device)
    sortino = mean_return / downside_std.clamp(min=_EPS) * periods.clamp(min=_EPS).sqrt()
    annual_return = mean_return * periods
    score = sortino + return_weight * annual_return
    loss = -score.mean()
    return loss, score, sortino, annual_return


def _sim_loop_state(
    closes: torch.Tensor,
    highs: torch.Tensor,
    lows: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    buy_frac: torch.Tensor,
    sell_frac: torch.Tensor,
    max_lev: torch.Tensor,
    can_short: torch.Tensor,
    can_long: torch.Tensor,
    initial_cash: float,
    initial_inventory: float,
    fee_buy: float,
    fee_sell: float,
    temperature: float,
    fill_buffer_pct: float,
    margin_cost_per_step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-step sim state for a flattened (B, T) batch.

    `buy_frac` / `sell_frac` are interpreted the same way as the training sim:
    trade intensities in [0, max_leverage], not already-normalized fractions.
    """
    B, T = closes.shape
    dtype = closes.dtype
    device = closes.device

    cash = torch.full((B,), initial_cash, dtype=dtype, device=device)
    inv = torch.full((B,), initial_inventory, dtype=dtype, device=device)
    prev_val = cash + inv * closes[:, 0]

    returns = torch.empty((B, T), dtype=dtype, device=device)
    portfolio_values = torch.empty((B, T), dtype=dtype, device=device)
    cash_path = torch.empty((B, T), dtype=dtype, device=device)
    inventory_path = torch.empty((B, T), dtype=dtype, device=device)
    executed_buys = torch.empty((B, T), dtype=dtype, device=device)
    executed_sells = torch.empty((B, T), dtype=dtype, device=device)
    buy_fill_probability = torch.empty((B, T), dtype=dtype, device=device)
    sell_fill_probability = torch.empty((B, T), dtype=dtype, device=device)

    for t in range(T):
        close = closes[:, t]
        high = highs[:, t]
        low = lows[:, t]
        bp = buy_prices[:, t].clamp(min=_EPS)
        sp = sell_prices[:, t].clamp(min=_EPS)
        sl = max_lev[:, t].clamp(min=_EPS)
        b_int = torch.minimum(torch.clamp(buy_frac[:, t], min=0.0), sl)
        s_int = torch.minimum(torch.clamp(sell_frac[:, t], min=0.0), sl)
        bf = b_int / sl
        sf = s_int / sl

        scale = close.abs().clamp(min=1e-4)
        temp_scale = scale * temperature
        buy_thresh = bp * (1.0 - fill_buffer_pct) if fill_buffer_pct > 0 else bp
        buy_prob = torch.sigmoid((buy_thresh - low) / temp_scale.clamp(min=_EPS))
        sell_thresh = sp * (1.0 + fill_buffer_pct) if fill_buffer_pct > 0 else sp
        sell_prob = torch.sigmoid((high - sell_thresh) / temp_scale.clamp(min=_EPS))

        equity = cash + inv * close
        equity_pos = equity.clamp(min=_EPS)

        bp_fee = bp * fee_buy + _EPS
        max_buy_cash = torch.where(bp > 0, cash / bp_fee, torch.zeros_like(cash))
        target_notional = sl * equity_pos
        current_notional = inv * bp
        room = (target_notional - current_notional).clamp(min=0.0)
        leveraged_cap = torch.where(bp > 0, room / bp_fee, torch.zeros_like(cash))
        buy_cap = torch.where(sl <= 1.0 + 1e-6, max_buy_cash.clamp(min=0.0), leveraged_cap)
        buy_qty = bf * buy_cap

        if can_long.numel() > 0:
            cover_cap = (-inv).clamp(min=0.0)
            buy_qty = torch.where(can_long > 0.5, buy_qty, torch.minimum(buy_qty, cover_cap))

        long_to_close = inv.clamp(min=0.0)
        sp_fee = sp * fee_buy + _EPS
        max_short = torch.where(sp > 0, target_notional / sp_fee, torch.zeros_like(cash))
        cur_short = (-inv).clamp(min=0.0)
        short_open = (max_short - cur_short).clamp(min=0.0)
        sell_cap = long_to_close + torch.where(can_short > 0.5, short_open, torch.zeros_like(cash))
        sell_qty = sf * sell_cap

        exec_buys = buy_qty * buy_prob
        exec_sells = sell_qty * sell_prob

        cash = cash - exec_buys * bp * fee_buy + exec_sells * sp * fee_sell
        inv = inv + exec_buys - exec_sells

        if margin_cost_per_step > 0:
            pos_val = (inv * close).abs()
            eq = cash + inv * close
            margin_used = (pos_val - eq.clamp(min=0.0)).clamp(min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inv * close
        pnl = portfolio_value - prev_val
        ret = pnl / prev_val.clamp(min=_EPS)
        prev_val = portfolio_value

        returns[:, t] = ret
        portfolio_values[:, t] = portfolio_value
        cash_path[:, t] = cash
        inventory_path[:, t] = inv
        executed_buys[:, t] = exec_buys
        executed_sells[:, t] = exec_sells
        buy_fill_probability[:, t] = buy_prob
        sell_fill_probability[:, t] = sell_prob

    return (
        returns,
        portfolio_values,
        cash_path,
        inventory_path,
        executed_buys,
        executed_sells,
        buy_fill_probability,
        sell_fill_probability,
    )


def _sim_loop_sortino(
    closes: torch.Tensor,
    highs: torch.Tensor,
    lows: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    buy_frac: torch.Tensor,
    sell_frac: torch.Tensor,
    max_lev: torch.Tensor,
    can_short: torch.Tensor,
    can_long: torch.Tensor,
    initial_cash: float,
    initial_inventory: float,
    fee_buy: float,
    fee_sell: float,
    temperature: float,
    fill_buffer_pct: float,
    margin_cost_per_step: float,
    periods_per_year: float,
    return_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused sim loop + sortino loss. Returns (loss, score, sortino, annual_return).

    All batch dims are flattened: inputs are (B, T) tensors.
    """
    returns, *_rest = _sim_loop_state(
        closes,
        highs,
        lows,
        buy_prices,
        sell_prices,
        buy_frac,
        sell_frac,
        max_lev,
        can_short,
        can_long,
        initial_cash,
        initial_inventory,
        fee_buy,
        fee_sell,
        temperature,
        fill_buffer_pct,
        margin_cost_per_step,
    )
    return _sortino_from_returns(
        returns,
        periods_per_year=periods_per_year,
        return_weight=return_weight,
    )


# compile with reduce-overhead for maximum fusion
_compiled_sim_loss = None


def _compile_enabled() -> bool:
    if os.environ.get("TORCH_NO_COMPILE", ""):
        return False
    if not hasattr(torch, "compile"):
        return False
    if os.environ.get("TORCH_FORCE_COMPILE", ""):
        return True
    return bool(torch.cuda.is_available())


def get_compiled_sim_loss():
    """Lazily compile the fused sim+loss function."""
    global _compiled_sim_loss
    if _compiled_sim_loss is None:
        if not _compile_enabled():
            _compiled_sim_loss = _sim_loop_sortino
        else:
            try:
                _compiled_sim_loss = torch.compile(
                    _sim_loop_sortino,
                    mode="reduce-overhead",
                    fullgraph=False,  # allow Python control flow for margin check
                )
            except Exception:
                _compiled_sim_loss = _sim_loop_sortino
    return _compiled_sim_loss


def compiled_sim_and_loss(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    buy_frac: torch.Tensor,
    sell_frac: torch.Tensor,
    max_leverage: torch.Tensor,
    can_short: torch.Tensor,
    can_long: torch.Tensor,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    maker_fee: float = 0.001,
    temperature: float = 0.01,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
    periods_per_year: float = 8760.0,
    return_weight: float = 0.05,
    decision_lag_bars: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled sim+loss for binanceneural. Returns (loss, score, sortino, annual_return)."""

    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
        buy_frac = buy_frac[..., :-lag]
        sell_frac = sell_frac[..., :-lag]
        if max_leverage.ndim > 0:
            max_leverage = max_leverage[..., lag:]

    # flatten batch dims
    orig_shape = closes.shape[:-1]
    T = closes.shape[-1]
    closes_2d = closes.reshape(-1, T)
    highs_2d = highs.reshape(-1, T)
    lows_2d = lows.reshape(-1, T)
    bp_2d = buy_prices.reshape(-1, T)
    sp_2d = sell_prices.reshape(-1, T)
    bf_2d = buy_frac.reshape(-1, T)
    sf_2d = sell_frac.reshape(-1, T)
    ml_2d = max_leverage.reshape(-1, T) if max_leverage.ndim > 0 else max_leverage.expand_as(closes).reshape(-1, T)
    cs_flat = can_short.reshape(-1) if can_short.ndim > 0 else can_short.expand(orig_shape).reshape(-1)
    cl_flat = can_long.reshape(-1) if can_long.ndim > 0 else can_long.expand(orig_shape).reshape(-1)

    fee_buy = 1.0 + maker_fee
    fee_sell = 1.0 - maker_fee
    margin_cost_per_step = margin_annual_rate / periods_per_year if margin_annual_rate > 0 else 0.0

    fn = get_compiled_sim_loss()
    return fn(
        closes_2d, highs_2d, lows_2d, bp_2d, sp_2d,
        bf_2d, sf_2d, ml_2d, cs_flat, cl_flat,
        initial_cash, initial_inventory,
        fee_buy, fee_sell, temperature, fill_buffer_pct,
        margin_cost_per_step, periods_per_year, return_weight,
    )


def compiled_sim_trajectory(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    buy_frac: torch.Tensor,
    sell_frac: torch.Tensor,
    max_leverage: torch.Tensor,
    can_short: torch.Tensor,
    can_long: torch.Tensor,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    maker_fee: float = 0.001,
    temperature: float = 0.01,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
    periods_per_year: float = 8760.0,
    return_weight: float = 0.05,
    decision_lag_bars: int = 0,
) -> CompiledSimTrajectory:
    """Return full per-step trajectory for fused-sim validation and visualization."""
    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
        buy_frac = buy_frac[..., :-lag]
        sell_frac = sell_frac[..., :-lag]
        if max_leverage.ndim > 0:
            max_leverage = max_leverage[..., lag:]

    orig_shape = closes.shape[:-1]
    T = closes.shape[-1]
    closes_2d = closes.reshape(-1, T)
    highs_2d = highs.reshape(-1, T)
    lows_2d = lows.reshape(-1, T)
    bp_2d = buy_prices.reshape(-1, T)
    sp_2d = sell_prices.reshape(-1, T)
    bf_2d = buy_frac.reshape(-1, T)
    sf_2d = sell_frac.reshape(-1, T)
    ml_2d = max_leverage.reshape(-1, T) if max_leverage.ndim > 0 else max_leverage.expand_as(closes).reshape(-1, T)
    cs_flat = can_short.reshape(-1) if can_short.ndim > 0 else can_short.expand(orig_shape).reshape(-1)
    cl_flat = can_long.reshape(-1) if can_long.ndim > 0 else can_long.expand(orig_shape).reshape(-1)

    fee_buy = 1.0 + maker_fee
    fee_sell = 1.0 - maker_fee
    margin_cost_per_step = margin_annual_rate / periods_per_year if margin_annual_rate > 0 else 0.0

    (
        returns,
        portfolio_values,
        cash_path,
        inventory_path,
        executed_buys,
        executed_sells,
        buy_fill_probability,
        sell_fill_probability,
    ) = _sim_loop_state(
        closes_2d,
        highs_2d,
        lows_2d,
        bp_2d,
        sp_2d,
        bf_2d,
        sf_2d,
        ml_2d,
        cs_flat,
        cl_flat,
        initial_cash,
        initial_inventory,
        fee_buy,
        fee_sell,
        temperature,
        fill_buffer_pct,
        margin_cost_per_step,
    )
    loss, score, sortino, annual_return = _sortino_from_returns(
        returns,
        periods_per_year=periods_per_year,
        return_weight=return_weight,
    )

    def _restore(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*orig_shape, T)

    def _restore_metric(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*orig_shape)

    return CompiledSimTrajectory(
        returns=_restore(returns),
        portfolio_values=_restore(portfolio_values),
        cash_path=_restore(cash_path),
        inventory_path=_restore(inventory_path),
        executed_buys=_restore(executed_buys),
        executed_sells=_restore(executed_sells),
        buy_fill_probability=_restore(buy_fill_probability),
        sell_fill_probability=_restore(sell_fill_probability),
        closes=closes,
        highs=highs,
        lows=lows,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        loss=loss,
        score=_restore_metric(score),
        sortino=_restore_metric(sortino),
        annual_return=_restore_metric(annual_return),
    )
