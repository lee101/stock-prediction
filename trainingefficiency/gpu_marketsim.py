"""GPU-native (all-torch, no CPU) market simulator for fast evaluation."""
from __future__ import annotations

import torch

from differentiable_loss_utils import (
    _EPS,
    HOURLY_PERIODS_PER_YEAR,
    _safe_denominator as _safe_denom,
)


def gpu_simulate_binary(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    opens: torch.Tensor | None = None,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    buy_amounts: torch.Tensor,
    sell_amounts: torch.Tensor,
    maker_fee: float = 0.001,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    max_leverage: float = 1.0,
    can_short: bool = False,
    can_long: bool = True,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
    max_hold_bars: int = 0,
    market_order_entry: bool = False,
    decision_lag_bars: int = 0,
) -> dict:
    """Binary-fill portfolio simulation entirely on GPU.

    All inputs/outputs stay on the same device as `closes`.
    Returns dict with tensors: returns, portfolio_values, inventory_path,
    executed_buys, executed_sells, buy_fills, sell_fills.
    """
    if closes.ndim < 1:
        raise ValueError("Tensors must have at least a time dimension")

    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        if opens is not None:
            opens = opens[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
        buy_amounts = buy_amounts[..., :-lag]
        sell_amounts = sell_amounts[..., :-lag]

    margin_cost_per_step = margin_annual_rate / HOURLY_PERIODS_PER_YEAR

    batch_shape = closes.shape[:-1]
    steps = closes.shape[-1]
    device = closes.device
    dtype = closes.dtype

    fee = torch.as_tensor(maker_fee, dtype=dtype, device=device)
    fee_buy = 1.0 + fee
    fee_sell = 1.0 - fee
    max_lev = torch.as_tensor(max_leverage, dtype=dtype, device=device)
    can_short_f = torch.as_tensor(float(can_short), dtype=dtype, device=device)
    can_long_f = torch.as_tensor(float(can_long), dtype=dtype, device=device)

    cash = torch.full(batch_shape, initial_cash, dtype=dtype, device=device)
    inventory = torch.full(batch_shape, initial_inventory, dtype=dtype, device=device)
    bars_held = torch.zeros(batch_shape, dtype=dtype, device=device)
    prev_value = cash + inventory * closes[..., 0]

    returns_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    values_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    inventory_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_buy_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    exec_sell_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    buy_fill_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)
    sell_fill_out = torch.empty(*batch_shape, steps, dtype=dtype, device=device)

    b_prices_all = torch.clamp(buy_prices, min=_EPS)
    s_prices_all = torch.clamp(sell_prices, min=_EPS)

    if market_order_entry and opens is not None:
        b_prices_all = torch.clamp(opens, min=_EPS)

    b_int_all = torch.clamp(buy_amounts, min=0.0)
    s_int_all = torch.clamp(sell_amounts, min=0.0)
    b_int_all = torch.minimum(b_int_all, max_lev.expand_as(b_int_all))
    s_int_all = torch.minimum(s_int_all, max_lev.expand_as(s_int_all))
    b_frac_all = b_int_all / torch.clamp(max_lev, min=_EPS)
    s_frac_all = s_int_all / torch.clamp(max_lev, min=_EPS)

    _zero = torch.zeros(batch_shape, dtype=dtype, device=device)

    for idx in range(steps):
        close = closes[..., idx]
        high = highs[..., idx]
        low = lows[..., idx]
        bp = b_prices_all[..., idx]
        sp = s_prices_all[..., idx]
        bf = b_frac_all[..., idx]
        sf = s_frac_all[..., idx]
        bi = b_int_all[..., idx]
        si = s_int_all[..., idx]

        if max_hold_bars > 0:
            force_close = (bars_held >= max_hold_bars) & (torch.abs(inventory) > _EPS)
            force_price = close * 0.999
            long_mask = force_close & (inventory > 0)
            short_mask = force_close & (inventory < 0)
            cash = torch.where(
                long_mask,
                cash + inventory * force_price * fee_sell,
                cash,
            )
            cash = torch.where(
                short_mask,
                cash + inventory * force_price * fee_buy,
                cash,
            )
            inventory = torch.where(force_close, _zero, inventory)
            bars_held = torch.where(force_close, _zero, bars_held)

        # Fill logic
        if market_order_entry:
            buy_fill = bi > 0
        else:
            buy_threshold = bp * (1.0 - fill_buffer_pct) if fill_buffer_pct > 0 else bp
            buy_fill = (low <= buy_threshold) & (bi > 0)

        sell_threshold = sp * (1.0 + fill_buffer_pct) if fill_buffer_pct > 0 else sp
        sell_fill = (high >= sell_threshold) & (si > 0)

        equity = cash + inventory * close
        equity_pos = torch.clamp(equity, min=_EPS)

        # Buy capacity
        max_buy_cash = torch.where(
            bp > 0,
            cash / _safe_denom(bp * fee_buy),
            _zero,
        )
        target_notional = max_lev * equity_pos
        current_notional = inventory * bp
        leveraged_cap = torch.where(
            bp > 0,
            torch.clamp(target_notional - current_notional, min=0.0) / _safe_denom(bp * fee_buy),
            _zero,
        )
        buy_cap = torch.where(
            max_lev <= 1.0 + 1e-6,
            torch.clamp(max_buy_cash, min=0.0),
            leveraged_cap,
        )
        buy_qty = torch.where(buy_fill, bf * buy_cap, torch.zeros_like(cash))

        cover_cap = torch.clamp(-inventory, min=0.0)
        buy_qty = torch.where(
            can_long_f > 0.5,
            buy_qty,
            torch.minimum(buy_qty, cover_cap),
        )

        # Sell capacity
        long_to_close = torch.clamp(inventory, min=0.0)
        max_short_qty = torch.where(
            sp > 0,
            (max_lev * equity_pos) / _safe_denom(sp * fee_buy),
            _zero,
        )
        current_short = torch.clamp(-inventory, min=0.0)
        short_open_cap = torch.clamp(max_short_qty - current_short, min=0.0)
        sell_cap = long_to_close + torch.where(
            can_short_f > 0.5,
            short_open_cap,
            _zero,
        )
        sell_qty = torch.where(sell_fill, sf * sell_cap, torch.zeros_like(cash))

        cash = cash - buy_qty * bp * fee_buy + sell_qty * sp * fee_sell
        inventory = inventory + buy_qty - sell_qty

        if margin_cost_per_step > 0:
            pos_value = torch.abs(inventory * close)
            eq = cash + inventory * close
            margin_used = torch.clamp(pos_value - torch.clamp(eq, min=0.0), min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inventory * close
        ret = (portfolio_value - prev_value) / _safe_denom(prev_value)
        prev_value = portfolio_value

        # Track bars held
        if max_hold_bars > 0:
            has_pos = torch.abs(inventory) > _EPS
            bars_held = torch.where(has_pos, bars_held + 1, _zero)

        returns_out[..., idx] = ret
        values_out[..., idx] = portfolio_value
        inventory_out[..., idx] = inventory
        exec_buy_out[..., idx] = buy_qty
        exec_sell_out[..., idx] = sell_qty
        buy_fill_out[..., idx] = buy_fill.float()
        sell_fill_out[..., idx] = sell_fill.float()

    return {
        "returns": returns_out,
        "portfolio_values": values_out,
        "inventory_path": inventory_out,
        "executed_buys": exec_buy_out,
        "executed_sells": exec_sell_out,
        "buy_fills": buy_fill_out,
        "sell_fills": sell_fill_out,
        "final_cash": cash,
        "final_inventory": inventory,
    }


def compute_sortino_gpu(
    returns: torch.Tensor,
    periods_per_year: float = HOURLY_PERIODS_PER_YEAR,
) -> torch.Tensor:
    mean_ret = returns.mean(dim=-1)
    downside = torch.clamp(-returns, min=0.0)
    downside_std = (downside.square().mean(dim=-1) + _EPS).sqrt()
    sortino = mean_ret / torch.clamp(downside_std, min=_EPS)
    return sortino * (periods_per_year ** 0.5)


def compute_max_drawdown_gpu(returns: torch.Tensor) -> torch.Tensor:
    cum = torch.cumsum(returns, dim=-1)
    running_max = torch.cummax(cum, dim=-1).values
    drawdowns = running_max - cum
    return drawdowns.max(dim=-1).values


def compute_total_return_gpu(portfolio_values: torch.Tensor) -> torch.Tensor:
    return portfolio_values[..., -1] / _safe_denom(portfolio_values[..., 0]) - 1.0


def compute_num_trades_gpu(
    executed_buys: torch.Tensor,
    executed_sells: torch.Tensor,
) -> torch.Tensor:
    buys = (executed_buys > _EPS).float().sum(dim=-1)
    sells = (executed_sells > _EPS).float().sum(dim=-1)
    return buys + sells


def compute_metrics_gpu(sim_result: dict, periods_per_year: float = HOURLY_PERIODS_PER_YEAR) -> dict:
    """Compute all eval metrics from sim output, all on GPU."""
    returns = sim_result["returns"]
    values = sim_result["portfolio_values"]
    sortino = compute_sortino_gpu(returns, periods_per_year)
    max_dd = compute_max_drawdown_gpu(returns)
    total_ret = compute_total_return_gpu(values)
    num_trades = compute_num_trades_gpu(sim_result["executed_buys"], sim_result["executed_sells"])
    return {
        "sortino": sortino,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "num_trades": num_trades,
    }


def metrics_to_scalars(metrics: dict) -> dict:
    """Convert GPU metric tensors to Python floats (one .item() per metric)."""
    return {k: v.item() if hasattr(v, 'item') else float(v) for k, v in metrics.items()}


def gpu_evaluate_policy(
    model,
    features: torch.Tensor,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    opens: torch.Tensor,
    reference_close: torch.Tensor,
    chronos_high: torch.Tensor,
    chronos_low: torch.Tensor,
    *,
    maker_fee: float = 0.001,
    max_leverage: float = 1.0,
    margin_rate: float = 0.0,
    max_hold_bars: int = 6,
    fill_buffer_pct: float = 0.0005,
    can_short: bool = False,
    can_long: bool = True,
    market_order_entry: bool = False,
    decision_lag_bars: int = 0,
    initial_cash: float = 1.0,
    periods_per_year: float = HOURLY_PERIODS_PER_YEAR,
) -> dict:
    """Run policy inference + simulation entirely on GPU. Returns metrics dict."""
    with torch.inference_mode():
        outputs = model(features)
        decoded = model.decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )
        buy_prices = decoded["buy_price"]
        sell_prices = decoded["sell_price"]
        buy_amounts = decoded["buy_amount"]
        sell_amounts = decoded["sell_amount"]

        sim = gpu_simulate_binary(
            highs=highs,
            lows=lows,
            closes=closes,
            opens=opens,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
            buy_amounts=buy_amounts,
            sell_amounts=sell_amounts,
            maker_fee=maker_fee,
            max_leverage=max_leverage,
            margin_annual_rate=margin_rate,
            max_hold_bars=max_hold_bars,
            fill_buffer_pct=fill_buffer_pct,
            can_short=can_short,
            can_long=can_long,
            market_order_entry=market_order_entry,
            decision_lag_bars=decision_lag_bars,
            initial_cash=initial_cash,
        )

        gpu_metrics = compute_metrics_gpu(sim, periods_per_year)
        return metrics_to_scalars(gpu_metrics)


def gpu_evaluate_batch(
    models: list,
    features: torch.Tensor,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    opens: torch.Tensor,
    reference_close: torch.Tensor,
    chronos_high: torch.Tensor,
    chronos_low: torch.Tensor,
    **kwargs,
) -> list[dict]:
    """Evaluate multiple models/checkpoints. Returns list of metric dicts."""
    results = []
    for model in models:
        r = gpu_evaluate_policy(
            model, features, highs, lows, closes, opens,
            reference_close, chronos_high, chronos_low,
            **kwargs,
        )
        results.append(r)
    return results
