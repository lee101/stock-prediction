from __future__ import annotations

from typing import Mapping

import torch

from RLgpt.config import SimulatorConfig


def simulate_daily_plans(
    *,
    hourly_open: torch.Tensor,
    hourly_high: torch.Tensor,
    hourly_low: torch.Tensor,
    hourly_close: torch.Tensor,
    hourly_mask: torch.Tensor,
    daily_anchor: torch.Tensor,
    plans: Mapping[str, torch.Tensor],
    config: SimulatorConfig | None = None,
    init_inventory: torch.Tensor | None = None,
    init_cash: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or SimulatorConfig()
    _validate_ohlc_shapes(hourly_open, hourly_high, hourly_low, hourly_close, hourly_mask, daily_anchor)
    batch_size, steps, num_assets = hourly_open.shape
    required = {
        "allocation_logits",
        "center_offset_bps",
        "half_spread_bps",
        "max_long_fraction",
        "max_short_fraction",
        "trade_fraction",
    }
    missing = sorted(required - set(plans.keys()))
    if missing:
        raise ValueError(f"Plan dictionary is missing required keys: {missing}")

    device = hourly_open.device
    dtype = hourly_open.dtype
    inventory = (
        torch.zeros(batch_size, num_assets, device=device, dtype=dtype)
        if init_inventory is None
        else init_inventory.to(device=device, dtype=dtype).clone()
    )
    if inventory.shape != daily_anchor.shape:
        raise ValueError("init_inventory must have shape [batch, assets]")

    cash = (
        torch.full((batch_size,), float(cfg.initial_cash), device=device, dtype=dtype)
        if init_cash is None
        else init_cash.to(device=device, dtype=dtype).clone()
    )
    if cash.shape != (batch_size,):
        raise ValueError("init_cash must have shape [batch]")

    allocation = torch.softmax(plans["allocation_logits"], dim=-1)
    budget_scale = plans.get("budget_scale")
    if budget_scale is None:
        budget_scale = torch.ones(batch_size, 1, device=device, dtype=dtype)
    budget_scale = budget_scale.to(device=device, dtype=dtype).reshape(batch_size, 1).clamp(0.05, 1.0)
    shared_budget = budget_scale * float(cfg.shared_unit_budget)
    gross_cap = torch.minimum(
        allocation * shared_budget,
        torch.full_like(allocation, float(cfg.max_units_per_asset)),
    )
    max_long = gross_cap * plans["max_long_fraction"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    max_short = gross_cap * plans["max_short_fraction"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    trade_units = gross_cap * plans["trade_fraction"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if cfg.min_trade_units > 0:
        trade_units = torch.where(
            gross_cap > 0,
            torch.maximum(trade_units, torch.full_like(trade_units, float(cfg.min_trade_units))),
            trade_units,
        )

    buy_price, sell_price = _price_levels_from_plan(
        daily_anchor=daily_anchor,
        center_offset_bps=plans["center_offset_bps"].to(device=device, dtype=dtype),
        half_spread_bps=plans["half_spread_bps"].to(device=device, dtype=dtype).clamp(min=0.0),
    )

    fee_rate = float(cfg.maker_fee_bps) / 10_000.0
    slippage_rate = float(cfg.slippage_bps) / 10_000.0
    buffer_rate = float(cfg.fill_buffer_bps) / 10_000.0
    temperature = daily_anchor.abs() * (float(cfg.fill_temperature_bps) / 10_000.0) + 1e-6

    last_price = daily_anchor.clone()
    initial_equity = cash + (inventory * daily_anchor).sum(dim=-1)
    prev_equity = initial_equity.clone()

    pnl_steps: list[torch.Tensor] = []
    turnover_steps: list[torch.Tensor] = []

    for step in range(steps):
        mask = hourly_mask[:, step, :].to(device=device, dtype=dtype).clamp(0.0, 1.0)
        low = hourly_low[:, step, :].to(device=device, dtype=dtype)
        high = hourly_high[:, step, :].to(device=device, dtype=dtype)
        close = hourly_close[:, step, :].to(device=device, dtype=dtype)

        buy_buffer = buy_price * buffer_rate
        sell_buffer = sell_price * buffer_rate
        fill_prob_buy = torch.sigmoid((buy_price + buy_buffer - low) / temperature) * mask
        fill_prob_sell = torch.sigmoid((high - (sell_price - sell_buffer)) / temperature) * mask

        room_to_buy = torch.clamp(max_long - inventory, min=0.0)
        room_to_sell = torch.clamp(max_short + inventory, min=0.0)

        buy_qty = torch.minimum(trade_units * fill_prob_buy, room_to_buy)
        sell_qty = torch.minimum(trade_units * fill_prob_sell, room_to_sell)

        exec_buy_price = buy_price * (1.0 + slippage_rate)
        exec_sell_price = sell_price * (1.0 - slippage_rate)

        buy_notional = buy_qty * exec_buy_price
        buy_fee = buy_notional * fee_rate
        total_buy_cash = buy_notional.sum(dim=-1) + buy_fee.sum(dim=-1)
        affordability = torch.clamp(cash / (total_buy_cash + 1e-6), min=0.0, max=1.0)
        buy_qty = buy_qty * affordability.unsqueeze(-1)
        buy_notional = buy_qty * exec_buy_price
        buy_fee = buy_notional * fee_rate

        sell_notional = sell_qty * exec_sell_price
        sell_fee = sell_notional * fee_rate

        inventory = inventory + buy_qty - sell_qty
        cash = cash - buy_notional.sum(dim=-1) - buy_fee.sum(dim=-1)
        cash = cash + sell_notional.sum(dim=-1) - sell_fee.sum(dim=-1)

        last_price = torch.where(mask > 0.0, close, last_price)
        equity = cash + (inventory * last_price).sum(dim=-1)

        pnl_steps.append(equity - prev_equity)
        turnover_steps.append(buy_notional.sum(dim=-1) + sell_notional.sum(dim=-1))
        prev_equity = equity

    if not bool(cfg.carry_inventory):
        flatten_sell_qty = torch.clamp(inventory, min=0.0)
        flatten_buy_qty = torch.clamp(-inventory, min=0.0)
        flatten_sell_price = last_price * (1.0 - slippage_rate)
        flatten_buy_price = last_price * (1.0 + slippage_rate)
        flatten_sell_notional = flatten_sell_qty * flatten_sell_price
        flatten_buy_notional = flatten_buy_qty * flatten_buy_price
        flatten_sell_fee = flatten_sell_notional * fee_rate
        flatten_buy_fee = flatten_buy_notional * fee_rate
        cash = cash + flatten_sell_notional.sum(dim=-1) - flatten_sell_fee.sum(dim=-1)
        cash = cash - flatten_buy_notional.sum(dim=-1) - flatten_buy_fee.sum(dim=-1)
        inventory = torch.zeros_like(inventory)
        final_equity = cash.clone()
        pnl_steps.append(final_equity - prev_equity)
        turnover_steps.append(flatten_sell_notional.sum(dim=-1) + flatten_buy_notional.sum(dim=-1))
    else:
        final_equity = cash + (inventory * last_price).sum(dim=-1)

    pnl_path = torch.stack(pnl_steps, dim=1)
    turnover_path = torch.stack(turnover_steps, dim=1)
    return {
        "allocation": allocation,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "max_long": max_long,
        "max_short": max_short,
        "trade_units": trade_units,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "final_inventory": inventory,
        "pnl_path": pnl_path,
        "turnover_path": turnover_path,
    }


def compute_trading_objective(
    sim_out: Mapping[str, torch.Tensor],
    config: SimulatorConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or SimulatorConfig()
    pnl_path = sim_out["pnl_path"]
    turnover_path = sim_out["turnover_path"]
    final_inventory = sim_out["final_inventory"]
    initial_equity = sim_out["initial_equity"]
    final_equity = sim_out["final_equity"]

    total_raw_pnl = final_equity - initial_equity
    step_returns = pnl_path / initial_equity.unsqueeze(-1).clamp(min=1e-6)
    downside = torch.clamp(-step_returns, min=0.0)
    downside_dev = torch.sqrt((downside.square().mean(dim=-1)) + 1e-8)
    sortino_like = step_returns.mean(dim=-1) / (downside_dev + 1e-6)
    smooth_pnl = pnl_path.mean(dim=-1) / (pnl_path.std(dim=-1, unbiased=False) + 1e-6)
    turnover_norm = turnover_path.sum(dim=-1) / initial_equity.clamp(min=1e-6)
    inventory_penalty = final_inventory.abs().sum(dim=-1)

    score = (
        float(cfg.raw_pnl_weight) * (total_raw_pnl / initial_equity.clamp(min=1e-6))
        + float(cfg.smooth_pnl_weight) * smooth_pnl
        + float(cfg.downside_penalty) * sortino_like
        - float(cfg.turnover_penalty) * turnover_norm
        - float(cfg.inventory_penalty) * inventory_penalty
    )
    loss = -score.mean()
    metrics = {
        "loss": loss.detach(),
        "score": score.mean().detach(),
        "raw_pnl": total_raw_pnl.mean().detach(),
        "return_pct": ((total_raw_pnl / initial_equity.clamp(min=1e-6)) * 100.0).mean().detach(),
        "smooth_pnl": smooth_pnl.mean().detach(),
        "sortino_like": sortino_like.mean().detach(),
        "turnover_norm": turnover_norm.mean().detach(),
        "end_inventory_abs": inventory_penalty.mean().detach(),
        "final_equity": final_equity.mean().detach(),
    }
    return loss, metrics


def _price_levels_from_plan(
    *,
    daily_anchor: torch.Tensor,
    center_offset_bps: torch.Tensor,
    half_spread_bps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    center_frac = center_offset_bps / 10_000.0
    half_spread_frac = half_spread_bps / 10_000.0
    buy_price = daily_anchor * (1.0 + center_frac - half_spread_frac)
    sell_price = daily_anchor * (1.0 + center_frac + half_spread_frac)
    min_price = torch.full_like(daily_anchor, 1e-6)
    return torch.maximum(buy_price, min_price), torch.maximum(sell_price, min_price)


def _validate_ohlc_shapes(
    hourly_open: torch.Tensor,
    hourly_high: torch.Tensor,
    hourly_low: torch.Tensor,
    hourly_close: torch.Tensor,
    hourly_mask: torch.Tensor,
    daily_anchor: torch.Tensor,
) -> None:
    base_shape = hourly_open.shape
    for tensor in (hourly_high, hourly_low, hourly_close, hourly_mask):
        if tensor.shape != base_shape:
            raise ValueError("All hourly tensors must share the same shape [batch, steps, assets].")
    if daily_anchor.shape != (base_shape[0], base_shape[2]):
        raise ValueError("daily_anchor must have shape [batch, assets].")
