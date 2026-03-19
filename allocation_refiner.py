from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pufferlib_market.evaluate_multiperiod import load_policy
from pufferlib_market.hourly_replay import (
    FEATURES_PER_SYM,
    INITIAL_CASH,
    P_CLOSE,
    P_HIGH,
    P_LOW,
    DailySimResult,
    HourlyMarket,
    HourlyReplayResult,
    MktdData,
    Position,
    _apply_short_borrow_cost,
    _build_obs,
    _close_position,
    _compute_equity,
    _compute_sortino,
    _is_tradable,
    _open_long_limit,
    _open_short_limit,
    load_hourly_market,
    read_mktd,
)
from src.market_sim_early_exit import evaluate_drawdown_vs_profit_early_exit, print_early_exit

_EPS = 1e-8


@dataclass(frozen=True)
class FrozenTrace:
    observations: torch.Tensor
    latents: torch.Tensor
    selected_features: torch.Tensor
    portfolio_state: torch.Tensor
    confidence: torch.Tensor
    logit_gap: torch.Tensor
    selected_position: torch.Tensor
    directions: torch.Tensor
    action_active: torch.Tensor
    signed_returns: torch.Tensor
    short_mask: torch.Tensor
    target_ids: torch.Tensor
    actions: np.ndarray
    max_steps: int
    num_symbols: int


@dataclass(frozen=True)
class RefinedDailySimResult:
    actions: np.ndarray
    allocation_pcts: np.ndarray
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_hold_steps: float


@dataclass(frozen=True)
class RefinerObjectiveSummary:
    objective: float
    total_return: float
    sortino: float
    max_drawdown: float
    drawdown_excess: float
    mean_gross: float
    max_gross: float
    turnover_mean: float


@dataclass(frozen=True)
class RefinerConfig:
    max_leverage: float = 5.0
    fee_rate: float = 0.001
    short_borrow_apr: float = 0.0
    periods_per_year: float = 365.0
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 192
    drawdown_limit: float = 0.25
    sortino_weight: float = 0.15
    drawdown_penalty: float = 2.0
    drawdown_excess_penalty: float = 12.0
    smoothness_penalty: float = 0.2
    leverage_penalty: float = 0.05
    turnover_penalty: float = 0.02
    init_allocation_pct: float = 0.2
    train_max_leverage: float = 1.0
    fill_buffer_bps: float = 5.0
    seed: int = 42


class ResidualExposureHead(nn.Module):
    """Small residual head that refines gross exposure on top of a frozen policy."""

    def __init__(self, input_dim: int, hidden_size: int, *, init_allocation_pct: float) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

        init_pct = float(min(0.99, max(0.01, init_allocation_pct)))
        self.output.bias.data.fill_(math.log(init_pct / (1.0 - init_pct)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.in_norm(x))
        h = h + self.block(h)
        h = self.out_norm(h)
        return torch.sigmoid(self.output(h)).squeeze(-1)


def _policy_hidden(policy: nn.Module, obs_t: torch.Tensor) -> torch.Tensor:
    if hasattr(policy, "encoder"):
        return policy.encoder(obs_t)
    if hasattr(policy, "input_proj") and hasattr(policy, "blocks") and hasattr(policy, "out_norm"):
        h = policy.input_proj(obs_t)
        h = policy.blocks(h)
        return policy.out_norm(h)
    raise ValueError(f"Unsupported policy type for hidden extraction: {type(policy)!r}")


def _decode_legacy_action(action: int, num_symbols: int) -> tuple[int, int]:
    if action <= 0:
        return -1, 0
    if action <= num_symbols:
        return int(action - 1), 1
    if action <= 2 * num_symbols:
        return int(action - num_symbols - 1), -1
    return -1, 0


def _selected_symbol_features(obs: np.ndarray, *, sym_idx: int, num_symbols: int) -> np.ndarray:
    if sym_idx < 0:
        return np.zeros((FEATURES_PER_SYM,), dtype=np.float32)
    start = sym_idx * FEATURES_PER_SYM
    end = start + FEATURES_PER_SYM
    return obs[start:end].astype(np.float32, copy=False)


def _portfolio_state(obs: np.ndarray, *, num_symbols: int) -> tuple[np.ndarray, float]:
    base = num_symbols * FEATURES_PER_SYM
    state = obs[base : base + 5].astype(np.float32, copy=False)
    return state, float(base)


def _replay_price_return(data: MktdData, t: int, sym_idx: int, direction: int) -> float:
    if sym_idx < 0 or direction == 0:
        return 0.0
    current_close = float(data.prices[t, sym_idx, P_CLOSE])
    next_close = float(data.prices[t + 1, sym_idx, P_CLOSE])
    if current_close <= 0.0:
        return 0.0
    raw = (next_close - current_close) / current_close
    return float(direction) * float(raw)


def collect_frozen_trace(
    *,
    checkpoint_path: str,
    data_path: str,
    max_steps: int | None = None,
    fee_rate: float = 0.001,
    fill_buffer_bps: float = 5.0,
    max_leverage: float = 1.0,
    short_borrow_apr: float = 0.0,
    periods_per_year: float = 365.0,
    device: torch.device = torch.device("cpu"),
) -> FrozenTrace:
    data = read_mktd(data_path)
    steps = int(max_steps) if max_steps is not None else int(data.num_timesteps - 1)
    if steps < 1 or steps >= data.num_timesteps:
        raise ValueError(f"max_steps must be in [1, {data.num_timesteps - 1}], got {steps}")

    policy, _, _ = load_policy(checkpoint_path, data.num_symbols, device=device)
    policy.eval()

    cash = float(INITIAL_CASH)
    pos: Position | None = None
    hold_steps = 0
    num_symbols = data.num_symbols
    obs_rows: list[torch.Tensor] = []
    latent_rows: list[torch.Tensor] = []
    selected_rows: list[torch.Tensor] = []
    portfolio_rows: list[torch.Tensor] = []
    confidence_rows: list[float] = []
    gap_rows: list[float] = []
    selected_position_rows: list[float] = []
    direction_rows: list[float] = []
    action_active_rows: list[float] = []
    signed_return_rows: list[float] = []
    short_rows: list[float] = []
    target_id_rows: list[int] = []
    actions = np.zeros((steps,), dtype=np.int32)

    for step in range(steps):
        t = step
        obs_np = _build_obs(data, t, pos, cash, hold_steps, step, steps)
        obs_t = torch.from_numpy(obs_np.astype(np.float32, copy=False)).view(1, -1).to(device=device)
        with torch.no_grad():
            logits, _ = policy(obs_t)
            hidden = _policy_hidden(policy, obs_t)
            probs = torch.softmax(logits, dim=-1)
        action = int(torch.argmax(logits, dim=-1).item())
        conf = float(probs[0, action].item())
        if logits.shape[-1] >= 2:
            top2 = torch.topk(logits[0], k=2).values
            gap = float((top2[0] - top2[1]).item())
        else:
            gap = float(logits[0, action].item())

        sym_idx, direction = _decode_legacy_action(action, num_symbols)
        target_id = -1 if direction == 0 or sym_idx < 0 else int(sym_idx + (num_symbols if direction < 0 else 0))
        selected_feat = _selected_symbol_features(obs_np, sym_idx=sym_idx, num_symbols=num_symbols)
        portfolio_state, base = _portfolio_state(obs_np, num_symbols=num_symbols)
        selected_position = 0.0
        if sym_idx >= 0:
            selected_position = float(obs_np[int(base) + 5 + sym_idx])

        obs_rows.append(torch.from_numpy(obs_np.copy()))
        latent_rows.append(hidden.detach().cpu().view(-1))
        selected_rows.append(torch.from_numpy(selected_feat.copy()))
        portfolio_rows.append(torch.from_numpy(portfolio_state.copy()))
        confidence_rows.append(conf)
        gap_rows.append(gap)
        selected_position_rows.append(selected_position)
        direction_rows.append(float(direction))
        action_active_rows.append(1.0 if direction != 0 else 0.0)
        signed_return_rows.append(_replay_price_return(data, t, sym_idx, direction))
        short_rows.append(1.0 if direction < 0 else 0.0)
        target_id_rows.append(target_id)
        actions[step] = action

        cur_sym = pos.sym if pos is not None else -1
        cur_tradable = _is_tradable(data, t, cur_sym) if cur_sym >= 0 else True

        if action == 0:
            if pos is not None and cur_tradable:
                cash, _ = _close_position(cash, pos, float(data.prices[t, pos.sym, P_CLOSE]), fee_rate)
                pos = None
                hold_steps = 0
            elif pos is not None:
                hold_steps += 1
        elif direction != 0 and sym_idx >= 0:
            target_tradable = _is_tradable(data, t, sym_idx)
            same_target = (
                pos is not None
                and pos.sym == sym_idx
                and ((not pos.is_short and direction > 0) or (pos.is_short and direction < 0))
            )
            if same_target:
                hold_steps += 1
            elif not target_tradable:
                if pos is not None:
                    hold_steps += 1
            elif pos is not None and not cur_tradable:
                hold_steps += 1
            else:
                if pos is not None:
                    cash, _ = _close_position(cash, pos, float(data.prices[t, pos.sym, P_CLOSE]), fee_rate)
                close_px = float(data.prices[t, sym_idx, P_CLOSE])
                low_px = float(data.prices[t, sym_idx, P_LOW])
                high_px = float(data.prices[t, sym_idx, P_HIGH])
                if direction < 0:
                    cash, pos = _open_short_limit(
                        cash=cash,
                        sym=sym_idx,
                        close_price=close_px,
                        low_price=low_px,
                        high_price=high_px,
                        fee_rate=fee_rate,
                        max_leverage=max_leverage,
                        allocation_pct=1.0,
                        level_offset_bps=0.0,
                        fill_buffer_bps=fill_buffer_bps,
                    )
                else:
                    cash, pos = _open_long_limit(
                        cash=cash,
                        sym=sym_idx,
                        close_price=close_px,
                        low_price=low_px,
                        high_price=high_px,
                        fee_rate=fee_rate,
                        max_leverage=max_leverage,
                        allocation_pct=1.0,
                        level_offset_bps=0.0,
                        fill_buffer_bps=fill_buffer_bps,
                    )
                hold_steps = 0
        elif pos is not None:
            hold_steps += 1

        t_new = min(step + 1, data.num_timesteps - 1)
        if pos is not None:
            price_new = float(data.prices[t_new, pos.sym, P_CLOSE])
            cash, _ = _apply_short_borrow_cost(
                cash=cash,
                pos=pos,
                price=price_new,
                short_borrow_apr=short_borrow_apr,
                periods_per_year=periods_per_year,
            )

    return FrozenTrace(
        observations=torch.stack(obs_rows, dim=0).to(torch.float32),
        latents=torch.stack(latent_rows, dim=0).to(torch.float32),
        selected_features=torch.stack(selected_rows, dim=0).to(torch.float32),
        portfolio_state=torch.stack(portfolio_rows, dim=0).to(torch.float32),
        confidence=torch.tensor(confidence_rows, dtype=torch.float32),
        logit_gap=torch.tensor(gap_rows, dtype=torch.float32),
        selected_position=torch.tensor(selected_position_rows, dtype=torch.float32),
        directions=torch.tensor(direction_rows, dtype=torch.float32),
        action_active=torch.tensor(action_active_rows, dtype=torch.float32),
        signed_returns=torch.tensor(signed_return_rows, dtype=torch.float32),
        short_mask=torch.tensor(short_rows, dtype=torch.float32),
        target_ids=torch.tensor(target_id_rows, dtype=torch.long),
        actions=actions,
        max_steps=steps,
        num_symbols=num_symbols,
    )


def build_refiner_inputs(trace: FrozenTrace) -> torch.Tensor:
    base_alloc = torch.full_like(trace.confidence.unsqueeze(-1), 1.0)
    scalars = torch.stack(
        [
            trace.confidence,
            trace.logit_gap,
            trace.directions,
            trace.selected_position,
            trace.action_active,
        ],
        dim=-1,
    )
    return torch.cat(
        [
            trace.latents,
            trace.selected_features,
            trace.portfolio_state,
            scalars,
            base_alloc,
        ],
        dim=-1,
    )


def compute_refiner_objective(
    allocation_pcts: torch.Tensor,
    trace: FrozenTrace,
    config: RefinerConfig,
) -> tuple[torch.Tensor, RefinerObjectiveSummary]:
    alloc = torch.clamp(allocation_pcts, 0.0, 1.0) * trace.action_active
    gross = alloc * float(config.max_leverage)
    prev_gross = torch.cat([torch.zeros(1, dtype=gross.dtype, device=gross.device), gross[:-1]], dim=0)
    prev_ids = torch.cat(
        [torch.full((1,), -1, dtype=trace.target_ids.dtype, device=trace.target_ids.device), trace.target_ids[:-1]],
        dim=0,
    )

    same_target = (trace.target_ids == prev_ids) & (trace.target_ids >= 0)
    prev_active = prev_ids >= 0
    curr_active = trace.target_ids >= 0
    turnover = torch.where(
        curr_active & prev_active & same_target,
        torch.abs(gross - prev_gross),
        torch.where(
            curr_active & prev_active & (~same_target),
            gross + prev_gross,
            torch.where(curr_active, gross, prev_gross),
        ),
    )

    borrow_cost = trace.short_mask * gross * (float(config.short_borrow_apr) / max(float(config.periods_per_year), 1.0))
    fee_cost = float(config.fee_rate) * turnover
    step_returns = gross * trace.signed_returns - fee_cost - borrow_cost
    step_returns = torch.clamp(step_returns, min=-0.95)
    equity = torch.cumprod(1.0 + step_returns, dim=0)
    total_return = equity[-1] - 1.0

    downside = torch.clamp(-step_returns, min=0.0)
    downside_dev = torch.sqrt(torch.mean(downside * downside) + _EPS)
    sortino = (torch.mean(step_returns) / downside_dev) * math.sqrt(float(config.periods_per_year))

    peaks = torch.cummax(equity, dim=0).values
    drawdowns = (peaks - equity) / torch.clamp(peaks, min=_EPS)
    max_dd = torch.max(drawdowns)
    dd_excess = torch.clamp(max_dd - float(config.drawdown_limit), min=0.0)

    smoothness = torch.mean((gross[1:] - gross[:-1]) ** 2) if gross.numel() > 1 else torch.tensor(0.0, dtype=gross.dtype)
    leverage_pen = torch.mean(torch.clamp(gross - 1.0, min=0.0) ** 2)
    turnover_pen = torch.mean(turnover * turnover)

    objective = (
        total_return
        + float(config.sortino_weight) * sortino
        - float(config.drawdown_penalty) * max_dd
        - float(config.drawdown_excess_penalty) * dd_excess
        - float(config.smoothness_penalty) * smoothness
        - float(config.leverage_penalty) * leverage_pen
        - float(config.turnover_penalty) * turnover_pen
    )
    summary = RefinerObjectiveSummary(
        objective=float(objective.detach().cpu().item()),
        total_return=float(total_return.detach().cpu().item()),
        sortino=float(sortino.detach().cpu().item()),
        max_drawdown=float(max_dd.detach().cpu().item()),
        drawdown_excess=float(dd_excess.detach().cpu().item()),
        mean_gross=float(gross.mean().detach().cpu().item()),
        max_gross=float(gross.max().detach().cpu().item()),
        turnover_mean=float(turnover.mean().detach().cpu().item()),
    )
    return objective, summary


def fit_refiner(
    *,
    train_trace: FrozenTrace,
    val_trace: FrozenTrace,
    config: RefinerConfig,
    device: torch.device = torch.device("cpu"),
) -> tuple[ResidualExposureHead, dict[str, object]]:
    torch.manual_seed(int(config.seed))
    train_inputs = build_refiner_inputs(train_trace).to(device=device)
    val_inputs = build_refiner_inputs(val_trace).to(device=device)
    train_trace_gpu = FrozenTrace(
        observations=train_trace.observations.to(device=device),
        latents=train_trace.latents.to(device=device),
        selected_features=train_trace.selected_features.to(device=device),
        portfolio_state=train_trace.portfolio_state.to(device=device),
        confidence=train_trace.confidence.to(device=device),
        logit_gap=train_trace.logit_gap.to(device=device),
        selected_position=train_trace.selected_position.to(device=device),
        directions=train_trace.directions.to(device=device),
        action_active=train_trace.action_active.to(device=device),
        signed_returns=train_trace.signed_returns.to(device=device),
        short_mask=train_trace.short_mask.to(device=device),
        target_ids=train_trace.target_ids.to(device=device),
        actions=train_trace.actions,
        max_steps=train_trace.max_steps,
        num_symbols=train_trace.num_symbols,
    )
    val_trace_gpu = FrozenTrace(
        observations=val_trace.observations.to(device=device),
        latents=val_trace.latents.to(device=device),
        selected_features=val_trace.selected_features.to(device=device),
        portfolio_state=val_trace.portfolio_state.to(device=device),
        confidence=val_trace.confidence.to(device=device),
        logit_gap=val_trace.logit_gap.to(device=device),
        selected_position=val_trace.selected_position.to(device=device),
        directions=val_trace.directions.to(device=device),
        action_active=val_trace.action_active.to(device=device),
        signed_returns=val_trace.signed_returns.to(device=device),
        short_mask=val_trace.short_mask.to(device=device),
        target_ids=val_trace.target_ids.to(device=device),
        actions=val_trace.actions,
        max_steps=val_trace.max_steps,
        num_symbols=val_trace.num_symbols,
    )

    model = ResidualExposureHead(
        input_dim=int(train_inputs.shape[-1]),
        hidden_size=int(config.hidden_size),
        init_allocation_pct=float(config.init_allocation_pct),
    ).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_state = None
    best_val_objective = -float("inf")
    best_metrics: dict[str, object] = {}

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_alloc = model(train_inputs)
        train_objective, train_summary = compute_refiner_objective(train_alloc, train_trace_gpu, config)
        loss = -train_objective
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_alloc = model(val_inputs)
            val_objective, val_summary = compute_refiner_objective(val_alloc, val_trace_gpu, config)
        if float(val_objective.item()) > best_val_objective:
            best_val_objective = float(val_objective.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": int(epoch),
                "train_objective": train_summary.objective,
                "val_objective": val_summary.objective,
                "train_summary": asdict(train_summary),
                "val_summary": asdict(val_summary),
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def predict_allocations(
    model: ResidualExposureHead,
    trace: FrozenTrace,
    *,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model.eval()
    inputs = build_refiner_inputs(trace).to(device=device)
    with torch.no_grad():
        alloc = model(inputs).clamp(0.0, 1.0) * trace.action_active.to(device=device)
    return alloc.detach().cpu().numpy().astype(np.float64, copy=False)


def _target_notional(equity: float, allocation_pct: float, max_leverage: float) -> float:
    alloc = float(max(0.0, min(1.0, allocation_pct)))
    return float(max(0.0, equity) * float(max_leverage) * alloc)


def _position_target_id(pos: Position | None, num_symbols: int) -> int:
    if pos is None:
        return -1
    return int(pos.sym + (num_symbols if pos.is_short else 0))


def _gross_notional(pos: Position | None, price: float) -> float:
    if pos is None or price <= 0.0:
        return 0.0
    return float(abs(pos.qty) * price)


def _open_to_target_notional(
    *,
    cash: float,
    sym: int,
    price: float,
    fee_rate: float,
    is_short: bool,
    target_notional: float,
) -> tuple[float, Position | None]:
    if price <= 0.0 or target_notional <= 0.0:
        return float(cash), None
    denom = float(price) * (1.0 + float(fee_rate))
    if denom <= 0.0:
        return float(cash), None
    qty = float(target_notional) / denom
    if qty <= 0.0:
        return float(cash), None
    if is_short:
        cash += qty * float(price) * (1.0 - float(fee_rate))
    else:
        cash -= qty * denom
    return float(cash), Position(sym=int(sym), is_short=bool(is_short), qty=float(qty), entry_price=float(price))


def _rebalance_same_direction(
    *,
    cash: float,
    pos: Position,
    price: float,
    fee_rate: float,
    target_notional: float,
) -> tuple[float, Position | None, bool]:
    current_notional = _gross_notional(pos, price)
    if target_notional <= 0.0:
        cash, _ = _close_position(cash, pos, price, fee_rate)
        return float(cash), None, current_notional > 0.0

    if price <= 0.0:
        return float(cash), pos, False

    if math.isclose(current_notional, target_notional, rel_tol=1e-6, abs_tol=1e-8):
        return float(cash), pos, False

    delta = float(target_notional - current_notional)
    denom = float(price) * (1.0 + float(fee_rate))
    if denom <= 0.0:
        return float(cash), pos, False
    qty_delta = abs(delta) / float(price)
    if qty_delta <= 0.0:
        return float(cash), pos, False

    if delta > 0.0:
        add_qty = float(delta) / denom
        if add_qty <= 0.0:
            return float(cash), pos, False
        if pos.is_short:
            cash += add_qty * float(price) * (1.0 - float(fee_rate))
        else:
            cash -= add_qty * denom
        pos = Position(sym=pos.sym, is_short=pos.is_short, qty=float(pos.qty + add_qty), entry_price=pos.entry_price)
        return float(cash), pos, True

    reduce_qty = min(float(pos.qty), qty_delta)
    if reduce_qty <= 0.0:
        return float(cash), pos, False
    if pos.is_short:
        cash -= reduce_qty * float(price) * (1.0 + float(fee_rate))
    else:
        cash += reduce_qty * float(price) * (1.0 - float(fee_rate))
    remaining = float(pos.qty - reduce_qty)
    if remaining <= 1e-10:
        return float(cash), None, True
    pos = Position(sym=pos.sym, is_short=pos.is_short, qty=remaining, entry_price=pos.entry_price)
    return float(cash), pos, True


def simulate_daily_actions_with_allocations(
    *,
    data: MktdData,
    actions: np.ndarray,
    allocation_pcts: np.ndarray,
    max_steps: int,
    fee_rate: float = 0.001,
    max_leverage: float = 5.0,
    periods_per_year: float = 365.0,
    short_borrow_apr: float = 0.0,
    fill_buffer_bps: float = 5.0,
    initial_cash: float = INITIAL_CASH,
) -> RefinedDailySimResult:
    if int(actions.shape[0]) != int(max_steps):
        raise ValueError(f"actions length {actions.shape[0]} != max_steps {max_steps}")
    if int(allocation_pcts.shape[0]) != int(max_steps):
        raise ValueError(f"allocation_pcts length {allocation_pcts.shape[0]} != max_steps {max_steps}")
    if max_steps < 1 or max_steps >= data.num_timesteps:
        raise ValueError(f"max_steps must be in [1, {data.num_timesteps - 1}] (got {max_steps})")

    cash = float(initial_cash)
    pos: Position | None = None
    hold_steps = 0
    peak_equity = float(initial_cash)
    max_dd = 0.0
    initial_equity = float(initial_cash)
    num_trades = 0
    winning_trades = 0
    sum_ret = 0.0
    sum_neg_sq = 0.0
    ret_count = 0
    executed_allocs = np.zeros((max_steps,), dtype=np.float64)
    symbols = data.num_symbols
    equity_history: list[float] = [float(initial_cash)]

    for step in range(max_steps):
        t = step
        cur_sym = pos.sym if pos is not None else -1
        cur_tradable = _is_tradable(data, t, cur_sym) if cur_sym >= 0 else True
        price_cur = float(data.prices[t, cur_sym, P_CLOSE]) if cur_sym >= 0 else 0.0
        equity_before = _compute_equity(cash, pos, price_cur) if cur_sym >= 0 else float(cash)

        action = int(actions[step])
        alloc = float(max(0.0, min(1.0, allocation_pcts[step])))
        target_sym, direction = _decode_legacy_action(action, symbols)
        target_id = -1 if direction == 0 or target_sym < 0 else int(target_sym + (symbols if direction < 0 else 0))

        if action == 0 or alloc <= 0.0 or direction == 0 or target_sym < 0:
            if pos is not None and cur_tradable:
                cash, win = _close_position(cash, pos, float(data.prices[t, pos.sym, P_CLOSE]), fee_rate)
                num_trades += 1
                winning_trades += int(win)
                pos = None
                hold_steps = 0
            elif pos is not None:
                hold_steps += 1
            executed_allocs[step] = 0.0
        else:
            target_tradable = _is_tradable(data, t, target_sym)
            same_target = _position_target_id(pos, symbols) == target_id
            target_notional = _target_notional(equity_before, alloc, max_leverage)
            close_px = float(data.prices[t, target_sym, P_CLOSE])
            if not target_tradable:
                if pos is not None:
                    hold_steps += 1
            elif pos is not None and not cur_tradable and not same_target:
                hold_steps += 1
            elif same_target and pos is not None:
                cash, pos, changed = _rebalance_same_direction(
                    cash=cash,
                    pos=pos,
                    price=close_px,
                    fee_rate=fee_rate,
                    target_notional=target_notional,
                )
                if changed:
                    num_trades += 1
                else:
                    hold_steps += 1
                executed_allocs[step] = alloc
            else:
                if pos is not None:
                    cash, win = _close_position(cash, pos, float(data.prices[t, pos.sym, P_CLOSE]), fee_rate)
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
                cash, pos = _open_to_target_notional(
                    cash=cash,
                    sym=target_sym,
                    price=close_px,
                    fee_rate=fee_rate,
                    is_short=direction < 0,
                    target_notional=target_notional,
                )
                if pos is not None:
                    num_trades += 1
                    hold_steps = 0
                    executed_allocs[step] = alloc

        step_new = step + 1
        t_new = min(step_new, data.num_timesteps - 1)
        if pos is None:
            equity_after = float(cash)
        else:
            price_new = float(data.prices[t_new, pos.sym, P_CLOSE])
            cash, _ = _apply_short_borrow_cost(
                cash=cash,
                pos=pos,
                price=price_new,
                short_borrow_apr=short_borrow_apr,
                periods_per_year=periods_per_year,
            )
            equity_after = _compute_equity(cash, pos, price_new)

        ret = 0.0
        if equity_before > 1e-6:
            ret = (equity_after - equity_before) / equity_before
        sum_ret += ret
        ret_count += 1
        if ret < 0.0:
            sum_neg_sq += ret * ret
        peak_equity = max(peak_equity, equity_after)
        dd = (peak_equity - equity_after) / peak_equity if peak_equity > 0 else 0.0
        max_dd = max(max_dd, dd)
        equity_history.append(float(equity_after))
        early_exit = evaluate_drawdown_vs_profit_early_exit(
            equity_history,
            total_steps=max_steps + 1,
            label="pufferlib_market.simulate_daily_actions_with_allocations",
        )
        if early_exit.should_stop:
            print_early_exit(early_exit)

        done = (
            early_exit.should_stop
            or (step_new >= max_steps)
            or (t_new >= data.num_timesteps - 1)
            or (equity_after < initial_cash * 0.01)
        )
        if done:
            if pos is not None:
                price_end = float(data.prices[t_new, pos.sym, P_CLOSE])
                cash, win = _close_position(cash, pos, price_end, fee_rate)
                num_trades += 1
                winning_trades += int(win)
                pos = None
            final_equity = float(cash)
            total_return = (final_equity - initial_equity) / initial_equity
            sortino = 0.0
            if ret_count > 1 and sum_neg_sq > 0.0:
                mean_ret = sum_ret / ret_count
                downside_dev = float(np.sqrt(sum_neg_sq / ret_count))
                sortino = float(mean_ret / max(downside_dev, 1e-12) * np.sqrt(periods_per_year))
            win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0
            avg_hold = float(step_new / num_trades) if num_trades > 0 else 0.0
            return RefinedDailySimResult(
                actions=actions.copy(),
                allocation_pcts=executed_allocs.copy(),
                total_return=float(total_return),
                sortino=float(sortino),
                max_drawdown=float(max_dd),
                num_trades=int(num_trades),
                win_rate=float(win_rate),
                avg_hold_steps=float(avg_hold),
            )

    raise RuntimeError("simulate_daily_actions_with_allocations finished without terminal step")


def replay_hourly_actions_with_allocations(
    *,
    data: MktdData,
    actions: np.ndarray,
    allocation_pcts: np.ndarray,
    market: HourlyMarket,
    start_date: str,
    end_date: str,
    max_steps: int,
    fee_rate: float = 0.001,
    max_leverage: float = 5.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    initial_cash: float = INITIAL_CASH,
) -> HourlyReplayResult:
    symbols = [s.upper() for s in data.symbols]
    S = data.num_symbols
    if actions.shape[0] != max_steps or allocation_pcts.shape[0] != max_steps:
        raise ValueError("actions/allocation length mismatch")

    start_day = pd.to_datetime(start_date, utc=True).floor("D")
    end_day = pd.to_datetime(end_date, utc=True).floor("D")
    daily_days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    if len(daily_days) != data.num_timesteps:
        raise ValueError(
            f"Date range mismatch for daily data: days={len(daily_days)} timesteps={data.num_timesteps}. "
            "Provide the same start/end used during export_data_daily.py."
        )
    if max_steps >= len(daily_days):
        raise ValueError("max_steps must be < num_timesteps (needs t_new)")
    final_day_idx = max_steps

    ref_stock = None
    ref_idx = None
    if data.tradable is not None:
        for i, sym in enumerate(symbols):
            mask = data.tradable[:, i]
            if bool(np.any(mask == 0)):
                ref_stock = sym
                ref_idx = i
                break

    trade_ts: list[pd.Timestamp] = []
    for di, day in enumerate(daily_days):
        ts = day + pd.Timedelta(hours=23)
        if ref_stock is not None and ref_idx is not None and _is_tradable(data, di, ref_idx):
            day_mask = market.index.floor("D") == day
            ref_tr = market.tradable[ref_stock]
            candidates = market.index[day_mask & ref_tr]
            if len(candidates) > 0:
                ts = candidates.max()
        trade_ts.append(ts)
    trade_ts_set = {ts for ts in trade_ts}

    cash = float(initial_cash)
    pos: Position | None = None
    num_trades = 0
    winning_trades = 0
    num_orders = 0
    orders_by_day: dict[str, int] = {}
    equity_curve = np.zeros((len(market.index),), dtype=np.float64)
    peak_equity = float(initial_cash)
    max_dd = 0.0
    stopped_early = False
    last_hi = -1
    last_ts: pd.Timestamp | None = None

    def _count_order(day_ts: pd.Timestamp) -> None:
        nonlocal num_orders
        num_orders += 1
        key = str(day_ts.floor("D").date())
        orders_by_day[key] = orders_by_day.get(key, 0) + 1

    for hi, ts in enumerate(market.index):
        day = ts.floor("D")
        day_idx = int((day - start_day).days)

        if hi > 0 and pos is not None:
            px_carry = float(market.close[symbols[pos.sym]][hi])
            cash, _ = _apply_short_borrow_cost(
                cash=cash,
                pos=pos,
                price=px_carry,
                short_borrow_apr=short_borrow_apr,
                periods_per_year=periods_per_year,
            )

        if ts in trade_ts_set and 0 <= day_idx < max_steps:
            action = int(actions[day_idx])
            alloc = float(max(0.0, min(1.0, allocation_pcts[day_idx])))
            target_sym, direction = _decode_legacy_action(action, S)
            target_id = -1 if direction == 0 or target_sym < 0 else int(target_sym + (S if direction < 0 else 0))
            cur_sym = pos.sym if pos is not None else -1
            cur_day_tr = _is_tradable(data, day_idx, cur_sym) if cur_sym >= 0 else True
            cur_hr_tr = bool(market.tradable[symbols[cur_sym]][hi]) if cur_sym >= 0 else True
            cur_tradable = bool(cur_day_tr and cur_hr_tr)

            def _hour_price(sym_i: int) -> float:
                return float(market.close[symbols[sym_i]][hi])

            price_cur = _hour_price(cur_sym) if cur_sym >= 0 else 0.0
            equity_before = _compute_equity(cash, pos, price_cur) if cur_sym >= 0 else float(cash)

            if action == 0 or alloc <= 0.0 or direction == 0 or target_sym < 0:
                if pos is not None and cur_tradable:
                    cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                    _count_order(ts)
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
            else:
                target_day_tr = _is_tradable(data, day_idx, target_sym)
                target_hr_tr = bool(market.tradable[symbols[target_sym]][hi])
                target_tradable = bool(target_day_tr and target_hr_tr)
                same_target = _position_target_id(pos, S) == target_id
                target_notional = _target_notional(equity_before, alloc, max_leverage)

                if not target_tradable:
                    pass
                elif pos is not None and not cur_tradable and not same_target:
                    pass
                elif same_target and pos is not None:
                    cash, pos, changed = _rebalance_same_direction(
                        cash=cash,
                        pos=pos,
                        price=_hour_price(target_sym),
                        fee_rate=fee_rate,
                        target_notional=target_notional,
                    )
                    if changed:
                        _count_order(ts)
                        num_trades += 1
                else:
                    if pos is not None:
                        cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                        _count_order(ts)
                        num_trades += 1
                        winning_trades += int(win)
                        pos = None
                    cash, pos = _open_to_target_notional(
                        cash=cash,
                        sym=target_sym,
                        price=_hour_price(target_sym),
                        fee_rate=fee_rate,
                        is_short=direction < 0,
                        target_notional=target_notional,
                    )
                    if pos is not None:
                        _count_order(ts)
                        num_trades += 1

        if pos is None:
            equity = float(cash)
        else:
            px = float(market.close[symbols[pos.sym]][hi])
            equity = _compute_equity(cash, pos, px)
        equity_curve[hi] = equity
        last_hi = hi
        last_ts = ts
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = max(max_dd, dd)
        early_exit = evaluate_drawdown_vs_profit_early_exit(
            equity_curve[: hi + 1],
            total_steps=len(market.index),
            label="pufferlib_market.replay_hourly_actions_with_allocations",
        )
        if early_exit.should_stop:
            print_early_exit(early_exit)
            stopped_early = True
            break

    if stopped_early:
        if pos is not None and last_hi >= 0 and last_ts is not None:
            px_end = float(market.close[symbols[pos.sym]][last_hi])
            cash, win = _close_position(cash, pos, px_end, fee_rate)
            _count_order(last_ts)
            num_trades += 1
            winning_trades += int(win)
            pos = None
            equity_curve[last_hi] = float(cash)
        used_equity_curve = equity_curve[: max(last_hi + 1, 0)]
    else:
        final_close_ts = trade_ts[final_day_idx]
        if final_close_ts in market.index and pos is not None:
            hi_end = int(market.index.get_loc(final_close_ts))
            px_end = float(market.close[symbols[pos.sym]][hi_end])
            cash, win = _close_position(cash, pos, px_end, fee_rate)
            _count_order(final_close_ts)
            num_trades += 1
            winning_trades += int(win)
            pos = None
        used_equity_curve = equity_curve

    final_equity = float(cash)
    total_return = (final_equity - float(initial_cash)) / float(initial_cash)
    rets = (used_equity_curve[1:] - used_equity_curve[:-1]) / np.clip(used_equity_curve[:-1], 1e-12, None)
    sortino = _compute_sortino(rets.astype(np.float64, copy=False), periods_per_year)
    win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0
    return HourlyReplayResult(
        total_return=float(total_return),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        num_trades=int(num_trades),
        num_orders=int(num_orders),
        win_rate=float(win_rate),
        equity_curve=used_equity_curve,
        orders_by_day=orders_by_day,
    )


def _evaluate_refined_trace(
    *,
    checkpoint_path: str,
    data_path: str,
    start_date: str,
    end_date: str,
    hourly_data_root: str,
    model: ResidualExposureHead,
    config: RefinerConfig,
    device: torch.device,
) -> dict[str, object]:
    trace = collect_frozen_trace(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        fee_rate=config.fee_rate,
        fill_buffer_bps=config.fill_buffer_bps,
        max_leverage=config.train_max_leverage,
        short_borrow_apr=config.short_borrow_apr,
        periods_per_year=config.periods_per_year,
        device=device,
    )
    alloc = predict_allocations(model, trace, device=device)
    data = read_mktd(data_path)
    daily = simulate_daily_actions_with_allocations(
        data=data,
        actions=trace.actions,
        allocation_pcts=alloc,
        max_steps=trace.max_steps,
        fee_rate=config.fee_rate,
        max_leverage=config.max_leverage,
        periods_per_year=config.periods_per_year,
        short_borrow_apr=config.short_borrow_apr,
        fill_buffer_bps=config.fill_buffer_bps,
    )
    market = load_hourly_market(
        data.symbols,
        hourly_data_root,
        start=f"{start_date} 00:00",
        end=f"{end_date} 23:00",
    )
    hourly = replay_hourly_actions_with_allocations(
        data=data,
        actions=trace.actions,
        allocation_pcts=alloc,
        market=market,
        start_date=start_date,
        end_date=end_date,
        max_steps=trace.max_steps,
        fee_rate=config.fee_rate,
        max_leverage=config.max_leverage,
        periods_per_year=8760.0,
        short_borrow_apr=config.short_borrow_apr,
    )
    _, objective_summary = compute_refiner_objective(
        torch.from_numpy(alloc.astype(np.float32)),
        trace,
        config,
    )
    return {
        "checkpoint": checkpoint_path,
        "data_path": data_path,
        "date_range": {"start": start_date, "end": end_date},
        "daily": {
            "total_return": daily.total_return,
            "sortino": daily.sortino,
            "max_drawdown": daily.max_drawdown,
            "num_trades": daily.num_trades,
            "win_rate": daily.win_rate,
            "avg_hold_steps": daily.avg_hold_steps,
        },
        "hourly_replay": {
            "total_return": hourly.total_return,
            "sortino": hourly.sortino,
            "max_drawdown": hourly.max_drawdown,
            "num_trades": hourly.num_trades,
            "num_orders": hourly.num_orders,
            "win_rate": hourly.win_rate,
        },
        "allocation_summary": {
            "mean_alloc_pct": float(np.mean(alloc)),
            "max_alloc_pct": float(np.max(alloc)),
            "mean_gross": float(np.mean(alloc) * float(config.max_leverage)),
            "max_gross": float(np.max(alloc) * float(config.max_leverage)),
        },
        "objective_summary": asdict(objective_summary),
    }


def _save_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc residual allocation refiner for daily pufferlib_market policies")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--val-start-date", required=True)
    parser.add_argument("--val-end-date", required=True)
    parser.add_argument("--hourly-data-root", default="trainingdatahourly")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drawdown-limit", type=float, default=0.25)
    parser.add_argument("--drawdown-penalty", type=float, default=2.0)
    parser.add_argument("--drawdown-excess-penalty", type=float, default=12.0)
    parser.add_argument("--sortino-weight", type=float, default=0.15)
    parser.add_argument("--smoothness-penalty", type=float, default=0.2)
    parser.add_argument("--leverage-penalty", type=float, default=0.05)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-eval-data", default="")
    parser.add_argument("--extra-eval-name", default="")
    parser.add_argument("--extra-start-date", default="")
    parser.add_argument("--extra-end-date", default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = RefinerConfig(
        max_leverage=float(args.max_leverage),
        fee_rate=float(args.fee_rate),
        short_borrow_apr=float(args.short_borrow_apr),
        periods_per_year=float(args.periods_per_year),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_size=int(args.hidden_size),
        drawdown_limit=float(args.drawdown_limit),
        sortino_weight=float(args.sortino_weight),
        drawdown_penalty=float(args.drawdown_penalty),
        drawdown_excess_penalty=float(args.drawdown_excess_penalty),
        smoothness_penalty=float(args.smoothness_penalty),
        leverage_penalty=float(args.leverage_penalty),
        turnover_penalty=float(args.turnover_penalty),
        init_allocation_pct=min(1.0, max(0.01, 1.0 / max(float(args.max_leverage), 1.0))),
        fill_buffer_bps=float(args.fill_buffer_bps),
        seed=int(args.seed),
    )

    train_trace = collect_frozen_trace(
        checkpoint_path=args.checkpoint,
        data_path=args.train_data,
        fee_rate=config.fee_rate,
        fill_buffer_bps=config.fill_buffer_bps,
        max_leverage=config.train_max_leverage,
        short_borrow_apr=config.short_borrow_apr,
        periods_per_year=config.periods_per_year,
        device=device,
    )
    val_trace = collect_frozen_trace(
        checkpoint_path=args.checkpoint,
        data_path=args.val_data,
        fee_rate=config.fee_rate,
        fill_buffer_bps=config.fill_buffer_bps,
        max_leverage=config.train_max_leverage,
        short_borrow_apr=config.short_borrow_apr,
        periods_per_year=config.periods_per_year,
        device=device,
    )
    model, train_metrics = fit_refiner(
        train_trace=train_trace,
        val_trace=val_trace,
        config=config,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "checkpoint": str(args.checkpoint),
        },
        output_dir / "refiner.pt",
    )
    _save_report(output_dir / "train_metrics.json", train_metrics)

    val_report = _evaluate_refined_trace(
        checkpoint_path=args.checkpoint,
        data_path=args.val_data,
        start_date=args.val_start_date,
        end_date=args.val_end_date,
        hourly_data_root=args.hourly_data_root,
        model=model,
        config=config,
        device=device,
    )
    _save_report(output_dir / "val_report.json", val_report)

    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "config": asdict(config),
        "train_metrics": train_metrics,
        "val_report": val_report,
    }
    if args.extra_eval_data and args.extra_start_date and args.extra_end_date:
        extra_name = args.extra_eval_name or "extra_eval"
        extra_report = _evaluate_refined_trace(
            checkpoint_path=args.checkpoint,
            data_path=args.extra_eval_data,
            start_date=args.extra_start_date,
            end_date=args.extra_end_date,
            hourly_data_root=args.hourly_data_root,
            model=model,
            config=config,
            device=device,
        )
        report[extra_name] = extra_report
        _save_report(output_dir / f"{extra_name}.json", extra_report)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
