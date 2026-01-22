"""Differentiable market simulator with PnL tracking.

Simulates hourly trades with:
- Soft fill probabilities for differentiability
- PnL tracking over time for Chronos2 forecasting
- Temperature annealing for hard → soft transition
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from chronospnltrader.config import SimulationConfig, TrainingConfig


@dataclass
class TradeResult:
    """Result from simulating a single trade."""

    entry_filled: torch.Tensor  # (batch,) soft probability of entry
    exit_tp_filled: torch.Tensor  # (batch,) take-profit hit probability
    exit_forced: torch.Tensor  # (batch,) forced exit probability
    pnl: torch.Tensor  # (batch,) profit/loss
    returns: torch.Tensor  # (batch,) return percentage
    actual_hold_hours: torch.Tensor  # (batch,) weighted hold time
    fees: torch.Tensor  # (batch,) total fees paid


@dataclass
class SimulationHistory:
    """Tracks simulation history for PnL forecasting."""

    returns: List[torch.Tensor]  # Per-bar returns
    cumulative_pnl: torch.Tensor  # Running cumulative PnL
    trade_count: int
    win_count: int

    @classmethod
    def create(cls, device: torch.device) -> "SimulationHistory":
        return cls(
            returns=[],
            cumulative_pnl=torch.tensor(0.0, device=device),
            trade_count=0,
            win_count=0,
        )

    def update(self, returns: torch.Tensor) -> None:
        """Update history with new returns."""
        self.returns.append(returns.detach())
        self.cumulative_pnl = self.cumulative_pnl + returns.mean()
        self.trade_count += returns.numel()
        self.win_count += (returns > 0).sum().item()

    @property
    def win_rate(self) -> float:
        return self.win_count / max(1, self.trade_count)

    def get_pnl_tensor(self, max_history: int = 210) -> torch.Tensor:
        """Get PnL history as a tensor for forecasting."""
        if not self.returns:
            return torch.zeros(1)

        # Concatenate and take last max_history values
        all_returns = torch.cat([r.flatten() for r in self.returns])
        return all_returns[-max_history:]


def _soft_fill_prob(
    target_price: torch.Tensor,
    bar_price: torch.Tensor,
    reference_price: torch.Tensor,
    is_buy: bool,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Compute soft fill probability for limit orders.

    For buy: fills if bar_low <= buy_price
    For sell: fills if bar_high >= sell_price
    """
    price_diff = target_price - bar_price
    normalized_diff = price_diff / (reference_price * temperature + 1e-8)

    if is_buy:
        # Buy fills when bar_low <= buy_price
        fill_prob = torch.sigmoid(normalized_diff)
    else:
        # Sell fills when bar_high >= sell_price
        fill_prob = torch.sigmoid(-normalized_diff)

    return fill_prob


def simulate_trade(
    *,
    future_highs: torch.Tensor,  # (batch, 24)
    future_lows: torch.Tensor,  # (batch, 24)
    future_closes: torch.Tensor,  # (batch, 24)
    buy_price: torch.Tensor,  # (batch,)
    sell_price: torch.Tensor,  # (batch,)
    position_length: torch.Tensor,  # (batch,) 0-24 soft hours
    position_size: torch.Tensor,  # (batch,) 0-1
    reference_price: torch.Tensor,
    config: SimulationConfig,
    temperature: float = 0.01,
) -> TradeResult:
    """Simulate a single hourly trade decision.

    Flow:
    1. Hour 0: Check if buy fills (low <= buy_price)
    2. Hours 1 to position_length: Check if sell fills (high >= sell_price)
    3. At position_length: Forced exit at close with slippage

    position_length = 0 means skip trade.
    """
    batch_size = buy_price.size(0)
    max_hours = future_highs.size(1)
    device = buy_price.device

    # Soft skip signal: position_length < 0.5 = skip
    skip_weight = torch.sigmoid((0.5 - position_length) * 10)
    trade_weight = 1.0 - skip_weight

    # Entry check on hour 0
    entry_prob = _soft_fill_prob(
        buy_price,
        future_lows[:, 0],
        reference_price,
        is_buy=True,
        temperature=temperature,
    )
    entry_prob = entry_prob * trade_weight

    # Track holding probability
    cumulative_holding = torch.ones(batch_size, device=device)
    weighted_exit_price = torch.zeros(batch_size, device=device)
    tp_probability = torch.zeros(batch_size, device=device)
    forced_exit_prob = torch.zeros(batch_size, device=device)
    weighted_hold_hours = torch.zeros(batch_size, device=device)

    position_length_clamped = position_length.clamp(0.5, max_hours)

    for hour in range(1, max_hours):
        hour_high = future_highs[:, hour]
        hour_close = future_closes[:, hour]

        # Still within holding period?
        hours_remaining = position_length_clamped - hour
        still_holding_weight = torch.sigmoid(hours_remaining * 5)

        # Take-profit check
        tp_prob_now = _soft_fill_prob(
            sell_price,
            hour_high,
            reference_price,
            is_buy=False,
            temperature=temperature,
        )

        # TP this hour
        tp_this_hour = cumulative_holding * tp_prob_now * still_holding_weight

        tp_probability = tp_probability + tp_this_hour
        weighted_exit_price = weighted_exit_price + tp_this_hour * sell_price
        weighted_hold_hours = weighted_hold_hours + tp_this_hour * hour

        # Update holding
        cumulative_holding = cumulative_holding * (1 - tp_prob_now * still_holding_weight)

        # Forced exit at position_length
        exit_hour_proximity = 1.0 - torch.abs(position_length_clamped - hour).clamp(0, 1)
        forced_exit_this_hour = cumulative_holding * exit_hour_proximity * (1 - still_holding_weight)

        forced_exit_price = hour_close * (1 - config.forced_exit_slippage)
        weighted_exit_price = weighted_exit_price + forced_exit_this_hour * forced_exit_price
        weighted_hold_hours = weighted_hold_hours + forced_exit_this_hour * hour
        forced_exit_prob = forced_exit_prob + forced_exit_this_hour

        cumulative_holding = cumulative_holding * (1 - exit_hour_proximity)

    # Handle remaining at max hours
    final_forced = cumulative_holding
    final_exit_price = future_closes[:, -1] * (1 - config.forced_exit_slippage)
    weighted_exit_price = weighted_exit_price + final_forced * final_exit_price
    weighted_hold_hours = weighted_hold_hours + final_forced * max_hours
    forced_exit_prob = forced_exit_prob + final_forced

    total_exit_prob = (tp_probability + forced_exit_prob).clamp(max=1.0)

    # Compute returns
    effective_exit_price = weighted_exit_price / (total_exit_prob.clamp(min=1e-8))
    gross_return = (effective_exit_price - buy_price) / (buy_price + 1e-8)

    # Fees
    total_fees = 2 * config.maker_fee
    net_return = (gross_return - total_fees) * entry_prob * position_size

    actual_hold_hours = weighted_hold_hours * entry_prob

    return TradeResult(
        entry_filled=entry_prob,
        exit_tp_filled=tp_probability.clamp(0, 1),
        exit_forced=forced_exit_prob.clamp(0, 1),
        pnl=net_return * reference_price,
        returns=net_return,
        actual_hold_hours=actual_hold_hours,
        fees=torch.full_like(entry_prob, total_fees) * entry_prob,
    )


def compute_loss(
    result: TradeResult,
    pnl_history: torch.Tensor,
    length_probs: torch.Tensor,
    position_length: torch.Tensor,
    buy_offset: torch.Tensor,
    sell_offset: torch.Tensor,
    pnl_forecast: Dict[str, torch.Tensor],
    config: TrainingConfig,
) -> Dict[str, torch.Tensor]:
    """Compute training loss with Chronos2 PnL forecast as judge.

    Key innovation: The loss includes a term that maximizes the
    predicted next-day profitability from the PnL forecaster.
    This trains the model to generate trades that Chronos2
    predicts will be profitable.

    Components:
    1. Sortino ratio (risk-adjusted returns)
    2. PnL forecast alignment (Chronos2 judge)
    3. Raw returns
    4. Penalties (forced exit, no trade, spread utilization)
    """
    returns = result.returns
    mean_return = returns.mean()

    # Sortino ratio
    downside = torch.clamp(-returns, min=0.0)
    downside_std = torch.sqrt((downside ** 2).mean() + 1e-8)
    sortino = mean_return / downside_std * math.sqrt(24 * 365)

    # === KEY INNOVATION: PnL Forecast Loss ===
    # Maximize predicted profitability
    predicted_pnl = pnl_forecast["predicted_pnl"]
    confidence = pnl_forecast["confidence"]

    # The model should generate trades where:
    # 1. Chronos2 predicts positive PnL
    # 2. Actual returns align with prediction
    pnl_forecast_loss = -config.pnl_forecast_weight * (
        predicted_pnl.mean() +  # Maximize predicted PnL
        0.5 * (predicted_pnl * returns).mean()  # Alignment bonus
    )

    # Standard losses
    sortino_loss = -config.sortino_weight * sortino
    return_loss = -config.return_weight * mean_return * 24 * 365

    # Forced exit penalty
    forced_exit_rate = result.exit_forced.mean()
    forced_penalty = config.forced_exit_penalty * forced_exit_rate

    # No-trade penalty
    no_trade_prob = length_probs[:, 0].mean()
    no_trade_penalty = config.no_trade_penalty * no_trade_prob

    # Spread utilization
    avg_offset = (buy_offset + sell_offset).mean()
    min_offset = config.min_price_offset_pct
    max_offset = config.max_price_offset_pct
    offset_utilization = (avg_offset - min_offset) / (max_offset - min_offset + 1e-8)
    spread_util_loss = -config.spread_utilization * offset_utilization

    total_loss = (
        sortino_loss
        + pnl_forecast_loss
        + return_loss
        + forced_penalty
        + no_trade_penalty
        + spread_util_loss
    )

    return {
        "loss": total_loss,
        "sortino_loss": sortino_loss,
        "pnl_forecast_loss": pnl_forecast_loss,
        "return_loss": return_loss,
        "forced_penalty": forced_penalty,
        "no_trade_penalty": no_trade_penalty,
        "spread_util_loss": spread_util_loss,
        # Metrics
        "sortino": sortino,
        "mean_return": mean_return,
        "predicted_pnl": predicted_pnl.mean(),
        "confidence": confidence.mean(),
        "forced_exit_rate": forced_exit_rate,
        "no_trade_rate": no_trade_prob,
        "avg_offset": avg_offset,
        "tp_rate": result.exit_tp_filled.mean(),
        "avg_hold_hours": result.actual_hold_hours.mean(),
        "avg_position_length": position_length.mean(),
        "entry_rate": result.entry_filled.mean(),
    }


def simulate_batch(
    *,
    batch: Dict[str, torch.Tensor],
    actions: Dict[str, torch.Tensor],
    config: SimulationConfig,
    temperature: float = 0.01,
) -> TradeResult:
    """Convenience function to simulate a batch of trades."""
    return simulate_trade(
        future_highs=batch["future_highs"],
        future_lows=batch["future_lows"],
        future_closes=batch["future_closes"],
        buy_price=actions["buy_price"],
        sell_price=actions["sell_price"],
        position_length=actions["position_length"],
        position_size=actions["position_size"],
        reference_price=batch["current_close"],
        config=config,
        temperature=temperature,
    )


def run_simulation_30_days(
    data_module,
    model,
    config: SimulationConfig,
    device: torch.device,
    use_simple_algo: bool = False,
) -> Dict[str, float]:
    """Run simulation over 30 days (~210 hours) for evaluation.

    Returns metrics including PnL, Sortino, win rate.
    """
    model.eval()
    history = SimulationHistory.create(device)

    dataloader = data_module.val_dataloader(batch_size=1, num_workers=0)

    total_pnl = 0.0
    returns_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_simple_algo:
                # Use simple algorithm
                from chronospnltrader.simple_algo import simple_algo_batch_decide
                from chronospnltrader.config import SimpleAlgoConfig

                actions = simple_algo_batch_decide(
                    current_close=batch["current_close"],
                    chronos_high=batch["chronos_high"][:, -1],
                    chronos_low=batch["chronos_low"][:, -1],
                    pnl_history=batch["pnl_history"],
                    config=SimpleAlgoConfig(),
                    sim_config=config,
                )
                actions["position_length"] = actions["hold_hours"]
            else:
                # Use neural model
                outputs = model(batch["features"])
                actions = model.decode_actions(
                    outputs,
                    batch["current_close"],
                    temperature=0.001,  # Hard decisions
                )

            result = simulate_batch(
                batch=batch,
                actions=actions,
                config=config,
                temperature=0.001,
            )

            pnl = result.returns.item()
            total_pnl += pnl
            returns_list.append(pnl)
            history.update(result.returns)

    # Calculate metrics
    returns_arr = torch.tensor(returns_list)
    mean_return = returns_arr.mean().item()
    std_return = returns_arr.std().item() + 1e-8
    downside = torch.clamp(-returns_arr, min=0.0)
    downside_std = torch.sqrt((downside ** 2).mean()).item() + 1e-8

    sharpe = mean_return / std_return * math.sqrt(24 * 365)
    sortino = mean_return / downside_std * math.sqrt(24 * 365)

    return {
        "total_pnl": total_pnl,
        "mean_return": mean_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": history.win_rate,
        "trade_count": history.trade_count,
    }
