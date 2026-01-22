"""Simple Chronos2-based trading algorithm.

This is the baseline algorithm that:
1. Uses Chronos2 to forecast next-period high/low
2. Sets buy price near predicted low, sell near predicted high
3. Tracks its own PnL over time
4. Uses Chronos2 to forecast if tomorrow will be profitable
5. Skips trades when predicted PnL is negative

This serves as a simpler alternative to the neural model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from chronospnltrader.config import SimpleAlgoConfig, SimulationConfig
from chronospnltrader.forecaster import Chronos2Forecaster, PnLForecast, PriceForecast

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Decision for a single trade."""

    should_trade: bool
    buy_price: float
    sell_price: float
    position_size: float
    hold_hours: int
    reason: str

    # Forecasts that led to decision
    price_forecast: Optional[PriceForecast] = None
    pnl_forecast: Optional[PnLForecast] = None


@dataclass
class SimpleAlgoState:
    """State tracking for the simple algorithm."""

    pnl_history: List[float] = field(default_factory=list)
    cumulative_pnl: float = 0.0
    trades_executed: int = 0
    trades_skipped: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def recent_pnl(self) -> np.ndarray:
        """Get recent PnL for forecasting (last 30 days ~210 hours)."""
        return np.array(self.pnl_history[-210:], dtype=np.float32)

    def update(self, pnl: float) -> None:
        """Update state with new PnL."""
        self.pnl_history.append(pnl)
        self.cumulative_pnl += pnl

        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1


class SimpleChronosAlgo:
    """Simple trading algorithm using Chronos2 for price and PnL forecasting.

    Strategy:
    1. Get Chronos2 price forecast (predicted high/low)
    2. Set buy_price = predicted_low * (1 + entry_buffer)
    3. Set sell_price = predicted_high * (1 - exit_buffer)
    4. Get Chronos2 PnL forecast from historical PnL
    5. If predicted PnL < 0, skip the trade
    6. Otherwise, execute and track result
    """

    def __init__(
        self,
        config: SimpleAlgoConfig,
        sim_config: SimulationConfig,
        forecaster: Chronos2Forecaster,
    ) -> None:
        self.config = config
        self.sim_config = sim_config
        self.forecaster = forecaster
        self.state = SimpleAlgoState()

    def decide(
        self,
        current_close: float,
        chronos_high: float,
        chronos_low: float,
        context_df: Optional = None,
    ) -> TradeDecision:
        """Make a trade decision based on Chronos2 forecasts.

        Args:
            current_close: Current close price
            chronos_high: Chronos2 predicted high
            chronos_low: Chronos2 predicted low
            context_df: Optional price context for live forecasting

        Returns:
            TradeDecision with buy/sell prices and whether to trade
        """
        # Calculate expected prices
        buy_price = chronos_low * (1 + self.config.entry_buffer_pct)
        sell_price = chronos_high * (1 - self.config.exit_buffer_pct)

        # Ensure buy < sell (minimum spread = 2 * maker_fee)
        min_spread = 2 * self.sim_config.maker_fee
        if sell_price <= buy_price * (1 + min_spread):
            spread_center = (buy_price + sell_price) / 2
            buy_price = spread_center * (1 - min_spread / 2)
            sell_price = spread_center * (1 + min_spread / 2)

        # Calculate expected return
        expected_return = (sell_price - buy_price) / buy_price - 2 * self.sim_config.maker_fee

        # Check minimum return threshold
        if expected_return < self.config.min_predicted_return:
            return TradeDecision(
                should_trade=False,
                buy_price=buy_price,
                sell_price=sell_price,
                position_size=0.0,
                hold_hours=0,
                reason=f"Expected return {expected_return:.4f} below threshold",
            )

        # Check PnL forecast if we have history
        pnl_forecast = None
        if len(self.state.pnl_history) >= 10:
            pnl_forecast = self.forecaster.forecast_pnl(
                self.state.recent_pnl,
                prediction_length=self.config.pnl_forecast_horizon,
            )

            if pnl_forecast is not None:
                # Skip if predicted PnL is negative
                if pnl_forecast.predicted_pnl < self.config.pnl_min_profit_threshold:
                    return TradeDecision(
                        should_trade=False,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        position_size=0.0,
                        hold_hours=0,
                        reason=f"Predicted PnL {pnl_forecast.predicted_pnl:.6f} below threshold",
                        pnl_forecast=pnl_forecast,
                    )

                # Check downside risk
                if pnl_forecast.downside_risk < -self.config.max_predicted_drawdown:
                    return TradeDecision(
                        should_trade=False,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        position_size=0.0,
                        hold_hours=0,
                        reason=f"Downside risk {pnl_forecast.downside_risk:.4f} too high",
                        pnl_forecast=pnl_forecast,
                    )

        # Calculate position size based on confidence
        confidence = 1.0
        if pnl_forecast is not None:
            # Higher predicted PnL = larger position
            pnl_factor = min(1.0, max(0.1, pnl_forecast.predicted_pnl * 100 + 0.5))
            confidence = pnl_factor * self.config.position_scale

        position_size = min(self.sim_config.max_position_size, confidence)

        # Estimate hold time based on price range
        price_range = (chronos_high - chronos_low) / current_close
        hold_hours = max(1, min(24, int(7 / (price_range * 10 + 0.1))))

        return TradeDecision(
            should_trade=True,
            buy_price=buy_price,
            sell_price=sell_price,
            position_size=position_size,
            hold_hours=hold_hours,
            reason="All checks passed",
            pnl_forecast=pnl_forecast,
        )

    def record_result(self, pnl: float, did_trade: bool) -> None:
        """Record the result of a trade decision."""
        if did_trade:
            self.state.trades_executed += 1
            self.state.update(pnl)
        else:
            self.state.trades_skipped += 1
            # Still record 0 PnL for skipped trades
            self.state.pnl_history.append(0.0)

    def get_stats(self) -> Dict[str, float]:
        """Get algorithm statistics."""
        return {
            "cumulative_pnl": self.state.cumulative_pnl,
            "win_rate": self.state.win_rate,
            "trades_executed": float(self.state.trades_executed),
            "trades_skipped": float(self.state.trades_skipped),
            "skip_rate": self.state.trades_skipped / max(1, self.state.trades_executed + self.state.trades_skipped),
        }

    def reset(self) -> None:
        """Reset algorithm state."""
        self.state = SimpleAlgoState()


def create_simple_algo(
    config: SimpleAlgoConfig,
    sim_config: SimulationConfig,
    forecaster: Chronos2Forecaster,
) -> SimpleChronosAlgo:
    """Factory function to create simple algorithm."""
    return SimpleChronosAlgo(config, sim_config, forecaster)


def simple_algo_batch_decide(
    current_close: torch.Tensor,
    chronos_high: torch.Tensor,
    chronos_low: torch.Tensor,
    pnl_history: torch.Tensor,
    config: SimpleAlgoConfig,
    sim_config: SimulationConfig,
) -> Dict[str, torch.Tensor]:
    """Batch decision making for the simple algorithm (differentiable).

    This is the differentiable version for use during training.
    Allows gradients to flow through the decision process.

    Args:
        current_close: (batch,) current close prices
        chronos_high: (batch,) predicted highs
        chronos_low: (batch,) predicted lows
        pnl_history: (batch, history_len) PnL history
        config: Algorithm config
        sim_config: Simulation config

    Returns:
        Dict with buy_price, sell_price, position_size, hold_hours
    """
    device = current_close.device
    batch_size = current_close.size(0)

    # Calculate prices
    buy_price = chronos_low * (1 + config.entry_buffer_pct)
    sell_price = chronos_high * (1 - config.exit_buffer_pct)

    # Ensure minimum spread
    min_spread = 2 * sim_config.maker_fee
    spread = (sell_price - buy_price) / buy_price
    too_tight = spread < min_spread

    # Adjust prices when spread is too tight
    center = (buy_price + sell_price) / 2
    adjusted_buy = center * (1 - min_spread / 2)
    adjusted_sell = center * (1 + min_spread / 2)

    buy_price = torch.where(too_tight, adjusted_buy, buy_price)
    sell_price = torch.where(too_tight, adjusted_sell, sell_price)

    # Expected return
    expected_return = (sell_price - buy_price) / buy_price - 2 * sim_config.maker_fee

    # PnL-based confidence
    if pnl_history.size(1) >= 10:
        # Exponential weights
        history_len = pnl_history.size(1)
        weights = torch.exp(torch.linspace(-2, 0, history_len, device=device))
        weights = weights / weights.sum()

        # Predicted PnL (simple weighted average)
        predicted_pnl = (pnl_history * weights).sum(dim=1)

        # Confidence scaling
        pnl_std = pnl_history.std(dim=1) + 1e-8
        confidence = torch.sigmoid(predicted_pnl / pnl_std)
    else:
        predicted_pnl = torch.zeros(batch_size, device=device)
        confidence = torch.ones(batch_size, device=device) * 0.5

    # Should trade? (soft decision for differentiability)
    return_ok = torch.sigmoid((expected_return - config.min_predicted_return) * 100)
    pnl_ok = torch.sigmoid((predicted_pnl - config.pnl_min_profit_threshold) * 1000)
    should_trade = return_ok * pnl_ok

    # Position size
    position_size = should_trade * confidence * config.position_scale
    position_size = position_size.clamp(0, sim_config.max_position_size)

    # Hold hours (based on expected range)
    price_range = (chronos_high - chronos_low) / current_close
    hold_hours = (7 / (price_range * 10 + 0.1)).clamp(1, 24)

    return {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "position_size": position_size,
        "hold_hours": hold_hours,
        "should_trade": should_trade,
        "expected_return": expected_return,
        "predicted_pnl": predicted_pnl,
        "confidence": confidence,
    }
