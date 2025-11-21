from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .simulator import (
    HourlyCryptoMarketSimulator,
    SimulationConfig,
    SimulationResult,
    TradeRecord,
)


@dataclass
class DailyPnlProbeConfig:
    """Configuration for daily PnL-based probe trading strategy."""
    probe_trade_amount: float = 0.01  # Minimum probe trade size when in probe mode
    min_daily_pnl_to_exit_probe: float = 0.0  # Daily PnL threshold to exit probe mode ($)


class DailyPnlProbeSimulator(HourlyCryptoMarketSimulator):
    """Market simulator with daily PnL-based probe trading strategy.

    Daily PnL probe trading logic:
    - Track cumulative PnL for each day (resets at midnight UTC)
    - If daily PnL goes negative, enter "probe mode" (trade at minimum size)
    - Stay in probe mode until daily PnL returns to positive
    - Once positive, exit probe mode and resume full trading

    This provides drawdown protection by limiting losses on bad days.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        probe_config: Optional[DailyPnlProbeConfig] = None,
    ) -> None:
        super().__init__(config)
        self.probe_config = probe_config or DailyPnlProbeConfig()

    def run(self, bars: pd.DataFrame, actions: pd.DataFrame) -> SimulationResult:
        """Run simulation with daily PnL-based probe trading strategy."""
        frame = self._prepare_frame(bars, actions)
        cash = self.config.initial_cash
        inventory = 0.0
        equity_values: List[float] = []
        per_hour_rows: List[dict] = []
        trades: List[TradeRecord] = []

        # Daily PnL tracking
        current_day: Optional[pd.Timestamp] = None
        daily_start_portfolio_value = self.config.initial_cash
        daily_pnl = 0.0
        in_probe_mode = False
        probe_mode_switches = 0
        hours_in_probe = 0
        hours_in_full = 0

        for row in frame.itertuples(index=False):
            # Get current day (UTC midnight)
            row_day = pd.Timestamp(row.timestamp).normalize()

            # Reset daily PnL at midnight UTC
            if current_day is None or row_day > current_day:
                current_day = row_day
                daily_start_portfolio_value = cash + inventory * row.close
                daily_pnl = 0.0
                # Note: We keep probe mode state across days if still negative overall
                # Only the daily PnL resets, not the probe mode state

            # Calculate current daily PnL
            current_portfolio_value = cash + inventory * row.close
            daily_pnl = current_portfolio_value - daily_start_portfolio_value

            # Determine if we should be in probe mode
            # Enter probe mode if daily PnL is negative
            # Exit probe mode if daily PnL becomes positive
            was_in_probe = in_probe_mode
            if daily_pnl < 0:
                in_probe_mode = True
            elif daily_pnl >= self.probe_config.min_daily_pnl_to_exit_probe:
                in_probe_mode = False

            # Track mode switches
            if was_in_probe != in_probe_mode:
                probe_mode_switches += 1

            # Get base trade intensity from model
            base_buy_intensity = float(np.clip(getattr(row, "buy_amount", getattr(row, "trade_amount", 0.0)), 0.0, 1.0))
            base_sell_intensity = float(np.clip(getattr(row, "sell_amount", getattr(row, "trade_amount", 0.0)), 0.0, 1.0))

            # Apply probe mode filter
            if in_probe_mode and (base_buy_intensity > 0 or base_sell_intensity > 0):
                adjusted_buy = self.probe_config.probe_trade_amount
                adjusted_sell = self.probe_config.probe_trade_amount
                hours_in_probe += 1
            else:
                adjusted_buy = base_buy_intensity
                adjusted_sell = base_sell_intensity
                hours_in_full += 1

            buy_fill = bool(row.low <= row.buy_price and adjusted_buy > 0)
            sell_fill = bool(row.high >= row.sell_price and adjusted_sell > 0)

            executed_buy = 0.0
            executed_sell = 0.0

            if buy_fill:
                max_buy = cash / (row.buy_price * (1 + self.config.maker_fee)) if row.buy_price > 0 else 0.0
                executed_buy = adjusted_buy * max_buy

            if sell_fill:
                executed_sell = adjusted_sell * max(0.0, inventory)

            # Execute buy
            if executed_buy > 0:
                cost = executed_buy * row.buy_price * (1 + self.config.maker_fee)
                cash -= cost
                inventory += executed_buy
                trades.append(
                    TradeRecord(
                        timestamp=row.timestamp,
                        side="buy",
                        price=float(row.buy_price),
                        quantity=executed_buy,
                        cash_after=cash,
                        inventory_after=inventory,
                    )
                )

            # Execute sell
            if executed_sell > 0:
                proceeds = executed_sell * row.sell_price * (1 - self.config.maker_fee)
                cash += proceeds
                inventory -= executed_sell
                trades.append(
                    TradeRecord(
                        timestamp=row.timestamp,
                        side="sell",
                        price=float(row.sell_price),
                        quantity=executed_sell,
                        cash_after=cash,
                        inventory_after=inventory,
                    )
                )

            portfolio_value = cash + inventory * row.close
            equity_values.append(portfolio_value)
            per_hour_rows.append(
                {
                    "timestamp": row.timestamp,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "inventory": inventory,
                    "buy_filled": float(executed_buy > 0),
                    "sell_filled": float(executed_sell > 0),
                    "buy_intensity": adjusted_buy,
                    "sell_intensity": adjusted_sell,
                    "base_buy_intensity": base_buy_intensity,
                    "base_sell_intensity": base_sell_intensity,
                    "in_probe_mode": float(in_probe_mode),
                    "daily_pnl": daily_pnl,
                }
            )

        equity_curve = pd.Series(equity_values, index=frame["timestamp"].values)
        per_hour = pd.DataFrame(per_hour_rows)
        metrics = self._compute_metrics(equity_curve)

        # Add daily PnL probe trading metrics
        metrics["probe_mode_switches"] = probe_mode_switches
        metrics["hours_in_probe"] = hours_in_probe
        metrics["hours_in_full"] = hours_in_full
        metrics["probe_mode_pct"] = (hours_in_probe / (hours_in_probe + hours_in_full) * 100) if (hours_in_probe + hours_in_full) > 0 else 0.0

        return SimulationResult(
            equity_curve=equity_curve,
            trades=trades,
            per_hour=per_hour,
            final_cash=cash,
            final_inventory=inventory,
            metrics=metrics,
        )
