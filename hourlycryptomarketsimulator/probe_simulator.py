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
class ProbeTradeConfig:
    """Configuration for probe trading strategy."""
    probe_trade_amount: float = 0.01  # Minimum probe trade size (1% of normal)
    lookback_trades: int = 2  # Number of recent trades to consider
    min_avg_pnl_pct: float = 0.0  # Minimum average PnL% to trade full size


class ProbeTradingSimulator(HourlyCryptoMarketSimulator):
    """Market simulator with probe trading strategy.

    Probe trading logic:
    - Tracks last 1-2 completed trades and their PnL%
    - If average PnL% is positive, execute full trade
    - Otherwise, execute minimal "probe" trade to test the market
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        probe_config: Optional[ProbeTradeConfig] = None,
    ) -> None:
        super().__init__(config)
        self.probe_config = probe_config or ProbeTradeConfig()
        self.completed_trades: List[dict] = []

    def run(self, bars: pd.DataFrame, actions: pd.DataFrame) -> SimulationResult:
        """Run simulation with probe trading strategy."""
        frame = self._prepare_frame(bars, actions)
        cash = self.config.initial_cash
        inventory = 0.0
        equity_values: List[float] = []
        per_hour_rows: List[dict] = []
        trades: List[TradeRecord] = []

        # Track trades for probe logic
        self.completed_trades = []

        for row in frame.itertuples(index=False):
            # Get base trade intensity from model
            base_intensity = float(np.clip(getattr(row, "trade_amount", 0.0), 0.0, 1.0))

            # Apply probe trading filter
            adjusted_intensity = self._apply_probe_filter(base_intensity)

            buy_fill = bool(row.low <= row.buy_price and adjusted_intensity > 0)
            sell_fill = bool(row.high >= row.sell_price and adjusted_intensity > 0)

            executed_buy = 0.0
            executed_sell = 0.0

            if buy_fill:
                max_buy = cash / (row.buy_price * (1 + self.config.maker_fee)) if row.buy_price > 0 else 0.0
                executed_buy = adjusted_intensity * max_buy

            if sell_fill:
                executed_sell = adjusted_intensity * max(0.0, inventory)

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
                # Track for probe logic (entry price)
                self._record_trade_entry(
                    timestamp=row.timestamp,
                    price=float(row.buy_price),
                    quantity=executed_buy,
                    cost=cost,
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
                # Track for probe logic (exit and calculate PnL)
                self._record_trade_exit(
                    timestamp=row.timestamp,
                    price=float(row.sell_price),
                    quantity=executed_sell,
                    proceeds=proceeds,
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
                    "trade_intensity": adjusted_intensity,
                    "base_intensity": base_intensity,
                }
            )

        equity_curve = pd.Series(equity_values, index=frame["timestamp"].values)
        per_hour = pd.DataFrame(per_hour_rows)
        metrics = self._compute_metrics(equity_curve)

        # Add probe trading metrics
        metrics["total_probe_trades"] = sum(
            1 for t in self.completed_trades if t.get("was_probe", False)
        )
        metrics["total_full_trades"] = sum(
            1 for t in self.completed_trades if not t.get("was_probe", False)
        )

        return SimulationResult(
            equity_curve=equity_curve,
            trades=trades,
            per_hour=per_hour,
            final_cash=cash,
            final_inventory=inventory,
            metrics=metrics,
        )

    def _apply_probe_filter(self, base_intensity: float) -> float:
        """Apply probe trading logic to adjust trade intensity.

        Returns:
            Adjusted intensity (either probe size or full size)
        """
        if base_intensity == 0.0:
            return 0.0

        # Get recent trade performance
        avg_pnl_pct = self._calculate_avg_recent_pnl()

        # Decide: full trade or probe trade
        if avg_pnl_pct is None:
            # No trade history - use probe trade
            return self.probe_config.probe_trade_amount
        elif avg_pnl_pct >= self.probe_config.min_avg_pnl_pct:
            # Positive average PnL - full trade
            return base_intensity
        else:
            # Negative average PnL - probe trade
            return self.probe_config.probe_trade_amount

    def _calculate_avg_recent_pnl(self) -> Optional[float]:
        """Calculate average PnL% of recent completed trades.

        Returns:
            Average PnL% or None if no completed trades
        """
        if not self.completed_trades:
            return None

        # Get last N completed trades
        recent = self.completed_trades[-self.probe_config.lookback_trades:]

        # Filter only trades with PnL calculated (round trips)
        with_pnl = [t for t in recent if "pnl_pct" in t]

        if not with_pnl:
            return None

        # Calculate average PnL%
        avg_pnl = sum(t["pnl_pct"] for t in with_pnl) / len(with_pnl)
        return avg_pnl

    def _record_trade_entry(
        self,
        timestamp: pd.Timestamp,
        price: float,
        quantity: float,
        cost: float,
    ) -> None:
        """Record a buy trade entry."""
        # Store entry for later PnL calculation
        if not hasattr(self, "_pending_entry"):
            self._pending_entry = None

        self._pending_entry = {
            "timestamp": timestamp,
            "entry_price": price,
            "quantity": quantity,
            "cost": cost,
        }

    def _record_trade_exit(
        self,
        timestamp: pd.Timestamp,
        price: float,
        quantity: float,
        proceeds: float,
    ) -> None:
        """Record a sell trade exit and calculate PnL."""
        if not hasattr(self, "_pending_entry") or self._pending_entry is None:
            # No entry to match - skip
            return

        entry = self._pending_entry

        # Calculate PnL%
        pnl_pct = (proceeds - entry["cost"]) / entry["cost"] * 100 if entry["cost"] > 0 else 0.0

        # Determine if this was a probe trade
        was_probe = quantity < (entry["quantity"] * 0.5)  # Heuristic: probe if much smaller

        # Record completed trade
        self.completed_trades.append(
            {
                "entry_timestamp": entry["timestamp"],
                "exit_timestamp": timestamp,
                "entry_price": entry["entry_price"],
                "exit_price": price,
                "quantity": quantity,
                "cost": entry["cost"],
                "proceeds": proceeds,
                "pnl_pct": pnl_pct,
                "was_probe": was_probe,
            }
        )

        # Clear pending entry
        self._pending_entry = None
