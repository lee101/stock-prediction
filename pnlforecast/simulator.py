"""Strategy PnL simulator for historical backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SimulationConfigPnL, StrategyConfigPnL
from .strategy import StrategyThresholds

logger = logging.getLogger(__name__)


@dataclass
class DayTrade:
    """Record of a single day's trading activity."""

    trade_date: date
    open_price: float
    high: float
    low: float
    close: float
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    trade_type: str = "no_trade"  # round_trip, buy_pending, sell_close, no_trade
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    fees: float = 0.0
    position_held: bool = False


@dataclass
class StrategyPnLResult:
    """Result of simulating a strategy over multiple days."""

    strategy: StrategyThresholds
    symbol: str
    start_date: date
    end_date: date

    # PnL series
    daily_pnl: List[float] = field(default_factory=list)
    cumulative_pnl: List[float] = field(default_factory=list)
    dates: List[date] = field(default_factory=list)

    # Trade statistics
    trades: List[DayTrade] = field(default_factory=list)
    total_trades: int = 0
    round_trips: int = 0
    winning_trades: int = 0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_fees: float = 0.0

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if self.round_trips == 0:
            return 0.0
        return self.winning_trades / self.round_trips

    @property
    def avg_trade_pnl(self) -> float:
        """Average PnL per round-trip trade."""
        if self.round_trips == 0:
            return 0.0
        return self.total_net_pnl / self.round_trips

    @property
    def pnl_series(self) -> pd.Series:
        """Get daily PnL as pandas Series."""
        if not self.dates:
            return pd.Series(dtype=float)
        return pd.Series(self.daily_pnl, index=pd.DatetimeIndex(self.dates))

    @property
    def cumulative_series(self) -> pd.Series:
        """Get cumulative PnL as pandas Series."""
        if not self.dates:
            return pd.Series(dtype=float)
        return pd.Series(self.cumulative_pnl, index=pd.DatetimeIndex(self.dates))


class StrategySimulator:
    """Simulates buy-low/sell-high strategies on historical data."""

    def __init__(
        self,
        strategy_config: StrategyConfigPnL,
        sim_config: SimulationConfigPnL,
    ) -> None:
        self.strategy_config = strategy_config
        self.sim_config = sim_config

    def simulate_strategy(
        self,
        strategy: StrategyThresholds,
        symbol: str,
        price_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        is_crypto: bool = False,
    ) -> StrategyPnLResult:
        """Simulate a single strategy on historical data.

        Args:
            strategy: The threshold strategy to simulate
            symbol: Trading symbol
            price_data: DataFrame with columns [date, open, high, low, close]
            start_date: Start date for simulation
            end_date: End date for simulation
            is_crypto: Whether this is a crypto asset

        Returns:
            StrategyPnLResult with full simulation results
        """
        fee_pct = self.strategy_config.get_fee_pct(is_crypto)

        # Filter data to date range
        if "date" in price_data.columns:
            mask = (price_data["date"] >= start_date) & (price_data["date"] <= end_date)
        else:
            # Use timestamp
            start_ts = pd.Timestamp(start_date, tz="UTC")
            end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            mask = (price_data["timestamp"] >= start_ts) & (price_data["timestamp"] < end_ts)

        filtered = price_data[mask].copy()
        if filtered.empty:
            return StrategyPnLResult(
                strategy=strategy,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        # Simulation state
        result = StrategyPnLResult(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        holding_position = False
        entry_price = 0.0
        cumulative = 0.0

        for _, row in filtered.iterrows():
            trade_date = row["date"] if "date" in row.index else row["timestamp"].date()
            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            buy_price = strategy.get_buy_price(open_price)
            sell_price = strategy.get_sell_price(open_price)

            can_buy = low <= buy_price
            can_sell = high >= sell_price

            day_trade = DayTrade(
                trade_date=trade_date,
                open_price=open_price,
                high=high,
                low=low,
                close=close,
            )

            day_pnl = 0.0

            if not holding_position:
                # Looking to enter
                if can_buy and can_sell:
                    # Round-trip in single day
                    day_trade.buy_price = buy_price
                    day_trade.sell_price = sell_price
                    day_trade.trade_type = "round_trip"

                    gross = (sell_price - buy_price) / buy_price
                    fees = 2 * fee_pct
                    net = gross - fees

                    day_trade.gross_pnl = gross
                    day_trade.fees = fees
                    day_trade.net_pnl = net

                    day_pnl = net
                    result.round_trips += 1
                    result.total_trades += 1
                    if net > 0:
                        result.winning_trades += 1

                elif can_buy:
                    # Buy but can't sell today - hold position
                    day_trade.buy_price = buy_price
                    day_trade.trade_type = "buy_pending"
                    day_trade.position_held = True

                    holding_position = True
                    entry_price = buy_price
                    result.total_trades += 1

                else:
                    # No entry
                    day_trade.trade_type = "no_trade"

            else:
                # Already holding - looking to exit
                day_trade.position_held = True

                # Check if we can sell at target
                if can_sell:
                    # Sell at target
                    day_trade.sell_price = sell_price
                    day_trade.trade_type = "sell_close"

                    gross = (sell_price - entry_price) / entry_price
                    fees = 2 * fee_pct  # Entry + exit
                    net = gross - fees

                    day_trade.gross_pnl = gross
                    day_trade.fees = fees
                    day_trade.net_pnl = net

                    day_pnl = net
                    result.round_trips += 1
                    if net > 0:
                        result.winning_trades += 1

                    holding_position = False
                    entry_price = 0.0

                else:
                    # Still holding - mark-to-market but no realized PnL
                    day_trade.trade_type = "holding"
                    # Unrealized P&L (not counted in daily_pnl)
                    day_trade.gross_pnl = (close - entry_price) / entry_price

            cumulative += day_pnl

            result.trades.append(day_trade)
            result.dates.append(trade_date)
            result.daily_pnl.append(day_pnl)
            result.cumulative_pnl.append(cumulative)

            result.total_gross_pnl += day_trade.gross_pnl
            result.total_net_pnl += day_trade.net_pnl
            result.total_fees += day_trade.fees

        return result

    def simulate_all_strategies(
        self,
        strategies: List[StrategyThresholds],
        symbol: str,
        price_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        is_crypto: bool = False,
    ) -> Dict[str, StrategyPnLResult]:
        """Simulate multiple strategies on the same data.

        Args:
            strategies: List of strategies to simulate
            symbol: Trading symbol
            price_data: DataFrame with OHLC data
            start_date: Start date
            end_date: End date
            is_crypto: Whether crypto asset

        Returns:
            Dict mapping strategy_id -> StrategyPnLResult
        """
        results = {}
        for strategy in strategies:
            result = self.simulate_strategy(
                strategy=strategy,
                symbol=symbol,
                price_data=price_data,
                start_date=start_date,
                end_date=end_date,
                is_crypto=is_crypto,
            )
            results[strategy.strategy_id] = result

        return results


def get_strategy_pnl_history(
    result: StrategyPnLResult,
    n_days: int = 7,
) -> np.ndarray:
    """Extract recent PnL history for forecasting.

    Args:
        result: Strategy simulation result
        n_days: Number of recent days to extract

    Returns:
        NumPy array of daily PnL values (most recent n_days)
    """
    if len(result.daily_pnl) < n_days:
        # Pad with zeros if not enough history
        padding = [0.0] * (n_days - len(result.daily_pnl))
        return np.array(padding + result.daily_pnl)

    return np.array(result.daily_pnl[-n_days:])


def get_strategy_cumulative_history(
    result: StrategyPnLResult,
    n_days: int = 7,
) -> np.ndarray:
    """Extract recent cumulative PnL history for forecasting.

    Args:
        result: Strategy simulation result
        n_days: Number of recent days to extract

    Returns:
        NumPy array of cumulative PnL values
    """
    if len(result.cumulative_pnl) < n_days:
        # Extend with zeros at the beginning
        n_pad = n_days - len(result.cumulative_pnl)
        padded = [0.0] * n_pad + result.cumulative_pnl
        return np.array(padded)

    return np.array(result.cumulative_pnl[-n_days:])
