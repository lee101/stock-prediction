"""Long-term daily market simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfigLong, ForecastConfigLong, SimulationConfigLong
from .data import DailyDataLoader, get_trading_calendar, is_crypto_symbol
from .forecaster import Chronos2Forecaster, DailyForecasts, SymbolForecast

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    quantity: float
    entry_price: float
    entry_date: date

    @property
    def notional_value(self) -> float:
        """Entry notional value."""
        return self.quantity * self.entry_price


@dataclass
class TradeRecord:
    """Record of a single trade."""

    timestamp: date
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    notional: float
    fee: float
    cash_after: float
    portfolio_value_after: float


@dataclass
class DayResult:
    """Results for a single trading day."""

    date: date
    starting_cash: float
    ending_cash: float
    starting_portfolio_value: float
    ending_portfolio_value: float
    positions_held: List[str]
    trades_executed: List[TradeRecord]
    daily_return: float
    forecasts_used: Optional[DailyForecasts] = None


@dataclass
class SimulationResult:
    """Complete simulation results."""

    start_date: date
    end_date: date
    initial_cash: float
    final_cash: float
    final_portfolio_value: float

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade statistics
    total_trades: int
    total_days: int
    avg_daily_return: float

    # Detailed results
    equity_curve: pd.Series
    daily_results: List[DayResult]
    all_trades: List[TradeRecord]

    # Per-symbol performance
    symbol_returns: Dict[str, float] = field(default_factory=dict)


class LongTermDailySimulator:
    """Simulates long-term daily trading strategy.

    Strategy:
    - Each trading day, get Chronos2 forecasts for all symbols
    - Rank symbols by predicted % growth
    - Buy top N symbols with equal weight
    - Close all positions at end of day (or carry over if same symbol)
    - Track performance over entire simulation period
    """

    def __init__(
        self,
        data_loader: DailyDataLoader,
        forecaster: Chronos2Forecaster,
        sim_config: SimulationConfigLong,
    ) -> None:
        self.data_loader = data_loader
        self.forecaster = forecaster
        self.config = sim_config

        # State
        self.cash = sim_config.initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.daily_results: List[DayResult] = []
        self.equity_values: List[Tuple[date, float]] = []

    def reset(self) -> None:
        """Reset simulator state."""
        self.cash = self.config.initial_cash
        self.positions.clear()
        self.trades.clear()
        self.daily_results.clear()
        self.equity_values.clear()

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            Total portfolio value (cash + positions)
        """
        position_value = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            position_value += pos.quantity * price

        return self.cash + position_value

    def close_position(
        self,
        symbol: str,
        price: float,
        trade_date: date,
    ) -> Optional[TradeRecord]:
        """Close a position.

        Args:
            symbol: Symbol to close
            price: Exit price
            trade_date: Date of trade

        Returns:
            TradeRecord or None if no position
        """
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        quantity = pos.quantity
        notional = quantity * price

        # Calculate fee
        fee = notional * self.config.maker_fee
        proceeds = notional - fee

        self.cash += proceeds

        trade = TradeRecord(
            timestamp=trade_date,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            cash_after=self.cash,
            portfolio_value_after=self.cash,  # Updated later
        )
        self.trades.append(trade)

        return trade

    def open_position(
        self,
        symbol: str,
        price: float,
        notional_amount: float,
        trade_date: date,
    ) -> Optional[TradeRecord]:
        """Open a new position.

        Args:
            symbol: Symbol to buy
            price: Entry price
            notional_amount: Amount to invest
            trade_date: Date of trade

        Returns:
            TradeRecord or None if insufficient cash
        """
        # Calculate fee
        fee = notional_amount * self.config.maker_fee
        total_cost = notional_amount + fee

        if total_cost > self.cash:
            # Adjust notional to fit available cash
            notional_amount = self.cash / (1 + self.config.maker_fee)
            fee = notional_amount * self.config.maker_fee
            total_cost = notional_amount + fee

        if notional_amount <= 0:
            return None

        quantity = notional_amount / price
        self.cash -= total_cost

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=trade_date,
        )

        trade = TradeRecord(
            timestamp=trade_date,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            notional=notional_amount,
            fee=fee,
            cash_after=self.cash,
            portfolio_value_after=self.cash + notional_amount,  # Updated later
        )
        self.trades.append(trade)

        return trade

    def simulate_day(
        self,
        trade_date: date,
        forecasts: DailyForecasts,
    ) -> DayResult:
        """Simulate a single trading day.

        Args:
            trade_date: Date to simulate
            forecasts: Forecasts for this date

        Returns:
            DayResult with day's performance
        """
        # Get current prices for all symbols
        prices = {}
        for symbol in self.data_loader.get_available_symbols_on_date(trade_date):
            price_data = self.data_loader.get_price_on_date(symbol, trade_date)
            if price_data:
                prices[symbol] = price_data["close"]

        # Calculate starting portfolio value
        starting_cash = self.cash
        starting_value = self.get_portfolio_value(prices)

        # Get top N symbols to hold today
        top_symbols = forecasts.get_top_n_symbols(
            n=self.config.top_n,
            metric="predicted_return",
            min_return=self.config.min_predicted_return,
        )

        day_trades = []

        # Close positions not in top N
        symbols_to_close = [s for s in self.positions if s not in top_symbols]
        for symbol in symbols_to_close:
            if symbol in prices:
                trade = self.close_position(symbol, prices[symbol], trade_date)
                if trade:
                    day_trades.append(trade)

        # Open positions for top N symbols not already held
        if top_symbols:
            # Equal weight allocation
            available_cash = self.cash
            weight = 1.0 / len(top_symbols) if self.config.equal_weight else self.config.max_position_size
            allocation_per_symbol = available_cash * weight

            for symbol in top_symbols:
                if symbol in self.positions:
                    continue  # Already holding

                if symbol not in prices:
                    continue  # No price data

                trade = self.open_position(
                    symbol,
                    prices[symbol],
                    allocation_per_symbol,
                    trade_date,
                )
                if trade:
                    day_trades.append(trade)

        # Calculate ending portfolio value
        ending_value = self.get_portfolio_value(prices)
        daily_return = (ending_value - starting_value) / starting_value if starting_value > 0 else 0.0

        # Record equity curve
        self.equity_values.append((trade_date, ending_value))

        result = DayResult(
            date=trade_date,
            starting_cash=starting_cash,
            ending_cash=self.cash,
            starting_portfolio_value=starting_value,
            ending_portfolio_value=ending_value,
            positions_held=list(self.positions.keys()),
            trades_executed=day_trades,
            daily_return=daily_return,
            forecasts_used=forecasts,
        )
        self.daily_results.append(result)

        return result

    def run(
        self,
        start_date: date,
        end_date: date,
        progress_callback: Optional[callable] = None,
    ) -> SimulationResult:
        """Run the full simulation.

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            progress_callback: Optional callback(day_num, total_days, day_result)

        Returns:
            SimulationResult with full simulation metrics
        """
        self.reset()

        # Build trading calendar (use stock calendar, crypto fills gaps)
        trading_days = get_trading_calendar(start_date, end_date, is_crypto=False)
        crypto_only_days = get_trading_calendar(start_date, end_date, is_crypto=True)

        # Also include weekend days if we have crypto symbols
        all_days = sorted(set(trading_days + crypto_only_days))

        logger.info(
            "Running simulation from %s to %s (%d days)",
            start_date,
            end_date,
            len(all_days),
        )

        for day_idx, trade_date in enumerate(all_days):
            # Get available symbols for this day
            available_symbols = self.data_loader.get_tradable_symbols_on_date(trade_date)

            if not available_symbols:
                logger.debug("No tradable symbols on %s, skipping", trade_date)
                continue

            # Check if it's a stock trading day or crypto-only
            is_stock_day = trade_date in trading_days

            # Filter symbols appropriately
            if not is_stock_day:
                # Weekend - only trade crypto
                available_symbols = [s for s in available_symbols if is_crypto_symbol(s)]

            if not available_symbols:
                continue

            # Generate forecasts
            forecasts = self.forecaster.forecast_all_symbols(trade_date, available_symbols)

            if not forecasts.forecasts:
                logger.warning("No forecasts generated for %s", trade_date)
                continue

            # Simulate the day
            day_result = self.simulate_day(trade_date, forecasts)

            if progress_callback:
                progress_callback(day_idx + 1, len(all_days), day_result)

            # Log progress periodically
            if (day_idx + 1) % 20 == 0:
                logger.info(
                    "Day %d/%d: %s, Portfolio: $%.2f, Return: %.2f%%",
                    day_idx + 1,
                    len(all_days),
                    trade_date,
                    day_result.ending_portfolio_value,
                    day_result.daily_return * 100,
                )

        # Compute final metrics
        return self._compute_results(start_date, end_date)

    def _compute_results(self, start_date: date, end_date: date) -> SimulationResult:
        """Compute final simulation results and metrics."""
        if not self.equity_values:
            return SimulationResult(
                start_date=start_date,
                end_date=end_date,
                initial_cash=self.config.initial_cash,
                final_cash=self.cash,
                final_portfolio_value=self.cash,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                total_days=0,
                avg_daily_return=0.0,
                equity_curve=pd.Series(dtype=float),
                daily_results=[],
                all_trades=[],
                symbol_returns={},
            )

        # Build equity curve
        dates, values = zip(*self.equity_values)
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        # Calculate returns
        daily_returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] - self.config.initial_cash) / self.config.initial_cash

        # Annualized return
        n_days = len(equity_curve)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe ratio (assuming 0 risk-free rate)
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0

        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0

        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Win rate
        winning_days = (daily_returns > 0).sum()
        total_trading_days = len(daily_returns)
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0.0

        # Per-symbol returns
        symbol_returns = self._compute_symbol_returns()

        return SimulationResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.config.initial_cash,
            final_cash=self.cash,
            final_portfolio_value=equity_curve.iloc[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            total_days=n_days,
            avg_daily_return=mean_return,
            equity_curve=equity_curve,
            daily_results=self.daily_results,
            all_trades=self.trades,
            symbol_returns=symbol_returns,
        )

    def _compute_symbol_returns(self) -> Dict[str, float]:
        """Compute realized returns per symbol."""
        symbol_pnl: Dict[str, float] = {}
        symbol_invested: Dict[str, float] = {}

        for trade in self.trades:
            if trade.symbol not in symbol_pnl:
                symbol_pnl[trade.symbol] = 0.0
                symbol_invested[trade.symbol] = 0.0

            if trade.side == "buy":
                symbol_invested[trade.symbol] += trade.notional
                symbol_pnl[trade.symbol] -= trade.notional + trade.fee
            else:  # sell
                symbol_pnl[trade.symbol] += trade.notional - trade.fee

        # Convert to returns
        symbol_returns = {}
        for symbol in symbol_pnl:
            invested = symbol_invested.get(symbol, 0.0)
            if invested > 0:
                symbol_returns[symbol] = symbol_pnl[symbol] / invested

        return symbol_returns


def run_simulation(
    data_config: DataConfigLong,
    forecast_config: ForecastConfigLong,
    sim_config: SimulationConfigLong,
    progress_callback: Optional[callable] = None,
) -> SimulationResult:
    """Run a complete long-term simulation.

    Args:
        data_config: Data configuration
        forecast_config: Forecast configuration
        sim_config: Simulation configuration
        progress_callback: Optional progress callback

    Returns:
        SimulationResult with full metrics
    """
    # Initialize components
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    forecaster = Chronos2Forecaster(data_loader, forecast_config)

    simulator = LongTermDailySimulator(data_loader, forecaster, sim_config)

    try:
        result = simulator.run(
            start_date=data_config.start_date,
            end_date=data_config.end_date,
            progress_callback=progress_callback,
        )
        return result
    finally:
        forecaster.unload()
