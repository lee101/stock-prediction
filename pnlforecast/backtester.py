"""Main backtester for PnL Forecast meta-strategy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DataConfigPnL,
    ForecastConfigPnL,
    StrategyConfigPnL,
    SimulationConfigPnL,
    FullConfigPnL,
)
from .strategy import StrategyThresholds, generate_threshold_strategies
from .simulator import StrategySimulator, StrategyPnLResult
from .selector import (
    PnLForecaster,
    StrategySelector,
    PnLForecast,
    DailySelection,
)

logger = logging.getLogger(__name__)


# Import data loader from marketsimlong
try:
    from marketsimlong.data import (
        DailyDataLoader,
        is_crypto_symbol,
        get_trading_calendar,
    )
    from marketsimlong.config import DataConfigLong
except ImportError:
    logger.warning("Could not import marketsimlong.data, using local implementation")
    DailyDataLoader = None


@dataclass
class DailyResult:
    """Result for a single trading day."""

    trade_date: date
    symbol: str
    selected_strategy: Optional[StrategyThresholds]
    forecast: Optional[PnLForecast]
    actual_pnl: float  # Realized PnL for the day
    predicted_pnl: float  # What was predicted
    trade_executed: bool
    trade_type: str  # round_trip, buy_pending, sell_close, no_trade
    portfolio_value: float
    cash: float


@dataclass
class SymbolResult:
    """Aggregated result for a single symbol."""

    symbol: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    daily_results: List[DailyResult]
    strategy_usage: Dict[str, int]  # strategy_id -> times selected

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_daily_pnl(self) -> float:
        if not self.daily_results:
            return 0.0
        return self.total_pnl / len(self.daily_results)


@dataclass
class BacktestResult:
    """Complete backtest results."""

    start_date: date
    end_date: date
    initial_cash: float
    final_portfolio_value: float

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade statistics
    total_trades: int
    total_days: int

    # Per-symbol results
    symbol_results: Dict[str, SymbolResult]

    # Equity curve
    equity_curve: pd.Series
    daily_returns: pd.Series

    # Strategy analysis
    strategy_selection_counts: Dict[str, int]
    prediction_accuracy: float  # Correlation of predicted vs actual PnL


class PnLForecastBacktester:
    """Full backtester for the PnL Forecast meta-strategy.

    This backtester:
    1. For each trading day, generates OHLC forecasts using Chronos2
    2. Simulates all threshold strategies on historical data
    3. Uses Chronos2 to forecast each strategy's future PnL
    4. Selects the best strategy based on forecasted PnL
    5. Executes the selected strategy for the day
    6. Tracks overall performance
    """

    def __init__(
        self,
        config: FullConfigPnL,
    ) -> None:
        self.config = config
        self.data_loader = None
        self.pnl_forecaster = None
        self.strategy_simulator = None
        self.strategy_selector = None

        # Pre-generate strategies
        self.stock_strategies = generate_threshold_strategies(
            config.strategy, is_crypto=False
        )
        self.crypto_strategies = generate_threshold_strategies(
            config.strategy, is_crypto=True
        )

        # State
        self.cash = config.simulation.initial_cash
        self.positions: Dict[str, dict] = {}  # symbol -> position info
        self.equity_values: List[Tuple[date, float]] = []

    def _initialize(self) -> None:
        """Initialize all components."""
        # Data loader
        if DailyDataLoader is not None:
            data_config_long = DataConfigLong(
                stock_symbols=self.config.data.stock_symbols,
                crypto_symbols=self.config.data.crypto_symbols,
                data_root=self.config.data.data_root,
                start_date=self.config.data.start_date,
                end_date=self.config.data.end_date,
                context_days=self.config.data.context_days,
            )
            self.data_loader = DailyDataLoader(data_config_long)
            self.data_loader.load_all_symbols()
        else:
            raise RuntimeError("DailyDataLoader not available")

        # PnL forecaster
        self.pnl_forecaster = PnLForecaster(self.config.forecast)

        # Strategy simulator
        self.strategy_simulator = StrategySimulator(
            self.config.strategy,
            self.config.simulation,
        )

        # Strategy selector
        self.strategy_selector = StrategySelector(
            self.config.strategy,
            self.config.simulation,
        )

        logger.info(
            "Initialized backtester with %d stock strategies, %d crypto strategies",
            len(self.stock_strategies),
            len(self.crypto_strategies),
        )

    def _get_strategies_for_symbol(self, symbol: str) -> List[StrategyThresholds]:
        """Get appropriate strategies for a symbol."""
        if is_crypto_symbol(symbol):
            return self.crypto_strategies
        return self.stock_strategies

    def _simulate_strategies_for_symbol(
        self,
        symbol: str,
        target_date: date,
        history_days: int,
    ) -> Dict[str, StrategyPnLResult]:
        """Simulate all strategies for a symbol using historical data.

        Args:
            symbol: Trading symbol
            target_date: Current date (simulate up to day before)
            history_days: Number of historical days to simulate

        Returns:
            Dict of strategy_id -> StrategyPnLResult
        """
        strategies = self._get_strategies_for_symbol(symbol)
        is_crypto = is_crypto_symbol(symbol)

        # Get historical price data
        start_history = target_date - timedelta(days=history_days + 30)  # Buffer
        context_df = self.data_loader.get_context_for_date(
            symbol,
            target_date,
            context_days=history_days + 30,
        )

        if context_df.empty:
            return {}

        # Simulate each strategy
        sim_start = target_date - timedelta(days=history_days)
        sim_end = target_date - timedelta(days=1)

        results = self.strategy_simulator.simulate_all_strategies(
            strategies=strategies,
            symbol=symbol,
            price_data=context_df,
            start_date=sim_start,
            end_date=sim_end,
            is_crypto=is_crypto,
        )

        return results

    def _select_strategy_for_day(
        self,
        symbol: str,
        target_date: date,
        strategy_results: Dict[str, StrategyPnLResult],
    ) -> Optional[Tuple[StrategyThresholds, PnLForecast]]:
        """Select the best strategy for a symbol on a given day.

        Args:
            symbol: Trading symbol
            target_date: Date to trade
            strategy_results: Historical simulation results for all strategies

        Returns:
            Tuple of (selected_strategy, forecast) or None
        """
        # Get PnL forecasts for all strategies
        pnl_forecasts = self.pnl_forecaster.forecast_all_strategies(
            strategy_results, target_date
        )

        if not pnl_forecasts:
            return None

        # Select best strategy
        selection = self.strategy_selector.select_best_strategy(
            pnl_forecasts, strategy_results
        )

        if selection is None:
            return None

        strategy_id, forecast = selection

        # Get the actual strategy object
        strategies = self._get_strategies_for_symbol(symbol)
        selected_strategy = None
        for s in strategies:
            if s.strategy_id == strategy_id:
                selected_strategy = s
                break

        if selected_strategy is None:
            return None

        return selected_strategy, forecast

    def _execute_trade(
        self,
        symbol: str,
        strategy: StrategyThresholds,
        price_data: dict,
        trade_date: date,
    ) -> Tuple[float, str]:
        """Execute a trade using the selected strategy.

        Args:
            symbol: Trading symbol
            strategy: Selected strategy
            price_data: OHLC data for the day
            trade_date: Trading date

        Returns:
            Tuple of (realized_pnl, trade_type)
        """
        is_crypto = is_crypto_symbol(symbol)
        fee_pct = self.config.strategy.get_fee_pct(is_crypto)

        open_price = price_data["open"]
        high = price_data["high"]
        low = price_data["low"]

        # Check if we have an existing position
        position = self.positions.get(symbol)

        if position is None:
            # No position - try to enter and exit
            pnl, trade_type = strategy.compute_trade_pnl(
                open_price, high, low, fee_pct
            )

            if trade_type == "buy_pending":
                # Enter position, will close later
                entry_price = strategy.get_buy_price(open_price)
                position_size = self.cash * self.config.simulation.position_size_pct
                entry_cost = position_size * (1 + fee_pct)

                if entry_cost <= self.cash:
                    self.positions[symbol] = {
                        "entry_price": entry_price,
                        "entry_date": trade_date,
                        "size": position_size,
                        "strategy_id": strategy.strategy_id,
                    }
                    self.cash -= entry_cost
                    return 0.0, "buy_pending"

            elif trade_type == "round_trip":
                # Complete round-trip
                position_size = self.cash * self.config.simulation.position_size_pct
                realized_pnl = position_size * pnl
                self.cash += realized_pnl  # Add net P&L
                return pnl, "round_trip"

            return 0.0, trade_type

        else:
            # Have position - try to exit
            sell_price = strategy.get_sell_price(open_price)

            if high >= sell_price:
                # Can exit at target
                entry_price = position["entry_price"]
                position_size = position["size"]

                gross_pnl = (sell_price - entry_price) / entry_price
                net_pnl = gross_pnl - (2 * fee_pct)

                realized_pnl = position_size * net_pnl
                self.cash += position_size + realized_pnl

                del self.positions[symbol]
                return net_pnl, "sell_close"

            else:
                # Still holding
                return 0.0, "holding"

    def run(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> BacktestResult:
        """Run the full backtest.

        Args:
            start_date: Start date (default: config start)
            end_date: End date (default: config end)
            symbols: Symbols to trade (default: all from config)
            progress_callback: Optional callback(day_num, total_days, date)

        Returns:
            BacktestResult with full metrics
        """
        self._initialize()

        start = start_date or self.config.data.start_date
        end = end_date or self.config.data.end_date
        trade_symbols = symbols or list(self.config.data.all_symbols)

        # Get trading calendar
        trading_days = get_trading_calendar(start, end, is_crypto=False)
        crypto_days = get_trading_calendar(start, end, is_crypto=True)
        all_days = sorted(set(trading_days + crypto_days))

        logger.info(
            "Running backtest from %s to %s (%d days, %d symbols)",
            start, end, len(all_days), len(trade_symbols),
        )

        # Track results
        symbol_daily_results: Dict[str, List[DailyResult]] = {
            s: [] for s in trade_symbols
        }
        strategy_counts: Dict[str, int] = {}
        predictions: List[Tuple[float, float]] = []  # (predicted, actual)

        self.cash = self.config.simulation.initial_cash
        self.positions.clear()
        self.equity_values.clear()

        history_days = self.config.simulation.max_pnl_history_days

        for day_idx, trade_date in enumerate(all_days):
            is_stock_day = trade_date in trading_days

            for symbol in trade_symbols:
                is_crypto = is_crypto_symbol(symbol)

                # Skip stocks on weekends
                if not is_stock_day and not is_crypto:
                    continue

                # Check if we have price data for this day
                price_data = self.data_loader.get_price_on_date(symbol, trade_date)
                if price_data is None:
                    continue

                # Simulate all strategies on historical data
                strategy_results = self._simulate_strategies_for_symbol(
                    symbol, trade_date, history_days
                )

                if not strategy_results:
                    continue

                # Select best strategy using PnL forecasting
                selection = self._select_strategy_for_day(
                    symbol, trade_date, strategy_results
                )

                predicted_pnl = 0.0
                selected_strategy = None
                forecast = None

                if selection:
                    selected_strategy, forecast = selection
                    predicted_pnl = forecast.predicted_pnl_change

                    # Track strategy usage
                    sid = selected_strategy.strategy_id
                    strategy_counts[sid] = strategy_counts.get(sid, 0) + 1

                # Execute trade
                actual_pnl = 0.0
                trade_type = "no_trade"

                if selected_strategy:
                    actual_pnl, trade_type = self._execute_trade(
                        symbol, selected_strategy, price_data, trade_date
                    )

                    # Track prediction accuracy
                    if trade_type in ("round_trip", "sell_close"):
                        predictions.append((predicted_pnl, actual_pnl))

                # Calculate portfolio value
                portfolio_value = self.cash
                for pos_symbol, pos in self.positions.items():
                    pos_price = self.data_loader.get_price_on_date(pos_symbol, trade_date)
                    if pos_price:
                        current_value = pos["size"] * (pos_price["close"] / pos["entry_price"])
                        portfolio_value += current_value

                # Record daily result
                daily_result = DailyResult(
                    trade_date=trade_date,
                    symbol=symbol,
                    selected_strategy=selected_strategy,
                    forecast=forecast,
                    actual_pnl=actual_pnl,
                    predicted_pnl=predicted_pnl,
                    trade_executed=trade_type in ("round_trip", "buy_pending", "sell_close"),
                    trade_type=trade_type,
                    portfolio_value=portfolio_value,
                    cash=self.cash,
                )
                symbol_daily_results[symbol].append(daily_result)

            # Record equity curve
            portfolio_value = self.cash
            for pos_symbol, pos in self.positions.items():
                pos_price = self.data_loader.get_price_on_date(pos_symbol, trade_date)
                if pos_price:
                    current_value = pos["size"] * (pos_price["close"] / pos["entry_price"])
                    portfolio_value += current_value

            self.equity_values.append((trade_date, portfolio_value))

            # Progress callback
            if progress_callback:
                progress_callback(day_idx + 1, len(all_days), trade_date)

            # Log progress
            if (day_idx + 1) % self.config.simulation.log_interval_days == 0:
                logger.info(
                    "Day %d/%d: %s, Portfolio: $%.2f",
                    day_idx + 1,
                    len(all_days),
                    trade_date,
                    portfolio_value,
                )

        # Compute final results
        return self._compute_results(
            start, end, trade_symbols, symbol_daily_results,
            strategy_counts, predictions
        )

    def _compute_results(
        self,
        start_date: date,
        end_date: date,
        symbols: List[str],
        symbol_daily_results: Dict[str, List[DailyResult]],
        strategy_counts: Dict[str, int],
        predictions: List[Tuple[float, float]],
    ) -> BacktestResult:
        """Compute final backtest metrics."""
        if not self.equity_values:
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_cash=self.config.simulation.initial_cash,
                final_portfolio_value=self.cash,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                total_days=0,
                symbol_results={},
                equity_curve=pd.Series(dtype=float),
                daily_returns=pd.Series(dtype=float),
                strategy_selection_counts={},
                prediction_accuracy=0.0,
            )

        # Build equity curve
        dates, values = zip(*self.equity_values)
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))
        daily_returns = equity_curve.pct_change().dropna()

        # Performance metrics
        initial_cash = self.config.simulation.initial_cash
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - initial_cash) / initial_cash

        n_days = len(equity_curve)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe ratio
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0

        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Aggregate symbol results
        symbol_results = {}
        total_trades = 0
        winning_trades = 0

        for symbol, daily_results in symbol_daily_results.items():
            trades = sum(1 for r in daily_results if r.trade_executed)
            wins = sum(1 for r in daily_results if r.actual_pnl > 0)
            total_pnl = sum(r.actual_pnl for r in daily_results)

            # Strategy usage for this symbol
            usage = {}
            for r in daily_results:
                if r.selected_strategy:
                    sid = r.selected_strategy.strategy_id
                    usage[sid] = usage.get(sid, 0) + 1

            symbol_results[symbol] = SymbolResult(
                symbol=symbol,
                total_pnl=total_pnl,
                total_trades=trades,
                winning_trades=wins,
                daily_results=daily_results,
                strategy_usage=usage,
            )

            total_trades += trades
            winning_trades += wins

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Prediction accuracy (correlation)
        prediction_accuracy = 0.0
        if len(predictions) > 5:
            pred_arr = np.array([p[0] for p in predictions])
            actual_arr = np.array([p[1] for p in predictions])
            if np.std(pred_arr) > 0 and np.std(actual_arr) > 0:
                prediction_accuracy = float(np.corrcoef(pred_arr, actual_arr)[0, 1])

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            final_portfolio_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            total_days=n_days,
            symbol_results=symbol_results,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            strategy_selection_counts=strategy_counts,
            prediction_accuracy=prediction_accuracy,
        )

    def cleanup(self) -> None:
        """Release resources."""
        if self.pnl_forecaster:
            self.pnl_forecaster.unload()


def run_backtest(
    config: Optional[FullConfigPnL] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    symbols: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None,
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        config: Full configuration (default: use defaults)
        start_date: Override start date
        end_date: Override end date
        symbols: Override symbols
        progress_callback: Progress callback

    Returns:
        BacktestResult
    """
    if config is None:
        config = FullConfigPnL()

    backtester = PnLForecastBacktester(config)

    try:
        result = backtester.run(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            progress_callback=progress_callback,
        )
        return result
    finally:
        backtester.cleanup()


def save_results(result: BacktestResult, output_path: Path) -> None:
    """Save backtest results to JSON.

    Args:
        result: BacktestResult to save
        output_path: Path to save to
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
        "initial_cash": result.initial_cash,
        "final_portfolio_value": result.final_portfolio_value,
        "total_return": result.total_return,
        "total_return_pct": result.total_return * 100,
        "annualized_return": result.annualized_return,
        "annualized_return_pct": result.annualized_return * 100,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "max_drawdown_pct": result.max_drawdown * 100,
        "win_rate": result.win_rate,
        "win_rate_pct": result.win_rate * 100,
        "total_trades": result.total_trades,
        "total_days": result.total_days,
        "prediction_accuracy": result.prediction_accuracy,
        "strategy_selection_counts": result.strategy_selection_counts,
        "symbol_summary": {
            symbol: {
                "total_pnl": sr.total_pnl,
                "total_pnl_pct": sr.total_pnl * 100,
                "total_trades": sr.total_trades,
                "win_rate": sr.win_rate,
                "win_rate_pct": sr.win_rate * 100,
            }
            for symbol, sr in result.symbol_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved results to %s", output_path)
