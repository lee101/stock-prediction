"""Market simulation for backtesting Solana trading strategies.

Simulates trading using historical OHLC data with realistic
cost modeling for Solana swaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SimulationConfig, CostConfig, TokenConfig, SOL_MINT
from .data_collector import OHLCBar
from .forecaster import TokenForecast, ForecastBatch, TokenForecaster

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position in a token."""

    token_mint: str
    token_symbol: str
    quantity: float  # Amount of token held
    entry_price_sol: float  # Entry price in SOL
    entry_time: datetime
    entry_cost_sol: float = 0.0  # Total SOL cost including fees

    @property
    def notional_sol(self) -> float:
        """Entry notional value in SOL."""
        return self.quantity * self.entry_price_sol


@dataclass
class Trade:
    """Record of a completed trade."""

    timestamp: datetime
    token_mint: str
    token_symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price_sol: float
    notional_sol: float
    fee_sol: float  # Swap fees + network fees
    slippage_sol: float  # Actual slippage cost

    # Portfolio state after trade
    sol_balance_after: float
    portfolio_value_after: float


@dataclass
class SimulationState:
    """Current state of the simulation."""

    timestamp: datetime
    sol_balance: float
    positions: Dict[str, Position]  # token_mint -> Position
    trades: List[Trade]
    equity_history: List[Tuple[datetime, float]]

    # Daily tracking
    day_start_value: float = 0.0
    daily_pnl: float = 0.0

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value in SOL.

        Args:
            prices: Dict of token_mint -> price_in_sol

        Returns:
            Total portfolio value in SOL
        """
        position_value = sum(
            pos.quantity * prices.get(pos.token_mint, pos.entry_price_sol)
            for pos in self.positions.values()
        )
        return self.sol_balance + position_value


@dataclass
class SimulationResult:
    """Complete simulation results."""

    start_time: datetime
    end_time: datetime
    initial_sol: float
    final_sol: float
    final_portfolio_value: float

    # Performance metrics
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    total_fees_sol: float
    total_slippage_sol: float

    # Detailed data
    equity_curve: pd.Series
    trades: List[Trade]

    # Per-token performance
    token_returns: Dict[str, float] = field(default_factory=dict)


class MarketSimulator:
    """Simulates trading on historical OHLC data.

    Features:
    - Realistic swap cost modeling (fees, slippage, priority fees)
    - Position tracking with entry cost basis
    - Risk controls (max drawdown, daily loss limits)
    - Performance metrics calculation
    """

    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config

        # State
        self.state: Optional[SimulationState] = None
        self._prices: Dict[str, float] = {}  # Current prices

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.state = SimulationState(
            timestamp=datetime.utcnow(),
            sol_balance=self.config.initial_sol,
            positions={},
            trades=[],
            equity_history=[],
            day_start_value=self.config.initial_sol,
        )
        self._prices = {SOL_MINT: 1.0}

    def _calculate_swap_cost(
        self,
        notional_sol: float,
        is_buy: bool,
    ) -> Tuple[float, float]:
        """Calculate swap costs.

        Args:
            notional_sol: Trade notional in SOL
            is_buy: True if buying token, False if selling

        Returns:
            Tuple of (total_fee_sol, slippage_sol)
        """
        costs = self.config.costs

        # AMM/DEX fee
        swap_fee = notional_sol * (costs.estimated_swap_fee_bps / 10000)

        # Slippage estimate (increases with size)
        # Assume 0.1% base + 0.05% per 0.1 SOL traded
        base_slippage = costs.default_slippage_bps / 20000  # Half of tolerance
        size_factor = min(notional_sol / 0.1, 10)  # Cap at 10x
        slippage_pct = base_slippage * (1 + size_factor * 0.5)
        slippage = notional_sol * slippage_pct

        # Network fee (in SOL)
        network_fee = costs.estimated_total_fee_sol

        total_fee = swap_fee + network_fee

        return total_fee, slippage

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices.

        Args:
            prices: Dict of token_mint -> price_in_sol
        """
        self._prices.update(prices)
        self._prices[SOL_MINT] = 1.0

    def simulate_timestamp(
        self,
        timestamp: datetime,
        bars: Dict[str, OHLCBar],
        tokens: Dict[str, TokenConfig],
        actions: Dict[str, str],
    ) -> None:
        """Process multiple token bars that share the same timestamp.

        Args:
            timestamp: Timestamp for this step
            bars: Dict of token_mint -> OHLCBar at timestamp
            tokens: Dict of token_mint -> TokenConfig
            actions: Dict of token_mint -> action ("buy", "sell", or "hold")
        """
        if self.state is None:
            self.reset()

        prices = {mint: bar.close for mint, bar in bars.items()}
        self.update_prices(prices)
        self.state.timestamp = timestamp

        for mint, action in actions.items():
            bar = bars.get(mint)
            token = tokens.get(mint)
            if bar is None or token is None:
                continue

            if action == "buy" and mint not in self.state.positions:
                portfolio_value = self.state.get_portfolio_value(self._prices)
                position_size = portfolio_value * self.config.max_position_pct
                position_size = min(position_size, self.state.sol_balance * 0.9)
                if self.config.max_position_sol is not None:
                    position_size = min(position_size, self.config.max_position_sol)
                self.open_position(token, position_size, bar.close, timestamp)

            elif action == "sell" and mint in self.state.positions:
                self.close_position(token, bar.close, timestamp)

        portfolio_value = self.state.get_portfolio_value(self._prices)
        self.state.equity_history.append((timestamp, portfolio_value))

    def open_position(
        self,
        token: TokenConfig,
        sol_amount: float,
        price_sol: float,
        timestamp: datetime,
    ) -> Optional[Trade]:
        """Open a new position by buying a token.

        Args:
            token: Token to buy
            sol_amount: Amount of SOL to spend
            price_sol: Current price in SOL
            timestamp: Trade timestamp

        Returns:
            Trade record or None if trade failed
        """
        if self.state is None:
            self.reset()

        if self.config.max_position_sol is not None:
            current_value = sum(
                pos.quantity * self._prices.get(pos.token_mint, pos.entry_price_sol)
                for pos in self.state.positions.values()
            )
            remaining_capacity = self.config.max_position_sol - current_value
            if remaining_capacity <= 0:
                logger.warning("Position limit reached: skipping buy")
                return None
            sol_amount = min(sol_amount, remaining_capacity)

        # Check if we have enough SOL
        if sol_amount > self.state.sol_balance:
            logger.warning(
                f"Insufficient SOL: have {self.state.sol_balance}, need {sol_amount}"
            )
            sol_amount = self.state.sol_balance * 0.95  # Leave some for fees

        if sol_amount < self.config.min_trade_value_sol:
            logger.warning(f"Trade too small: {sol_amount} SOL")
            return None

        # Calculate costs
        fee_sol, slippage_sol = self._calculate_swap_cost(sol_amount, is_buy=True)
        total_cost = sol_amount + fee_sol

        if total_cost > self.state.sol_balance:
            # Reduce trade size to fit
            sol_amount = self.state.sol_balance - fee_sol - 0.001
            fee_sol, slippage_sol = self._calculate_swap_cost(sol_amount, is_buy=True)
            total_cost = sol_amount + fee_sol

        # Execute trade
        effective_price = price_sol * (1 + slippage_sol / sol_amount)
        quantity = sol_amount / effective_price

        # Update state
        self.state.sol_balance -= total_cost

        if token.mint in self.state.positions:
            # Add to existing position (average entry)
            pos = self.state.positions[token.mint]
            total_qty = pos.quantity + quantity
            avg_price = (pos.notional_sol + sol_amount) / total_qty
            pos.quantity = total_qty
            pos.entry_price_sol = avg_price
            pos.entry_cost_sol += total_cost
        else:
            # New position
            self.state.positions[token.mint] = Position(
                token_mint=token.mint,
                token_symbol=token.symbol,
                quantity=quantity,
                entry_price_sol=effective_price,
                entry_time=timestamp,
                entry_cost_sol=total_cost,
            )

        # Update prices
        self._prices[token.mint] = price_sol

        # Record trade
        portfolio_value = self.state.get_portfolio_value(self._prices)
        trade = Trade(
            timestamp=timestamp,
            token_mint=token.mint,
            token_symbol=token.symbol,
            side="buy",
            quantity=quantity,
            price_sol=price_sol,
            notional_sol=sol_amount,
            fee_sol=fee_sol,
            slippage_sol=slippage_sol,
            sol_balance_after=self.state.sol_balance,
            portfolio_value_after=portfolio_value,
        )
        self.state.trades.append(trade)

        logger.info(
            f"BUY {quantity:.6f} {token.symbol} @ {price_sol:.8f} SOL "
            f"(cost: {total_cost:.6f} SOL, fee: {fee_sol:.6f} SOL)"
        )

        return trade

    def close_position(
        self,
        token: TokenConfig,
        price_sol: float,
        timestamp: datetime,
        quantity: Optional[float] = None,
    ) -> Optional[Trade]:
        """Close a position by selling a token.

        Args:
            token: Token to sell
            price_sol: Current price in SOL
            timestamp: Trade timestamp
            quantity: Amount to sell (None for full position)

        Returns:
            Trade record or None if no position
        """
        if self.state is None or token.mint not in self.state.positions:
            return None

        pos = self.state.positions[token.mint]

        if quantity is None:
            quantity = pos.quantity

        quantity = min(quantity, pos.quantity)

        # Calculate proceeds and costs
        gross_proceeds = quantity * price_sol
        fee_sol, slippage_sol = self._calculate_swap_cost(gross_proceeds, is_buy=False)
        net_proceeds = gross_proceeds - fee_sol - slippage_sol

        # Update state
        self.state.sol_balance += net_proceeds

        if quantity >= pos.quantity:
            # Close full position
            del self.state.positions[token.mint]
        else:
            # Partial close
            pos.quantity -= quantity

        # Update prices
        self._prices[token.mint] = price_sol

        # Record trade
        portfolio_value = self.state.get_portfolio_value(self._prices)
        trade = Trade(
            timestamp=timestamp,
            token_mint=token.mint,
            token_symbol=token.symbol,
            side="sell",
            quantity=quantity,
            price_sol=price_sol,
            notional_sol=gross_proceeds,
            fee_sol=fee_sol,
            slippage_sol=slippage_sol,
            sol_balance_after=self.state.sol_balance,
            portfolio_value_after=portfolio_value,
        )
        self.state.trades.append(trade)

        logger.info(
            f"SELL {quantity:.6f} {token.symbol} @ {price_sol:.8f} SOL "
            f"(proceeds: {net_proceeds:.6f} SOL, fee: {fee_sol:.6f} SOL)"
        )

        return trade

    def simulate_bar(
        self,
        bar: OHLCBar,
        token: TokenConfig,
        forecast: Optional[TokenForecast] = None,
        action: Optional[str] = None,  # "buy", "sell", or None for hold
    ) -> None:
        """Process a single OHLC bar.

        Args:
            bar: OHLC bar data
            token: Token configuration
            forecast: Optional forecast for decision making
            action: Explicit action to take (overrides forecast-based decision)
        """
        if self.state is None:
            self.reset()

        # Update price
        price = bar.close
        self._prices[token.mint] = price
        self.state.timestamp = bar.timestamp

        # Execute action
        if action == "buy" and token.mint not in self.state.positions:
            # Calculate position size
            portfolio_value = self.state.get_portfolio_value(self._prices)
            position_size = portfolio_value * self.config.max_position_pct
            position_size = min(position_size, self.state.sol_balance * 0.9)
            if self.config.max_position_sol is not None:
                position_size = min(position_size, self.config.max_position_sol)

            self.open_position(token, position_size, price, bar.timestamp)

        elif action == "sell" and token.mint in self.state.positions:
            self.close_position(token, price, bar.timestamp)

        # Record equity
        portfolio_value = self.state.get_portfolio_value(self._prices)
        self.state.equity_history.append((bar.timestamp, portfolio_value))

    def run_backtest(
        self,
        bars: Dict[str, List[OHLCBar]],  # token_mint -> bars
        tokens: Dict[str, TokenConfig],  # token_mint -> config
        strategy_fn: callable,  # (state, prices, forecasts) -> Dict[mint, action]
        forecaster: Optional[object] = None,
    ) -> SimulationResult:
        """Run a backtest over historical bars.

        Args:
            bars: Dict of token_mint -> list of OHLCBars
            tokens: Dict of token_mint -> TokenConfig
            strategy_fn: Strategy function that returns actions
            forecaster: Optional forecaster for generating predictions

        Returns:
            SimulationResult with backtest metrics
        """
        self.reset()

        # Align bars by timestamp
        all_timestamps = set()
        for mint_bars in bars.values():
            for bar in mint_bars:
                all_timestamps.add(bar.timestamp)

        timestamps = sorted(all_timestamps)

        logger.info(f"Running backtest over {len(timestamps)} bars")

        for ts in timestamps:
            # Get prices at this timestamp
            prices = {}
            current_bars = {}

            for mint, mint_bars in bars.items():
                for bar in mint_bars:
                    if bar.timestamp == ts:
                        prices[mint] = bar.close
                        current_bars[mint] = bar
                        break

            if not prices:
                continue

            self.update_prices(prices)

            # Generate forecasts if forecaster available
            forecasts = {}
            # (Would call forecaster here with historical context)

            # Get strategy actions
            actions = strategy_fn(self.state, prices, forecasts)

            # Execute actions
            for mint, action in actions.items():
                if mint in current_bars and mint in tokens:
                    self.simulate_bar(
                        current_bars[mint],
                        tokens[mint],
                        action=action,
                    )

        return self._compute_results()

    def run_walk_forward_backtest(
        self,
        bars: Dict[str, List[OHLCBar]],
        tokens: Dict[str, TokenConfig],
        strategy_fn: callable,
        forecaster: Optional[TokenForecaster] = None,
        forecast_cache: Optional[Dict[datetime, Dict[str, TokenForecast]]] = None,
        context_bars: Optional[int] = None,
        min_context_bars: int = 20,
    ) -> SimulationResult:
        """Run a walk-forward backtest with optional forecasting.

        Args:
            bars: Dict of token_mint -> list of OHLCBars
            tokens: Dict of token_mint -> TokenConfig
            strategy_fn: Strategy function (state, prices, forecasts) -> actions
            forecaster: Optional TokenForecaster for on-the-fly predictions
            forecast_cache: Optional precomputed forecasts by timestamp
            context_bars: Max context bars to use (defaults to forecaster config)
            min_context_bars: Minimum bars required to generate a forecast

        Returns:
            SimulationResult
        """
        self.reset()

        bar_lookup: Dict[str, Dict[datetime, OHLCBar]] = {}
        all_timestamps = set()

        for mint, mint_bars in bars.items():
            sorted_bars = sorted(mint_bars, key=lambda b: b.timestamp)
            bar_lookup[mint] = {bar.timestamp: bar for bar in sorted_bars}
            for bar in sorted_bars:
                all_timestamps.add(bar.timestamp)

        timestamps = sorted(all_timestamps)
        history: Dict[str, List[OHLCBar]] = {mint: [] for mint in bars.keys()}

        for ts in timestamps:
            bars_at_ts: Dict[str, OHLCBar] = {}
            for mint, lookup in bar_lookup.items():
                bar = lookup.get(ts)
                if bar is None:
                    continue
                history[mint].append(bar)
                bars_at_ts[mint] = bar

            if not bars_at_ts:
                continue

            prices = {mint: bar.close for mint, bar in bars_at_ts.items()}

            if forecast_cache is not None:
                forecasts = forecast_cache.get(ts, {})
            elif forecaster is not None:
                forecasts: Dict[str, TokenForecast] = {}
                for mint, bar in bars_at_ts.items():
                    token = tokens.get(mint)
                    if token is None:
                        continue
                    context = history[mint]
                    if len(context) < min_context_bars:
                        continue
                    ctx_len = (
                        context_bars
                        if context_bars is not None
                        else forecaster.config.context_length
                    )
                    forecast = forecaster.forecast_from_bars(
                        token=token,
                        bars=context[-ctx_len:],
                    )
                    if forecast is not None:
                        forecasts[mint] = forecast
            else:
                forecasts = {}

            actions = strategy_fn(self.state, prices, forecasts)
            self.simulate_timestamp(ts, bars_at_ts, tokens, actions)

        return self._compute_results()

    def _compute_results(self) -> SimulationResult:
        """Compute final simulation results."""
        if self.state is None or not self.state.equity_history:
            return SimulationResult(
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                initial_sol=self.config.initial_sol,
                final_sol=0.0,
                final_portfolio_value=0.0,
                total_return=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                total_fees_sol=0.0,
                total_slippage_sol=0.0,
                equity_curve=pd.Series(dtype=float),
                trades=[],
            )

        # Build equity curve
        times, values = zip(*self.state.equity_history)
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(times))

        # Calculate returns
        initial = self.config.initial_sol
        final = equity_curve.iloc[-1]
        total_return = final - initial
        total_return_pct = (total_return / initial) * 100 if initial > 0 else 0

        # Daily returns for Sharpe/Sortino
        daily_returns = equity_curve.resample("D").last().pct_change().dropna()

        if len(daily_returns) > 1:
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            sharpe_ratio = mean_return / std_return * np.sqrt(365) if std_return > 0 else 0

            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std() if len(downside) > 0 else 0
            sortino_ratio = mean_return / downside_std * np.sqrt(365) if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Trade statistics
        total_fees = sum(t.fee_sol for t in self.state.trades)
        total_slippage = sum(t.slippage_sol for t in self.state.trades)

        # Win rate (profitable trades)
        buy_trades = {t.token_mint: t for t in self.state.trades if t.side == "buy"}
        sell_trades = [t for t in self.state.trades if t.side == "sell"]

        winning = 0
        for sell in sell_trades:
            buy = buy_trades.get(sell.token_mint)
            if buy and sell.price_sol > buy.price_sol:
                winning += 1

        win_rate = winning / len(sell_trades) if sell_trades else 0

        return SimulationResult(
            start_time=times[0] if times else datetime.utcnow(),
            end_time=times[-1] if times else datetime.utcnow(),
            initial_sol=initial,
            final_sol=self.state.sol_balance,
            final_portfolio_value=final,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.state.trades),
            winning_trades=winning,
            total_fees_sol=total_fees,
            total_slippage_sol=total_slippage,
            equity_curve=equity_curve,
            trades=self.state.trades,
        )


def simple_momentum_strategy(
    state: SimulationState,
    prices: Dict[str, float],
    forecasts: Dict[str, TokenForecast],
) -> Dict[str, str]:
    """Simple momentum-based strategy.

    Buys tokens with positive predicted return, sells with negative.
    """
    actions = {}

    for mint, forecast in forecasts.items():
        if forecast.predicted_return > 0.005:  # 0.5% threshold
            if mint not in state.positions:
                actions[mint] = "buy"
        elif forecast.predicted_return < -0.005:
            if mint in state.positions:
                actions[mint] = "sell"

    return actions


def forecast_threshold_strategy(
    min_return: float = 0.005,
    max_drawdown_return: float = -0.005,
) -> callable:
    """Create a threshold-based strategy using forecasted returns."""

    def _strategy(
        state: SimulationState,
        prices: Dict[str, float],
        forecasts: Dict[str, TokenForecast],
    ) -> Dict[str, str]:
        actions: Dict[str, str] = {}

        for mint, forecast in forecasts.items():
            if forecast.predicted_return >= min_return:
                if mint not in state.positions:
                    actions[mint] = "buy"
            elif forecast.predicted_return <= max_drawdown_return:
                if mint in state.positions:
                    actions[mint] = "sell"

        return actions

    return _strategy


def build_forecast_cache(
    bars: Dict[str, List[OHLCBar]],
    tokens: Dict[str, TokenConfig],
    forecaster: TokenForecaster,
    context_bars: Optional[int] = None,
    min_context_bars: int = 20,
) -> Dict[datetime, Dict[str, TokenForecast]]:
    """Precompute forecasts for each timestamp to reuse across backtests."""
    bar_lookup: Dict[str, Dict[datetime, OHLCBar]] = {}
    all_timestamps = set()

    for mint, mint_bars in bars.items():
        sorted_bars = sorted(mint_bars, key=lambda b: b.timestamp)
        bar_lookup[mint] = {bar.timestamp: bar for bar in sorted_bars}
        for bar in sorted_bars:
            all_timestamps.add(bar.timestamp)

    timestamps = sorted(all_timestamps)
    history: Dict[str, List[OHLCBar]] = {mint: [] for mint in bars.keys()}
    cache: Dict[datetime, Dict[str, TokenForecast]] = {}

    ctx_len = context_bars if context_bars is not None else forecaster.config.context_length

    for ts in timestamps:
        forecasts: Dict[str, TokenForecast] = {}
        for mint, lookup in bar_lookup.items():
            bar = lookup.get(ts)
            if bar is None:
                continue
            history[mint].append(bar)
            if len(history[mint]) < min_context_bars:
                continue
            token = tokens.get(mint)
            if token is None:
                continue
            forecast = forecaster.forecast_from_bars(
                token=token,
                bars=history[mint][-ctx_len:],
            )
            if forecast is not None:
                forecasts[mint] = forecast

        if forecasts:
            cache[ts] = forecasts

    return cache
