"""Hourly market simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfigHourly, ForecastConfigHourly, SimulationConfigHourly
from .data import HourlyDataLoader, is_crypto_symbol
from .forecaster import Chronos2HourlyForecaster, HourlyForecasts

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp

    @property
    def notional_value(self) -> float:
        return self.quantity * self.entry_price


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    fee: float
    cash_after: float
    portfolio_value_after: float


@dataclass
class HourResult:
    timestamp: pd.Timestamp
    starting_cash: float
    ending_cash: float
    starting_portfolio_value: float
    ending_portfolio_value: float
    positions_held: List[str]
    trades_executed: List[TradeRecord]
    period_return: float
    risk_penalty: float = 0.0
    forecasts_used: Optional[HourlyForecasts] = None


@dataclass
class SimulationResultHourly:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    initial_cash: float
    final_cash: float
    final_portfolio_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_periods: int
    avg_period_return: float
    equity_curve: pd.Series
    hourly_results: List[HourResult]
    all_trades: List[TradeRecord]
    symbol_returns: Dict[str, float] = field(default_factory=dict)
    total_margin_interest_paid: float = 0.0
    total_risk_penalty: float = 0.0


class HourlySimulator:
    def __init__(
        self,
        data_loader: HourlyDataLoader,
        forecaster: Chronos2HourlyForecaster,
        sim_config: SimulationConfigHourly,
    ) -> None:
        self.data_loader = data_loader
        self.forecaster = forecaster
        self.config = sim_config

        self.cash = sim_config.initial_cash
        self.margin_borrowed = 0.0
        self.total_margin_interest_paid = 0.0
        self.total_risk_penalties = 0.0
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.hourly_results: List[HourResult] = []
        self.equity_values: List[Tuple[pd.Timestamp, float]] = []

    def reset(self) -> None:
        self.cash = self.config.initial_cash
        self.margin_borrowed = 0.0
        self.total_margin_interest_paid = 0.0
        self.total_risk_penalties = 0.0
        self.positions.clear()
        self.trades.clear()
        self.hourly_results.clear()
        self.equity_values.clear()

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        position_value = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            position_value += pos.quantity * price
        return self.cash + position_value - self.margin_borrowed

    def close_position(
        self,
        symbol: str,
        price: float,
        trade_time: pd.Timestamp,
    ) -> Optional[TradeRecord]:
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        quantity = pos.quantity
        notional = quantity * price

        fee = notional * self.config.maker_fee
        proceeds = notional - fee
        self.cash += proceeds

        if self.margin_borrowed > 0 and self.cash > self.config.initial_cash:
            excess = self.cash - self.config.initial_cash
            payback = min(excess, self.margin_borrowed)
            self.margin_borrowed -= payback
            self.cash -= payback

        trade = TradeRecord(
            timestamp=trade_time,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            cash_after=self.cash,
            portfolio_value_after=self.cash,
        )
        self.trades.append(trade)
        return trade

    def open_position(
        self,
        symbol: str,
        price: float,
        notional_amount: float,
        trade_time: pd.Timestamp,
    ) -> Optional[TradeRecord]:
        fee = notional_amount * self.config.maker_fee
        total_cost = notional_amount + fee

        is_stock = not is_crypto_symbol(symbol)
        can_use_leverage = (is_stock or not self.config.leverage_stocks_only) and self.config.leverage > 1.0

        if total_cost > self.cash:
            if can_use_leverage:
                shortfall = total_cost - self.cash
                max_margin = self.config.initial_cash * (self.config.leverage - 1)
                available_margin = max_margin - self.margin_borrowed

                if shortfall <= available_margin:
                    self.margin_borrowed += shortfall
                    self.cash += shortfall
                else:
                    max_affordable = self.cash + available_margin
                    notional_amount = max_affordable / (1 + self.config.maker_fee)
                    fee = notional_amount * self.config.maker_fee
                    total_cost = notional_amount + fee
                    margin_needed = total_cost - self.cash
                    if margin_needed > 0:
                        self.margin_borrowed += margin_needed
                        self.cash += margin_needed
            else:
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
            entry_time=trade_time,
        )

        trade = TradeRecord(
            timestamp=trade_time,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            notional=notional_amount,
            fee=fee,
            cash_after=self.cash,
            portfolio_value_after=self.cash + notional_amount,
        )
        self.trades.append(trade)
        return trade

    def get_max_hold_hours(self, symbol: str) -> int:
        if self.config.max_hold_hours_per_symbol:
            return self.config.max_hold_hours_per_symbol.get(symbol, self.config.max_hold_hours)
        return self.config.max_hold_hours

    def simulate_hour(
        self,
        trade_time: pd.Timestamp,
        forecasts: HourlyForecasts,
    ) -> HourResult:
        prices: Dict[str, float] = {}
        for symbol in self.data_loader.get_available_symbols_at(trade_time):
            price_data = self.data_loader.get_price_on_timestamp(symbol, trade_time)
            if price_data:
                prices[symbol] = price_data["close"]

        starting_cash = self.cash
        starting_value = self.get_portfolio_value(prices)

        top_symbols = forecasts.get_top_n_symbols(
            n=self.config.top_n,
            metric="predicted_return",
            min_return=self.config.min_predicted_return,
        )

        hour_trades: List[TradeRecord] = []

        for symbol, pos in list(self.positions.items()):
            max_hold = self.get_max_hold_hours(symbol)
            if max_hold > 0:
                hours_held = int((trade_time - pos.entry_time).total_seconds() // 3600)
                if hours_held >= max_hold and symbol in prices:
                    trade = self.close_position(symbol, prices[symbol], trade_time)
                    if trade:
                        hour_trades.append(trade)

        symbols_to_close = [s for s in self.positions if s not in top_symbols]
        for symbol in symbols_to_close:
            if symbol in prices:
                trade = self.close_position(symbol, prices[symbol], trade_time)
                if trade:
                    hour_trades.append(trade)

        if top_symbols:
            portfolio_value = self.get_portfolio_value(prices)
            max_position_value = portfolio_value * self.config.leverage
            current_position_value = sum(
                pos.quantity * prices.get(pos.symbol, pos.entry_price)
                for pos in self.positions.values()
            )
            available_to_allocate = max_position_value - current_position_value

            weight = 1.0 / len(top_symbols) if self.config.equal_weight else self.config.max_position_size
            allocation_per_symbol = available_to_allocate * weight

            for symbol in top_symbols:
                if symbol in self.positions:
                    continue
                if symbol not in prices:
                    continue
                trade = self.open_position(symbol, prices[symbol], allocation_per_symbol, trade_time)
                if trade:
                    hour_trades.append(trade)

        if self.margin_borrowed > 0:
            hourly_interest = self.margin_borrowed * self.config.hourly_margin_rate
            self.cash -= hourly_interest
            self.total_margin_interest_paid += hourly_interest

        risk_penalty = 0.0
        if (
            self.config.hold_penalty_rate > 0.0
            and self.config.hold_penalty_start_hours > 0
            and self.positions
        ):
            for symbol, pos in self.positions.items():
                hours_held = int((trade_time - pos.entry_time).total_seconds() // 3600)
                if hours_held >= self.config.hold_penalty_start_hours:
                    price = prices.get(symbol, pos.entry_price)
                    notional = pos.quantity * price
                    risk_penalty += notional * self.config.hold_penalty_rate

        if self.config.leverage_penalty_rate > 0.0 and self.config.leverage_soft_cap > 0.0:
            equity = self.get_portfolio_value(prices)
            if equity > 0:
                exposure = sum(
                    pos.quantity * prices.get(pos.symbol, pos.entry_price)
                    for pos in self.positions.values()
                )
                leverage_ratio = exposure / equity
                if leverage_ratio > self.config.leverage_soft_cap:
                    excess = leverage_ratio - self.config.leverage_soft_cap
                    risk_penalty += equity * excess * self.config.leverage_penalty_rate

        if risk_penalty > 0.0:
            self.cash -= risk_penalty
            self.total_risk_penalties += risk_penalty

        ending_value = self.get_portfolio_value(prices)
        period_return = (ending_value - starting_value) / starting_value if starting_value > 0 else 0.0

        self.equity_values.append((trade_time, ending_value))

        result = HourResult(
            timestamp=trade_time,
            starting_cash=starting_cash,
            ending_cash=self.cash,
            starting_portfolio_value=starting_value,
            ending_portfolio_value=ending_value,
            positions_held=list(self.positions.keys()),
            trades_executed=hour_trades,
            period_return=period_return,
            risk_penalty=risk_penalty,
            forecasts_used=forecasts,
        )
        self.hourly_results.append(result)
        return result

    def run(
        self,
        start_date: date,
        end_date: date,
        progress_callback: Optional[callable] = None,
    ) -> SimulationResultHourly:
        self.reset()

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        all_hours = pd.date_range(start=start_ts, end=end_ts, freq="h")

        logger.info("Running hourly simulation from %s to %s (%d hours)", start_ts, end_ts, len(all_hours))

        for idx, trade_time in enumerate(all_hours):
            available_symbols = self.data_loader.get_tradable_symbols_at(trade_time)
            if not available_symbols:
                continue

            forecasts = self.forecaster.forecast_all_symbols(trade_time, available_symbols)
            if not forecasts.forecasts:
                continue

            hour_result = self.simulate_hour(trade_time, forecasts)

            if progress_callback:
                progress_callback(idx + 1, len(all_hours), hour_result)

            if (idx + 1) % 120 == 0:
                logger.info(
                    "Hour %d/%d: %s, Portfolio: $%.2f, Return: %.4f",
                    idx + 1,
                    len(all_hours),
                    trade_time,
                    hour_result.ending_portfolio_value,
                    hour_result.period_return,
                )

        return self._compute_results(start_ts, end_ts)

    def _compute_results(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> SimulationResultHourly:
        if not self.equity_values:
            return SimulationResultHourly(
                start_time=start_ts,
                end_time=end_ts,
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
                total_periods=0,
                avg_period_return=0.0,
                equity_curve=pd.Series(dtype=float),
                hourly_results=[],
                all_trades=[],
                symbol_returns={},
                total_margin_interest_paid=self.total_margin_interest_paid,
                total_risk_penalty=self.total_risk_penalties,
            )

        timestamps, values = zip(*self.equity_values)
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(timestamps))

        period_returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] - self.config.initial_cash) / self.config.initial_cash

        n_periods = len(equity_curve)
        annualized_return = (1 + total_return) ** (self.config.trading_hours_per_year / max(n_periods, 1)) - 1

        mean_return = period_returns.mean()
        std_return = period_returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(self.config.trading_hours_per_year) if std_return > 0 else 0.0

        downside_returns = period_returns[period_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        sortino_ratio = mean_return / downside_std * np.sqrt(self.config.trading_hours_per_year) if downside_std > 0 else 0.0

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        winning_periods = (period_returns > 0).sum()
        total_trading_periods = len(period_returns)
        win_rate = winning_periods / total_trading_periods if total_trading_periods > 0 else 0.0

        symbol_returns = self._compute_symbol_returns()

        return SimulationResultHourly(
            start_time=start_ts,
            end_time=end_ts,
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
            total_periods=n_periods,
            avg_period_return=mean_return,
            equity_curve=equity_curve,
            hourly_results=self.hourly_results,
            all_trades=self.trades,
            symbol_returns=symbol_returns,
            total_margin_interest_paid=self.total_margin_interest_paid,
            total_risk_penalty=self.total_risk_penalties,
        )

    def _compute_symbol_returns(self) -> Dict[str, float]:
        symbol_pnl: Dict[str, float] = {}
        symbol_invested: Dict[str, float] = {}

        for trade in self.trades:
            if trade.symbol not in symbol_pnl:
                symbol_pnl[trade.symbol] = 0.0
                symbol_invested[trade.symbol] = 0.0

            if trade.side == "buy":
                symbol_invested[trade.symbol] += trade.notional
                symbol_pnl[trade.symbol] -= trade.notional + trade.fee
            else:
                symbol_pnl[trade.symbol] += trade.notional - trade.fee

        symbol_returns = {}
        for symbol in symbol_pnl:
            invested = symbol_invested.get(symbol, 0.0)
            if invested > 0:
                symbol_returns[symbol] = symbol_pnl[symbol] / invested
        return symbol_returns


def run_simulation(
    data_config: DataConfigHourly,
    forecast_config: ForecastConfigHourly,
    sim_config: SimulationConfigHourly,
    progress_callback: Optional[callable] = None,
) -> SimulationResultHourly:
    data_loader = HourlyDataLoader(data_config)
    data_loader.load_all_symbols()

    forecaster = Chronos2HourlyForecaster(data_loader, forecast_config)
    simulator = HourlySimulator(data_loader, forecaster, sim_config)
    try:
        return simulator.run(
            start_date=data_config.start_date,
            end_date=data_config.end_date,
            progress_callback=progress_callback,
        )
    finally:
        forecaster.unload()
