"""Strategy comparison with realistic capital constraints and Chronos-adaptive thresholds."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    """Different strategy modes to compare."""
    BUY_AND_HOLD = "buy_and_hold"  # Just hold the asset
    FIXED_THRESHOLD = "fixed_threshold"  # Fixed buy-low/sell-high thresholds
    CHRONOS_ADAPTIVE = "chronos_adaptive"  # Use Chronos predicted high/low for thresholds
    CHRONOS_SYMBOL_SELECT = "chronos_symbol_select"  # Select best symbol via Chronos
    CHRONOS_FULL = "chronos_full"  # Adaptive thresholds + symbol selection + PnL forecast


@dataclass
class TradeResult:
    """Result of a single trade."""
    trade_date: date
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    pnl_pct: float
    trade_type: str  # round_trip, hold, no_trade
    fees_pct: float
    strategy_info: dict = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result of running a strategy over the full period."""
    strategy_name: str
    mode: StrategyMode
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    trades: List[TradeResult]
    equity_curve: pd.Series
    daily_returns: pd.Series

    # For alpha calculation
    baseline_return_pct: float = 0.0
    alpha_pct: float = 0.0


class RealisticBacktester:
    """Backtester with realistic single-portfolio capital constraints.

    Key differences from original:
    1. Single capital pool - can only be in one position at a time
    2. Uses Chronos predicted high/low for adaptive thresholds
    3. Compares multiple strategies fairly
    """

    def __init__(
        self,
        data_loader,
        initial_capital: float = 100_000.0,
        stock_fee_pct: float = 0.0003,  # 3bp per side
        crypto_fee_pct: float = 0.0008,  # 8bp per side
    ):
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.stock_fee_pct = stock_fee_pct
        self.crypto_fee_pct = crypto_fee_pct

        # Chronos forecaster (lazy loaded)
        self._forecaster = None

    def _get_fee_pct(self, symbol: str) -> float:
        """Get fee percentage for symbol."""
        from marketsimlong.data import is_crypto_symbol
        return self.crypto_fee_pct if is_crypto_symbol(symbol) else self.stock_fee_pct

    def _ensure_forecaster(self):
        """Lazy load Chronos forecaster."""
        if self._forecaster is not None:
            return

        from marketsimlong.forecaster import Chronos2Forecaster
        from marketsimlong.config import ForecastConfigLong

        config = ForecastConfigLong(
            model_id="amazon/chronos-2",
            device_map="cuda",
            prediction_length=1,
            context_length=512,
            use_multivariate=True,
        )
        self._forecaster = Chronos2Forecaster(self.data_loader, config)

    def _get_chronos_forecast(self, symbol: str, target_date: date) -> Optional[dict]:
        """Get Chronos OHLC forecast for a symbol."""
        self._ensure_forecaster()

        forecast = self._forecaster.forecast_symbol(symbol, target_date)
        if forecast is None:
            return None

        return {
            "predicted_close": forecast.predicted_close,
            "predicted_high": forecast.predicted_high,
            "predicted_low": forecast.predicted_low,
            "predicted_return": forecast.predicted_return,
            "current_close": forecast.current_close,
        }

    def run_buy_and_hold(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> StrategyResult:
        """Run buy-and-hold baseline strategy."""

        # Get price data
        start_price = None
        end_price = None
        equity_values = []

        current = start_date
        while current <= end_date:
            price_data = self.data_loader.get_price_on_date(symbol, current)
            if price_data:
                if start_price is None:
                    start_price = price_data["close"]
                end_price = price_data["close"]
                equity = self.initial_capital * (price_data["close"] / start_price)
                equity_values.append((current, equity))
            current += timedelta(days=1)

        if start_price is None or end_price is None:
            return self._empty_result(f"buy_hold_{symbol}", StrategyMode.BUY_AND_HOLD)

        # Calculate metrics
        total_return = (end_price - start_price) / start_price
        final_capital = self.initial_capital * (1 + total_return)

        # Build equity curve
        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))
        daily_returns = equity_curve.pct_change().dropna()

        # Calculate Sharpe
        sharpe = self._calculate_sharpe(daily_returns)
        max_dd = self._calculate_max_drawdown(equity_curve)
        n_days = len(equity_curve)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        return StrategyResult(
            strategy_name=f"buy_hold_{symbol}",
            mode=StrategyMode.BUY_AND_HOLD,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate=1.0 if total_return > 0 else 0.0,
            total_trades=1,
            trades=[],
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )

    def run_fixed_threshold(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        buy_threshold_pct: float = 0.005,  # Buy 0.5% below open
        sell_threshold_pct: float = 0.005,  # Sell 0.5% above open
    ) -> StrategyResult:
        """Run fixed threshold buy-low/sell-high strategy on single symbol."""

        fee_pct = self._get_fee_pct(symbol)
        capital = self.initial_capital
        equity_values = []
        trades = []

        holding = False
        entry_price = 0.0
        entry_date = None

        current = start_date
        while current <= end_date:
            price_data = self.data_loader.get_price_on_date(symbol, current)
            if not price_data:
                current += timedelta(days=1)
                continue

            open_price = price_data["open"]
            high = price_data["high"]
            low = price_data["low"]
            close = price_data["close"]

            buy_price = open_price * (1 - buy_threshold_pct)
            sell_price = open_price * (1 + sell_threshold_pct)

            can_buy = low <= buy_price
            can_sell = high >= sell_price

            trade_pnl = 0.0
            trade_type = "no_trade"

            if not holding:
                if can_buy and can_sell:
                    # Round-trip in single day
                    gross_pnl = (sell_price - buy_price) / buy_price
                    net_pnl = gross_pnl - (2 * fee_pct)
                    trade_pnl = net_pnl
                    capital *= (1 + net_pnl)
                    trade_type = "round_trip"

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=symbol,
                        entry_price=buy_price,
                        exit_price=sell_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="round_trip",
                        fees_pct=2 * fee_pct * 100,
                    ))

                elif can_buy:
                    # Enter position
                    holding = True
                    entry_price = buy_price
                    entry_date = current
                    trade_type = "enter"
            else:
                # Holding - check exit
                if can_sell:
                    gross_pnl = (sell_price - entry_price) / entry_price
                    net_pnl = gross_pnl - (2 * fee_pct)
                    trade_pnl = net_pnl
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=symbol,
                        entry_price=entry_price,
                        exit_price=sell_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="exit",
                        fees_pct=2 * fee_pct * 100,
                    ))

                    holding = False
                    entry_price = 0.0
                    trade_type = "exit"
                else:
                    trade_type = "holding"

            equity_values.append((current, capital))
            current += timedelta(days=1)

        return self._build_result(
            f"fixed_{symbol}_b{int(buy_threshold_pct*10000)}s{int(sell_threshold_pct*10000)}",
            StrategyMode.FIXED_THRESHOLD,
            capital,
            equity_values,
            trades,
        )

    def run_chronos_adaptive(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        threshold_buffer_pct: float = 0.001,  # Buffer from predicted high/low
    ) -> StrategyResult:
        """Run strategy using Chronos predicted high/low for adaptive thresholds."""

        self._ensure_forecaster()
        fee_pct = self._get_fee_pct(symbol)
        capital = self.initial_capital
        equity_values = []
        trades = []

        holding = False
        entry_price = 0.0

        current = start_date
        while current <= end_date:
            price_data = self.data_loader.get_price_on_date(symbol, current)
            if not price_data:
                current += timedelta(days=1)
                continue

            # Get Chronos forecast
            forecast = self._get_chronos_forecast(symbol, current)
            if forecast is None:
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            open_price = price_data["open"]
            high = price_data["high"]
            low = price_data["low"]

            # Adaptive thresholds based on Chronos predictions
            # Buy near predicted low, sell near predicted high
            predicted_low = forecast["predicted_low"]
            predicted_high = forecast["predicted_high"]

            # Set buy price slightly above predicted low (to increase fill probability)
            buy_price = predicted_low * (1 + threshold_buffer_pct)
            # Set sell price slightly below predicted high
            sell_price = predicted_high * (1 - threshold_buffer_pct)

            # Sanity check - buy must be below sell
            if buy_price >= sell_price:
                # Skip this day - prediction suggests no profitable trade
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            can_buy = low <= buy_price
            can_sell = high >= sell_price

            if not holding:
                if can_buy and can_sell:
                    # Round-trip
                    gross_pnl = (sell_price - buy_price) / buy_price
                    net_pnl = gross_pnl - (2 * fee_pct)

                    if net_pnl > 0:  # Only take profitable trades
                        capital *= (1 + net_pnl)
                        trades.append(TradeResult(
                            trade_date=current,
                            symbol=symbol,
                            entry_price=buy_price,
                            exit_price=sell_price,
                            pnl_pct=net_pnl * 100,
                            trade_type="round_trip",
                            fees_pct=2 * fee_pct * 100,
                            strategy_info={
                                "predicted_low": predicted_low,
                                "predicted_high": predicted_high,
                                "actual_low": low,
                                "actual_high": high,
                            }
                        ))

                elif can_buy:
                    holding = True
                    entry_price = buy_price

            else:
                if can_sell:
                    gross_pnl = (sell_price - entry_price) / entry_price
                    net_pnl = gross_pnl - (2 * fee_pct)
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=symbol,
                        entry_price=entry_price,
                        exit_price=sell_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="exit",
                        fees_pct=2 * fee_pct * 100,
                    ))

                    holding = False

            equity_values.append((current, capital))
            current += timedelta(days=1)

        return self._build_result(
            f"chronos_adaptive_{symbol}",
            StrategyMode.CHRONOS_ADAPTIVE,
            capital,
            equity_values,
            trades,
        )

    def run_chronos_symbol_select(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        buy_threshold_pct: float = 0.005,
        sell_threshold_pct: float = 0.005,
    ) -> StrategyResult:
        """Select best symbol each day using Chronos predicted return."""

        self._ensure_forecaster()
        capital = self.initial_capital
        equity_values = []
        trades = []

        holding = False
        entry_price = 0.0
        entry_symbol = None

        current = start_date
        while current <= end_date:
            # Get forecasts for all symbols
            best_symbol = None
            best_return = -float("inf")

            for symbol in symbols:
                forecast = self._get_chronos_forecast(symbol, current)
                if forecast and forecast["predicted_return"] > best_return:
                    best_return = forecast["predicted_return"]
                    best_symbol = symbol

            if best_symbol is None:
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            price_data = self.data_loader.get_price_on_date(best_symbol, current)
            if not price_data:
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            fee_pct = self._get_fee_pct(best_symbol)
            open_price = price_data["open"]
            high = price_data["high"]
            low = price_data["low"]

            buy_price = open_price * (1 - buy_threshold_pct)
            sell_price = open_price * (1 + sell_threshold_pct)

            can_buy = low <= buy_price
            can_sell = high >= sell_price

            # If holding different symbol, close first
            if holding and entry_symbol != best_symbol:
                # Force close at current close
                close_data = self.data_loader.get_price_on_date(entry_symbol, current)
                if close_data:
                    exit_price = close_data["close"]
                    gross_pnl = (exit_price - entry_price) / entry_price
                    old_fee = self._get_fee_pct(entry_symbol)
                    net_pnl = gross_pnl - (2 * old_fee)
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=entry_symbol,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="switch_exit",
                        fees_pct=2 * old_fee * 100,
                    ))
                holding = False

            if not holding:
                if can_buy and can_sell:
                    gross_pnl = (sell_price - buy_price) / buy_price
                    net_pnl = gross_pnl - (2 * fee_pct)
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=best_symbol,
                        entry_price=buy_price,
                        exit_price=sell_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="round_trip",
                        fees_pct=2 * fee_pct * 100,
                        strategy_info={"predicted_return": best_return},
                    ))

                elif can_buy and best_return > 0:  # Only enter if predicted return positive
                    holding = True
                    entry_price = buy_price
                    entry_symbol = best_symbol

            else:
                if can_sell:
                    gross_pnl = (sell_price - entry_price) / entry_price
                    net_pnl = gross_pnl - (2 * fee_pct)
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=entry_symbol,
                        entry_price=entry_price,
                        exit_price=sell_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="exit",
                        fees_pct=2 * fee_pct * 100,
                    ))
                    holding = False

            equity_values.append((current, capital))
            current += timedelta(days=1)

        return self._build_result(
            "chronos_symbol_select",
            StrategyMode.CHRONOS_SYMBOL_SELECT,
            capital,
            equity_values,
            trades,
        )

    def run_chronos_full(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        threshold_buffer_pct: float = 0.001,
    ) -> StrategyResult:
        """Full Chronos strategy: symbol selection + adaptive thresholds."""

        self._ensure_forecaster()
        capital = self.initial_capital
        equity_values = []
        trades = []

        holding = False
        entry_price = 0.0
        entry_symbol = None

        current = start_date
        while current <= end_date:
            # Get forecasts for all symbols and pick best
            best_symbol = None
            best_forecast = None
            best_expected_pnl = -float("inf")

            for symbol in symbols:
                forecast = self._get_chronos_forecast(symbol, current)
                if forecast is None:
                    continue

                # Expected PnL = (predicted_high - predicted_low) / predicted_low - fees
                predicted_range = forecast["predicted_high"] - forecast["predicted_low"]
                expected_pnl = predicted_range / forecast["predicted_low"] - (2 * self._get_fee_pct(symbol))

                if expected_pnl > best_expected_pnl and expected_pnl > 0:
                    best_expected_pnl = expected_pnl
                    best_symbol = symbol
                    best_forecast = forecast

            if best_symbol is None or best_forecast is None:
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            price_data = self.data_loader.get_price_on_date(best_symbol, current)
            if not price_data:
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            fee_pct = self._get_fee_pct(best_symbol)
            high = price_data["high"]
            low = price_data["low"]

            # Adaptive thresholds
            buy_price = best_forecast["predicted_low"] * (1 + threshold_buffer_pct)
            sell_price = best_forecast["predicted_high"] * (1 - threshold_buffer_pct)

            if buy_price >= sell_price:
                equity_values.append((current, capital))
                current += timedelta(days=1)
                continue

            can_buy = low <= buy_price
            can_sell = high >= sell_price

            # Handle symbol switch
            if holding and entry_symbol != best_symbol:
                close_data = self.data_loader.get_price_on_date(entry_symbol, current)
                if close_data:
                    exit_price = close_data["close"]
                    gross_pnl = (exit_price - entry_price) / entry_price
                    old_fee = self._get_fee_pct(entry_symbol)
                    net_pnl = gross_pnl - (2 * old_fee)
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=entry_symbol,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="switch_exit",
                        fees_pct=2 * old_fee * 100,
                    ))
                holding = False

            if not holding:
                if can_buy and can_sell:
                    gross_pnl = (sell_price - buy_price) / buy_price
                    net_pnl = gross_pnl - (2 * fee_pct)

                    if net_pnl > 0:
                        capital *= (1 + net_pnl)
                        trades.append(TradeResult(
                            trade_date=current,
                            symbol=best_symbol,
                            entry_price=buy_price,
                            exit_price=sell_price,
                            pnl_pct=net_pnl * 100,
                            trade_type="round_trip",
                            fees_pct=2 * fee_pct * 100,
                            strategy_info={
                                "predicted_low": best_forecast["predicted_low"],
                                "predicted_high": best_forecast["predicted_high"],
                                "actual_low": low,
                                "actual_high": high,
                                "expected_pnl": best_expected_pnl,
                            }
                        ))

                elif can_buy:
                    holding = True
                    entry_price = buy_price
                    entry_symbol = best_symbol

            else:
                if can_sell:
                    gross_pnl = (sell_price - entry_price) / entry_price
                    net_pnl = gross_pnl - (2 * fee_pct)
                    capital *= (1 + net_pnl)

                    trades.append(TradeResult(
                        trade_date=current,
                        symbol=entry_symbol,
                        entry_price=entry_price,
                        exit_price=sell_price,
                        pnl_pct=net_pnl * 100,
                        trade_type="exit",
                        fees_pct=2 * fee_pct * 100,
                    ))
                    holding = False

            equity_values.append((current, capital))
            current += timedelta(days=1)

        return self._build_result(
            "chronos_full",
            StrategyMode.CHRONOS_FULL,
            capital,
            equity_values,
            trades,
        )

    def _build_result(
        self,
        name: str,
        mode: StrategyMode,
        final_capital: float,
        equity_values: List[Tuple[date, float]],
        trades: List[TradeResult],
    ) -> StrategyResult:
        """Build StrategyResult from simulation data."""

        if not equity_values:
            return self._empty_result(name, mode)

        dates, values = zip(*equity_values)
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))
        daily_returns = equity_curve.pct_change().dropna()

        total_return = (final_capital - self.initial_capital) / self.initial_capital
        n_days = len(equity_curve)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        sharpe = self._calculate_sharpe(daily_returns)
        max_dd = self._calculate_max_drawdown(equity_curve)

        winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
        win_rate = winning_trades / len(trades) if trades else 0.0

        return StrategyResult(
            strategy_name=name,
            mode=mode,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate=win_rate,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )

    def _empty_result(self, name: str, mode: StrategyMode) -> StrategyResult:
        """Return empty result for failed strategies."""
        return StrategyResult(
            strategy_name=name,
            mode=mode,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            total_trades=0,
            trades=[],
            equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
        )

    def _calculate_sharpe(self, daily_returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(daily_returns) < 2:
            return 0.0
        mean = daily_returns.mean()
        std = daily_returns.std()
        if std == 0:
            return 0.0
        return float(mean / std * np.sqrt(252))

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return abs(float(drawdown.min()))

    def cleanup(self):
        """Release resources."""
        if self._forecaster:
            self._forecaster.unload()
            self._forecaster = None


def run_comparison(
    symbols: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
    baseline_symbol: str = "BTCUSD",
) -> Dict[str, StrategyResult]:
    """Run comparison of all strategies.

    Args:
        symbols: Symbols to trade
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        baseline_symbol: Symbol for baseline comparison

    Returns:
        Dict of strategy_name -> StrategyResult
    """
    from marketsimlong.data import DailyDataLoader
    from marketsimlong.config import DataConfigLong

    # Load data
    config = DataConfigLong(
        stock_symbols=tuple(s for s in symbols if not s.endswith("USD")),
        crypto_symbols=tuple(s for s in symbols if s.endswith("USD")),
        start_date=start_date,
        end_date=end_date,
    )
    data_loader = DailyDataLoader(config)
    data_loader.load_all_symbols()

    backtester = RealisticBacktester(data_loader, initial_capital)
    results = {}

    try:
        # 1. Buy and hold baseline
        logger.info("Running buy-and-hold baseline on %s...", baseline_symbol)
        baseline = backtester.run_buy_and_hold(baseline_symbol, start_date, end_date)
        results["buy_hold_baseline"] = baseline

        # 2. Buy and hold best performers
        for symbol in ["SOLUSD", "UNIUSD"]:
            if symbol in symbols:
                logger.info("Running buy-and-hold on %s...", symbol)
                results[f"buy_hold_{symbol}"] = backtester.run_buy_and_hold(symbol, start_date, end_date)

        # 3. Fixed threshold on best symbol
        logger.info("Running fixed threshold on SOLUSD...")
        results["fixed_SOLUSD"] = backtester.run_fixed_threshold(
            "SOLUSD", start_date, end_date,
            buy_threshold_pct=0.005, sell_threshold_pct=0.005
        )

        # 4. Chronos adaptive on single symbol
        logger.info("Running Chronos adaptive on SOLUSD...")
        results["chronos_adaptive_SOLUSD"] = backtester.run_chronos_adaptive(
            "SOLUSD", start_date, end_date
        )

        # 5. Chronos symbol selection
        logger.info("Running Chronos symbol selection...")
        results["chronos_symbol_select"] = backtester.run_chronos_symbol_select(
            symbols, start_date, end_date
        )

        # 6. Full Chronos (adaptive + symbol select)
        logger.info("Running full Chronos strategy...")
        results["chronos_full"] = backtester.run_chronos_full(
            symbols, start_date, end_date
        )

        # Calculate alpha for each strategy vs baseline
        baseline_return = baseline.total_return_pct
        for name, result in results.items():
            result.baseline_return_pct = baseline_return
            result.alpha_pct = result.total_return_pct - baseline_return

    finally:
        backtester.cleanup()

    return results


def print_comparison(results: Dict[str, StrategyResult]) -> None:
    """Print comparison table."""

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 100)
    print(f"{'Strategy':<30} {'Return':>10} {'Ann.Ret':>10} {'Alpha':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
    print("-" * 100)

    # Sort by total return
    sorted_results = sorted(results.items(), key=lambda x: x[1].total_return_pct, reverse=True)

    for name, r in sorted_results:
        print(f"{name:<30} {r.total_return_pct:>9.2f}% {r.annualized_return_pct:>9.2f}% {r.alpha_pct:>9.2f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>7.2f}% {r.win_rate*100:>7.1f}% {r.total_trades:>8}")

    print("=" * 100)


def save_comparison(results: Dict[str, StrategyResult], output_path: Path) -> None:
    """Save comparison results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, r in results.items():
        data[name] = {
            "total_return_pct": r.total_return_pct,
            "annualized_return_pct": r.annualized_return_pct,
            "alpha_pct": r.alpha_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown_pct": r.max_drawdown_pct,
            "win_rate": r.win_rate,
            "total_trades": r.total_trades,
            "initial_capital": r.initial_capital,
            "final_capital": r.final_capital,
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved comparison to %s", output_path)
