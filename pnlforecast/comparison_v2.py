"""Strategy comparison v2 - Rigorous simulator with proper position tracking.

Key fixes:
1. Tracks unrealized P&L in equity curve
2. Option to force exit at close if sell not hit (no overnight risk)
3. Uses original entry day's sell target (not shifting targets)
4. Liquidates all positions at end of backtest
5. Proper unit test cases for edge scenarios
"""

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


class ExitMode(Enum):
    """How to handle positions that don't hit sell target."""
    SAME_DAY_ONLY = "same_day_only"  # Force exit at close if sell not hit
    HOLD_WITH_ORIGINAL_TARGET = "hold_original"  # Keep original sell target
    HOLD_WITH_ADAPTIVE_TARGET = "hold_adaptive"  # Use new day's sell target


@dataclass
class Position:
    """Tracks an open position."""
    symbol: str
    entry_date: date
    entry_price: float
    target_sell_price: float  # Original sell target
    quantity: float = 1.0


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_date: date
    exit_date: date
    symbol: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    fees_pct: float
    trade_type: str  # round_trip, exit_at_target, exit_at_close, forced_liquidation
    days_held: int
    strategy_info: dict = field(default_factory=dict)


@dataclass
class DailyState:
    """Daily portfolio state for equity curve."""
    date: date
    capital: float  # Cash + realized P&L
    position_value: float  # Mark-to-market value of open positions
    total_equity: float  # capital + position_value
    has_position: bool
    position_symbol: Optional[str] = None


@dataclass
class StrategyResultV2:
    """Enhanced result with proper position tracking."""
    strategy_name: str
    exit_mode: ExitMode
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    round_trip_same_day: int
    multi_day_exits: int
    forced_exits_at_close: int
    trades: List[TradeResult]
    daily_states: List[DailyState]


class RigorousBacktester:
    """Backtester with proper position tracking and edge case handling."""

    def __init__(
        self,
        data_loader,
        initial_capital: float = 100_000.0,
        stock_fee_pct: float = 0.0003,  # 3bp per side
        crypto_fee_pct: float = 0.0008,  # 8bp per side
        leverage: float = 1.0,  # 1.0 = no leverage, 2.0 = 2x
        margin_rate_annual: float = 0.0625,  # 6.25% annual
    ):
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.stock_fee_pct = stock_fee_pct
        self.crypto_fee_pct = crypto_fee_pct
        self.leverage = leverage
        self.margin_rate_daily = margin_rate_annual / 365
        self._forecaster = None

    def _get_fee_pct(self, symbol: str) -> float:
        from marketsimlong.data import is_crypto_symbol
        return self.crypto_fee_pct if is_crypto_symbol(symbol) else self.stock_fee_pct

    def _ensure_forecaster(self):
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

    def run_chronos_full_v2(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        threshold_buffer_pct: float = 0.001,
        exit_mode: ExitMode = ExitMode.SAME_DAY_ONLY,
        top_n: int = 1,  # Select top N symbols
    ) -> StrategyResultV2:
        """Run chronos_full with rigorous position tracking.

        Args:
            symbols: List of symbols to trade
            start_date: Start date
            end_date: End date
            threshold_buffer_pct: Buffer from predicted high/low
            exit_mode: How to handle positions that don't hit target
            top_n: Number of top symbols to consider (1 = best only)
        """
        self._ensure_forecaster()
        from marketsimlong.data import is_crypto_symbol

        capital = self.initial_capital
        margin_borrowed = 0.0  # Track margin usage
        total_margin_interest = 0.0
        position: Optional[Position] = None
        trades: List[TradeResult] = []
        daily_states: List[DailyState] = []

        # Stats
        round_trip_same_day = 0
        multi_day_exits = 0
        forced_exits = 0

        current = start_date
        while current <= end_date:
            # Get forecasts and rank by expected PnL
            symbol_forecasts = []
            for symbol in symbols:
                forecast = self._get_chronos_forecast(symbol, current)
                if forecast is None:
                    continue

                predicted_range = forecast["predicted_high"] - forecast["predicted_low"]
                expected_pnl = predicted_range / forecast["predicted_low"] - (2 * self._get_fee_pct(symbol))

                if expected_pnl > 0:
                    symbol_forecasts.append({
                        "symbol": symbol,
                        "forecast": forecast,
                        "expected_pnl": expected_pnl,
                        "buy_price": forecast["predicted_low"] * (1 + threshold_buffer_pct),
                        "sell_price": forecast["predicted_high"] * (1 - threshold_buffer_pct),
                    })

            # Sort by expected PnL, take top N
            symbol_forecasts.sort(key=lambda x: x["expected_pnl"], reverse=True)
            top_symbols = symbol_forecasts[:top_n]

            # Get price data for position if holding
            position_value = 0.0
            if position:
                pos_price_data = self.data_loader.get_price_on_date(position.symbol, current)
                if pos_price_data:
                    position_value = capital * (pos_price_data["close"] / position.entry_price - 1)
                    position_value = max(position_value, -capital)  # Can't lose more than capital

            # Check if should exit current position
            if position:
                pos_price_data = self.data_loader.get_price_on_date(position.symbol, current)

                if pos_price_data:
                    high = pos_price_data["high"]
                    close = pos_price_data["close"]
                    fee_pct = self._get_fee_pct(position.symbol)

                    # Determine sell price based on exit mode
                    if exit_mode == ExitMode.HOLD_WITH_ORIGINAL_TARGET:
                        sell_target = position.target_sell_price
                    elif exit_mode == ExitMode.HOLD_WITH_ADAPTIVE_TARGET:
                        # Use today's forecast for same symbol
                        today_forecast = self._get_chronos_forecast(position.symbol, current)
                        if today_forecast:
                            sell_target = today_forecast["predicted_high"] * (1 - threshold_buffer_pct)
                        else:
                            sell_target = position.target_sell_price
                    else:  # SAME_DAY_ONLY - but we're holding from before, force exit
                        sell_target = float("inf")  # Force exit at close below

                    can_sell = high >= sell_target

                    if can_sell:
                        # Exit at target
                        exit_price = sell_target
                        gross_pnl = (exit_price - position.entry_price) / position.entry_price
                        net_pnl = gross_pnl - (2 * fee_pct)
                        # Apply leverage (stocks only)
                        is_stock = not is_crypto_symbol(position.symbol)
                        effective_leverage = self.leverage if is_stock else 1.0
                        leveraged_pnl = net_pnl * effective_leverage
                        capital *= (1 + leveraged_pnl)
                        # Pay back margin
                        if margin_borrowed > 0:
                            payback = min(margin_borrowed, capital - self.initial_capital)
                            if payback > 0:
                                margin_borrowed -= payback

                        days_held = (current - position.entry_date).days
                        trade_type = "exit_at_target" if days_held > 0 else "round_trip"

                        if days_held > 0:
                            multi_day_exits += 1

                        trades.append(TradeResult(
                            entry_date=position.entry_date,
                            exit_date=current,
                            symbol=position.symbol,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            pnl_pct=leveraged_pnl * 100,
                            fees_pct=2 * fee_pct * 100,
                            trade_type=trade_type,
                            days_held=days_held,
                        ))
                        position = None

                    elif exit_mode == ExitMode.SAME_DAY_ONLY:
                        # Position held from before - this shouldn't happen with SAME_DAY_ONLY
                        # Force exit at close
                        exit_price = close
                        gross_pnl = (exit_price - position.entry_price) / position.entry_price
                        net_pnl = gross_pnl - (2 * fee_pct)
                        # Apply leverage (stocks only)
                        is_stock = not is_crypto_symbol(position.symbol)
                        effective_leverage = self.leverage if is_stock else 1.0
                        leveraged_pnl = net_pnl * effective_leverage
                        capital *= (1 + leveraged_pnl)
                        # Pay back margin
                        if margin_borrowed > 0:
                            payback = min(margin_borrowed, capital - self.initial_capital)
                            if payback > 0:
                                margin_borrowed -= payback

                        days_held = (current - position.entry_date).days
                        forced_exits += 1

                        trades.append(TradeResult(
                            entry_date=position.entry_date,
                            exit_date=current,
                            symbol=position.symbol,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            pnl_pct=leveraged_pnl * 100,
                            fees_pct=2 * fee_pct * 100,
                            trade_type="forced_exit_at_close",
                            days_held=days_held,
                        ))
                        position = None

            # Try to enter new position if not holding
            if position is None and top_symbols:
                for sym_data in top_symbols:
                    symbol = sym_data["symbol"]
                    price_data = self.data_loader.get_price_on_date(symbol, current)
                    if not price_data:
                        continue

                    buy_price = sym_data["buy_price"]
                    sell_price = sym_data["sell_price"]

                    if buy_price >= sell_price:
                        continue

                    low = price_data["low"]
                    high = price_data["high"]
                    close = price_data["close"]
                    fee_pct = self._get_fee_pct(symbol)

                    can_buy = low <= buy_price
                    can_sell = high >= sell_price

                    if can_buy and can_sell:
                        # Round-trip same day
                        gross_pnl = (sell_price - buy_price) / buy_price
                        net_pnl = gross_pnl - (2 * fee_pct)

                        if net_pnl > 0:
                            # Apply leverage (stocks only)
                            is_stock = not is_crypto_symbol(symbol)
                            effective_leverage = self.leverage if is_stock else 1.0
                            leveraged_pnl = net_pnl * effective_leverage
                            capital *= (1 + leveraged_pnl)
                            round_trip_same_day += 1

                            trades.append(TradeResult(
                                entry_date=current,
                                exit_date=current,
                                symbol=symbol,
                                entry_price=buy_price,
                                exit_price=sell_price,
                                pnl_pct=leveraged_pnl * 100,
                                fees_pct=2 * fee_pct * 100,
                                trade_type="round_trip",
                                days_held=0,
                                strategy_info={
                                    "predicted_low": sym_data["forecast"]["predicted_low"],
                                    "predicted_high": sym_data["forecast"]["predicted_high"],
                                    "actual_low": low,
                                    "actual_high": high,
                                    "expected_pnl": sym_data["expected_pnl"],
                                    "leverage": effective_leverage,
                                }
                            ))
                        break  # Only take one trade per day

                    elif can_buy:
                        if exit_mode == ExitMode.SAME_DAY_ONLY:
                            # Buy but force exit at close same day
                            gross_pnl = (close - buy_price) / buy_price
                            net_pnl = gross_pnl - (2 * fee_pct)
                            # Apply leverage (stocks only)
                            is_stock = not is_crypto_symbol(symbol)
                            effective_leverage = self.leverage if is_stock else 1.0
                            leveraged_pnl = net_pnl * effective_leverage
                            capital *= (1 + leveraged_pnl)
                            forced_exits += 1

                            trades.append(TradeResult(
                                entry_date=current,
                                exit_date=current,
                                symbol=symbol,
                                entry_price=buy_price,
                                exit_price=close,
                                pnl_pct=leveraged_pnl * 100,
                                fees_pct=2 * fee_pct * 100,
                                trade_type="forced_exit_at_close",
                                days_held=0,
                                strategy_info={
                                    "predicted_low": sym_data["forecast"]["predicted_low"],
                                    "predicted_high": sym_data["forecast"]["predicted_high"],
                                    "actual_low": low,
                                    "actual_high": high,
                                    "expected_pnl": sym_data["expected_pnl"],
                                    "target_sell": sell_price,
                                    "actual_exit": close,
                                    "leverage": effective_leverage,
                                }
                            ))
                        else:
                            # Enter and hold
                            position = Position(
                                symbol=symbol,
                                entry_date=current,
                                entry_price=buy_price,
                                target_sell_price=sell_price,
                            )
                        break

            # Calculate position value for equity curve
            position_value = 0.0
            if position:
                pos_price_data = self.data_loader.get_price_on_date(position.symbol, current)
                if pos_price_data:
                    # Unrealized P&L based on current close
                    unrealized_pnl = (pos_price_data["close"] - position.entry_price) / position.entry_price
                    # Apply leverage for position value calculation (stocks only)
                    is_stock = not is_crypto_symbol(position.symbol)
                    effective_leverage = self.leverage if is_stock else 1.0
                    position_value = capital * unrealized_pnl * effective_leverage

                    # Calculate margin borrowed for leveraged stock positions
                    if is_stock and effective_leverage > 1.0:
                        borrowed_this_position = capital * (effective_leverage - 1)
                        margin_borrowed = max(margin_borrowed, borrowed_this_position)

            # Apply daily margin interest if holding leveraged position
            if margin_borrowed > 0:
                daily_interest = margin_borrowed * self.margin_rate_daily
                capital -= daily_interest
                total_margin_interest += daily_interest

            daily_states.append(DailyState(
                date=current,
                capital=capital,
                position_value=position_value,
                total_equity=capital + position_value - margin_borrowed,
                has_position=position is not None,
                position_symbol=position.symbol if position else None,
            ))

            current += timedelta(days=1)

        # Force liquidate any remaining position at end
        if position:
            last_date = end_date
            pos_price_data = self.data_loader.get_price_on_date(position.symbol, last_date)
            if pos_price_data is None:
                # Try previous day
                for i in range(1, 10):
                    check_date = last_date - timedelta(days=i)
                    pos_price_data = self.data_loader.get_price_on_date(position.symbol, check_date)
                    if pos_price_data:
                        last_date = check_date
                        break

            if pos_price_data:
                exit_price = pos_price_data["close"]
                fee_pct = self._get_fee_pct(position.symbol)
                gross_pnl = (exit_price - position.entry_price) / position.entry_price
                net_pnl = gross_pnl - (2 * fee_pct)
                capital *= (1 + net_pnl)

                days_held = (last_date - position.entry_date).days

                trades.append(TradeResult(
                    entry_date=position.entry_date,
                    exit_date=last_date,
                    symbol=position.symbol,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl_pct=net_pnl * 100,
                    fees_pct=2 * fee_pct * 100,
                    trade_type="end_of_backtest_liquidation",
                    days_held=days_held,
                ))

        # Build result
        leverage_str = f"_{self.leverage:.0f}x" if self.leverage > 1.0 else ""
        return self._build_result(
            f"chronos_full_v2_{exit_mode.value}_top{top_n}{leverage_str}",
            exit_mode,
            capital,
            daily_states,
            trades,
            round_trip_same_day,
            multi_day_exits,
            forced_exits,
        )

    def _build_result(
        self,
        name: str,
        exit_mode: ExitMode,
        final_capital: float,
        daily_states: List[DailyState],
        trades: List[TradeResult],
        round_trip_same_day: int,
        multi_day_exits: int,
        forced_exits: int,
    ) -> StrategyResultV2:

        if not daily_states:
            return StrategyResultV2(
                strategy_name=name,
                exit_mode=exit_mode,
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                total_trades=0,
                round_trip_same_day=0,
                multi_day_exits=0,
                forced_exits_at_close=0,
                trades=[],
                daily_states=[],
            )

        # Use total_equity (includes unrealized) for proper equity curve
        equity_values = [(s.date, s.total_equity) for s in daily_states]
        dates, values = zip(*equity_values)
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))
        daily_returns = equity_curve.pct_change().dropna()

        total_return = (final_capital - self.initial_capital) / self.initial_capital
        n_days = len(equity_curve)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe
        if len(daily_returns) < 2 or daily_returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(float(drawdown.min()))

        winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
        win_rate = winning_trades / len(trades) if trades else 0.0

        return StrategyResultV2(
            strategy_name=name,
            exit_mode=exit_mode,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate=win_rate,
            total_trades=len(trades),
            round_trip_same_day=round_trip_same_day,
            multi_day_exits=multi_day_exits,
            forced_exits_at_close=forced_exits,
            trades=trades,
            daily_states=daily_states,
        )

    def cleanup(self):
        if self._forecaster:
            self._forecaster.unload()
            self._forecaster = None


def run_rigorous_comparison(
    symbols: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
) -> Dict[str, StrategyResultV2]:
    """Run rigorous comparison with multiple exit modes and top_n values."""
    from marketsimlong.data import DailyDataLoader
    from marketsimlong.config import DataConfigLong

    config = DataConfigLong(
        stock_symbols=tuple(s for s in symbols if not s.endswith("USD")),
        crypto_symbols=tuple(s for s in symbols if s.endswith("USD")),
        start_date=start_date,
        end_date=end_date,
    )
    data_loader = DailyDataLoader(config)
    data_loader.load_all_symbols()

    backtester = RigorousBacktester(data_loader, initial_capital)
    results = {}

    try:
        # Test different configurations
        for exit_mode in ExitMode:
            for top_n in [1, 2, 3]:
                logger.info("Running chronos_full_v2 with exit_mode=%s, top_n=%d...", exit_mode.value, top_n)
                result = backtester.run_chronos_full_v2(
                    symbols, start_date, end_date,
                    exit_mode=exit_mode,
                    top_n=top_n,
                )
                results[result.strategy_name] = result

                logger.info(
                    "  -> Return: %.2f%%, Trades: %d (RT: %d, Multi: %d, Forced: %d)",
                    result.total_return_pct,
                    result.total_trades,
                    result.round_trip_same_day,
                    result.multi_day_exits,
                    result.forced_exits_at_close,
                )

    finally:
        backtester.cleanup()

    return results


def print_rigorous_comparison(results: Dict[str, StrategyResultV2]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 140)
    print("RIGOROUS STRATEGY COMPARISON - V2")
    print("=" * 140)
    print(f"{'Strategy':<45} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'RT':>5} {'Multi':>5} {'Forced':>6}")
    print("-" * 140)

    sorted_results = sorted(results.items(), key=lambda x: x[1].total_return_pct, reverse=True)

    for name, r in sorted_results:
        print(f"{name:<45} {r.total_return_pct:>9.2f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>7.2f}% {r.win_rate*100:>7.1f}% {r.total_trades:>7} {r.round_trip_same_day:>5} {r.multi_day_exits:>5} {r.forced_exits_at_close:>6}")

    print("=" * 140)
    print("\nLegend: RT=Round Trip Same Day, Multi=Multi-Day Exit at Target, Forced=Forced Exit at Close")


def analyze_trades(result: StrategyResultV2) -> None:
    """Analyze trades in detail."""
    print(f"\n{'='*80}")
    print(f"TRADE ANALYSIS: {result.strategy_name}")
    print(f"{'='*80}")

    if not result.trades:
        print("No trades")
        return

    # Group by trade type
    by_type = {}
    for t in result.trades:
        by_type.setdefault(t.trade_type, []).append(t)

    print("\nBy Trade Type:")
    for ttype, trades in sorted(by_type.items()):
        avg_pnl = np.mean([t.pnl_pct for t in trades])
        win_rate = sum(1 for t in trades if t.pnl_pct > 0) / len(trades)
        total_pnl = sum(t.pnl_pct for t in trades)
        avg_days = np.mean([t.days_held for t in trades])

        print(f"  {ttype:<25}: {len(trades):>4} trades, "
              f"avg PnL: {avg_pnl:>6.2f}%, "
              f"total PnL: {total_pnl:>8.2f}%, "
              f"win rate: {win_rate*100:>5.1f}%, "
              f"avg days: {avg_days:.1f}")

    # Show worst trades
    print("\nWorst 5 Trades:")
    worst = sorted(result.trades, key=lambda t: t.pnl_pct)[:5]
    for t in worst:
        print(f"  {t.entry_date} {t.symbol:<8} {t.trade_type:<20} PnL: {t.pnl_pct:>6.2f}% "
              f"(entry: ${t.entry_price:.2f}, exit: ${t.exit_price:.2f}, days: {t.days_held})")

    # Show best trades
    print("\nBest 5 Trades:")
    best = sorted(result.trades, key=lambda t: t.pnl_pct, reverse=True)[:5]
    for t in best:
        print(f"  {t.entry_date} {t.symbol:<8} {t.trade_type:<20} PnL: {t.pnl_pct:>6.2f}% "
              f"(entry: ${t.entry_price:.2f}, exit: ${t.exit_price:.2f}, days: {t.days_held})")
