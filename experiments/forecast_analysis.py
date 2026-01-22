"""Forecast Analysis and Meta-Algorithm Experimentation.

Analyze Chronos2 forecast quality and test different selection algorithms
inspired by dynamic programming and optimal packing theory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimlong.config import DataConfigLong, ForecastConfigLong
from marketsimlong.data import DailyDataLoader, is_crypto_symbol
from marketsimlong.forecaster import Chronos2Forecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastRecord:
    """Record of a single day's forecast vs actual outcome."""
    date: date
    symbol: str

    # Forecast values
    predicted_low: float
    predicted_high: float
    predicted_close: float
    predicted_return: float  # (high - low) / low

    # Actual values
    actual_open: float
    actual_low: float
    actual_high: float
    actual_close: float
    actual_return: float  # (close - open) / open

    # Derived metrics
    direction_correct: bool  # Did we predict direction correctly?
    low_error_pct: float  # (predicted_low - actual_low) / actual_low
    high_error_pct: float  # (predicted_high - actual_high) / actual_high

    # If we traded this forecast
    theoretical_pnl: float = 0.0  # If bought at predicted_low, sold at predicted_high


@dataclass
class DaySnapshot:
    """All forecasts and outcomes for a single day."""
    date: date
    forecasts: List[ForecastRecord]

    def get_top_n_by_predicted_return(self, n: int) -> List[ForecastRecord]:
        """Get top N symbols by predicted return."""
        return sorted(self.forecasts, key=lambda x: x.predicted_return, reverse=True)[:n]

    def get_top_n_by_actual_return(self, n: int) -> List[ForecastRecord]:
        """Get top N symbols by actual return (hindsight)."""
        return sorted(self.forecasts, key=lambda x: x.actual_return, reverse=True)[:n]


class ForecastAnalyzer:
    """Analyze forecast quality and test meta-algorithms."""

    def __init__(
        self,
        data_loader: DailyDataLoader,
        forecaster: Chronos2Forecaster,
    ):
        self.data_loader = data_loader
        self.forecaster = forecaster
        self.daily_snapshots: List[DaySnapshot] = []

    def collect_forecasts(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None,
    ) -> List[DaySnapshot]:
        """Collect forecasts and actual outcomes for date range.

        Args:
            start_date: Start date
            end_date: End date
            symbols: Symbols to analyze (None = all available)

        Returns:
            List of DaySnapshot objects
        """
        self.daily_snapshots = []

        current = start_date
        day_count = 0

        while current <= end_date:
            # Get available symbols for this day
            available = self.data_loader.get_tradable_symbols_on_date(current)
            if symbols:
                available = [s for s in available if s in symbols]

            if not available:
                current += timedelta(days=1)
                continue

            # Generate forecasts
            forecasts_obj = self.forecaster.forecast_all_symbols(current, available)

            if not forecasts_obj.forecasts:
                current += timedelta(days=1)
                continue

            # Get actual outcomes (next day or same day close)
            records = []
            for symbol, forecast in forecasts_obj.forecasts.items():
                # Get actual price data for this day
                price_data = self.data_loader.get_price_on_date(symbol, current)
                if not price_data:
                    continue

                actual_open = price_data.get("open", price_data["close"])
                actual_low = price_data.get("low", price_data["close"])
                actual_high = price_data.get("high", price_data["close"])
                actual_close = price_data["close"]

                actual_return = (actual_close - actual_open) / actual_open if actual_open > 0 else 0

                # Calculate errors
                low_error = (forecast.predicted_low - actual_low) / actual_low if actual_low > 0 else 0
                high_error = (forecast.predicted_high - actual_high) / actual_high if actual_high > 0 else 0

                # Direction correct if both predict same sign
                direction_correct = (forecast.predicted_return > 0) == (actual_return > 0)

                # Theoretical PnL: buy at predicted low, sell at predicted high
                # But capped by actual range
                buy_price = max(forecast.predicted_low, actual_low)  # Can't buy below actual low
                sell_price = min(forecast.predicted_high, actual_high)  # Can't sell above actual high
                theoretical_pnl = (sell_price - buy_price) / buy_price if buy_price > 0 else 0

                record = ForecastRecord(
                    date=current,
                    symbol=symbol,
                    predicted_low=forecast.predicted_low,
                    predicted_high=forecast.predicted_high,
                    predicted_close=forecast.predicted_close,
                    predicted_return=forecast.predicted_return,
                    actual_open=actual_open,
                    actual_low=actual_low,
                    actual_high=actual_high,
                    actual_close=actual_close,
                    actual_return=actual_return,
                    direction_correct=direction_correct,
                    low_error_pct=low_error,
                    high_error_pct=high_error,
                    theoretical_pnl=theoretical_pnl,
                )
                records.append(record)

            if records:
                self.daily_snapshots.append(DaySnapshot(date=current, forecasts=records))
                day_count += 1

                if day_count % 5 == 0:
                    logger.info("Collected %d days of forecasts (up to %s)", day_count, current)

            current += timedelta(days=1)

        logger.info("Collected %d total days of forecasts", len(self.daily_snapshots))
        return self.daily_snapshots

    def compute_forecast_accuracy_stats(self) -> Dict:
        """Compute overall forecast accuracy statistics."""
        all_records = [r for day in self.daily_snapshots for r in day.forecasts]

        if not all_records:
            return {}

        direction_accuracy = sum(1 for r in all_records if r.direction_correct) / len(all_records)

        low_errors = [r.low_error_pct for r in all_records]
        high_errors = [r.high_error_pct for r in all_records]
        predicted_returns = [r.predicted_return for r in all_records]
        actual_returns = [r.actual_return for r in all_records]
        theoretical_pnls = [r.theoretical_pnl for r in all_records]

        # Correlation between predicted and actual returns
        correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]

        return {
            "total_forecasts": len(all_records),
            "total_days": len(self.daily_snapshots),
            "direction_accuracy": direction_accuracy,
            "low_error_mean": np.mean(low_errors),
            "low_error_std": np.std(low_errors),
            "high_error_mean": np.mean(high_errors),
            "high_error_std": np.std(high_errors),
            "predicted_return_mean": np.mean(predicted_returns),
            "actual_return_mean": np.mean(actual_returns),
            "return_correlation": correlation,
            "theoretical_pnl_mean": np.mean(theoretical_pnls),
            "theoretical_pnl_std": np.std(theoretical_pnls),
        }

    def compute_per_symbol_stats(self) -> pd.DataFrame:
        """Compute accuracy stats per symbol."""
        symbol_records: Dict[str, List[ForecastRecord]] = {}

        for day in self.daily_snapshots:
            for record in day.forecasts:
                if record.symbol not in symbol_records:
                    symbol_records[record.symbol] = []
                symbol_records[record.symbol].append(record)

        rows = []
        for symbol, records in symbol_records.items():
            direction_acc = sum(1 for r in records if r.direction_correct) / len(records)
            pred_returns = [r.predicted_return for r in records]
            actual_returns = [r.actual_return for r in records]
            corr = np.corrcoef(pred_returns, actual_returns)[0, 1] if len(records) > 1 else 0

            rows.append({
                "symbol": symbol,
                "n_days": len(records),
                "direction_accuracy": direction_acc,
                "pred_return_mean": np.mean(pred_returns),
                "actual_return_mean": np.mean(actual_returns),
                "return_correlation": corr,
                "theoretical_pnl_mean": np.mean([r.theoretical_pnl for r in records]),
            })

        return pd.DataFrame(rows).sort_values("return_correlation", ascending=False)


class MetaAlgorithmBacktester:
    """Test different meta-algorithms for symbol selection."""

    def __init__(self, daily_snapshots: List[DaySnapshot], initial_cash: float = 100_000):
        self.snapshots = daily_snapshots
        self.initial_cash = initial_cash

    def backtest_top_n(
        self,
        n: int = 4,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Simple top-N by predicted return strategy.

        Args:
            n: Number of top symbols to buy
            fee_pct: Fee percentage per trade

        Returns:
            (equity_curve, stats)
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        for day in self.snapshots:
            top_n = day.get_top_n_by_predicted_return(n)

            if not top_n:
                equity_values.append((day.date, cash))
                continue

            # Equal weight allocation
            per_position = cash / len(top_n)
            day_pnl = 0.0

            for record in top_n:
                # Simulate: buy at open, sell at close (simplified)
                trade_pnl = per_position * record.actual_return
                trade_pnl -= per_position * fee_pct * 2  # Entry + exit fee
                day_pnl += trade_pnl
                total_trades += 1
                if record.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((day.date, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        total_return = (cash - self.initial_cash) / self.initial_cash

        stats = {
            "strategy": f"top_{n}_predicted",
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "final_equity": cash,
        }

        return equity_curve, stats

    def backtest_correlation_aware(
        self,
        n: int = 4,
        correlation_threshold: float = 0.7,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Select top symbols but avoid highly correlated ones.

        Inspired by portfolio diversification theory.
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        # Pre-compute return correlations between symbols
        symbol_returns: Dict[str, List[float]] = {}
        for day in self.snapshots:
            for record in day.forecasts:
                if record.symbol not in symbol_returns:
                    symbol_returns[record.symbol] = []
                symbol_returns[record.symbol].append(record.actual_return)

        # Build correlation matrix
        symbols = list(symbol_returns.keys())
        n_symbols = len(symbols)
        corr_matrix = np.eye(n_symbols)

        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                if i < j:
                    r1, r2 = symbol_returns[s1], symbol_returns[s2]
                    min_len = min(len(r1), len(r2))
                    if min_len > 5:
                        corr = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr

        symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        for day in self.snapshots:
            # Sort by predicted return
            sorted_forecasts = sorted(day.forecasts, key=lambda x: x.predicted_return, reverse=True)

            # Greedy selection avoiding correlated symbols
            selected = []
            for record in sorted_forecasts:
                if len(selected) >= n:
                    break

                # Check correlation with already selected
                is_diverse = True
                for sel in selected:
                    idx1 = symbol_to_idx.get(record.symbol)
                    idx2 = symbol_to_idx.get(sel.symbol)
                    if idx1 is not None and idx2 is not None:
                        if abs(corr_matrix[idx1, idx2]) > correlation_threshold:
                            is_diverse = False
                            break

                if is_diverse:
                    selected.append(record)

            if not selected:
                equity_values.append((day.date, cash))
                continue

            # Equal weight allocation
            per_position = cash / len(selected)
            day_pnl = 0.0

            for record in selected:
                trade_pnl = per_position * record.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if record.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((day.date, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        total_return = (cash - self.initial_cash) / self.initial_cash

        stats = {
            "strategy": f"correlation_aware_{n}",
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "final_equity": cash,
        }

        return equity_curve, stats

    def backtest_kelly_sizing(
        self,
        n: int = 4,
        lookback_days: int = 10,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Use Kelly criterion for position sizing based on recent win rate.

        Kelly fraction = (p * b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        # Track recent performance per symbol
        symbol_history: Dict[str, List[float]] = {}

        for day_idx, day in enumerate(self.snapshots):
            top_n = day.get_top_n_by_predicted_return(n)

            if not top_n:
                equity_values.append((day.date, cash))
                continue

            # Calculate Kelly fraction for each symbol
            allocations = []
            for record in top_n:
                history = symbol_history.get(record.symbol, [])

                if len(history) < 3:
                    # Not enough history, use equal weight
                    kelly_fraction = 1.0 / n
                else:
                    recent = history[-lookback_days:]
                    wins = [r for r in recent if r > 0]
                    losses = [r for r in recent if r <= 0]

                    p = len(wins) / len(recent) if recent else 0.5
                    avg_win = np.mean(wins) if wins else 0.01
                    avg_loss = abs(np.mean(losses)) if losses else 0.01
                    b = avg_win / avg_loss if avg_loss > 0 else 1

                    kelly_fraction = (p * b - (1 - p)) / b if b > 0 else 0
                    kelly_fraction = max(0, min(0.5, kelly_fraction))  # Cap at 50%

                allocations.append((record, kelly_fraction))

            # Normalize allocations
            total_kelly = sum(k for _, k in allocations)
            if total_kelly <= 0:
                total_kelly = 1.0

            day_pnl = 0.0
            for record, kelly in allocations:
                position_size = cash * (kelly / total_kelly)
                trade_pnl = position_size * record.actual_return
                trade_pnl -= position_size * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if record.actual_return > 0:
                    winning_trades += 1

                # Update history
                if record.symbol not in symbol_history:
                    symbol_history[record.symbol] = []
                symbol_history[record.symbol].append(record.actual_return)

            cash += day_pnl
            equity_values.append((day.date, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        total_return = (cash - self.initial_cash) / self.initial_cash

        stats = {
            "strategy": f"kelly_sizing_{n}",
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "final_equity": cash,
        }

        return equity_curve, stats

    def backtest_momentum_filter(
        self,
        n: int = 4,
        momentum_days: int = 5,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Only trade symbols with positive recent momentum.

        Combines forecast signal with momentum confirmation.
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        # Track recent returns per symbol
        symbol_returns: Dict[str, List[float]] = {}

        for day in self.snapshots:
            # Filter by momentum
            momentum_filtered = []
            for record in day.forecasts:
                recent = symbol_returns.get(record.symbol, [])[-momentum_days:]
                if len(recent) >= momentum_days:
                    momentum = sum(recent)
                    if momentum > 0:  # Only positive momentum
                        momentum_filtered.append((record, momentum))
                else:
                    # Not enough history, include with neutral momentum
                    momentum_filtered.append((record, 0))

            # Sort by predicted return among momentum-positive
            sorted_forecasts = sorted(
                momentum_filtered,
                key=lambda x: x[0].predicted_return * (1 + x[1]),  # Weight by momentum
                reverse=True
            )[:n]

            selected = [r for r, _ in sorted_forecasts]

            if not selected:
                equity_values.append((day.date, cash))
                # Update histories anyway
                for record in day.forecasts:
                    if record.symbol not in symbol_returns:
                        symbol_returns[record.symbol] = []
                    symbol_returns[record.symbol].append(record.actual_return)
                continue

            per_position = cash / len(selected)
            day_pnl = 0.0

            for record in selected:
                trade_pnl = per_position * record.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if record.actual_return > 0:
                    winning_trades += 1

            # Update all symbol histories
            for record in day.forecasts:
                if record.symbol not in symbol_returns:
                    symbol_returns[record.symbol] = []
                symbol_returns[record.symbol].append(record.actual_return)

            cash += day_pnl
            equity_values.append((day.date, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        total_return = (cash - self.initial_cash) / self.initial_cash

        stats = {
            "strategy": f"momentum_filter_{n}",
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "final_equity": cash,
        }

        return equity_curve, stats

    def backtest_hindsight_optimal(
        self,
        n: int = 4,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Oracle strategy - pick top N by ACTUAL return (upper bound)."""
        cash = self.initial_cash
        equity_values = []
        total_trades = 0

        for day in self.snapshots:
            top_n = day.get_top_n_by_actual_return(n)

            if not top_n:
                equity_values.append((day.date, cash))
                continue

            per_position = cash / len(top_n)
            day_pnl = 0.0

            for record in top_n:
                trade_pnl = per_position * record.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1

            cash += day_pnl
            equity_values.append((day.date, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        total_return = (cash - self.initial_cash) / self.initial_cash

        stats = {
            "strategy": f"hindsight_optimal_{n}",
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": 1.0,  # Always winning by definition
            "final_equity": cash,
        }

        return equity_curve, stats


def plot_results(
    equity_curves: Dict[str, pd.Series],
    stats: Dict[str, Dict],
    output_path: Optional[Path] = None,
):
    """Plot comparison of strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Equity curves
    ax1 = axes[0, 0]
    for name, curve in equity_curves.items():
        ax1.plot(curve.index, curve.values, label=name)
    ax1.set_title("Equity Curves")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Returns bar chart
    ax2 = axes[0, 1]
    names = list(stats.keys())
    returns = [stats[n]["total_return"] * 100 for n in names]
    colors = ["green" if r > 0 else "red" for r in returns]
    bars = ax2.bar(range(len(names)), returns, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Total Returns (%)")
    ax2.set_ylabel("Return %")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    # Win rate bar chart
    ax3 = axes[1, 0]
    win_rates = [stats[n]["win_rate"] * 100 for n in names]
    ax3.bar(range(len(names)), win_rates, color="steelblue")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax3.set_title("Win Rates (%)")
    ax3.set_ylabel("Win Rate %")
    ax3.axhline(y=50, color="red", linestyle="--", linewidth=1, label="50%")
    ax3.grid(True, alpha=0.3, axis="y")

    # Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = []
    for name in names:
        s = stats[name]
        table_data.append([
            name[:20],
            f"{s['total_return']*100:.2f}%",
            f"{s['win_rate']*100:.1f}%",
            f"${s['final_equity']:,.0f}",
            s['total_trades'],
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=["Strategy", "Return", "Win%", "Final $", "Trades"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", output_path)

    plt.show()


def run_analysis(
    n_days: int = 30,
    top_n: int = 4,
    output_dir: str = "reports/forecast_analysis",
):
    """Run complete forecast analysis and strategy comparison.

    Args:
        n_days: Number of days to analyze
        top_n: Number of top symbols for strategies
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup
    end_date = date(2025, 12, 31)
    start_date = end_date - timedelta(days=n_days + 10)  # Buffer for weekends

    data_config = DataConfigLong(
        start_date=start_date,
        end_date=end_date,
    )
    forecast_config = ForecastConfigLong()

    # Load data
    logger.info("Loading data...")
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    logger.info("Loading Chronos2 forecaster...")
    forecaster = Chronos2Forecaster(data_loader, forecast_config)

    try:
        # Collect forecasts
        analyzer = ForecastAnalyzer(data_loader, forecaster)
        logger.info("Collecting forecasts for %d days...", n_days)
        snapshots = analyzer.collect_forecasts(start_date, end_date)

        # Compute accuracy stats
        logger.info("Computing forecast accuracy stats...")
        accuracy_stats = analyzer.compute_forecast_accuracy_stats()

        print("\n" + "=" * 60)
        print("FORECAST ACCURACY STATISTICS")
        print("=" * 60)
        for key, value in accuracy_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Per-symbol stats
        symbol_stats = analyzer.compute_per_symbol_stats()
        print("\n" + "=" * 60)
        print("PER-SYMBOL FORECAST QUALITY (sorted by correlation)")
        print("=" * 60)
        print(symbol_stats.to_string(index=False))

        # Save stats
        with open(output_path / "forecast_stats.json", "w") as f:
            json.dump(accuracy_stats, f, indent=2)
        symbol_stats.to_csv(output_path / "symbol_stats.csv", index=False)

        # Run strategy backtests
        logger.info("Running strategy backtests...")
        backtester = MetaAlgorithmBacktester(snapshots)

        equity_curves = {}
        all_stats = {}

        # Test different strategies
        strategies = [
            ("top_n", lambda: backtester.backtest_top_n(n=top_n)),
            ("corr_aware", lambda: backtester.backtest_correlation_aware(n=top_n)),
            ("kelly", lambda: backtester.backtest_kelly_sizing(n=top_n)),
            ("momentum", lambda: backtester.backtest_momentum_filter(n=top_n)),
            ("hindsight", lambda: backtester.backtest_hindsight_optimal(n=top_n)),
        ]

        for name, strategy_fn in strategies:
            curve, stats = strategy_fn()
            equity_curves[name] = curve
            all_stats[name] = stats
            logger.info("  %s: Return=%.2f%%, WinRate=%.1f%%",
                       name, stats["total_return"]*100, stats["win_rate"]*100)

        # Plot results
        plot_results(equity_curves, all_stats, output_path / "strategy_comparison.png")

        # Save strategy stats
        with open(output_path / "strategy_stats.json", "w") as f:
            json.dump(all_stats, f, indent=2)

        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)
        for name, stats in sorted(all_stats.items(), key=lambda x: x[1]["total_return"], reverse=True):
            print(f"  {name:15s}: Return={stats['total_return']*100:>7.2f}%, "
                  f"WinRate={stats['win_rate']*100:>5.1f}%, "
                  f"Final=${stats['final_equity']:>10,.0f}")

        logger.info("Results saved to %s", output_path)

    finally:
        forecaster.unload()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forecast analysis and strategy comparison")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--top-n", type=int, default=4, help="Number of top symbols")
    parser.add_argument("--output-dir", type=str, default="reports/forecast_analysis")

    args = parser.parse_args()

    run_analysis(n_days=args.days, top_n=args.top_n, output_dir=args.output_dir)
