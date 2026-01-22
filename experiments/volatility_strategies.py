"""Volatility and Quantile-Based Trading Strategies.

Explore different ways to use Chronos2 predictions:
1. Quantile spread as volatility signal
2. Mean reversion on negatively correlated symbols
3. Volatility breakout trading
4. Regime detection
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimlong.config import DataConfigLong, ForecastConfigLong
from marketsimlong.data import DailyDataLoader, is_crypto_symbol
from marketsimlong.forecaster import Chronos2Forecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantileForecast:
    """Forecast with quantile information."""
    symbol: str
    date: date

    # Quantile predictions (from Chronos2)
    q10: float  # 10th percentile (low estimate)
    q50: float  # 50th percentile (median)
    q90: float  # 90th percentile (high estimate)

    # Derived signals
    volatility: float  # (q90 - q10) / q50 - predicted range
    upside: float  # (q90 - q50) / q50
    downside: float  # (q50 - q10) / q50
    skew: float  # upside - downside (positive = bullish)

    # Actual outcomes
    actual_open: float = 0.0
    actual_high: float = 0.0
    actual_low: float = 0.0
    actual_close: float = 0.0
    actual_return: float = 0.0
    actual_volatility: float = 0.0  # (high - low) / open


class QuantileForecaster:
    """Generate quantile-based forecasts using Chronos2."""

    def __init__(self, data_loader: DailyDataLoader, model_id: str = "amazon/chronos-t5-small"):
        self.data_loader = data_loader
        self.model_id = model_id
        self.pipeline = None

    def load_model(self):
        """Load Chronos2 pipeline."""
        if self.pipeline is not None:
            return

        import torch
        from chronos import ChronosPipeline

        logger.info("Loading Chronos2 model: %s", self.model_id)
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32,
        )

    def unload(self):
        """Unload model."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forecast_symbol(
        self,
        symbol: str,
        forecast_date: date,
        context_length: int = 512,
        prediction_length: int = 1,
    ) -> Optional[QuantileForecast]:
        """Generate quantile forecast for a symbol.

        Args:
            symbol: Trading symbol
            forecast_date: Date to forecast
            context_length: Historical context length
            prediction_length: Days ahead to forecast

        Returns:
            QuantileForecast or None
        """
        self.load_model()

        import torch

        # Get historical close prices
        history = self.data_loader.get_price_history(
            symbol,
            forecast_date - timedelta(days=context_length + 50),
            forecast_date - timedelta(days=1),
        )

        if history is None or len(history) < 100:
            return None

        closes = history["close"].values[-context_length:]
        context = torch.tensor(closes, dtype=torch.float32).unsqueeze(0)

        # Generate quantile forecasts
        quantiles, mean = self.pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        # Extract predictions for day 1
        q10 = float(quantiles[0, 0, 0])  # batch 0, quantile 0, day 0
        q50 = float(quantiles[0, 1, 0])  # batch 0, quantile 1, day 0
        q90 = float(quantiles[0, 2, 0])  # batch 0, quantile 2, day 0

        # Derived signals
        volatility = (q90 - q10) / q50 if q50 > 0 else 0
        upside = (q90 - q50) / q50 if q50 > 0 else 0
        downside = (q50 - q10) / q50 if q50 > 0 else 0
        skew = upside - downside

        # Get actual outcomes
        actual_data = self.data_loader.get_price_on_date(symbol, forecast_date)
        if actual_data:
            actual_open = actual_data.get("open", actual_data["close"])
            actual_high = actual_data.get("high", actual_data["close"])
            actual_low = actual_data.get("low", actual_data["close"])
            actual_close = actual_data["close"]
            actual_return = (actual_close - actual_open) / actual_open if actual_open > 0 else 0
            actual_volatility = (actual_high - actual_low) / actual_open if actual_open > 0 else 0
        else:
            actual_open = actual_high = actual_low = actual_close = 0
            actual_return = actual_volatility = 0

        return QuantileForecast(
            symbol=symbol,
            date=forecast_date,
            q10=q10,
            q50=q50,
            q90=q90,
            volatility=volatility,
            upside=upside,
            downside=downside,
            skew=skew,
            actual_open=actual_open,
            actual_high=actual_high,
            actual_low=actual_low,
            actual_close=actual_close,
            actual_return=actual_return,
            actual_volatility=actual_volatility,
        )


class VolatilityStrategies:
    """Test various volatility-based trading strategies."""

    def __init__(self, forecasts: List[QuantileForecast], initial_cash: float = 100_000):
        self.forecasts = forecasts
        self.initial_cash = initial_cash

        # Group forecasts by date
        self.by_date: Dict[date, List[QuantileForecast]] = {}
        for f in forecasts:
            if f.date not in self.by_date:
                self.by_date[f.date] = []
            self.by_date[f.date].append(f)

        self.dates = sorted(self.by_date.keys())

    def strategy_high_volatility(
        self,
        n: int = 4,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Buy symbols with highest predicted volatility.

        Thesis: High volatility = more opportunity for profit
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        for d in self.dates:
            day_forecasts = self.by_date[d]

            # Sort by predicted volatility (highest first)
            sorted_f = sorted(day_forecasts, key=lambda x: x.volatility, reverse=True)[:n]

            if not sorted_f:
                equity_values.append((d, cash))
                continue

            per_position = cash / len(sorted_f)
            day_pnl = 0.0

            for f in sorted_f:
                trade_pnl = per_position * f.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if f.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((d, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        return equity_curve, {
            "strategy": "high_volatility",
            "total_return": (cash - self.initial_cash) / self.initial_cash,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": total_trades,
            "final_equity": cash,
        }

    def strategy_low_volatility(
        self,
        n: int = 4,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Buy symbols with lowest predicted volatility.

        Thesis: Low volatility = lower risk, steadier gains
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        for d in self.dates:
            day_forecasts = self.by_date[d]

            # Sort by predicted volatility (lowest first)
            sorted_f = sorted(day_forecasts, key=lambda x: x.volatility)[:n]

            if not sorted_f:
                equity_values.append((d, cash))
                continue

            per_position = cash / len(sorted_f)
            day_pnl = 0.0

            for f in sorted_f:
                trade_pnl = per_position * f.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if f.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((d, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        return equity_curve, {
            "strategy": "low_volatility",
            "total_return": (cash - self.initial_cash) / self.initial_cash,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": total_trades,
            "final_equity": cash,
        }

    def strategy_bullish_skew(
        self,
        n: int = 4,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Buy symbols with most bullish skew (upside > downside).

        Thesis: Asymmetric upside potential
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        for d in self.dates:
            day_forecasts = self.by_date[d]

            # Sort by skew (most bullish first)
            sorted_f = sorted(day_forecasts, key=lambda x: x.skew, reverse=True)[:n]

            if not sorted_f:
                equity_values.append((d, cash))
                continue

            per_position = cash / len(sorted_f)
            day_pnl = 0.0

            for f in sorted_f:
                trade_pnl = per_position * f.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if f.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((d, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        return equity_curve, {
            "strategy": "bullish_skew",
            "total_return": (cash - self.initial_cash) / self.initial_cash,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": total_trades,
            "final_equity": cash,
        }

    def strategy_mean_reversion(
        self,
        n: int = 4,
        lookback: int = 5,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Mean reversion: buy losers, short winners.

        Thesis: Overreactions revert to mean
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        # Track recent returns per symbol
        symbol_returns: Dict[str, List[float]] = {}

        for d in self.dates:
            day_forecasts = self.by_date[d]

            # Calculate recent performance
            scored = []
            for f in day_forecasts:
                recent = symbol_returns.get(f.symbol, [])[-lookback:]
                if len(recent) >= lookback:
                    recent_return = sum(recent)
                    # Negative = oversold, buy
                    # Positive = overbought, avoid (or short)
                    scored.append((f, -recent_return))  # Invert so oversold ranks higher
                else:
                    scored.append((f, 0))

            # Sort by mean reversion score (most oversold first)
            sorted_f = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
            selected = [f for f, _ in sorted_f]

            if not selected:
                equity_values.append((d, cash))
                for f in day_forecasts:
                    if f.symbol not in symbol_returns:
                        symbol_returns[f.symbol] = []
                    symbol_returns[f.symbol].append(f.actual_return)
                continue

            per_position = cash / len(selected)
            day_pnl = 0.0

            for f in selected:
                trade_pnl = per_position * f.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if f.actual_return > 0:
                    winning_trades += 1

            # Update histories
            for f in day_forecasts:
                if f.symbol not in symbol_returns:
                    symbol_returns[f.symbol] = []
                symbol_returns[f.symbol].append(f.actual_return)

            cash += day_pnl
            equity_values.append((d, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        return equity_curve, {
            "strategy": "mean_reversion",
            "total_return": (cash - self.initial_cash) / self.initial_cash,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": total_trades,
            "final_equity": cash,
        }

    def strategy_volatility_breakout(
        self,
        n: int = 4,
        vol_threshold: float = 0.05,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Only trade when predicted volatility exceeds threshold.

        Thesis: Trade during high-vol periods, sit out low-vol
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        for d in self.dates:
            day_forecasts = self.by_date[d]

            # Filter by volatility threshold, sort by skew
            high_vol = [f for f in day_forecasts if f.volatility > vol_threshold]
            sorted_f = sorted(high_vol, key=lambda x: x.skew, reverse=True)[:n]

            if not sorted_f:
                equity_values.append((d, cash))
                continue

            per_position = cash / len(sorted_f)
            day_pnl = 0.0

            for f in sorted_f:
                trade_pnl = per_position * f.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if f.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((d, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        return equity_curve, {
            "strategy": "volatility_breakout",
            "total_return": (cash - self.initial_cash) / self.initial_cash,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": total_trades,
            "final_equity": cash,
        }

    def strategy_contrarian_skew(
        self,
        n: int = 4,
        fee_pct: float = 0.001,
    ) -> Tuple[pd.Series, Dict]:
        """Contrarian: buy bearish skew (model predicts down, we bet up).

        Thesis: If model is negatively correlated, invert the signal
        """
        cash = self.initial_cash
        equity_values = []
        total_trades = 0
        winning_trades = 0

        for d in self.dates:
            day_forecasts = self.by_date[d]

            # Sort by skew (most bearish first - we're being contrarian)
            sorted_f = sorted(day_forecasts, key=lambda x: x.skew)[:n]

            if not sorted_f:
                equity_values.append((d, cash))
                continue

            per_position = cash / len(sorted_f)
            day_pnl = 0.0

            for f in sorted_f:
                trade_pnl = per_position * f.actual_return
                trade_pnl -= per_position * fee_pct * 2
                day_pnl += trade_pnl
                total_trades += 1
                if f.actual_return > 0:
                    winning_trades += 1

            cash += day_pnl
            equity_values.append((d, cash))

        dates, values = zip(*equity_values) if equity_values else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))

        return equity_curve, {
            "strategy": "contrarian_skew",
            "total_return": (cash - self.initial_cash) / self.initial_cash,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_trades": total_trades,
            "final_equity": cash,
        }


def run_quick_test_with_existing_forecaster():
    """Use existing Chronos2Forecaster but extract more signal types."""
    from experiments.forecast_analysis import ForecastAnalyzer, MetaAlgorithmBacktester

    # Use 60 days for testing
    end_date = date(2025, 12, 31)
    start_date = end_date - timedelta(days=70)

    data_config = DataConfigLong(start_date=start_date, end_date=end_date)
    forecast_config = ForecastConfigLong()

    logger.info("Loading data...")
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    logger.info("Loading forecaster...")
    forecaster = Chronos2Forecaster(data_loader, forecast_config)

    try:
        # Collect forecasts
        logger.info("Collecting forecasts...")
        analyzer = ForecastAnalyzer(data_loader, forecaster)
        snapshots = analyzer.collect_forecasts(start_date, end_date)

        # Convert to QuantileForecast format (using predicted high/low as proxy)
        quantile_forecasts = []
        for day in snapshots:
            for rec in day.forecasts:
                qf = QuantileForecast(
                    symbol=rec.symbol,
                    date=rec.date,
                    q10=rec.predicted_low,
                    q50=rec.predicted_close,
                    q90=rec.predicted_high,
                    volatility=(rec.predicted_high - rec.predicted_low) / rec.predicted_close if rec.predicted_close > 0 else 0,
                    upside=(rec.predicted_high - rec.predicted_close) / rec.predicted_close if rec.predicted_close > 0 else 0,
                    downside=(rec.predicted_close - rec.predicted_low) / rec.predicted_close if rec.predicted_close > 0 else 0,
                    skew=0,  # Will calculate
                    actual_open=rec.actual_open,
                    actual_high=rec.actual_high,
                    actual_low=rec.actual_low,
                    actual_close=rec.actual_close,
                    actual_return=rec.actual_return,
                    actual_volatility=(rec.actual_high - rec.actual_low) / rec.actual_open if rec.actual_open > 0 else 0,
                )
                qf.skew = qf.upside - qf.downside
                quantile_forecasts.append(qf)

        logger.info("Collected %d quantile forecasts", len(quantile_forecasts))

        # Analyze volatility prediction quality
        pred_vols = [f.volatility for f in quantile_forecasts]
        actual_vols = [f.actual_volatility for f in quantile_forecasts]
        vol_corr = np.corrcoef(pred_vols, actual_vols)[0, 1]

        # Analyze skew prediction quality
        skews = [f.skew for f in quantile_forecasts]
        returns = [f.actual_return for f in quantile_forecasts]
        skew_corr = np.corrcoef(skews, returns)[0, 1]

        print("\n" + "=" * 60)
        print("SIGNAL QUALITY ANALYSIS")
        print("=" * 60)
        print(f"Volatility prediction correlation: {vol_corr:.4f}")
        print(f"Skew -> Return correlation: {skew_corr:.4f}")
        print(f"Mean predicted volatility: {np.mean(pred_vols)*100:.2f}%")
        print(f"Mean actual volatility: {np.mean(actual_vols)*100:.2f}%")

        # Run strategy tests
        strategies = VolatilityStrategies(quantile_forecasts)

        results = {}

        # Test each strategy
        for name, method in [
            ("high_vol", strategies.strategy_high_volatility),
            ("low_vol", strategies.strategy_low_volatility),
            ("bullish_skew", strategies.strategy_bullish_skew),
            ("contrarian_skew", strategies.strategy_contrarian_skew),
            ("mean_reversion", strategies.strategy_mean_reversion),
            ("vol_breakout", strategies.strategy_volatility_breakout),
        ]:
            curve, stats = method(n=4)
            results[name] = stats
            logger.info("  %s: Return=%.2f%%, WinRate=%.1f%%",
                       name, stats["total_return"]*100, stats["win_rate"]*100)

        # Also run baseline
        backtester = MetaAlgorithmBacktester(snapshots)
        _, baseline = backtester.backtest_top_n(n=4)
        results["baseline_top4"] = baseline

        _, hindsight = backtester.backtest_hindsight_optimal(n=4)
        results["hindsight"] = hindsight

        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)
        print(f"{'Strategy':<20} {'Return':>10} {'WinRate':>10} {'Trades':>8}")
        print("-" * 50)
        for name, stats in sorted(results.items(), key=lambda x: x[1]["total_return"], reverse=True):
            print(f"{name:<20} {stats['total_return']*100:>9.2f}% {stats['win_rate']*100:>9.1f}% {stats['total_trades']:>8}")

        return results

    finally:
        forecaster.unload()


if __name__ == "__main__":
    run_quick_test_with_existing_forecaster()
