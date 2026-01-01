"""V5 Backtesting for hourly crypto trading.

Walk-forward validation over 10 days of hourly data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from neuralhourlytradingv5.config import SimulationConfigV5
from neuralhourlytradingv5.data import FeatureNormalizer, HOURLY_FEATURES_V5
from neuralhourlytradingv5.model import HourlyCryptoPolicyV5


@dataclass
class TradeRecord:
    """Record of a single trade."""

    entry_timestamp: pd.Timestamp
    entry_price: float
    exit_timestamp: Optional[pd.Timestamp]
    exit_price: Optional[float]
    position_size: float
    position_length_target: int  # Hours planned to hold
    actual_hold_hours: int
    exit_type: str  # "tp" or "forced"
    pnl: float
    return_pct: float
    fees: float


@dataclass
class BacktestResult:
    """Result from backtesting."""

    equity_curve: pd.Series
    trades: List[TradeRecord]
    metrics: Dict[str, float] = field(default_factory=dict)


class HourlyMarketSimulatorV5:
    """Walk-forward market simulator for V5 hourly trading."""

    def __init__(self, config: SimulationConfigV5) -> None:
        self.config = config
        self.maker_fee = config.maker_fee
        self.forced_exit_slippage = config.forced_exit_slippage

    def run(
        self,
        model: HourlyCryptoPolicyV5,
        bars: pd.DataFrame,
        features: np.ndarray,
        normalizer: FeatureNormalizer,
        sequence_length: int = 168,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            model: Trained model
            bars: DataFrame with timestamp, open, high, low, close, volume
            features: Normalized feature matrix
            normalizer: Feature normalizer (for reference)
            sequence_length: Sequence length for model input

        Returns:
            BacktestResult with equity curve and trades
        """
        model.eval()
        device = next(model.parameters()).device

        # Ensure enough data
        if len(bars) < sequence_length + 24:
            raise ValueError(
                f"Need at least {sequence_length + 24} bars, got {len(bars)}"
            )

        cash = self.config.initial_cash
        inventory = 0.0
        equity_values: List[float] = []
        timestamps: List[pd.Timestamp] = []
        trades: List[TradeRecord] = []

        # Current position tracking
        current_position: Optional[Dict] = None

        # Walk forward hour by hour
        for i in range(sequence_length, len(bars) - 24):
            current_bar = bars.iloc[i]
            current_ts = pd.to_datetime(current_bar["timestamp"])
            current_close = float(current_bar["close"])
            current_high = float(current_bar["high"])
            current_low = float(current_bar["low"])

            # Get model input (last sequence_length hours)
            feature_seq = features[i - sequence_length : i].copy()
            feature_tensor = (
                torch.from_numpy(feature_seq).unsqueeze(0).float().contiguous().to(device)
            )
            ref_close_tensor = torch.tensor([current_close], device=device)

            # Get model predictions
            with torch.no_grad():
                outputs = model(feature_tensor)
                actions = model.get_hard_actions(outputs, ref_close_tensor)

            position_length = int(actions["position_length"].item())
            position_size = float(actions["position_size"].item())
            buy_price = float(actions["buy_price"].item())
            sell_price = float(actions["sell_price"].item())

            # Process existing position first
            if current_position is not None:
                hours_held = current_position["hours_held"]
                target_hours = current_position["target_hours"]

                # Check take-profit
                if current_high >= current_position["sell_price"]:
                    # TP hit - exit at sell_price
                    exit_price = current_position["sell_price"]
                    exit_value = (
                        inventory * exit_price * (1 - self.maker_fee)
                    )
                    pnl = exit_value - current_position["entry_value"]

                    trades.append(
                        TradeRecord(
                            entry_timestamp=current_position["entry_ts"],
                            entry_price=current_position["entry_price"],
                            exit_timestamp=current_ts,
                            exit_price=exit_price,
                            position_size=current_position["position_size"],
                            position_length_target=target_hours,
                            actual_hold_hours=hours_held,
                            exit_type="tp",
                            pnl=pnl,
                            return_pct=pnl / current_position["entry_value"],
                            fees=self.maker_fee * 2 * exit_value,
                        )
                    )

                    cash += exit_value
                    inventory = 0.0
                    current_position = None

                elif hours_held >= target_hours:
                    # Forced exit at close with slippage
                    exit_price = current_close * (1 - self.forced_exit_slippage)
                    exit_value = inventory * exit_price * (1 - self.maker_fee)
                    pnl = exit_value - current_position["entry_value"]

                    trades.append(
                        TradeRecord(
                            entry_timestamp=current_position["entry_ts"],
                            entry_price=current_position["entry_price"],
                            exit_timestamp=current_ts,
                            exit_price=exit_price,
                            position_size=current_position["position_size"],
                            position_length_target=target_hours,
                            actual_hold_hours=hours_held,
                            exit_type="forced",
                            pnl=pnl,
                            return_pct=pnl / current_position["entry_value"],
                            fees=self.maker_fee * 2 * exit_value,
                        )
                    )

                    cash += exit_value
                    inventory = 0.0
                    current_position = None

                else:
                    # Still holding, increment hours
                    current_position["hours_held"] += 1

            # Check for new entry (only if no current position)
            if current_position is None and position_length > 0:
                # Check if entry fills
                if current_low <= buy_price:
                    # Entry fills at buy_price
                    # Compute entry_value such that entry_cost = cash * position_size
                    # This accounts for fees in the position sizing
                    max_spend = cash * position_size
                    entry_value = max_spend / (1 + self.maker_fee)
                    entry_cost = max_spend  # entry_value * (1 + fee) = max_spend

                    if entry_cost <= cash:
                        inventory = entry_value / buy_price
                        cash -= entry_cost

                        current_position = {
                            "entry_ts": current_ts,
                            "entry_price": buy_price,
                            "entry_value": entry_value,
                            "sell_price": sell_price,
                            "position_size": position_size,
                            "target_hours": position_length,
                            "hours_held": 0,
                        }

            # Record equity
            portfolio_value = cash + inventory * current_close
            equity_values.append(portfolio_value)
            timestamps.append(current_ts)

        # Close any remaining position at end
        if current_position is not None and inventory > 0:
            final_close = float(bars.iloc[-1]["close"])
            exit_value = inventory * final_close * (1 - self.maker_fee)
            pnl = exit_value - current_position["entry_value"]

            trades.append(
                TradeRecord(
                    entry_timestamp=current_position["entry_ts"],
                    entry_price=current_position["entry_price"],
                    exit_timestamp=pd.to_datetime(bars.iloc[-1]["timestamp"]),
                    exit_price=final_close,
                    position_size=current_position["position_size"],
                    position_length_target=current_position["target_hours"],
                    actual_hold_hours=current_position["hours_held"],
                    exit_type="forced",
                    pnl=pnl,
                    return_pct=pnl / current_position["entry_value"],
                    fees=self.maker_fee * 2 * exit_value,
                )
            )

            cash += exit_value
            equity_values[-1] = cash

        equity_curve = pd.Series(equity_values, index=timestamps)
        metrics = self._compute_metrics(equity_curve, trades)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
        )

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[TradeRecord],
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        if equity_curve.empty:
            return {"total_return": 0.0, "sortino": 0.0}

        # Returns
        values = equity_curve.values
        returns = np.diff(values) / np.clip(values[:-1], 1e-8, None)
        mean_ret = returns.mean() if len(returns) else 0.0

        # Downside deviation
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) else 0.0

        # Annualized Sortino
        sortino = (
            mean_ret / downside_std * np.sqrt(24 * 365) if downside_std > 0 else 0.0
        )

        # Total return
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

        # Trade statistics
        if trades:
            tp_trades = [t for t in trades if t.exit_type == "tp"]
            forced_trades = [t for t in trades if t.exit_type == "forced"]
            winning_trades = [t for t in trades if t.pnl > 0]

            tp_rate = len(tp_trades) / len(trades)
            win_rate = len(winning_trades) / len(trades)
            avg_hold_hours = np.mean([t.actual_hold_hours for t in trades])
            avg_return = np.mean([t.return_pct for t in trades])
            avg_pnl = np.mean([t.pnl for t in trades])
        else:
            tp_rate = win_rate = avg_hold_hours = avg_return = avg_pnl = 0.0

        return {
            "total_return": float(total_return),
            "sortino": float(sortino),
            "mean_hourly_return": float(mean_ret),
            "num_trades": len(trades),
            "tp_rate": float(tp_rate),
            "win_rate": float(win_rate),
            "avg_hold_hours": float(avg_hold_hours),
            "avg_trade_return": float(avg_return),
            "avg_trade_pnl": float(avg_pnl),
            "num_tp_exits": len([t for t in trades if t.exit_type == "tp"]),
            "num_forced_exits": len([t for t in trades if t.exit_type == "forced"]),
        }


def run_10day_validation(
    model: HourlyCryptoPolicyV5,
    bars: pd.DataFrame,
    features: np.ndarray,
    normalizer: FeatureNormalizer,
    config: Optional[SimulationConfigV5] = None,
    sequence_length: int = 168,
) -> BacktestResult:
    """
    Run 10-day walk-forward validation.

    Args:
        model: Trained model
        bars: DataFrame with last 10 days + sequence_length of hourly data
        features: Corresponding feature matrix
        normalizer: Feature normalizer
        config: Simulation config (uses defaults if None)
        sequence_length: Model input sequence length

    Returns:
        BacktestResult
    """
    if config is None:
        config = SimulationConfigV5()

    simulator = HourlyMarketSimulatorV5(config)

    return simulator.run(
        model=model,
        bars=bars,
        features=features,
        normalizer=normalizer,
        sequence_length=sequence_length,
    )


def print_backtest_summary(result: BacktestResult) -> None:
    """Print human-readable backtest summary."""
    m = result.metrics

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    print(f"\nPerformance:")
    print(f"  Total Return:    {m.get('total_return', 0) * 100:+.2f}%")
    print(f"  Sortino Ratio:   {m.get('sortino', 0):.2f}")
    print(f"  Mean Hourly:     {m.get('mean_hourly_return', 0) * 100:.4f}%")

    print(f"\nTrades:")
    print(f"  Total Trades:    {m.get('num_trades', 0)}")
    print(f"  Win Rate:        {m.get('win_rate', 0) * 100:.1f}%")
    print(f"  TP Rate:         {m.get('tp_rate', 0) * 100:.1f}%")
    print(f"  TP Exits:        {m.get('num_tp_exits', 0)}")
    print(f"  Forced Exits:    {m.get('num_forced_exits', 0)}")

    print(f"\nHolding:")
    print(f"  Avg Hold Hours:  {m.get('avg_hold_hours', 0):.1f}")
    print(f"  Avg Trade Return:{m.get('avg_trade_return', 0) * 100:+.2f}%")

    print("=" * 50 + "\n")
