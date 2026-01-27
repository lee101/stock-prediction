"""Scipy optimizer for Chronos2 trading thresholds."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)


@dataclass
class OptimizedParams:
    """Optimized trading parameters."""

    buy_threshold: float  # Predicted return threshold to buy
    sell_threshold: float  # Predicted return threshold to sell (negative = loss)
    upside_ratio_min: float  # Minimum upside/downside ratio to buy
    hold_bars: int  # Bars to hold before allowing sell
    prediction_length: int  # Chronos prediction horizon

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizedParams":
        return cls(**d)


def simulate_strategy(
    prices: np.ndarray,
    forecasts: list,  # List of (predicted_return, upside_ratio) tuples
    params: OptimizedParams,
    cost_bps: float = 130.0,
) -> Tuple[float, dict]:
    """Simulate trading strategy and return final return.

    Args:
        prices: Price series
        forecasts: Pre-computed forecasts aligned with prices
        params: Trading parameters
        cost_bps: Round-trip cost in basis points

    Returns:
        (total_return, stats_dict)
    """
    n = len(prices)
    cost_mult = 1 - (cost_bps / 10000)

    holding = False
    entry_price = 0.0
    entry_idx = 0
    equity = 1.0
    trades = []

    for i in range(n):
        if forecasts[i] is None:
            continue

        pred_return, upside_ratio = forecasts[i]
        current_price = prices[i]

        if not holding:
            # Check buy conditions
            if (pred_return >= params.buy_threshold and
                upside_ratio >= params.upside_ratio_min):
                holding = True
                entry_price = current_price
                entry_idx = i

        else:
            # Check sell conditions
            bars_held = i - entry_idx
            current_return = (current_price - entry_price) / entry_price

            should_sell = False
            # Sell if predicted return is negative (expecting drop)
            if pred_return <= params.sell_threshold:
                should_sell = True
            # Sell if we've held long enough and have profit
            elif bars_held >= params.hold_bars and current_return > 0:
                should_sell = True

            if should_sell:
                # Execute sell
                trade_return = (current_price / entry_price) * cost_mult - 1
                equity *= (1 + trade_return)
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "return": trade_return,
                    "bars_held": bars_held,
                })
                holding = False

    # Close any open position at end
    if holding:
        current_price = prices[-1]
        trade_return = (current_price / entry_price) * cost_mult - 1
        equity *= (1 + trade_return)
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "return": trade_return,
            "bars_held": n - 1 - entry_idx,
        })

    total_return = equity - 1
    win_trades = [t for t in trades if t["return"] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0

    return total_return, {
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_return": np.mean([t["return"] for t in trades]) if trades else 0,
        "avg_bars_held": np.mean([t["bars_held"] for t in trades]) if trades else 0,
    }


def objective_function(
    x: np.ndarray,
    prices: np.ndarray,
    forecasts: list,
    prediction_length: int,
) -> float:
    """Objective function to minimize (negative return)."""
    params = OptimizedParams(
        buy_threshold=x[0],
        sell_threshold=x[1],
        upside_ratio_min=x[2],
        hold_bars=int(round(x[3])),
        prediction_length=prediction_length,
    )

    total_return, _ = simulate_strategy(prices, forecasts, params)

    # Minimize negative return (maximize return)
    return -total_return


def optimize_thresholds(
    prices: np.ndarray,
    forecasts: list,
    prediction_length: int = 6,
    method: str = "differential_evolution",
) -> Tuple[OptimizedParams, dict]:
    """Find optimal trading thresholds.

    Args:
        prices: Training price series
        forecasts: Pre-computed forecasts
        prediction_length: Forecast horizon used
        method: Optimization method

    Returns:
        (OptimizedParams, optimization_stats)
    """
    # Parameter bounds
    bounds = [
        (0.001, 0.10),   # buy_threshold: 0.1% to 10%
        (-0.10, -0.001), # sell_threshold: -10% to -0.1%
        (0.5, 5.0),      # upside_ratio_min: 0.5 to 5.0
        (1, 20),         # hold_bars: 1 to 20
    ]

    logger.info(f"Optimizing with {method}...")

    if method == "differential_evolution":
        result = differential_evolution(
            objective_function,
            bounds,
            args=(prices, forecasts, prediction_length),
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            workers=1,
            disp=True,
        )
    else:
        # Use scipy minimize with multiple starts
        best_result = None
        for _ in range(10):
            x0 = [
                np.random.uniform(b[0], b[1]) for b in bounds
            ]
            result = minimize(
                objective_function,
                x0,
                args=(prices, forecasts, prediction_length),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        result = best_result

    params = OptimizedParams(
        buy_threshold=result.x[0],
        sell_threshold=result.x[1],
        upside_ratio_min=result.x[2],
        hold_bars=int(round(result.x[3])),
        prediction_length=prediction_length,
    )

    # Get final stats
    _, stats = simulate_strategy(prices, forecasts, params)

    return params, {
        "optimization_success": result.success,
        "objective_value": result.fun,
        **stats,
    }


def load_and_prepare_data(
    ohlc_path: Path,
    mint: str,
) -> pd.DataFrame:
    """Load and deduplicate OHLC data."""
    df = pd.read_csv(ohlc_path)
    df = df[df["token_mint"] == mint].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def precompute_forecasts(
    prices: np.ndarray,
    context_length: int = 64,
    prediction_length: int = 6,
) -> list:
    """Precompute all forecasts using Chronos2.

    Returns list of (predicted_return, upside_ratio) tuples.
    """
    from .forecaster import DirectForecaster

    forecaster = DirectForecaster(
        prediction_length=prediction_length,
        context_length=context_length,
    )

    n = len(prices)
    forecasts = [None] * n

    logger.info(f"Precomputing {n - context_length} forecasts...")

    for i in range(context_length, n):
        window = prices[i - context_length:i]
        result = forecaster.forecast(window)

        if result is not None:
            forecasts[i] = (result.predicted_return, result.upside_ratio)

        if (i - context_length) % 100 == 0:
            logger.info(f"  Progress: {i - context_length}/{n - context_length}")

    return forecasts


def main():
    parser = argparse.ArgumentParser(description="Optimize Chronos2 trading thresholds")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--prediction-length", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("bagsdirect/optimized_params.json"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load data
    logger.info(f"Loading data from {args.ohlc}")
    df = load_and_prepare_data(args.ohlc, args.mint)
    logger.info(f"Loaded {len(df)} bars")

    # Split
    train_size = int(len(df) * args.train_split)
    train_df = df.iloc[:train_size]
    logger.info(f"Training on {len(train_df)} bars")

    prices = train_df["close"].to_numpy(dtype=np.float64)

    # Precompute forecasts
    forecasts = precompute_forecasts(
        prices,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    # Optimize
    params, stats = optimize_thresholds(
        prices,
        forecasts,
        prediction_length=args.prediction_length,
    )

    print("\n=== Optimization Results ===")
    print(f"Buy threshold: {params.buy_threshold:.4f} ({params.buy_threshold*100:.2f}%)")
    print(f"Sell threshold: {params.sell_threshold:.4f} ({params.sell_threshold*100:.2f}%)")
    print(f"Upside ratio min: {params.upside_ratio_min:.2f}")
    print(f"Hold bars: {params.hold_bars}")
    print(f"\nTraining performance:")
    print(f"  Return: {stats['total_return']*100:.2f}%")
    print(f"  Trades: {stats['num_trades']}")
    print(f"  Win rate: {stats['win_rate']*100:.1f}%")

    # Save params
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "params": params.to_dict(),
            "training_stats": stats,
            "config": {
                "mint": args.mint,
                "train_split": args.train_split,
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
            },
        }, f, indent=2)

    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
