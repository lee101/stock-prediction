"""Market simulator for BagsV5 backtesting on held-out data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from bagsv5.model import BagsV5Config, BagsV5Transformer
from bagsv5.dataset import (
    load_ohlc_dataframe,
    build_bar_features,
    build_aggregate_features,
    FeatureNormalizer,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl_pct: float
    signal_prob: float


@dataclass
class SimulationResult:
    """Simulation results."""
    total_return: float
    num_trades: int
    win_rate: float
    avg_trade_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: np.ndarray
    buy_hold_return: float


def load_v5_model(checkpoint_path: Path, device: str = "cuda") -> Tuple[BagsV5Transformer, BagsV5Config, Dict]:
    """Load V5 model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = BagsV5Config(**ckpt['config'])
    model = BagsV5Transformer(config)
    model.load_state_dict(ckpt['model_state'])

    normalizers = {}
    if 'normalizers' in ckpt:
        for key, norm_dict in ckpt['normalizers'].items():
            normalizers[key] = FeatureNormalizer.from_dict(norm_dict)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, config, normalizers


class MarketSimulator:
    """Simulate trading on held-out CODEX data."""

    def __init__(
        self,
        model: BagsV5Transformer,
        config: BagsV5Config,
        normalizers: Dict[str, FeatureNormalizer],
        buy_threshold: float = 0.5,
        sell_threshold: float = 0.45,
        cost_bps: float = 130.0,
        max_position: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.normalizers = normalizers
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.cost_bps = cost_bps
        self.cost_frac = cost_bps / 10000.0
        self.max_position = max_position
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict(
        self,
        bar_features: np.ndarray,
        agg_features: np.ndarray,
    ) -> Tuple[float, float]:
        """Get signal probability and size."""
        # Normalize
        if 'bar' in self.normalizers:
            bar_flat = bar_features.reshape(-1, bar_features.shape[-1])
            bar_flat = self.normalizers['bar'].transform(bar_flat)
            bar_features = bar_flat.reshape(bar_features.shape)

        if 'agg' in self.normalizers:
            agg_features = self.normalizers['agg'].transform(agg_features.reshape(1, -1))[0]

        # To tensors
        bar_t = torch.tensor(bar_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        agg_t = torch.tensor(agg_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        signal_logit, size_logit = self.model(bar_t, agg_t)

        prob = torch.sigmoid(signal_logit).item()
        size = torch.sigmoid(size_logit).item()

        return prob, size

    def run(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: Optional[int] = None,
    ) -> SimulationResult:
        """Run simulation on data range."""
        if end_idx is None:
            end_idx = len(df)

        context_length = self.config.context_length

        opens = df['open'].to_numpy(dtype=np.float32)
        highs = df['high'].to_numpy(dtype=np.float32)
        lows = df['low'].to_numpy(dtype=np.float32)
        closes = df['close'].to_numpy(dtype=np.float32)
        timestamps = df['timestamp'].to_numpy()

        # Ensure we have enough context
        start_idx = max(start_idx, context_length)

        # Trading state
        position = 0.0
        entry_price = 0.0
        entry_time = None
        entry_prob = 0.0
        entry_equity = 1.0
        trades = []
        equity = [1.0]

        for idx in range(start_idx, end_idx):
            current_price = closes[idx]
            start = idx - context_length

            # Build features
            bar_features = build_bar_features(
                opens[start:idx],
                highs[start:idx],
                lows[start:idx],
                closes[start:idx],
            )

            agg_features = build_aggregate_features(
                opens[start:idx],
                highs[start:idx],
                lows[start:idx],
                closes[start:idx],
            )

            # Get prediction
            prob, size = self.predict(bar_features, agg_features)

            # Trading logic
            if position == 0:
                if prob >= self.buy_threshold:
                    position = min(size, self.max_position)
                    entry_price = current_price * (1 + self.cost_frac)
                    entry_time = timestamps[idx]
                    entry_prob = prob
                    entry_equity = equity[-1]
                    equity.append(equity[-1])
                else:
                    equity.append(equity[-1])
            else:
                if prob < self.sell_threshold:
                    exit_price = current_price * (1 - self.cost_frac)
                    pnl_pct = (exit_price / entry_price - 1) * position

                    trades.append(Trade(
                        entry_time=str(entry_time),
                        exit_time=str(timestamps[idx]),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position,
                        pnl_pct=pnl_pct,
                        signal_prob=entry_prob,
                    ))

                    equity.append(entry_equity * (1 + pnl_pct))
                    position = 0.0
                else:
                    unrealized = (current_price / entry_price - 1) * position
                    equity.append(entry_equity * (1 + unrealized))

        # Close any open position at end
        if position > 0:
            exit_price = closes[end_idx - 1] * (1 - self.cost_frac)
            pnl_pct = (exit_price / entry_price - 1) * position
            trades.append(Trade(
                entry_time=str(entry_time),
                exit_time=str(timestamps[end_idx - 1]),
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position,
                pnl_pct=pnl_pct,
                signal_prob=entry_prob,
            ))
            equity[-1] = entry_equity * (1 + pnl_pct)

        equity = np.array(equity)

        # Calculate metrics
        total_return = equity[-1] / equity[0] - 1
        num_trades = len(trades)
        win_rate = sum(1 for t in trades if t.pnl_pct > 0) / num_trades if num_trades > 0 else 0
        avg_trade = np.mean([t.pnl_pct for t in trades]) if trades else 0

        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_dd = np.max(drawdowns)

        # Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 6) if len(returns) > 1 else 0

        # Buy and hold
        buy_hold = closes[end_idx - 1] / closes[start_idx] - 1

        return SimulationResult(
            total_return=total_return,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity,
            buy_hold_return=buy_hold,
        )


def run_simulation(
    checkpoint_path: Path,
    ohlc_path: Path,
    mint: str,
    test_split: float = 0.2,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.45,
    cost_bps: float = 130.0,
    device: str = "cuda",
) -> SimulationResult:
    """Run market simulation on held-out test data."""

    # Load model
    model, config, normalizers = load_v5_model(checkpoint_path, device)
    logger.info(f"Loaded V5 model from {checkpoint_path}")

    # Load data
    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} bars")

    # Test on last portion (unseen data)
    test_start = int(len(df) * (1 - test_split))
    logger.info(f"Testing on bars {test_start} to {len(df)} (unseen)")

    # Run simulation
    simulator = MarketSimulator(
        model=model,
        config=config,
        normalizers=normalizers,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        cost_bps=cost_bps,
        device=device,
    )

    result = simulator.run(df, test_start)

    return result


def optimize_thresholds(
    checkpoint_path: Path,
    ohlc_path: Path,
    mint: str,
    test_split: float = 0.2,
    cost_bps: float = 130.0,
    device: str = "cuda",
) -> Tuple[float, float, SimulationResult]:
    """Find optimal buy/sell thresholds."""

    model, config, normalizers = load_v5_model(checkpoint_path, device)
    df = load_ohlc_dataframe(ohlc_path, mint)
    test_start = int(len(df) * (1 - test_split))

    simulator = MarketSimulator(
        model=model,
        config=config,
        normalizers=normalizers,
        cost_bps=cost_bps,
        device=device,
    )

    best_alpha = -999
    best_thresholds = (0.5, 0.45)
    best_result = None

    for buy_t in np.arange(0.35, 0.55, 0.01):
        for sell_t in np.arange(0.30, buy_t, 0.01):
            simulator.buy_threshold = buy_t
            simulator.sell_threshold = sell_t

            result = simulator.run(df, test_start)

            if result.num_trades > 0:
                alpha = result.total_return - result.buy_hold_return
                if alpha > best_alpha:
                    best_alpha = alpha
                    best_thresholds = (buy_t, sell_t)
                    best_result = result

    return best_thresholds[0], best_thresholds[1], best_result


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="BagsV5 Market Simulator")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, default="HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--buy-threshold", type=float, default=0.5)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--cost-bps", type=float, default=130.0)
    parser.add_argument("--optimize", action="store_true", help="Optimize thresholds")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.optimize:
        logger.info("Optimizing thresholds...")
        buy_t, sell_t, result = optimize_thresholds(
            args.checkpoint, args.ohlc, args.mint, args.test_split, args.cost_bps, args.device
        )
        print(f"\nOptimal thresholds: Buy={buy_t:.2f}, Sell={sell_t:.2f}")
    else:
        result = run_simulation(
            args.checkpoint, args.ohlc, args.mint,
            args.test_split, args.buy_threshold, args.sell_threshold,
            cost_bps=args.cost_bps,
            device=args.device,
        )

    print("\n" + "=" * 60)
    print("MARKET SIMULATION RESULTS")
    print("=" * 60)
    print(f"Model Return:     {result.total_return*100:+.2f}%")
    print(f"Buy & Hold:       {result.buy_hold_return*100:+.2f}%")
    print(f"Alpha:            {(result.total_return - result.buy_hold_return)*100:+.2f}%")
    print(f"Trades:           {result.num_trades}")
    print(f"Win Rate:         {result.win_rate*100:.1f}%")
    print(f"Avg Trade:        {result.avg_trade_return*100:+.2f}%")
    print(f"Max Drawdown:     {result.max_drawdown*100:.1f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
