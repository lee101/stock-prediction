"""V5 Runtime for inference with ramp-into-position execution.

Key features:
- Load trained model and predict target portfolio
- Ramp-into-position: Gradual execution throughout the day
- Portfolio watcher: Monitor and adjust positions
- Integration with Alpaca API for live trading
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger

from neuraldailyv5.config import DailyTrainingConfigV5, PolicyConfigV5
from neuraldailyv5.model import PortfolioPolicyV5


@dataclass
class PortfolioTarget:
    """Target portfolio allocation."""
    weights: Dict[str, float]  # symbol -> weight
    confidence: float
    volatility: Dict[str, float]  # symbol -> predicted volatility
    timestamp: datetime


@dataclass
class ExecutionOrder:
    """Order to execute for rebalancing."""
    symbol: str
    side: str  # "buy" or "sell"
    notional: float  # Dollar amount to trade
    limit_price: Optional[float] = None
    time_in_force: str = "day"


class PortfolioPredictor:
    """
    V5 Portfolio predictor for inference.

    Loads trained model and predicts target portfolio weights.
    """

    def __init__(
        self,
        checkpoint_path: str,
        symbols: Tuple[str, ...],
        device: str = None,
    ):
        self.symbols = symbols
        self.num_assets = len(symbols)

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Asset class mapping
        self.asset_class = torch.tensor([
            1.0 if s.upper().endswith("USD") or "-USD" in s.upper() else 0.0
            for s in symbols
        ], device=self.device)

    def _load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Extract config
        if "config" in checkpoint:
            config = checkpoint["config"]
            if hasattr(config, "get_policy_config"):
                # Infer input_dim from model state
                state_dict = checkpoint["model_state_dict"]
                for key in state_dict:
                    if "embed" in key and "weight" in key:
                        input_dim = state_dict[key].shape[1] // config.patch_size
                        break
                else:
                    input_dim = 20  # Default

                policy_config = config.get_policy_config(input_dim, self.num_assets)
            else:
                policy_config = PolicyConfigV5(num_assets=self.num_assets)
        else:
            policy_config = PolicyConfigV5(num_assets=self.num_assets)

        # Create model
        self.model = PortfolioPolicyV5(policy_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded model from {path}")

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,  # (seq_len, num_features) or (batch, seq_len, num_features)
    ) -> PortfolioTarget:
        """
        Predict target portfolio weights.

        Args:
            features: Normalized feature tensor

        Returns:
            PortfolioTarget with weights and metadata
        """
        # Ensure batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)

        features = features.to(self.device)

        # Forward pass
        outputs = self.model(features, return_latents=False)

        # Extract weights
        weights = outputs["weights"][0].cpu().numpy()  # (num_assets,)
        confidence = outputs.get("confidence", torch.tensor([[0.5]]))[0, 0].item()

        # Extract volatility if available
        if "volatility" in outputs:
            volatility = {
                sym: outputs["volatility"][0, i].item()
                for i, sym in enumerate(self.symbols)
            }
        else:
            volatility = {sym: 0.0 for sym in self.symbols}

        # Convert to dict
        weight_dict = {
            sym: float(weights[i])
            for i, sym in enumerate(self.symbols)
        }

        return PortfolioTarget(
            weights=weight_dict,
            confidence=confidence,
            volatility=volatility,
            timestamp=datetime.now(),
        )


class PortfolioWatcher:
    """
    Watches current portfolio and generates rebalancing orders.

    Implements ramp-into-position execution:
    1. Get target portfolio from predictor
    2. Compare with current positions
    3. Generate orders to gradually move toward target
    4. Split execution across multiple periods
    """

    def __init__(
        self,
        predictor: PortfolioPredictor,
        symbols: Tuple[str, ...],
        *,
        rebalance_threshold: float = 0.02,
        ramp_periods: int = 4,
        max_single_trade_pct: float = 0.25,
    ):
        self.predictor = predictor
        self.symbols = symbols
        self.rebalance_threshold = rebalance_threshold
        self.ramp_periods = ramp_periods
        self.max_single_trade_pct = max_single_trade_pct

        self.current_target: Optional[PortfolioTarget] = None
        self.execution_progress = 0  # 0 to ramp_periods

    def update_target(self, features: torch.Tensor) -> PortfolioTarget:
        """Update target portfolio from new features."""
        self.current_target = self.predictor.predict(features)
        self.execution_progress = 0
        return self.current_target

    def get_rebalance_orders(
        self,
        current_positions: Dict[str, float],  # symbol -> market value
        portfolio_value: float,
        current_prices: Dict[str, float],  # symbol -> price
    ) -> List[ExecutionOrder]:
        """
        Generate rebalancing orders for current execution period.

        Args:
            current_positions: Current position values
            portfolio_value: Total portfolio value
            current_prices: Current prices for limit orders

        Returns:
            List of orders to execute
        """
        if self.current_target is None:
            return []

        if self.execution_progress >= self.ramp_periods:
            return []

        orders = []

        # Calculate current weights
        current_weights = {
            sym: current_positions.get(sym, 0.0) / portfolio_value
            for sym in self.symbols
        }

        # Calculate target weights for this period
        # Gradually move from current to target
        progress = (self.execution_progress + 1) / self.ramp_periods

        for sym in self.symbols:
            current_w = current_weights.get(sym, 0.0)
            target_w = self.current_target.weights.get(sym, 0.0)

            # Intermediate target for this period
            period_target = current_w + progress * (target_w - current_w)

            # Weight difference
            diff = period_target - current_w

            # Check threshold
            if abs(diff) < self.rebalance_threshold:
                continue

            # Limit single trade size
            max_trade = self.max_single_trade_pct * portfolio_value
            trade_value = min(abs(diff) * portfolio_value, max_trade)

            # Create order
            side = "buy" if diff > 0 else "sell"
            price = current_prices.get(sym)

            order = ExecutionOrder(
                symbol=sym,
                side=side,
                notional=trade_value,
                limit_price=price,
            )
            orders.append(order)

        self.execution_progress += 1
        return orders


def load_runtime(
    checkpoint_path: str,
    symbols: Tuple[str, ...] = None,
    device: str = None,
) -> Tuple[PortfolioPredictor, PortfolioWatcher]:
    """
    Load V5 runtime components.

    Args:
        checkpoint_path: Path to model checkpoint
        symbols: List of symbols (default: from checkpoint)
        device: Device to use

    Returns:
        predictor, watcher: Runtime components
    """
    if symbols is None:
        # Try to load from checkpoint metadata
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" in checkpoint and hasattr(checkpoint["config"], "dataset"):
            symbols = checkpoint["config"].dataset.symbols
        else:
            symbols = (
                "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
                "BTCUSD", "ETHUSD",
            )

    predictor = PortfolioPredictor(checkpoint_path, symbols, device)
    watcher = PortfolioWatcher(predictor, symbols)

    return predictor, watcher


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predictor, watcher = load_runtime(args.checkpoint, device=args.device)
    logger.info(f"Loaded runtime for {len(predictor.symbols)} symbols")
    logger.info(f"Symbols: {predictor.symbols}")
