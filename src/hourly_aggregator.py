"""Hourly Aggregator - Aggregates opportunities across crypto and stock models.

Picks the highest expected return opportunity across all symbols,
respecting market hours for stocks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from src.date_utils import is_nyse_trading_day_now

logger = logging.getLogger(__name__)


@dataclass
class SymbolOpportunity:
    """Trading opportunity for a single symbol."""

    symbol: str
    asset_class: str  # "crypto" or "stock"
    buy_price: float
    sell_price: float
    position_length: int  # 0 = skip, 1-24 = hours
    position_size: float  # Neural-learned allocation weight (0-1)
    expected_return_pct: float  # (sell - buy) / buy - fees
    risk_adjusted_return: float  # expected_return * position_size
    is_tradable_now: bool  # Market hours check for stocks
    timestamp: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"SymbolOpportunity({self.symbol}, "
            f"exp_ret={self.expected_return_pct:.4%}, "
            f"risk_adj={self.risk_adjusted_return:.4%}, "
            f"size={self.position_size:.2%}, "
            f"hold={self.position_length}h)"
        )


class HourlyAggregator:
    """Aggregates opportunities across crypto + stock models."""

    # Fee structures
    CRYPTO_MAKER_FEE = 0.0008  # 8 bps
    STOCK_MAKER_FEE = 0.0002   # 2 bps

    def __init__(
        self,
        crypto_checkpoint: Optional[str] = None,
        stock_checkpoint: Optional[str] = None,
        crypto_symbols: Optional[List[str]] = None,
        stock_symbols: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        """
        Initialize the aggregator with model checkpoints.

        Args:
            crypto_checkpoint: Path to crypto V5 model checkpoint
            stock_checkpoint: Path to stock V5 model checkpoint
            crypto_symbols: List of crypto symbols to trade
            stock_symbols: List of stock symbols to trade
            device: Device to run models on
        """
        self.device = device
        self.crypto_checkpoint = crypto_checkpoint
        self.stock_checkpoint = stock_checkpoint

        # Default symbols
        self.crypto_symbols = crypto_symbols or [
            "BTCUSD", "ETHUSD", "LINKUSD", "UNIUSD", "SOLUSD"
        ]
        self.stock_symbols = stock_symbols or self._get_all_stock_symbols()

        # Lazy-load models
        self._crypto_model = None
        self._crypto_normalizer = None
        self._crypto_features = None
        self._stock_model = None
        self._stock_normalizer = None
        self._stock_features = None

    def _get_all_stock_symbols(self) -> List[str]:
        """Get all available stock symbols."""
        data_root = Path("trainingdatahourly/stocks")
        if not data_root.exists():
            return []
        return sorted([f.stem for f in data_root.glob("*.csv")])

    def _load_crypto_model(self) -> None:
        """Lazy-load crypto model."""
        if self._crypto_model is not None:
            return

        if self.crypto_checkpoint is None:
            logger.warning("No crypto checkpoint specified")
            return

        from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
        from neuralhourlytradingv5.data import FeatureNormalizer

        logger.info(f"Loading crypto model from {self.crypto_checkpoint}")
        checkpoint = torch.load(
            self.crypto_checkpoint, map_location="cpu", weights_only=False
        )

        # Get config
        policy_config = checkpoint["config"]["policy"]

        # Create model
        model = HourlyCryptoPolicyV5(policy_config)

        # Handle torch.compile prefix
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self._crypto_model = model
        self._crypto_normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        self._crypto_features = checkpoint["feature_columns"]

    def _load_stock_model(self) -> None:
        """Lazy-load stock model."""
        if self._stock_model is not None:
            return

        if self.stock_checkpoint is None:
            logger.warning("No stock checkpoint specified")
            return

        from neuralhourlystocksv5.model import HourlyStockPolicyV5
        from neuralhourlystocksv5.data import StockFeatureNormalizer

        logger.info(f"Loading stock model from {self.stock_checkpoint}")
        checkpoint = torch.load(
            self.stock_checkpoint, map_location="cpu", weights_only=False
        )

        # Get config
        policy_config = checkpoint["config"]["policy"]

        # Create model
        model = HourlyStockPolicyV5(policy_config)

        # Handle torch.compile prefix
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self._stock_model = model
        self._stock_normalizer = StockFeatureNormalizer.from_dict(checkpoint["normalizer"])
        self._stock_features = checkpoint["feature_columns"]

    def _get_crypto_plan(
        self, symbol: str
    ) -> Optional[Tuple[float, float, int, float, pd.Timestamp]]:
        """Get trading plan for a crypto symbol.

        Returns:
            Tuple of (buy_price, sell_price, position_length, position_size, timestamp)
            or None if no opportunity.
        """
        self._load_crypto_model()
        if self._crypto_model is None:
            return None

        from neuralhourlytradingv5.data import HOURLY_FEATURES_V5
        from alpaca_data_wrapper import append_recent_crypto_data

        # Load fresh data
        try:
            append_recent_crypto_data([symbol])
            data_path = Path("trainingdatahourly/crypto") / f"{symbol}.csv"
            if not data_path.exists():
                return None

            df = pd.read_csv(data_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Add missing features with defaults
            for feat in HOURLY_FEATURES_V5:
                if feat not in df.columns:
                    df[feat] = 0.0

            # Get last 168 bars
            if len(df) < 168:
                return None

            features = df[list(self._crypto_features)].tail(168).values
            current_bar = df.iloc[-1]
            current_close = float(current_bar["close"])
            current_ts = pd.to_datetime(current_bar["timestamp"])

            # Normalize and inference
            features_norm = self._crypto_normalizer.transform(features)
            feature_tensor = (
                torch.from_numpy(features_norm)
                .unsqueeze(0)
                .float()
                .contiguous()
                .to(self.device)
            )
            ref_close_tensor = torch.tensor([current_close], device=self.device)

            with torch.no_grad():
                outputs = self._crypto_model(feature_tensor)
                actions = self._crypto_model.get_hard_actions(outputs, ref_close_tensor)

            position_length = int(actions["position_length"].item())
            position_size = float(actions["position_size"].item())
            buy_price = float(actions["buy_price"].item())
            sell_price = float(actions["sell_price"].item())

            return (buy_price, sell_price, position_length, position_size, current_ts)

        except Exception as e:
            logger.warning(f"Error getting crypto plan for {symbol}: {e}")
            return None

    def _get_stock_plan(
        self, symbol: str
    ) -> Optional[Tuple[float, float, int, float, pd.Timestamp]]:
        """Get trading plan for a stock symbol.

        Returns:
            Tuple of (buy_price, sell_price, position_length, position_size, timestamp)
            or None if no opportunity.
        """
        self._load_stock_model()
        if self._stock_model is None:
            return None

        from neuralhourlystocksv5.data import HOURLY_FEATURES_STOCKS_V5

        # Load data
        try:
            data_path = Path("trainingdatahourly/stocks") / f"{symbol}.csv"
            if not data_path.exists():
                return None

            df = pd.read_csv(data_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Add missing features with defaults
            for feat in HOURLY_FEATURES_STOCKS_V5:
                if feat not in df.columns:
                    df[feat] = 0.0

            # Get last 168 bars
            if len(df) < 168:
                return None

            features = df[list(self._stock_features)].tail(168).values
            current_bar = df.iloc[-1]
            current_close = float(current_bar["close"])
            current_ts = pd.to_datetime(current_bar["timestamp"])

            # Normalize and inference
            features_norm = self._stock_normalizer.transform(features)
            feature_tensor = (
                torch.from_numpy(features_norm)
                .unsqueeze(0)
                .float()
                .contiguous()
                .to(self.device)
            )
            ref_close_tensor = torch.tensor([current_close], device=self.device)

            with torch.no_grad():
                outputs = self._stock_model(feature_tensor)
                actions = self._stock_model.get_hard_actions(outputs, ref_close_tensor)

            position_length = int(actions["position_length"].item())
            position_size = float(actions["position_size"].item())
            buy_price = float(actions["buy_price"].item())
            sell_price = float(actions["sell_price"].item())

            return (buy_price, sell_price, position_length, position_size, current_ts)

        except Exception as e:
            logger.warning(f"Error getting stock plan for {symbol}: {e}")
            return None

    def _calculate_opportunity(
        self,
        symbol: str,
        asset_class: str,
        buy_price: float,
        sell_price: float,
        position_length: int,
        position_size: float,
        timestamp: pd.Timestamp,
    ) -> SymbolOpportunity:
        """Calculate expected return and create opportunity object."""
        # Fee structure
        maker_fee = self.CRYPTO_MAKER_FEE if asset_class == "crypto" else self.STOCK_MAKER_FEE
        round_trip_fees = 2 * maker_fee

        # Gross return
        gross_return = (sell_price - buy_price) / buy_price

        # Net return after fees
        net_return = gross_return - round_trip_fees

        # Risk-adjusted (weighted by model confidence)
        risk_adjusted = net_return * position_size

        # Market hours check
        if asset_class == "crypto":
            is_tradable = True  # Crypto is 24/7
        else:
            is_tradable = is_nyse_trading_day_now()

        return SymbolOpportunity(
            symbol=symbol,
            asset_class=asset_class,
            buy_price=buy_price,
            sell_price=sell_price,
            position_length=position_length,
            position_size=position_size,
            expected_return_pct=net_return,
            risk_adjusted_return=risk_adjusted,
            is_tradable_now=is_tradable,
            timestamp=timestamp,
        )

    def get_all_opportunities(self) -> List[SymbolOpportunity]:
        """
        Run both models on all symbols and return all opportunities.

        Returns:
            List of SymbolOpportunity objects, one per symbol that has a
            non-zero position_length.
        """
        opportunities = []
        now = datetime.now(timezone.utc)
        is_stock_market_open = is_nyse_trading_day_now(now)

        # Crypto opportunities (24/7 tradable)
        if self.crypto_checkpoint:
            for symbol in self.crypto_symbols:
                plan = self._get_crypto_plan(symbol)
                if plan and plan[2] > 0:  # position_length > 0
                    opp = self._calculate_opportunity(
                        symbol, "crypto",
                        plan[0], plan[1], plan[2], plan[3], plan[4]
                    )
                    opportunities.append(opp)
                    logger.info(f"Crypto {symbol}: {opp}")

        # Stock opportunities (only during market hours)
        if self.stock_checkpoint and is_stock_market_open:
            for symbol in self.stock_symbols:
                plan = self._get_stock_plan(symbol)
                if plan and plan[2] > 0:  # position_length > 0
                    opp = self._calculate_opportunity(
                        symbol, "stock",
                        plan[0], plan[1], plan[2], plan[3], plan[4]
                    )
                    opportunities.append(opp)
                    logger.debug(f"Stock {symbol}: {opp}")

        logger.info(f"Found {len(opportunities)} opportunities")
        return opportunities

    def rank_by_expected_return(
        self, opportunities: List[SymbolOpportunity]
    ) -> List[SymbolOpportunity]:
        """
        Rank opportunities by risk-adjusted expected return.

        Returns:
            Sorted list with highest expected return first.
        """
        # Filter: only tradable opportunities
        tradeable = [opp for opp in opportunities if opp.is_tradable_now]

        # Sort by risk-adjusted return (descending)
        return sorted(tradeable, key=lambda x: -x.risk_adjusted_return)

    def get_best_opportunity(self) -> Optional[SymbolOpportunity]:
        """
        Get the single best trading opportunity.

        Returns:
            The highest risk-adjusted return opportunity, or None if no opportunities.
        """
        opportunities = self.get_all_opportunities()
        ranked = self.rank_by_expected_return(opportunities)
        return ranked[0] if ranked else None

    def get_top_opportunities(
        self,
        n: int = 3,
        min_expected_return_bps: int = 10,
    ) -> List[SymbolOpportunity]:
        """
        Get the top N trading opportunities.

        Args:
            n: Maximum number of opportunities to return
            min_expected_return_bps: Minimum expected return in basis points

        Returns:
            List of top opportunities, sorted by risk-adjusted return.
        """
        opportunities = self.get_all_opportunities()
        ranked = self.rank_by_expected_return(opportunities)

        # Filter by minimum expected return
        min_return = min_expected_return_bps / 10000
        filtered = [opp for opp in ranked if opp.expected_return_pct >= min_return]

        return filtered[:n]

    def get_opportunity_for_symbol(self, symbol: str) -> Optional[SymbolOpportunity]:
        """
        Get opportunity for a specific symbol.

        Args:
            symbol: The trading symbol

        Returns:
            SymbolOpportunity if available, None otherwise.
        """
        # Check if it's crypto
        if symbol in self.crypto_symbols:
            plan = self._get_crypto_plan(symbol)
            if plan and plan[2] > 0:
                return self._calculate_opportunity(
                    symbol, "crypto",
                    plan[0], plan[1], plan[2], plan[3], plan[4]
                )
            return None

        # Check if it's stock
        if symbol in self.stock_symbols:
            plan = self._get_stock_plan(symbol)
            if plan and plan[2] > 0:
                return self._calculate_opportunity(
                    symbol, "stock",
                    plan[0], plan[1], plan[2], plan[3], plan[4]
                )
            return None

        return None
