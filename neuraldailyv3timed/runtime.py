"""V3 Timed: Inference runtime with explicit exit timing.

Key V3 features:
- TradingPlan includes exit_days and exit_timestamp
- Model predicts when to exit, not just at what price
- Exit deadline is computed at plan creation time
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch
from loguru import logger

from neuraldailytraining.data import FeatureNormalizer
from neuraldailyv3timed.config import DailyDatasetConfigV3, PolicyConfigV3, SimulationConfig
from neuraldailyv3timed.data import SymbolFrameBuilderV3
from neuraldailyv3timed.model import MultiSymbolPolicyV3, create_group_mask


@dataclass
class TradingPlan:
    """Trading plan output from the V3 model.

    V3 addition: exit_days and exit_timestamp for explicit exit timing.
    """

    symbol: str
    timestamp: pd.Timestamp      # When the plan was created
    buy_price: float             # Entry limit price
    sell_price: float            # Take profit target
    trade_amount: float          # Position size (fraction of equity)
    reference_close: float       # Reference price for calculations
    exit_days: float             # Maximum hold duration (1-10 days)
    exit_timestamp: pd.Timestamp # Absolute deadline for exit
    confidence: float = 1.0
    asset_class: float = 0.0     # 0 = stock, 1 = crypto


def compute_exit_timestamp(
    entry_time: pd.Timestamp,
    exit_days: float,
    is_crypto: bool = False,
) -> pd.Timestamp:
    """
    Compute exit timestamp from entry time and exit_days.

    For stocks: counts trading days (weekdays)
    For crypto: counts calendar days (24/7)

    Args:
        entry_time: When position is opened
        exit_days: Number of days to hold (1-10)
        is_crypto: True for crypto (calendar days), False for stocks (trading days)

    Returns:
        Exit deadline timestamp
    """
    exit_days_int = int(round(exit_days))
    exit_days_int = max(1, min(10, exit_days_int))  # Clamp to 1-10

    if is_crypto:
        # Crypto: calendar days
        exit_time = entry_time + timedelta(days=exit_days_int)
    else:
        # Stocks: trading days (skip weekends)
        days_added = 0
        current = entry_time
        while days_added < exit_days_int:
            current += timedelta(days=1)
            # Skip weekends (5=Saturday, 6=Sunday)
            if current.weekday() < 5:
                days_added += 1
        exit_time = current

    # Set exit time to end of trading day
    # For stocks: 4 PM ET (market close)
    # For crypto: midnight UTC
    if is_crypto:
        exit_time = exit_time.replace(hour=23, minute=59, second=59)
    else:
        # Approximate market close in UTC (4 PM ET = 9 PM UTC in winter)
        exit_time = exit_time.replace(hour=21, minute=0, second=0)

    return exit_time


class DailyTradingRuntimeV3:
    """
    V3 inference runtime with explicit exit timing.

    Key V3 features:
    - Predicts exit_days (1-10) for each trade
    - Computes exit_timestamp when plan is created
    - Uses temperature=0 (binary fills) to match production
    - Supports cross-symbol attention
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        dataset_config: Optional[DailyDatasetConfigV3] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        risk_threshold: Optional[float] = None,
        non_tradable: Optional[Sequence[str]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.non_tradable: Set[str] = set(s.upper() for s in (non_tradable or []))

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Verify V3 checkpoint
        version = checkpoint.get("version", "v1")
        if version != "v3":
            logger.warning(f"Loading {version} checkpoint with V3 runtime - may have issues")

        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load config
        policy_config_dict = checkpoint.get("policy_config", {})
        self.policy_config = PolicyConfigV3(**policy_config_dict)

        # Load feature columns and normalizer
        self.feature_columns = checkpoint.get("feature_columns", [])
        normalizer_data = checkpoint.get("normalizer", {})
        if normalizer_data:
            self.normalizer = FeatureNormalizer(
                mean=np.array(normalizer_data["mean"], dtype=np.float32),
                std=np.array(normalizer_data["std"], dtype=np.float32),
            )
        else:
            # Dummy normalizer (no-op)
            self.normalizer = FeatureNormalizer(
                mean=np.zeros(len(self.feature_columns), dtype=np.float32),
                std=np.ones(len(self.feature_columns), dtype=np.float32),
            )

        # Build model
        self.model = MultiSymbolPolicyV3(self.policy_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Build data builder
        if dataset_config is None:
            dataset_config = DailyDatasetConfigV3()
        self.dataset_config = dataset_config
        self._builder = SymbolFrameBuilderV3(
            data_root=dataset_config.data_root,
            forecast_cache_dir=dataset_config.forecast_cache_dir,
            feature_columns=self.feature_columns,
            require_forecasts=dataset_config.require_forecasts,
            forecast_fill_strategy=dataset_config.forecast_fill_strategy,
            forecast_cache_writeback=dataset_config.forecast_cache_writeback,
            include_weekly_features=dataset_config.include_weekly_features,
        )

        # Simulation config (uses training defaults)
        training_config = checkpoint.get("config", {})
        self.sim_config = SimulationConfig(
            maker_fee=training_config.get("maker_fee", 0.0008),
            equity_max_leverage=training_config.get("equity_max_leverage", 2.0),
            crypto_max_leverage=training_config.get("crypto_max_leverage", 1.0),
            leverage_fee_rate=training_config.get("leverage_fee_rate", 0.065),
            max_hold_days=training_config.get("max_hold_days", 10),
            min_hold_days=training_config.get("min_hold_days", 1),
            forced_exit_slippage=training_config.get("forced_exit_slippage", 0.001),
        )

        logger.info(f"V3 Timed Runtime loaded from {self.checkpoint_path}")
        logger.info(f"Device: {self.device}, Features: {len(self.feature_columns)}")
        logger.info(f"Max hold days: {self.sim_config.max_hold_days}")

    def plan_for_symbol(
        self,
        symbol: str,
        *,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Optional[TradingPlan]:
        """
        Generate trading plan for a single symbol.

        Args:
            symbol: Stock/crypto symbol
            as_of: Optional cutoff date for backtesting

        Returns:
            TradingPlan or None if no valid data
        """
        plans = self.plan_batch([symbol], as_of=as_of)
        return plans[0] if plans else None

    def plan_batch(
        self,
        symbols: Sequence[str],
        *,
        as_of: Optional[pd.Timestamp] = None,
        non_tradable_override: Optional[Set[str]] = None,
    ) -> List[TradingPlan]:
        """
        Generate trading plans for multiple symbols.

        Args:
            symbols: List of symbols
            as_of: Optional cutoff date for backtesting
            non_tradable_override: Override non-tradable set for this call

        Returns:
            List of TradingPlan objects with exit_days and exit_timestamp
        """
        non_tradable = non_tradable_override if non_tradable_override is not None else self.non_tradable

        # Build batch of feature tensors
        batch = self._builder.build_batch(
            list(symbols),
            self.dataset_config.sequence_length,
            self.normalizer,
            as_of=as_of,
        )

        if batch is None:
            return []

        # Move to device
        features = batch["features"].to(self.device)
        reference_close = batch["reference_close"].to(self.device)
        chronos_high = batch["chronos_high"].to(self.device)
        chronos_low = batch["chronos_low"].to(self.device)
        asset_class = batch["asset_class"].to(self.device)
        symbol_indices = batch["symbol_indices"]

        # Create group mask if using cross-attention
        group_mask = None
        if self.policy_config.use_cross_attention and features.shape[0] > 1:
            # Simple group: all symbols in same group for inference
            group_ids = torch.zeros(features.shape[0], dtype=torch.long, device=self.device)
            group_mask = create_group_mask(group_ids)

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(features, group_mask=group_mask)

            actions = self.model.decode_actions(
                outputs,
                reference_close=reference_close,
                chronos_high=chronos_high,
                chronos_low=chronos_low,
                asset_class=asset_class,
            )

        # Current timestamp
        now = as_of if as_of else pd.Timestamp.now(tz="UTC")

        # Extract last timestep actions and build plans
        plans = []
        for i, sym_idx in enumerate(symbol_indices):
            symbol = symbols[sym_idx].upper()

            # Skip non-tradable
            if symbol in non_tradable:
                continue

            buy_price = float(actions["buy_price"][i, -1])
            sell_price = float(actions["sell_price"][i, -1])
            trade_amount = float(actions["trade_amount"][i, -1])
            confidence = float(actions["confidence"][i, -1])
            exit_days = float(actions["exit_days"][i, -1])  # V3: exit timing
            ref_close = float(reference_close[i, -1])
            asset_flag = float(asset_class[i])

            # Apply risk threshold
            if self.risk_threshold is not None:
                trade_amount = min(trade_amount, self.risk_threshold)

            # Apply confidence threshold
            if self.confidence_threshold is not None and confidence < self.confidence_threshold:
                continue

            # Validate prices
            if sell_price <= buy_price:
                continue
            if buy_price <= 0 or sell_price <= 0:
                continue

            # V3: Compute exit timestamp from exit_days
            is_crypto = asset_flag > 0.5
            exit_timestamp = compute_exit_timestamp(now, exit_days, is_crypto=is_crypto)

            plans.append(TradingPlan(
                symbol=symbol,
                timestamp=now,
                buy_price=buy_price,
                sell_price=sell_price,
                trade_amount=trade_amount,
                reference_close=ref_close,
                exit_days=exit_days,
                exit_timestamp=exit_timestamp,
                confidence=confidence,
                asset_class=asset_flag,
            ))

        return plans

    def generate_plans(self, symbols: Sequence[str]) -> List[TradingPlan]:
        """Alias for plan_batch for compatibility with V1/V2 interface."""
        return self.plan_batch(symbols)
