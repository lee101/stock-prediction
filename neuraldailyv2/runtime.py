"""V2 Inference runtime using unified simulation.

The key V2 innovation is using temperature=0 (binary fills) at inference,
which exactly matches what the model was trained toward via temperature annealing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch
from loguru import logger

from neuraldailytraining.data import FeatureNormalizer
from neuraldailyv2.config import DailyDatasetConfigV2, PolicyConfigV2, SimulationConfig
from neuraldailyv2.data import SymbolFrameBuilderV2
from neuraldailyv2.model import MultiSymbolPolicyV2, create_group_mask


@dataclass
class TradingPlan:
    """Trading plan output from the model."""

    symbol: str
    timestamp: pd.Timestamp
    buy_price: float
    sell_price: float
    trade_amount: float
    reference_close: float
    confidence: float = 1.0
    asset_class: float = 0.0  # 0 = stock, 1 = crypto


class DailyTradingRuntimeV2:
    """
    V2 inference runtime using unified simulation.

    Key V2 features:
    - Uses temperature=0 (binary fills) to match production
    - Consistent with training simulation
    - Supports cross-symbol attention
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        dataset_config: Optional[DailyDatasetConfigV2] = None,
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

        # Verify V2 checkpoint
        version = checkpoint.get("version", "v1")
        if version != "v2":
            logger.warning(f"Loading {version} checkpoint with V2 runtime - may have issues")

        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load config
        policy_config_dict = checkpoint.get("policy_config", {})
        self.policy_config = PolicyConfigV2(**policy_config_dict)

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
        self.model = MultiSymbolPolicyV2(self.policy_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Build data builder
        if dataset_config is None:
            dataset_config = DailyDatasetConfigV2()
        self.dataset_config = dataset_config
        self._builder = SymbolFrameBuilderV2(
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
        )

        logger.info(f"V2 Runtime loaded from {self.checkpoint_path}")
        logger.info(f"Device: {self.device}, Features: {len(self.feature_columns)}")

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
            List of TradingPlan objects
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

            # Get timestamp from data
            timestamp = as_of if as_of else pd.Timestamp.now(tz="UTC")

            plans.append(TradingPlan(
                symbol=symbol,
                timestamp=timestamp,
                buy_price=buy_price,
                sell_price=sell_price,
                trade_amount=trade_amount,
                reference_close=ref_close,
                confidence=confidence,
                asset_class=asset_flag,
            ))

        return plans

    def generate_plans(self, symbols: Sequence[str]) -> List[TradingPlan]:
        """Alias for plan_batch for compatibility with V1 interface."""
        return self.plan_batch(symbols)
