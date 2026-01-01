"""V4: Inference runtime with multi-window Chronos-2 architecture.

Key V4 features:
- Multi-window predictions with trimmed mean aggregation
- Quantile-based price targets
- Learned position sizing from uncertainty
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch
from loguru import logger

from neuraldailytraining.data import FeatureNormalizer
from neuraldailyv4.config import DailyDatasetConfigV4, PolicyConfigV4, SimulationConfigV4
from neuraldailyv4.model import MultiSymbolPolicyV4, create_group_mask

# Import NYSE calendar for holiday detection
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.date_utils import is_nyse_open_on_date


@dataclass
class TradingPlanV4:
    """Trading plan output from the V4 model.

    V4 features: multi-window aggregated predictions with quantiles.
    """

    symbol: str
    timestamp: pd.Timestamp      # When the plan was created
    buy_price: float             # Entry limit price (from quantile)
    sell_price: float            # Take profit target (from quantile)
    trade_amount: float          # Position size (learned from uncertainty)
    reference_close: float       # Reference price for calculations
    exit_days: float             # Aggregated exit timing
    exit_timestamp: pd.Timestamp # Absolute deadline for exit
    confidence: float = 1.0      # Model confidence
    asset_class: float = 0.0     # 0 = stock, 1 = crypto


def compute_exit_timestamp(
    entry_time: pd.Timestamp,
    exit_days: float,
    is_crypto: bool = False,
) -> pd.Timestamp:
    """Compute exit timestamp from entry time and exit_days."""
    exit_days_int = int(round(exit_days))
    exit_days_int = max(1, min(20, exit_days_int))  # V4 supports up to 20 days

    if is_crypto:
        exit_time = entry_time + timedelta(days=exit_days_int)
    else:
        # Stocks: trading days (skip weekends)
        days_added = 0
        current = entry_time
        while days_added < exit_days_int:
            current += timedelta(days=1)
            if current.weekday() < 5:
                days_added += 1
        exit_time = current

    # Set exit time to end of trading day
    if is_crypto:
        exit_time = exit_time.replace(hour=23, minute=59, second=59)
    else:
        exit_time = exit_time.replace(hour=21, minute=0, second=0)

    return exit_time


class DailyTradingRuntimeV4:
    """
    V4 inference runtime with Chronos-2 multi-window architecture.

    Key V4 features:
    - Multi-window predictions with trimmed mean
    - Quantile-based price targets
    - Learned position sizing
    """

    # Default non-tradable symbols (poor backtest performance)
    # Equities:
    #   AMZN: consistently worst (-$55 in 60-day backtest, 40% TP rate)
    #   NFLX: -102% in 90-day backtest (4 trades, -25.58% avg) - model fails on NFLX
    #   Note: TSLA excluded improves Sharpe 0.076→0.127 but reduces trades 65→51
    # Crypto:
    #   BTCUSD: -22.72% total, 43.5% TP rate - model doesn't work for BTC
    #   AVAXUSD: -23.89% total, 0% TP rate - very bad
    #   Note: LINKUSD is actually profitable (+30.37%), keep it tradable
    # V8 analysis (90-day backtest Dec 2025):
    #   ORCL: -102% return, Sortino -1.16 - extremely bad
    #   NOW: -178% return, Sortino -0.38 - worst performer
    #   DDOG: -76% return, Sortino -0.73 - very bad
    #   PYPL: -31% return, Sortino -0.72 - consistently loses
    #   INTC: -30% return, Sortino -0.57 - avoid
    #   COIN: -31% return, Sortino -0.16 - bad for model
    DEFAULT_NON_TRADABLE = {
        # Original exclusions
        "META", "MSFT", "AMZN", "NFLX",
        # Crypto exclusions
        "BTCUSD", "AVAXUSD",
        # V8 additions (negative Sortino in 90-day backtest)
        "ORCL", "NOW", "DDOG", "PYPL", "INTC", "COIN",
        # Borderline but safer to exclude
        "AVGO", "COST", "CRM", "HD",
        # V9 analysis - consistently loses money
        "MRVL",  # -8% in 90-day backtest across both V8 and V9
    }

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        dataset_config: Optional[DailyDatasetConfigV4] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        risk_threshold: Optional[float] = None,
        non_tradable: Optional[Sequence[str]] = None,
        use_default_non_tradable: bool = True,
        trade_crypto: bool = False,  # Set True to trade crypto (16 bps round-trip fees)
        trade_crypto_on_holidays: bool = True,  # Auto-enable crypto when NYSE is closed
        max_position_override: Optional[float] = None,  # Override max position (e.g., 0.30 for 30%)
        max_exit_days: Optional[int] = None,  # Override max exit days (e.g., 2 for faster exits)
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.max_position_override = max_position_override
        self.max_exit_days = max_exit_days

        # Build non-tradable set
        self.non_tradable: Set[str] = set()
        if use_default_non_tradable:
            self.non_tradable.update(self.DEFAULT_NON_TRADABLE)
        if non_tradable:
            self.non_tradable.update(s.upper() for s in non_tradable)

        # Crypto trading flag (disabled by default - enable with trade_crypto=True)
        # Note: Actual fees are ~16 bps round-trip (8 bps per leg), not 1.6%
        self.trade_crypto = trade_crypto
        self.trade_crypto_on_holidays = trade_crypto_on_holidays

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load config
        config = checkpoint.get("config", {})
        policy_config_dict = config.get("policy", {})
        self.policy_config = PolicyConfigV4(**policy_config_dict)

        # Apply max_position override if specified
        if self.max_position_override is not None:
            self.policy_config.max_position = self.max_position_override
            logger.info(f"Overriding max_position to {self.max_position_override:.1%}")

        # Log max_exit_days override (applied in plan_batch)
        if self.max_exit_days is not None:
            logger.info(f"Overriding max_exit_days to {self.max_exit_days}")

        # Load feature columns and normalizer
        self.feature_columns = checkpoint.get("feature_columns", [])
        normalizer_data = checkpoint.get("normalizer", None)

        if normalizer_data is not None:
            if isinstance(normalizer_data, dict):
                self.normalizer = FeatureNormalizer(
                    mean=np.array(normalizer_data["mean"], dtype=np.float32),
                    std=np.array(normalizer_data["std"], dtype=np.float32),
                )
            elif hasattr(normalizer_data, "mean") and hasattr(normalizer_data, "std"):
                # Already a FeatureNormalizer
                self.normalizer = normalizer_data
            else:
                self.normalizer = FeatureNormalizer(
                    mean=np.zeros(len(self.feature_columns), dtype=np.float32),
                    std=np.ones(len(self.feature_columns), dtype=np.float32),
                )
        else:
            self.normalizer = FeatureNormalizer(
                mean=np.zeros(len(self.feature_columns), dtype=np.float32),
                std=np.ones(len(self.feature_columns), dtype=np.float32),
            )

        # Build model
        self.model = MultiSymbolPolicyV4(self.policy_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Build data builder - reuse V3's builder
        from neuraldailyv3timed.data import SymbolFrameBuilderV3
        if dataset_config is None:
            dataset_config = DailyDatasetConfigV4()
        self.dataset_config = dataset_config
        self._builder = SymbolFrameBuilderV3(
            data_root=dataset_config.data_root if hasattr(dataset_config, 'data_root') else "data",
            forecast_cache_dir=dataset_config.forecast_cache_dir if hasattr(dataset_config, 'forecast_cache_dir') else None,
            feature_columns=self.feature_columns,
            require_forecasts=dataset_config.require_forecasts if hasattr(dataset_config, 'require_forecasts') else True,
            include_weekly_features=dataset_config.include_weekly_features if hasattr(dataset_config, 'include_weekly_features') else True,
        )

        # Simulation config
        sim_config_dict = config.get("simulation", {})
        self.sim_config = SimulationConfigV4(**sim_config_dict) if sim_config_dict else SimulationConfigV4()

        logger.info(f"V4 Runtime loaded from {self.checkpoint_path}")
        logger.info(f"Device: {self.device}, Features: {len(self.feature_columns)}")
        logger.info(f"Windows: {self.policy_config.num_windows}, Quantiles: {self.policy_config.num_quantiles}")
        if not self.trade_crypto:
            logger.info("Crypto trading disabled (trade_crypto=False)")

    @staticmethod
    def is_crypto_symbol(symbol: str) -> bool:
        """Check if symbol is a crypto asset (ends in USD and length > 4)."""
        s = symbol.upper()
        return s.endswith("USD") and len(s) > 4

    def plan_for_symbol(
        self,
        symbol: str,
        *,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Optional[TradingPlanV4]:
        """Generate trading plan for a single symbol."""
        plans = self.plan_batch([symbol], as_of=as_of)
        return plans[0] if plans else None

    def plan_batch(
        self,
        symbols: Sequence[str],
        *,
        as_of: Optional[pd.Timestamp] = None,
        non_tradable_override: Optional[Set[str]] = None,
    ) -> List[TradingPlanV4]:
        """Generate trading plans for multiple symbols."""
        non_tradable = non_tradable_override if non_tradable_override is not None else self.non_tradable

        # Current date for holiday check
        now = as_of if as_of else pd.Timestamp.now(tz="UTC")

        # Check if NYSE is closed today (holiday/weekend)
        nyse_open_today = is_nyse_open_on_date(now)

        # Determine if we should trade crypto
        # 1. If trade_crypto is explicitly enabled, always include crypto
        # 2. If trade_crypto_on_holidays is enabled and NYSE is closed, include crypto
        include_crypto = self.trade_crypto or (self.trade_crypto_on_holidays and not nyse_open_today)

        # Filter symbols based on crypto trading rules
        if not include_crypto:
            symbols = [s for s in symbols if not self.is_crypto_symbol(s)]
            if not symbols:
                return []
        elif not nyse_open_today and self.trade_crypto_on_holidays and not self.trade_crypto:
            # NYSE closed: only trade crypto on holidays (filter out stocks)
            logger.info(f"NYSE closed on {now.date()}, enabling crypto-only trading")
            symbols = [s for s in symbols if self.is_crypto_symbol(s)]
            if not symbols:
                return []

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

        # Create group mask
        group_mask = None
        if self.policy_config.use_cross_attention and features.shape[0] > 1:
            group_ids = torch.zeros(features.shape[0], dtype=torch.long, device=self.device)
            group_mask = create_group_mask(group_ids)

        # Get last timestep reference prices
        ref_close_last = reference_close[:, -1] if reference_close.dim() > 1 else reference_close
        ch_high_last = chronos_high[:, -1] if chronos_high.dim() > 1 else chronos_high
        ch_low_last = chronos_low[:, -1] if chronos_low.dim() > 1 else chronos_low

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(features, group_mask=group_mask)

            actions = self.model.decode_actions(
                outputs,
                reference_close=ref_close_last,
                chronos_high=ch_high_last,
                chronos_low=ch_low_last,
                asset_class=asset_class,
            )

        # Current timestamp
        now = as_of if as_of else pd.Timestamp.now(tz="UTC")

        # Build plans from aggregated outputs
        plans = []
        for i, sym_idx in enumerate(symbol_indices):
            symbol = symbols[sym_idx].upper()

            if symbol in non_tradable:
                continue

            # Get median quantile for prices (index 1 of 3 = q50)
            buy_quantiles = actions["buy_quantiles"][i]  # (num_windows, 3)
            sell_quantiles = actions["sell_quantiles"][i]  # (num_windows, 3)

            # Aggregate across windows using median quantile (q50)
            buy_prices = buy_quantiles[:, 1]  # q50 for each window
            sell_prices = sell_quantiles[:, 1]  # q50 for each window

            # Trimmed mean across windows
            buy_price = self._trimmed_mean(buy_prices)
            sell_price = self._trimmed_mean(sell_prices)

            # Minimum spread (2% default, adjustable)
            min_spread = 0.02
            min_sell = buy_price * (1 + min_spread)
            if sell_price < min_sell:
                sell_price = float(min_sell)

            # Position size and confidence from model
            pos_size_tensor = actions["position_size"][i]
            conf_tensor = actions["confidence"][i]
            exit_days_tensor = actions["exit_days"][i]

            # Handle tensor vs scalar conversion
            position_size = float(pos_size_tensor.item() if pos_size_tensor.dim() == 0 else pos_size_tensor.mean().item())
            confidence = float(conf_tensor.item() if conf_tensor.dim() == 0 else conf_tensor.mean().item())
            exit_days = float(exit_days_tensor.item() if exit_days_tensor.dim() == 0 else exit_days_tensor.mean().item())

            # Apply max_exit_days override (shorter exits = better Sharpe in backtests)
            if self.max_exit_days is not None:
                exit_days = min(exit_days, float(self.max_exit_days))

            ref_close_t = ref_close_last[i]
            asset_flag_t = asset_class[i]
            ref_close = float(ref_close_t.item() if ref_close_t.dim() == 0 else ref_close_t.mean().item())
            asset_flag = float(asset_flag_t.item() if asset_flag_t.dim() == 0 else asset_flag_t.mean().item())

            # Apply risk threshold
            if self.risk_threshold is not None:
                position_size = min(position_size, self.risk_threshold)

            # Apply confidence threshold
            if self.confidence_threshold is not None and confidence < self.confidence_threshold:
                continue

            # Validate prices
            if sell_price <= buy_price:
                continue
            if buy_price <= 0 or sell_price <= 0:
                continue

            # Compute exit timestamp
            is_crypto = asset_flag > 0.5
            exit_timestamp = compute_exit_timestamp(now, exit_days, is_crypto=is_crypto)

            plans.append(TradingPlanV4(
                symbol=symbol,
                timestamp=now,
                buy_price=float(buy_price),
                sell_price=float(sell_price),
                trade_amount=position_size,
                reference_close=ref_close,
                exit_days=exit_days,
                exit_timestamp=exit_timestamp,
                confidence=confidence,
                asset_class=asset_flag,
            ))

        return plans

    def _trimmed_mean(self, values: torch.Tensor, trim_fraction: float = 0.25) -> float:
        """Compute trimmed mean, removing top and bottom fraction."""
        if len(values) <= 2:
            return float(values.mean())

        sorted_vals = torch.sort(values)[0]
        n = len(sorted_vals)
        trim_count = int(n * trim_fraction)
        if trim_count == 0:
            return float(sorted_vals.mean())

        trimmed = sorted_vals[trim_count:-trim_count] if trim_count > 0 else sorted_vals
        return float(trimmed.mean())
