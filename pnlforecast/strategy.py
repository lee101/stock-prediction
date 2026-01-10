"""Strategy generation for buy-low/sell-high thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .config import StrategyConfigPnL

logger = logging.getLogger(__name__)


@dataclass
class StrategyThresholds:
    """Buy-low/sell-high threshold strategy.

    The strategy works as follows:
    - Buy when price drops below (open * (1 - buy_threshold_pct))
    - Sell when price rises above (open * (1 + sell_threshold_pct))

    Both thresholds are expressed as percentages (e.g., 0.002 = 0.2%).
    """

    buy_threshold_pct: float  # How far below open to buy (e.g., 0.002 = 0.2%)
    sell_threshold_pct: float  # How far above open to sell (e.g., 0.003 = 0.3%)
    strategy_id: str = ""

    def __post_init__(self):
        if not self.strategy_id:
            buy_bp = int(self.buy_threshold_pct * 10000)
            sell_bp = int(self.sell_threshold_pct * 10000)
            self.strategy_id = f"b{buy_bp}s{sell_bp}"

    @property
    def total_spread_pct(self) -> float:
        """Total spread between buy and sell thresholds."""
        return self.buy_threshold_pct + self.sell_threshold_pct

    def get_buy_price(self, open_price: float) -> float:
        """Calculate buy trigger price."""
        return open_price * (1 - self.buy_threshold_pct)

    def get_sell_price(self, open_price: float) -> float:
        """Calculate sell trigger price."""
        return open_price * (1 + self.sell_threshold_pct)

    def can_trade_intraday(self, open_price: float, high: float, low: float) -> Tuple[bool, bool]:
        """Check if both buy and sell can trigger in a single day.

        Args:
            open_price: Day's opening price
            high: Day's high price
            low: Day's low price

        Returns:
            Tuple of (can_buy, can_sell) booleans
        """
        buy_price = self.get_buy_price(open_price)
        sell_price = self.get_sell_price(open_price)

        can_buy = low <= buy_price
        can_sell = high >= sell_price

        return can_buy, can_sell

    def compute_trade_pnl(
        self,
        open_price: float,
        high: float,
        low: float,
        fee_pct: float,
    ) -> Tuple[float, str]:
        """Compute PnL for a round-trip trade on a single day.

        Strategy logic:
        1. If both buy and sell thresholds are hit, assume we buy at low threshold
           and sell at high threshold (best case for long strategy)
        2. If only buy is hit, we hold the position (PnL = close - buy_price)
        3. If only sell is hit, we can't enter (no trade)
        4. If neither is hit, no trade

        Args:
            open_price: Day's opening price
            high: Day's high price
            low: Day's low price
            fee_pct: Fee percentage per side

        Returns:
            Tuple of (pnl_pct, trade_type) where trade_type is
            "round_trip", "hold", or "no_trade"
        """
        buy_price = self.get_buy_price(open_price)
        sell_price = self.get_sell_price(open_price)

        can_buy = low <= buy_price
        can_sell = high >= sell_price

        if can_buy and can_sell:
            # Round-trip: buy at buy_price, sell at sell_price
            gross_pnl_pct = (sell_price - buy_price) / buy_price
            net_pnl_pct = gross_pnl_pct - (2 * fee_pct)  # Two-sided fees
            return net_pnl_pct, "round_trip"

        elif can_buy:
            # Buy triggered but can't sell - would need to hold
            # For simulation, we mark this as a pending position
            return 0.0, "buy_pending"

        elif can_sell:
            # Can't enter - no trade
            return 0.0, "no_entry"

        else:
            # Neither threshold hit - no trade
            return 0.0, "no_trade"

    def __hash__(self):
        return hash((self.buy_threshold_pct, self.sell_threshold_pct))

    def __eq__(self, other):
        if not isinstance(other, StrategyThresholds):
            return False
        return (
            abs(self.buy_threshold_pct - other.buy_threshold_pct) < 1e-9
            and abs(self.sell_threshold_pct - other.sell_threshold_pct) < 1e-9
        )


def generate_threshold_strategies(
    config: StrategyConfigPnL,
    is_crypto: bool = False,
) -> List[StrategyThresholds]:
    """Generate a grid of threshold strategies.

    Args:
        config: Strategy configuration
        is_crypto: Whether generating for crypto (affects min spread)

    Returns:
        List of valid StrategyThresholds
    """
    min_spread = config.get_min_spread_pct(is_crypto)
    step = config.threshold_step_pct / 100  # Convert to decimal

    # Generate threshold values from min to max
    min_thresh = config.min_threshold_pct / 100
    max_thresh = config.max_threshold_pct / 100

    # Create grid of thresholds
    n_steps = int((max_thresh - min_thresh) / step) + 1
    thresholds = [min_thresh + i * step for i in range(n_steps)]

    strategies = []
    for buy_thresh in thresholds:
        for sell_thresh in thresholds:
            # Skip if spread is too small (must cover round-trip fees)
            total_spread = buy_thresh + sell_thresh
            if total_spread < min_spread:
                continue

            # Skip if thresholds are inverted or too close
            # (buy and sell can't be the same or swapped)
            if sell_thresh <= 0 and buy_thresh <= 0:
                # At least one threshold must be positive
                continue

            strategy = StrategyThresholds(
                buy_threshold_pct=buy_thresh,
                sell_threshold_pct=sell_thresh,
            )
            strategies.append(strategy)

    logger.info(
        "Generated %d threshold strategies for %s (min_spread=%.4f%%)",
        len(strategies),
        "crypto" if is_crypto else "stock",
        min_spread * 100,
    )

    return strategies


def filter_profitable_strategies(
    strategies: List[StrategyThresholds],
    min_expected_pnl_pct: float = 0.0,
    fee_pct: float = 0.0008,
) -> List[StrategyThresholds]:
    """Filter strategies to only those with positive expected PnL.

    Args:
        strategies: List of strategies to filter
        min_expected_pnl_pct: Minimum expected PnL percentage
        fee_pct: Fee percentage per side

    Returns:
        Filtered list of strategies
    """
    filtered = []
    for strategy in strategies:
        # Simple expected PnL: spread minus fees
        expected_pnl = strategy.total_spread_pct - (2 * fee_pct)
        if expected_pnl >= min_expected_pnl_pct:
            filtered.append(strategy)

    return filtered


def rank_strategies_by_spread(
    strategies: List[StrategyThresholds],
    ascending: bool = False,
) -> List[StrategyThresholds]:
    """Rank strategies by their total spread.

    Args:
        strategies: List of strategies
        ascending: If True, smallest spread first

    Returns:
        Sorted list of strategies
    """
    return sorted(
        strategies,
        key=lambda s: s.total_spread_pct,
        reverse=not ascending,
    )
