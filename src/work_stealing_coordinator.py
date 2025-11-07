"""Work stealing coordinator for maxdiff order management."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import alpaca_wrapper
from loguru import logger

from src.work_stealing_config import (
    WORK_STEALING_COOLDOWN_SECONDS,
    WORK_STEALING_DRY_RUN,
    WORK_STEALING_ENABLED,
    WORK_STEALING_ENTRY_TOLERANCE_PCT,
    WORK_STEALING_FIGHT_COOLDOWN_SECONDS,
    WORK_STEALING_FIGHT_THRESHOLD,
    WORK_STEALING_FIGHT_WINDOW_SECONDS,
    WORK_STEALING_PROTECTION_PCT,
)


@dataclass
class OrderCandidate:
    """Candidate order for work stealing."""

    symbol: str
    side: str
    limit_price: float
    qty: float
    notional_value: float
    forecasted_pnl: float
    distance_pct: float  # How far from limit price
    mode: str  # 'probe', 'normal', etc
    order_id: str
    entry_strategy: Optional[str] = None


@dataclass
class StealRecord:
    """Record of a work steal event."""

    timestamp: datetime
    from_symbol: str
    to_symbol: str
    from_order_id: str
    to_forecasted_pnl: float
    from_forecasted_pnl: float


class WorkStealingCoordinator:
    """Coordinates work stealing between maxdiff orders."""

    def __init__(self):
        self._steal_history: List[StealRecord] = []
        self._cooldown_tracker: Dict[str, datetime] = {}  # symbol -> last_steal_time
        self._fight_tracker: Dict[Tuple[str, str], List[datetime]] = {}  # (sym1, sym2) -> steal_times

    def can_open_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
        current_price: Optional[float] = None,
    ) -> bool:
        """Check if we have capacity to open a new order.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            limit_price: Limit price for order
            qty: Quantity
            current_price: Current market price (for distance check)

        Returns:
            True if order can be opened without stealing
        """
        if not WORK_STEALING_ENABLED:
            return True

        needed_notional = abs(qty * limit_price)
        available_capacity = self._get_available_capacity()

        return available_capacity >= needed_notional

    def attempt_steal(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
        current_price: float,
        forecasted_pnl: float,
        mode: str = "normal",
        entry_strategy: Optional[str] = None,
    ) -> Optional[str]:
        """Attempt to steal work from worst order.

        Args:
            symbol: Symbol wanting to enter
            side: Trade side
            limit_price: Desired limit price
            qty: Quantity needed
            current_price: Current market price
            forecasted_pnl: Forecasted PnL for this trade
            mode: Trade mode ('probe', 'normal', etc)
            entry_strategy: Entry strategy name

        Returns:
            Symbol of canceled order if steal successful, None otherwise
        """
        if not WORK_STEALING_ENABLED:
            return None

        # Check if we're close enough to steal
        distance_pct = abs(current_price - limit_price) / limit_price
        if distance_pct > WORK_STEALING_ENTRY_TOLERANCE_PCT:
            logger.debug(
                f"{symbol}: Not close enough to steal (distance={distance_pct:.4f} "
                f"> threshold={WORK_STEALING_ENTRY_TOLERANCE_PCT})"
            )
            return None

        # Check cooldown
        if self._is_on_cooldown(symbol):
            logger.debug(f"{symbol}: On steal cooldown")
            return None

        # Find steal candidates
        candidates = self._get_steal_candidates()
        if not candidates:
            logger.debug(f"{symbol}: No steal candidates available")
            return None

        # Get furthest candidate (already sorted by distance descending)
        furthest_candidate = candidates[0]

        # Steal from furthest order - it's least likely to execute
        # We don't require better PnL because execution likelihood matters more
        # An order 5% away might have great PnL but will likely never fill
        logger.debug(
            f"{symbol}: Evaluating steal from {furthest_candidate.symbol} "
            f"(distance={furthest_candidate.distance_pct:.4f}, PnL={furthest_candidate.forecasted_pnl:.4f}) "
            f"vs mine (distance={distance_pct:.4f}, PnL={forecasted_pnl:.4f})"
        )

        # Check fighting - but allow PnL-based resolution
        if self._would_cause_fight(symbol, furthest_candidate.symbol):
            # If fighting, settle by PnL - better PnL wins
            if forecasted_pnl > furthest_candidate.forecasted_pnl:
                logger.info(
                    f"{symbol}: Fighting with {furthest_candidate.symbol} but PnL better "
                    f"({forecasted_pnl:.4f} > {furthest_candidate.forecasted_pnl:.4f}), allowing steal"
                )
            else:
                logger.warning(
                    f"{symbol}: Fighting with {furthest_candidate.symbol} and PnL not better, aborting steal"
                )
                return None

        # Execute steal
        logger.info(
            f"Work steal: {symbol} (dist={distance_pct:.4f}, PnL={forecasted_pnl:.4f}) stealing from "
            f"{furthest_candidate.symbol} (dist={furthest_candidate.distance_pct:.4f}, PnL={furthest_candidate.forecasted_pnl:.4f})"
        )

        if WORK_STEALING_DRY_RUN:
            logger.info(f"[DRY RUN] Would cancel order {furthest_candidate.order_id} for {furthest_candidate.symbol}")
            return furthest_candidate.symbol

        # Cancel the furthest order
        try:
            alpaca_wrapper.cancel_order(furthest_candidate.order_id)
        except Exception as exc:
            logger.error(f"Failed to cancel order {furthest_candidate.order_id}: {exc}")
            return None

        # Record the steal
        self._record_steal(
            from_symbol=furthest_candidate.symbol,
            to_symbol=symbol,
            from_order_id=furthest_candidate.order_id,
            to_forecasted_pnl=forecasted_pnl,
            from_forecasted_pnl=furthest_candidate.forecasted_pnl,
        )

        return furthest_candidate.symbol

    def is_protected(
        self,
        symbol: str,
        limit_price: float,
        current_price: float,
        mode: str = "normal",
    ) -> bool:
        """Check if an order is protected from work stealing.

        Args:
            symbol: Trading symbol
            limit_price: Order limit price
            current_price: Current market price
            mode: Trade mode

        Returns:
            True if order cannot be stolen
        """
        # Probe trades are always protected
        if mode == "probe":
            return True

        # Check if close to execution
        distance_pct = abs(current_price - limit_price) / limit_price
        if distance_pct <= WORK_STEALING_PROTECTION_PCT:
            return True

        # Check if recently stolen
        if self._is_on_cooldown(symbol):
            return True

        return False

    def get_best_orders_count(self, buying_power: float) -> int:
        """Calculate how many "best" orders should use tight tolerance.

        Args:
            buying_power: Available buying power

        Returns:
            Number of top orders that get tight tolerance
        """
        # With 2x leverage, we can have ~4 concurrent positions
        # Reserve capacity for top performers
        return 4

    def _get_available_capacity(self) -> float:
        """Calculate available buying power capacity.

        Returns:
            Available notional capacity in dollars
        """
        try:
            account = alpaca_wrapper.get_account()
            buying_power = float(getattr(account, "buying_power", 0.0))
        except Exception as exc:
            logger.warning(f"Failed to get account buying power: {exc}")
            return 0.0

        # Get current open orders notional
        try:
            orders = alpaca_wrapper.get_orders()
            open_notional = sum(
                abs(float(getattr(order, "qty", 0.0)) * float(getattr(order, "limit_price", 0.0)))
                for order in orders
                if getattr(order, "limit_price", None) is not None
            )
        except Exception as exc:
            logger.warning(f"Failed to calculate open orders notional: {exc}")
            open_notional = 0.0

        # 2x leverage limit
        max_capacity = buying_power * 2.0
        available = max_capacity - open_notional

        logger.debug(
            f"Capacity: buying_power={buying_power:.2f} "
            f"max={max_capacity:.2f} open={open_notional:.2f} available={available:.2f}"
        )

        return max(0.0, available)

    def _get_steal_candidates(self) -> List[OrderCandidate]:
        """Get list of orders that can be stolen, sorted by distance from limit (furthest first).

        The logic: Orders far from their limit are unlikely to execute, so we steal from
        them first regardless of PnL. An order 5% from limit with great PnL is less
        valuable than an order 0.5% from limit with mediocre PnL, because the close
        one will actually execute.

        Returns:
            List of candidate orders sorted by distance_pct descending (furthest first),
            with PnL as tiebreaker for similar distances
        """
        candidates = []

        try:
            orders = alpaca_wrapper.get_orders()
        except Exception as exc:
            logger.error(f"Failed to fetch orders: {exc}")
            return []

        # Load forecast data for PnL estimates
        try:
            from trade_stock_e2e import _load_latest_forecast_snapshot

            forecast_data = _load_latest_forecast_snapshot()
        except Exception as exc:
            logger.warning(f"Failed to load forecast data: {exc}")
            forecast_data = {}

        for order in orders:
            symbol = getattr(order, "symbol", None)
            if not symbol:
                continue

            limit_price = getattr(order, "limit_price", None)
            if limit_price is None:
                continue  # Not a limit order

            try:
                limit_price = float(limit_price)
                qty = float(getattr(order, "qty", 0.0))
            except (TypeError, ValueError):
                continue

            # Get current price
            try:
                current_price = float(getattr(order, "current_price", limit_price))
            except (TypeError, ValueError):
                current_price = limit_price

            # Determine mode (check if probe trade)
            # We'd need to query active_trades store for this
            # For now, assume normal unless notional is very small
            notional_value = abs(qty * limit_price)
            mode = "probe" if notional_value < 500 else "normal"

            # Check if protected
            if self.is_protected(symbol, limit_price, current_price, mode):
                continue

            # Get forecasted PnL
            forecast = forecast_data.get(symbol, {})
            forecasted_pnl = self._extract_forecasted_pnl(forecast)

            # Calculate distance
            distance_pct = abs(current_price - limit_price) / limit_price

            candidates.append(
                OrderCandidate(
                    symbol=symbol,
                    side=getattr(order, "side", "buy"),
                    limit_price=limit_price,
                    qty=qty,
                    notional_value=notional_value,
                    forecasted_pnl=forecasted_pnl,
                    distance_pct=distance_pct,
                    mode=mode,
                    order_id=getattr(order, "id", str(order)),
                    entry_strategy=None,  # Would need to load from active_trades
                )
            )

        # Sort by distance from limit (furthest first = highest distance_pct)
        # Use PnL as tiebreaker for orders at similar distances
        # Furthest orders are least likely to execute, so steal from them first
        candidates.sort(key=lambda c: (-c.distance_pct, c.forecasted_pnl))

        return candidates

    def _extract_forecasted_pnl(self, forecast: Dict) -> float:
        """Extract forecasted PnL from forecast data.

        Args:
            forecast: Forecast dict for symbol

        Returns:
            Forecasted PnL value (default 0.0)
        """
        # Try various PnL fields
        for field in [
            "maxdiff_forecasted_pnl",
            "maxdiffalwayson_forecasted_pnl",
            "highlow_forecasted_pnl",
            "all_signals_forecasted_pnl",
            "avg_return",
        ]:
            value = forecast.get(field)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on steal cooldown.

        Args:
            symbol: Trading symbol

        Returns:
            True if on cooldown
        """
        last_steal = self._cooldown_tracker.get(symbol)
        if last_steal is None:
            return False

        # Check if fighting cooldown applies
        fighting_cooldown = self._get_fighting_cooldown(symbol)
        cooldown_seconds = max(WORK_STEALING_COOLDOWN_SECONDS, fighting_cooldown)

        elapsed = (datetime.now(timezone.utc) - last_steal).total_seconds()
        return elapsed < cooldown_seconds

    def _get_fighting_cooldown(self, symbol: str) -> int:
        """Get extended cooldown if symbol involved in fighting.

        Args:
            symbol: Trading symbol

        Returns:
            Cooldown seconds (0 if not fighting)
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=WORK_STEALING_FIGHT_WINDOW_SECONDS)

        # Check all fight pairs involving this symbol
        for (sym1, sym2), steal_times in self._fight_tracker.items():
            if symbol not in (sym1, sym2):
                continue

            # Count recent steals
            recent_steals = [t for t in steal_times if t >= cutoff]
            if len(recent_steals) >= WORK_STEALING_FIGHT_THRESHOLD:
                return WORK_STEALING_FIGHT_COOLDOWN_SECONDS

        return 0

    def _would_cause_fight(self, from_symbol: str, to_symbol: str) -> bool:
        """Check if steal would cause a fight.

        Args:
            from_symbol: Symbol attempting steal
            to_symbol: Symbol being stolen from

        Returns:
            True if this would be considered fighting
        """
        pair = tuple(sorted([from_symbol, to_symbol]))
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=WORK_STEALING_FIGHT_WINDOW_SECONDS)

        steal_times = self._fight_tracker.get(pair, [])
        recent_steals = [t for t in steal_times if t >= cutoff]

        return len(recent_steals) >= WORK_STEALING_FIGHT_THRESHOLD - 1

    def _record_steal(
        self,
        from_symbol: str,
        to_symbol: str,
        from_order_id: str,
        to_forecasted_pnl: float,
        from_forecasted_pnl: float,
    ) -> None:
        """Record a work steal event.

        Args:
            from_symbol: Symbol stolen from
            to_symbol: Symbol that stole
            from_order_id: Canceled order ID
            to_forecasted_pnl: PnL forecast for new order
            from_forecasted_pnl: PnL forecast for canceled order
        """
        now = datetime.now(timezone.utc)

        # Add to history
        self._steal_history.append(
            StealRecord(
                timestamp=now,
                from_symbol=from_symbol,
                to_symbol=to_symbol,
                from_order_id=from_order_id,
                to_forecasted_pnl=to_forecasted_pnl,
                from_forecasted_pnl=from_forecasted_pnl,
            )
        )

        # Update cooldown
        self._cooldown_tracker[from_symbol] = now

        # Track fighting
        pair = tuple(sorted([from_symbol, to_symbol]))
        if pair not in self._fight_tracker:
            self._fight_tracker[pair] = []
        self._fight_tracker[pair].append(now)

        # Cleanup old fight records
        self._cleanup_fight_tracker()

    def _cleanup_fight_tracker(self) -> None:
        """Remove old fight records outside the window."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=WORK_STEALING_FIGHT_WINDOW_SECONDS * 2)

        for pair in list(self._fight_tracker.keys()):
            self._fight_tracker[pair] = [t for t in self._fight_tracker[pair] if t >= cutoff]
            if not self._fight_tracker[pair]:
                del self._fight_tracker[pair]


# Global coordinator instance
_coordinator: Optional[WorkStealingCoordinator] = None


def get_coordinator() -> WorkStealingCoordinator:
    """Get or create the global work stealing coordinator.

    Returns:
        WorkStealingCoordinator instance
    """
    global _coordinator
    if _coordinator is None:
        _coordinator = WorkStealingCoordinator()
    return _coordinator
