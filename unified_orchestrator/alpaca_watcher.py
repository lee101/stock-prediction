"""Intra-hour order watcher for Alpaca crypto trading.

Monitors paired buy+sell limit orders and oscillates between them:
- When a buy fills → place the corresponding TP sell
- When a sell fills → place a new buy if time remains in the hour
- Self-terminates at hour boundary or after max oscillations

Usage:
    watcher = AlpacaCryptoWatcher(alpaca_client, pairs, expiry_minutes=55)
    watcher.start()  # background thread
    ...
    watcher.stop()   # signal graceful shutdown
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class OrderPair:
    """A paired buy+sell order for one symbol."""
    symbol: str
    buy_price: float
    sell_price: float
    target_qty: float
    current_qty: float = 0.0
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    oscillations: int = 0
    max_oscillations: int = 5


class AlpacaCryptoWatcher(threading.Thread):
    """Background thread that monitors and oscillates paired crypto orders.

    Polls Alpaca order status every ``poll_interval`` seconds.  When a buy
    order fills, a take-profit sell is placed at ``pair.sell_price``.  When
    the sell fills, a new buy is placed if time remains before expiry.

    The thread terminates when:
    - ``stop()`` is called (sets the stop event)
    - ``expiry_minutes`` have elapsed since start
    - All pairs have completed their max oscillations
    """

    def __init__(
        self,
        alpaca_client,
        pairs: list[OrderPair],
        expiry_minutes: int = 55,
        poll_interval: int = 30,
        dry_run: bool = False,
    ) -> None:
        super().__init__(daemon=True, name="AlpacaCryptoWatcher")
        if alpaca_client is None:
            from alpaca.trading.client import TradingClient
            from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
            alpaca_client = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)
        self.alpaca = alpaca_client
        self.pairs = {p.symbol: p for p in pairs}
        self.expiry_minutes = expiry_minutes
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self.fill_log: list[dict] = []

    def stop(self) -> None:
        """Signal the watcher to shut down gracefully."""
        self._stop_event.set()

    @property
    def is_expired(self) -> bool:
        if self._start_time == 0:
            return False
        elapsed = (time.time() - self._start_time) / 60.0
        return elapsed >= self.expiry_minutes

    def run(self) -> None:
        self._start_time = time.time()
        logger.info(f"Watcher started: {len(self.pairs)} pairs, expiry={self.expiry_minutes}m")

        while not self._stop_event.is_set() and not self.is_expired:
            try:
                self._poll_orders()
            except Exception as e:
                logger.error(f"Watcher poll error: {e}")

            # Check if all pairs are done
            if all(p.oscillations >= p.max_oscillations for p in self.pairs.values()):
                logger.info("Watcher: all pairs completed max oscillations")
                break

            self._stop_event.wait(timeout=self.poll_interval)

        self._cancel_all_open()
        logger.info(f"Watcher stopped: {len(self.fill_log)} fills logged")

    def _poll_orders(self) -> None:
        """Check order statuses and react to fills."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        # Get all open and recently closed orders
        try:
            open_orders = self.alpaca.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception as e:
            logger.debug(f"Watcher: error fetching orders: {e}")
            return

        open_ids = {str(o.id) for o in open_orders}

        for sym, pair in self.pairs.items():
            if pair.oscillations >= pair.max_oscillations:
                continue

            # Check if buy order filled (was submitted but no longer open)
            if pair.buy_order_id and pair.buy_order_id not in open_ids:
                position_qty = self._get_position_qty(sym)
                if position_qty is None:
                    logger.warning(
                        f"Watcher: {sym} buy order {pair.buy_order_id} disappeared, "
                        "but position state is unavailable; deferring resolution"
                    )
                    continue
                if position_qty > 0.0:
                    fill_qty = float(position_qty)
                    logger.info(f"Watcher: {sym} buy filled @ ${pair.buy_price:.2f} qty={fill_qty:.8f}")
                    self.fill_log.append({
                        "symbol": sym, "side": "buy", "price": pair.buy_price,
                        "qty": fill_qty, "oscillation": pair.oscillations,
                    })
                    pair.current_qty = fill_qty
                    pair.buy_order_id = None
                    self._on_buy_fill(pair)
                else:
                    # Order was canceled or rejected, not filled
                    pair.buy_order_id = None

            # Check if sell order filled (skip if just placed by _on_buy_fill above)
            elif pair.sell_order_id and pair.sell_order_id not in open_ids:
                position_qty = self._get_position_qty(sym)
                if position_qty is None:
                    logger.warning(
                        f"Watcher: {sym} sell order {pair.sell_order_id} disappeared, "
                        "but position state is unavailable; deferring resolution"
                    )
                    continue
                if position_qty <= 0.0:
                    logger.info(f"Watcher: {sym} sell filled @ ${pair.sell_price:.2f}")
                    self.fill_log.append({
                        "symbol": sym, "side": "sell", "price": pair.sell_price,
                        "qty": pair.current_qty, "oscillation": pair.oscillations,
                    })
                    pair.current_qty = 0.0
                    pair.sell_order_id = None
                    pair.oscillations += 1
                    self._on_sell_fill(pair)
                else:
                    # Still has position — sell not fully filled, or it was canceled externally.
                    if pair.current_qty > 0.0 and position_qty < pair.current_qty:
                        logger.info(
                            f"Watcher: {sym} partial sell detected, remaining qty={position_qty:.8f}"
                        )
                        pair.current_qty = float(position_qty)
                    pair.sell_order_id = None

    def _check_position_exists(self, symbol: str) -> bool:
        """Check if we currently hold a position in this symbol.

        Returns True if a non-zero position exists.  On API error, returns
        False conservatively (do not assume a position exists) and logs the
        error so it is visible in production logs.
        """
        qty = self._get_position_qty(symbol)
        return bool(qty and qty > 0.0)

    def _get_position_qty(self, symbol: str) -> float | None:
        """Return current held quantity for a symbol.

        Returns:
            float: Current quantity, or ``0.0`` when no position exists.
            None: Position state could not be determined due to an API error.
        """
        try:
            positions = self.alpaca.get_all_positions()
            for pos in positions:
                sym_norm = str(pos.symbol).replace("/", "")
                if sym_norm == symbol:
                    qty = float(pos.qty)
                    return qty if qty > 0.0 else 0.0
        except Exception as e:
            logger.warning(f"Watcher: _get_position_qty({symbol}) API error: {e}")
            return None
        return 0.0

    def _on_buy_fill(self, pair: OrderPair) -> None:
        """React to a buy fill: place the TP sell order."""
        sell_qty = math.floor(pair.current_qty * 1e8 - 1) / 1e8
        if sell_qty <= 0:
            return

        logger.info(f"Watcher: {pair.symbol} placing TP sell {sell_qty:.8f} @ ${pair.sell_price:.2f}")
        if self.dry_run:
            pair.sell_order_id = f"dry-sell-{pair.symbol}-{pair.oscillations}"
            return

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req = LimitOrderRequest(
                symbol=pair.symbol,
                qty=round(sell_qty, 8),
                side=OrderSide.SELL,
                type="limit",
                time_in_force=TimeInForce.GTC,
                limit_price=round(pair.sell_price, 2),
            )
            result = self.alpaca.submit_order(req)
            pair.sell_order_id = str(result.id)
        except Exception as e:
            logger.error(f"Watcher: {pair.symbol} sell order error: {e}")

    def _on_sell_fill(self, pair: OrderPair) -> None:
        """React to a sell fill: place a new buy if time remains."""
        if self.is_expired or pair.oscillations >= pair.max_oscillations:
            logger.info(f"Watcher: {pair.symbol} sell filled, not re-entering (expired or max osc)")
            return

        logger.info(
            f"Watcher: {pair.symbol} re-entering buy @ ${pair.buy_price:.2f} "
            f"(osc {pair.oscillations}/{pair.max_oscillations})"
        )
        if self.dry_run:
            pair.buy_order_id = f"dry-buy-{pair.symbol}-{pair.oscillations}"
            return

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req = LimitOrderRequest(
                symbol=pair.symbol,
                qty=round(pair.target_qty, 8),
                side=OrderSide.BUY,
                type="limit",
                time_in_force=TimeInForce.GTC,
                limit_price=round(pair.buy_price, 2),
            )
            result = self.alpaca.submit_order(req)
            pair.buy_order_id = str(result.id)
        except Exception as e:
            logger.error(f"Watcher: {pair.symbol} re-buy error: {e}")

    def _cancel_all_open(self) -> None:
        """Cancel all open orders managed by this watcher."""
        for sym, pair in self.pairs.items():
            for order_id in [pair.buy_order_id, pair.sell_order_id]:
                if order_id and not order_id.startswith("dry-"):
                    try:
                        self.alpaca.cancel_order_by_id(order_id)
                        logger.info(f"Watcher: canceled {sym} order {order_id}")
                    except Exception as e:
                        logger.debug(f"Watcher: cancel error for {order_id}: {e}")
            pair.buy_order_id = None
            pair.sell_order_id = None
