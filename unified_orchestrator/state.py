"""Unified portfolio state across Alpaca (stocks) and Binance (crypto).

Builds a read-only snapshot each cycle from both broker APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

NEW_YORK = ZoneInfo("America/New_York")

# Market hours (NYSE)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_START = time(9, 1)


@dataclass
class Position:
    """A position on either broker."""
    symbol: str
    qty: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    broker: str = ""  # "alpaca" or "binance"

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price if self.current_price else self.qty * self.avg_price


@dataclass
class PendingOrder:
    """A pending limit order on either broker."""
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    limit_price: float
    broker: str = ""
    order_id: str = ""


@dataclass
class UnifiedPortfolioSnapshot:
    """Full portfolio state across both brokers."""
    # Alpaca
    alpaca_cash: float = 0.0
    alpaca_buying_power: float = 0.0
    alpaca_positions: dict[str, Position] = field(default_factory=dict)
    alpaca_pending_orders: list[PendingOrder] = field(default_factory=list)

    # Binance
    binance_fdusd: float = 0.0
    binance_usdt: float = 0.0
    binance_positions: dict[str, Position] = field(default_factory=dict)
    binance_pending_orders: list[PendingOrder] = field(default_factory=list)

    # Market context
    market_is_open: bool = False
    minutes_to_open: Optional[int] = None
    minutes_to_close: Optional[int] = None
    regime: str = "CRYPTO_ONLY"
    timestamp: str = ""

    @property
    def total_stock_value(self) -> float:
        return self.alpaca_cash + sum(p.market_value for p in self.alpaca_positions.values())

    @property
    def total_crypto_value(self) -> float:
        return (self.binance_fdusd + self.binance_usdt +
                sum(p.market_value for p in self.binance_positions.values()))

    @property
    def total_value(self) -> float:
        return self.total_stock_value + self.total_crypto_value


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------

def _is_nyse_weekday(dt: datetime) -> bool:
    """Check if date is Mon-Fri (doesn't check holidays)."""
    return dt.weekday() < 5


def determine_regime(now_utc: datetime) -> tuple[str, Optional[int], Optional[int]]:
    """Determine current market regime.

    Returns (regime, minutes_to_open, minutes_to_close).
    """
    ny = now_utc.astimezone(NEW_YORK)
    ny_time = ny.time()

    if not _is_nyse_weekday(ny):
        return "CRYPTO_ONLY", None, None

    market_open_dt = ny.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_dt = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    pre_market_dt = ny.replace(hour=9, minute=1, second=0, microsecond=0)

    mins_to_open = int((market_open_dt - ny).total_seconds() / 60)
    mins_to_close = int((market_close_dt - ny).total_seconds() / 60)

    if PRE_MARKET_START <= ny_time < MARKET_OPEN:
        return "PRE_MARKET", max(0, mins_to_open), None
    elif MARKET_OPEN <= ny_time < MARKET_CLOSE:
        return "STOCK_HOURS", 0, max(0, mins_to_close)
    elif time(16, 0) <= ny_time <= time(16, 5):
        return "POST_MARKET", None, 0
    else:
        # Calculate next open
        if ny_time >= MARKET_CLOSE:
            # Next day
            next_open_mins = mins_to_open + 24 * 60 if mins_to_open < 0 else None
        else:
            next_open_mins = max(0, mins_to_open) if mins_to_open > 0 else None
        return "CRYPTO_ONLY", next_open_mins, None


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------

def build_alpaca_snapshot(snapshot: UnifiedPortfolioSnapshot) -> None:
    """Populate Alpaca fields in the snapshot."""
    try:
        from alpaca.trading.client import TradingClient
        from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

        client = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

        # Account
        account = client.get_account()
        snapshot.alpaca_cash = float(account.cash)
        snapshot.alpaca_buying_power = float(account.buying_power)

        # Positions
        positions = client.get_all_positions()
        for pos in positions:
            snapshot.alpaca_positions[pos.symbol] = Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
                broker="alpaca",
            )

        # Open orders
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in orders:
            snapshot.alpaca_pending_orders.append(PendingOrder(
                symbol=order.symbol,
                side=order.side.value.lower(),
                qty=float(order.qty),
                limit_price=float(order.limit_price) if order.limit_price else 0.0,
                broker="alpaca",
                order_id=str(order.id),
            ))

    except Exception as e:
        logger.error(f"Failed to build Alpaca snapshot: {e}")


def build_binance_snapshot(snapshot: UnifiedPortfolioSnapshot) -> None:
    """Populate Binance fields in the snapshot."""
    try:
        from src.binan import binance_wrapper

        snapshot.binance_fdusd = binance_wrapper.get_asset_free_balance("FDUSD") or 0.0
        snapshot.binance_usdt = binance_wrapper.get_asset_free_balance("USDT") or 0.0

        # Positions - check each traded crypto
        CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "SUI", "AAVE"]
        for asset in CRYPTO_ASSETS:
            bal = binance_wrapper.get_asset_free_balance(asset) or 0.0
            if bal > 0:
                # Get current price
                pair = f"{asset}USDT"
                try:
                    price = float(binance_wrapper.get_symbol_price(pair))
                except Exception:
                    price = 0.0
                snapshot.binance_positions[asset] = Position(
                    symbol=f"{asset}USD",
                    qty=bal,
                    avg_price=0.0,  # Binance doesn't track avg entry
                    current_price=price,
                    unrealized_pnl=0.0,
                    broker="binance",
                )

        # Open orders
        try:
            open_orders = binance_wrapper.get_open_orders()
            for order in (open_orders or []):
                snapshot.binance_pending_orders.append(PendingOrder(
                    symbol=order.get("symbol", ""),
                    side=order.get("side", "").lower(),
                    qty=float(order.get("origQty", 0)),
                    limit_price=float(order.get("price", 0)),
                    broker="binance",
                    order_id=str(order.get("orderId", "")),
                ))
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Failed to build Binance snapshot: {e}")


def build_snapshot(now_utc: Optional[datetime] = None) -> UnifiedPortfolioSnapshot:
    """Build full unified snapshot from both brokers."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    regime, mins_to_open, mins_to_close = determine_regime(now_utc)

    snapshot = UnifiedPortfolioSnapshot(
        market_is_open=regime == "STOCK_HOURS",
        minutes_to_open=mins_to_open,
        minutes_to_close=mins_to_close,
        regime=regime,
        timestamp=now_utc.isoformat(),
    )

    build_alpaca_snapshot(snapshot)
    build_binance_snapshot(snapshot)

    return snapshot


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

STATE_FILE = Path("strategy_state/unified_state.json")


def save_snapshot(snapshot: UnifiedPortfolioSnapshot) -> None:
    """Persist snapshot to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": snapshot.timestamp,
        "regime": snapshot.regime,
        "alpaca_cash": snapshot.alpaca_cash,
        "alpaca_buying_power": snapshot.alpaca_buying_power,
        "alpaca_positions": {k: asdict(v) for k, v in snapshot.alpaca_positions.items()},
        "binance_fdusd": snapshot.binance_fdusd,
        "binance_usdt": snapshot.binance_usdt,
        "binance_positions": {k: asdict(v) for k, v in snapshot.binance_positions.items()},
        "market_is_open": snapshot.market_is_open,
        "minutes_to_open": snapshot.minutes_to_open,
        "minutes_to_close": snapshot.minutes_to_close,
        "total_stock_value": snapshot.total_stock_value,
        "total_crypto_value": snapshot.total_crypto_value,
        "total_value": snapshot.total_value,
    }
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(STATE_FILE)
