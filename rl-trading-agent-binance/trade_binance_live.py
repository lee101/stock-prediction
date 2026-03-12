"""
Production Binance Trading - RL+LLM Hybrid System.

Manages:
- Spot execution on FDUSD pairs for BTC/ETH and USDT pairs for altcoins
- Cross-margin execution on USDT pairs with Binance borrow/auto-repay side effects
- Automatic consolidation of tracked spot assets into cross margin when margin mode is active
- Automatic stablecoin swaps between FDUSD ↔ USDT for spot mode
- Hourly trading cycle with RL or LLM signals

Modes:
  --rl-checkpoint PATH   Use RL policy as primary signal (portfolio rotator)
  (default)              Use LLM signals per symbol

Usage:
  python rl-trading-agent-binance/trade_binance_live.py --dry-run
  python rl-trading-agent-binance/trade_binance_live.py --live --rl-checkpoint rl-trainingbinance/checkpoints/autoresearch_ema.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from src.binan import binance_wrapper
from src.binan.binance_margin import (
    cancel_margin_order,
    create_margin_order,
    get_margin_account,
    get_margin_asset_balance,
    get_margin_borrowed_balance,
    get_margin_free_balance,
    get_margin_trades,
    get_max_borrowable,
    get_open_margin_orders,
    margin_repay_all,
    transfer_margin_to_spot,
    transfer_spot_to_margin,
)
from src.binan.binance_conversion import (
    build_stable_quote_conversion_plan,
    execute_stable_quote_conversion,
)
from binanceneural.execution import (
    resolve_symbol_rules,
    quantize_qty,
    quantize_price,
    split_binance_symbol,
)
from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.gemini_wrapper import TradePlan

from rl_signal import RLSignalGenerator, PortfolioSnapshot, SYMBOLS as RL_SYMBOLS
from hybrid_prompt import (
    gather_symbol_contexts,
    build_allocation_prompt,
    call_gemini_allocation,
    AllocationPlan,
    PlanOutcome,
    SYMBOL_BINANCE_MAP,
)


# ---------------------------------------------------------------------------
# Symbol Configuration
# ---------------------------------------------------------------------------

@dataclass
class BinanceSymbolConfig:
    """Config for each tradeable symbol."""
    symbol: str           # Internal name (e.g. "BTCUSD")
    binance_pair: str     # Binance trading pair (e.g. "BTCFDUSD")
    quote_asset: str      # "FDUSD" or "USDT"
    base_asset: str       # "BTC", "ETH", etc.
    maker_fee: float      # 0.0 for FDUSD, 0.001 for USDT
    max_position_pct: float = 0.20  # max % of portfolio in this symbol


# BTC/ETH trade on FDUSD (zero maker fees)
# Altcoins trade on USDT
TRADING_SYMBOLS = {
    "BTCUSD": BinanceSymbolConfig("BTCUSD", "BTCFDUSD", "FDUSD", "BTC", 0.0, 0.25),
    "ETHUSD": BinanceSymbolConfig("ETHUSD", "ETHFDUSD", "FDUSD", "ETH", 0.0, 0.20),
    "SOLUSD": BinanceSymbolConfig("SOLUSD", "SOLUSDT", "USDT", "SOL", 0.001, 0.15),
    "DOGEUSD": BinanceSymbolConfig("DOGEUSD", "DOGEUSDT", "USDT", "DOGE", 0.001, 0.10),
    "SUIUSD": BinanceSymbolConfig("SUIUSD", "SUIUSDT", "USDT", "SUI", 0.001, 0.15),
    "AAVEUSD": BinanceSymbolConfig("AAVEUSD", "AAVEUSDT", "USDT", "AAVE", 0.001, 0.15),
}

# Minimum trade sizes
MIN_TRADE_USD = 12.0  # Binance min notional is typically $10
SUPPORTED_EXECUTION_MODES = {"auto", "spot", "margin"}
ACCOUNT_TRANSFER_RESERVES = {
    "USDT": 1e-6,
    "FDUSD": 1e-6,
    "BTC": 1e-8,
    "ETH": 1e-8,
    "SOL": 1e-8,
    "DOGE": 1e-5,
    "SUI": 1e-5,
    "AAVE": 1e-8,
}


@dataclass(frozen=True)
class AccountTransfer:
    asset: str
    amount: float


@dataclass(frozen=True)
class MarginCapitalSyncPlan:
    spot_to_margin: tuple[AccountTransfer, ...] = ()
    margin_to_spot: tuple[AccountTransfer, ...] = ()
    spot_fdusd_to_usdt: float = 0.0
    transfer_all_spot_usdt_to_margin: bool = False

    def has_actions(self) -> bool:
        return bool(
            self.spot_to_margin
            or self.margin_to_spot
            or self.spot_fdusd_to_usdt > 0.0
            or self.transfer_all_spot_usdt_to_margin
        )


# ---------------------------------------------------------------------------
# Portfolio State
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Track current portfolio state."""
    fdusd_balance: float = 0.0
    usdt_balance: float = 0.0
    borrowed_quotes: dict = field(default_factory=dict)  # {quote_asset: borrowed_qty}
    borrowable_quotes: dict = field(default_factory=dict)  # {quote_asset: extra borrow headroom}
    positions: dict = field(default_factory=dict)  # {base_asset: qty}
    total_value_usd: float = 0.0

    def available_quote(self, quote_asset: str) -> float:
        if quote_asset == "FDUSD":
            return self.fdusd_balance
        return self.usdt_balance


def _tracked_base_assets() -> tuple[str, ...]:
    return tuple(sorted({cfg.base_asset for cfg in TRADING_SYMBOLS.values()}))


def _transfer_reserve(asset: str) -> float:
    return ACCOUNT_TRANSFER_RESERVES.get(asset.upper(), 1e-8)


def _transferable_balance(asset: str, free_balance: float) -> float:
    return max(0.0, float(free_balance) - _transfer_reserve(asset))


def _build_margin_capital_sync_plan(
    spot_free: dict[str, float],
    margin_free: dict[str, float],
) -> MarginCapitalSyncPlan:
    spot_to_margin: list[AccountTransfer] = []
    for asset in _tracked_base_assets():
        amount = _transferable_balance(asset, spot_free.get(asset, 0.0))
        if amount > 0.0:
            spot_to_margin.append(AccountTransfer(asset=asset, amount=amount))

    spot_fdusd = _transferable_balance("FDUSD", spot_free.get("FDUSD", 0.0))
    margin_fdusd = _transferable_balance("FDUSD", margin_free.get("FDUSD", 0.0))
    margin_to_spot: list[AccountTransfer] = []
    spot_fdusd_to_usdt = 0.0
    if spot_fdusd + margin_fdusd >= MIN_TRADE_USD:
        if margin_fdusd > 0.0:
            margin_to_spot.append(AccountTransfer(asset="FDUSD", amount=margin_fdusd))
        spot_fdusd_to_usdt = spot_fdusd + margin_fdusd

    transferable_spot_usdt = _transferable_balance("USDT", spot_free.get("USDT", 0.0))
    transfer_all_spot_usdt_to_margin = transferable_spot_usdt > 0.0 or spot_fdusd_to_usdt >= MIN_TRADE_USD
    return MarginCapitalSyncPlan(
        spot_to_margin=tuple(spot_to_margin),
        margin_to_spot=tuple(margin_to_spot),
        spot_fdusd_to_usdt=spot_fdusd_to_usdt,
        transfer_all_spot_usdt_to_margin=transfer_all_spot_usdt_to_margin,
    )


def _sync_margin_capital(dry_run: bool) -> MarginCapitalSyncPlan:
    tracked_assets = ("USDT", "FDUSD", *_tracked_base_assets())
    spot_free = {
        asset: float(binance_wrapper.get_asset_free_balance(asset) or 0.0)
        for asset in tracked_assets
    }
    margin_free = {
        asset: float(get_margin_free_balance(asset) or 0.0)
        for asset in tracked_assets
    }
    plan = _build_margin_capital_sync_plan(spot_free, margin_free)
    if not plan.has_actions():
        return plan

    logger.info("Preparing capital for cross-margin execution")
    for transfer in plan.spot_to_margin:
        logger.info(f"  Spot -> margin: {transfer.amount:.8f} {transfer.asset}")
        if dry_run:
            continue
        transfer_spot_to_margin(transfer.asset, transfer.amount)

    for transfer in plan.margin_to_spot:
        logger.info(f"  Margin -> spot: {transfer.amount:.8f} {transfer.asset}")
        if dry_run:
            continue
        transfer_margin_to_spot(transfer.asset, transfer.amount)

    if plan.spot_fdusd_to_usdt >= MIN_TRADE_USD:
        logger.info(f"  Spot conversion: {plan.spot_fdusd_to_usdt:.2f} FDUSD -> USDT")
        if not dry_run:
            conversion = build_stable_quote_conversion_plan(
                from_asset="FDUSD",
                to_asset="USDT",
                amount=plan.spot_fdusd_to_usdt,
                available_pairs=["FDUSDUSDT"],
            )
            if conversion is None:
                raise RuntimeError("Could not build FDUSD -> USDT conversion plan for margin capital sync.")
            execute_stable_quote_conversion(conversion, dry_run=False)

    if plan.transfer_all_spot_usdt_to_margin:
        if dry_run:
            logger.info("  Spot -> margin: all transferable USDT after conversion")
        else:
            transferable_usdt = _transferable_balance(
                "USDT",
                float(binance_wrapper.get_asset_free_balance("USDT") or 0.0),
            )
            if transferable_usdt > 0.0:
                logger.info(f"  Spot -> margin: {transferable_usdt:.8f} USDT")
                transfer_spot_to_margin("USDT", transferable_usdt)

    return plan


def _execution_pair(sym_cfg: BinanceSymbolConfig, execution_mode: str) -> str:
    if execution_mode == "margin" and sym_cfg.quote_asset == "FDUSD":
        return f"{sym_cfg.base_asset}USDT"
    return sym_cfg.binance_pair


def _execution_quote_asset(sym_cfg: BinanceSymbolConfig, execution_mode: str) -> str:
    if execution_mode == "margin":
        return "USDT"
    return sym_cfg.quote_asset


def _execution_fee_bps(sym_cfg: BinanceSymbolConfig, execution_mode: str) -> int:
    fee = 0.001 if execution_mode == "margin" else sym_cfg.maker_fee
    return int(round(fee * 10000))


def _resolve_execution_mode(execution_mode: str, leverage: float) -> str:
    normalized = str(execution_mode or "auto").strip().lower()
    if normalized not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported execution mode {execution_mode!r}; "
            f"expected one of {sorted(SUPPORTED_EXECUTION_MODES)}."
        )
    if normalized == "auto":
        return "margin" if float(leverage) > 1.0 + 1e-9 else "spot"
    return normalized


def _effective_leverage(execution_mode: str, leverage: float) -> float:
    requested = max(0.0, float(leverage))
    if execution_mode == "spot":
        return _resolve_spot_leverage(requested)
    return requested


def _get_market_price(sym_cfg: BinanceSymbolConfig, execution_mode: str) -> float:
    return float(binance_wrapper.get_symbol_price(_execution_pair(sym_cfg, execution_mode)))


def get_portfolio_state(execution_mode: str = "spot") -> PortfolioState:
    """Fetch current portfolio from Binance spot or margin."""
    state = PortfolioState()
    if execution_mode == "margin":
        state.usdt_balance = get_margin_free_balance("USDT")
        state.borrowed_quotes["USDT"] = get_margin_borrowed_balance("USDT")
        state.borrowable_quotes["USDT"] = get_max_borrowable("USDT")

        margin_account = get_margin_account()
        try:
            total_net_btc = float(margin_account.get("totalNetAssetOfBtc", 0.0))
            btc_price = float(binance_wrapper.get_symbol_price("BTCUSDT"))
            state.total_value_usd = total_net_btc * btc_price
        except Exception:
            state.total_value_usd = 0.0

        for cfg in TRADING_SYMBOLS.values():
            asset_entry = get_margin_asset_balance(cfg.base_asset)
            if not asset_entry:
                continue
            qty = float(asset_entry.get("free", 0.0))
            if qty > 0:
                state.positions[cfg.base_asset] = qty
        return state

    state.fdusd_balance = binance_wrapper.get_asset_free_balance("FDUSD") or 0.0
    state.usdt_balance = binance_wrapper.get_asset_free_balance("USDT") or 0.0

    for cfg in TRADING_SYMBOLS.values():
        bal = binance_wrapper.get_asset_free_balance(cfg.base_asset) or 0.0
        if bal > 0:
            state.positions[cfg.base_asset] = bal

    account = binance_wrapper.get_account_value_usdt(include_locked=False)
    state.total_value_usd = account.get("total_usdt", 0.0) if isinstance(account, dict) else 0.0

    return state


def get_position_entry(
    sym_cfg: "BinanceSymbolConfig",
    *,
    execution_mode: str = "spot",
) -> tuple[float, Optional[datetime]]:
    """Get an approximate entry price/time for the active position from recent fills."""
    market_symbol = _execution_pair(sym_cfg, execution_mode)
    if execution_mode == "margin":
        trades = get_margin_trades(market_symbol, limit=20)
    else:
        trades = binance_wrapper.get_my_trades(market_symbol, limit=20)
    if not trades:
        return 0.0, None
    for t in reversed(trades):
        if t.get("isBuyer"):
            price = float(t.get("price", 0))
            ts = t.get("time", 0)
            open_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else None
            return price, open_time
    return 0.0, None


# ---------------------------------------------------------------------------
# Stablecoin Management
# ---------------------------------------------------------------------------

def ensure_quote_balance(
    needed_quote: str,
    needed_amount: float,
    state: PortfolioState,
    dry_run: bool = True,
) -> bool:
    """Ensure we have enough of the right stablecoin, converting if needed."""
    available = state.available_quote(needed_quote)
    if available >= needed_amount:
        return True

    shortfall = needed_amount - available

    # Determine source
    if needed_quote == "USDT":
        source = "FDUSD"
        source_available = state.fdusd_balance
    else:
        source = "USDT"
        source_available = state.usdt_balance

    if source_available < shortfall:
        logger.warning(f"Insufficient {source} ({source_available:.2f}) to convert {shortfall:.2f} to {needed_quote}")
        return False

    logger.info(f"Converting {shortfall:.2f} {source} → {needed_quote}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would convert {shortfall:.2f} {source} → {needed_quote}")
        if source == "FDUSD":
            state.fdusd_balance = max(0.0, state.fdusd_balance - shortfall)
            state.usdt_balance += shortfall
        else:
            state.usdt_balance = max(0.0, state.usdt_balance - shortfall)
            state.fdusd_balance += shortfall
        return True

    try:
        plan = build_stable_quote_conversion_plan(
            from_asset=source,
            to_asset=needed_quote,
            amount=shortfall,
            available_pairs=["FDUSDUSDT"],
        )
        if plan:
            result = execute_stable_quote_conversion(plan, dry_run=False)
            logger.info(f"  Conversion result: {result}")
            # Update state balances after conversion
            state.fdusd_balance = binance_wrapper.get_asset_free_balance("FDUSD") or 0.0
            state.usdt_balance = binance_wrapper.get_asset_free_balance("USDT") or 0.0
            return True
        else:
            logger.error(f"  No conversion plan found for {source} → {needed_quote}")
            return False
    except Exception as e:
        logger.error(f"  Conversion failed: {e}")
        return False


def _estimate_order_notional(order: Optional[dict], fallback: float) -> float:
    """Estimate quote notional reserved by a limit order."""
    if not order:
        return max(0.0, fallback)
    qty = order.get("qty", order.get("origQty", order.get("executedQty", order.get("quantity"))))
    price = order.get("price")
    try:
        notional = float(qty) * float(price)
    except (TypeError, ValueError):
        notional = 0.0
    if notional > 0:
        return notional
    return max(0.0, fallback)


def _reserve_quote_balance(state: PortfolioState, quote_asset: str, amount: float) -> None:
    """Reserve quote balance after placing a buy so later orders do not reuse it."""
    if amount <= 0:
        return
    if quote_asset == "FDUSD":
        state.fdusd_balance = max(0.0, state.fdusd_balance - amount)
    else:
        state.usdt_balance = max(0.0, state.usdt_balance - amount)


def _quote_buying_power(
    state: PortfolioState,
    quote_asset: str,
    *,
    execution_mode: str,
    effective_leverage: float,
) -> float:
    available = state.available_quote(quote_asset)
    if execution_mode != "margin" or effective_leverage <= 1.0 + 1e-9:
        return available
    return available + max(0.0, state.borrowable_quotes.get(quote_asset, 0.0))


def _reserve_buying_power(
    state: PortfolioState,
    quote_asset: str,
    amount: float,
    *,
    execution_mode: str,
) -> None:
    if amount <= 0:
        return
    free_quote = state.available_quote(quote_asset)
    from_free = min(free_quote, amount)
    if from_free > 0:
        _reserve_quote_balance(state, quote_asset, from_free)
    if execution_mode != "margin":
        return
    borrowed_remaining = max(0.0, amount - from_free)
    if borrowed_remaining <= 0:
        return
    current_headroom = max(0.0, state.borrowable_quotes.get(quote_asset, 0.0))
    state.borrowable_quotes[quote_asset] = max(0.0, current_headroom - borrowed_remaining)


def _load_open_orders(execution_mode: str) -> list[dict]:
    if execution_mode == "margin":
        return get_open_margin_orders()
    return binance_wrapper.get_open_orders()


def _cancel_open_order(execution_mode: str, symbol: str, order_id: int) -> None:
    if execution_mode == "margin":
        cancel_margin_order(symbol, order_id=order_id)
    else:
        binance_wrapper.cancel_order(symbol, order_id)


def _order_remaining_qty(order: dict) -> float:
    try:
        orig_qty = float(order.get("origQty", order.get("qty", order.get("quantity", 0.0))))
        executed_qty = float(order.get("executedQty", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, orig_qty - executed_qty)


def _order_remaining_notional(order: dict) -> float:
    try:
        price = float(order.get("price", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return _order_remaining_qty(order) * price


def _matching_open_orders(open_orders: list[dict], symbol: str, side: str) -> list[dict]:
    side_upper = side.upper()
    return [
        order for order in open_orders
        if str(order.get("symbol", "")).upper() == symbol.upper()
        and str(order.get("side", "")).upper() == side_upper
    ]


def _dedupe_side_orders(
    open_orders: list[dict],
    *,
    symbol: str,
    side: str,
    execution_mode: str,
    dry_run: bool,
    desired_qty: float | None = None,
    desired_notional: float | None = None,
) -> tuple[list[dict], bool]:
    matches = _matching_open_orders(open_orders, symbol, side)
    if not matches:
        return open_orders, False

    if desired_qty is not None:
        remaining_qty = sum(_order_remaining_qty(order) for order in matches)
        if remaining_qty >= max(0.0, desired_qty) * 0.98:
            logger.info(f"  Existing {side.upper()} order already covers desired qty on {symbol}")
            return open_orders, True
    if desired_notional is not None:
        remaining_notional = sum(_order_remaining_notional(order) for order in matches)
        if remaining_notional >= max(0.0, desired_notional) * 0.98:
            logger.info(f"  Existing {side.upper()} order already covers desired notional on {symbol}")
            return open_orders, True

    logger.info(f"  Replacing {len(matches)} existing {side.upper()} order(s) on {symbol}")
    for order in matches:
        order_id = order.get("orderId")
        if order_id is None:
            logger.warning(f"  Cannot cancel open order without orderId on {symbol}: {order}")
            return open_orders, True
        if dry_run:
            logger.info(f"    [DRY RUN] Would cancel order {order_id}")
            continue
        try:
            _cancel_open_order(execution_mode, symbol, int(order_id))
        except Exception as exc:
            logger.warning(f"  Failed to cancel open {side.upper()} order {order_id} on {symbol}: {exc}")
            return open_orders, True
    remaining = [order for order in open_orders if order not in matches]
    return remaining, False


def _cancel_stale_open_orders(execution_mode: str, dry_run: bool) -> list[dict]:
    try:
        open_orders = _load_open_orders(execution_mode)
        if not open_orders:
            return []
        logger.info(f"  {len(open_orders)} open {execution_mode} orders found")
        now_ms = int(time.time() * 1000)
        remaining: list[dict] = []
        for order in open_orders:
            placed_ms = order.get("time", order.get("updateTime", now_ms))
            age_hours = (now_ms - placed_ms) / 3600000
            if age_hours <= 2:
                remaining.append(order)
                continue
            if not dry_run:
                _cancel_open_order(execution_mode, order["symbol"], order["orderId"])
            logger.info(f"  Cancelled stale order {order['orderId']} ({age_hours:.1f}h old)")
        return remaining
    except Exception as e:
        logger.warning(f"  Failed to check open orders: {e}")
        return []


def _repay_margin_debt_if_flat(state: PortfolioState, dry_run: bool, execution_mode: str) -> None:
    if execution_mode != "margin":
        return
    for cfg in TRADING_SYMBOLS.values():
        asset_entry = get_margin_asset_balance(cfg.base_asset)
        if not asset_entry:
            continue
        try:
            net_asset = float(asset_entry.get("netAsset", 0.0))
        except (TypeError, ValueError):
            net_asset = 0.0
        if net_asset > 0.00000001:
            return
    borrowed_usdt = max(0.0, state.borrowed_quotes.get("USDT", 0.0))
    if borrowed_usdt <= 0.01:
        return
    logger.info(f"  Flat margin account with borrowed USDT={borrowed_usdt:.4f}; repaying")
    if dry_run:
        return
    try:
        margin_repay_all("USDT")
    except Exception as exc:
        logger.warning(f"  Margin repay failed: {exc}")


def _resolve_spot_leverage(leverage: float) -> float:
    """Clamp the spot execution path to sizes supportable without borrowing."""
    requested = max(0.0, float(leverage))
    effective = min(requested, 1.0)
    if requested > 1.0 + 1e-9:
        logger.warning(
            f"Spot execution does not support borrowed leverage; "
            f"clamping requested leverage {requested:.2f}x to {effective:.2f}x. "
            "Use a margin execution path for real leverage."
        )
    return effective


# ---------------------------------------------------------------------------
# Order Execution
# ---------------------------------------------------------------------------

def place_limit_buy(
    sym_cfg: BinanceSymbolConfig,
    price: float,
    amount_usd: float,
    execution_mode: str = "spot",
    dry_run: bool = True,
) -> Optional[dict]:
    """Place a limit buy order."""
    if execution_mode == "margin":
        return place_margin_limit_buy(sym_cfg, price, amount_usd, dry_run)

    market_symbol = _execution_pair(sym_cfg, execution_mode)
    rules = resolve_symbol_rules(market_symbol)
    qty = amount_usd / price
    qty = quantize_qty(qty, step_size=rules.step_size or 0.00001)
    price = quantize_price(price, tick_size=rules.tick_size or 0.01, side="BUY")

    notional = qty * price
    if notional < MIN_TRADE_USD:
        logger.warning(f"Order too small: {notional:.2f} < {MIN_TRADE_USD}")
        return None

    if rules.min_qty and qty < rules.min_qty:
        logger.warning(f"Qty {qty} below min {rules.min_qty} for {market_symbol}")
        return None

    logger.info(f"BUY {market_symbol}: qty={qty}, price={price}, notional={notional:.2f}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would place limit buy")
        return {"symbol": market_symbol, "side": "BUY", "qty": qty, "price": price, "dry_run": True}

    try:
        order = binance_wrapper.create_order(market_symbol, "BUY", qty, price)
        logger.info(f"  Order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        logger.error(f"  Order failed: {e}")
        return None


def place_limit_sell(
    sym_cfg: BinanceSymbolConfig,
    price: float,
    qty: float,
    execution_mode: str = "spot",
    dry_run: bool = True,
) -> Optional[dict]:
    """Place a limit sell (take-profit) order."""
    if execution_mode == "margin":
        return place_margin_limit_sell(sym_cfg, price, qty, dry_run)

    market_symbol = _execution_pair(sym_cfg, execution_mode)
    rules = resolve_symbol_rules(market_symbol)
    qty = quantize_qty(qty, step_size=rules.step_size or 0.00001)
    price = quantize_price(price, tick_size=rules.tick_size or 0.01, side="SELL")

    notional = qty * price
    if notional < MIN_TRADE_USD:
        logger.warning(f"Sell too small: {notional:.2f} < {MIN_TRADE_USD}")
        return None

    logger.info(f"SELL {market_symbol}: qty={qty}, price={price}, notional={notional:.2f}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would place limit sell")
        return {"symbol": market_symbol, "side": "SELL", "qty": qty, "price": price, "dry_run": True}

    try:
        order = binance_wrapper.create_order(market_symbol, "SELL", qty, price)
        logger.info(f"  Order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        logger.error(f"  Order failed: {e}")
        return None


def place_margin_limit_buy(
    sym_cfg: BinanceSymbolConfig,
    price: float,
    amount_usd: float,
    dry_run: bool = True,
) -> Optional[dict]:
    """Place a cross-margin limit buy that can borrow quote when needed."""
    market_symbol = _execution_pair(sym_cfg, "margin")
    rules = resolve_symbol_rules(market_symbol)
    qty = amount_usd / price
    qty = quantize_qty(qty, step_size=rules.step_size or 0.00001)
    price = quantize_price(price, tick_size=rules.tick_size or 0.01, side="BUY")

    notional = qty * price
    if notional < MIN_TRADE_USD:
        logger.warning(f"Margin order too small: {notional:.2f} < {MIN_TRADE_USD}")
        return None
    if rules.min_qty and qty < rules.min_qty:
        logger.warning(f"Qty {qty} below min {rules.min_qty} for {market_symbol}")
        return None

    logger.info(f"MARGIN BUY {market_symbol}: qty={qty}, price={price}, notional={notional:.2f}")
    if dry_run:
        logger.info("  [DRY RUN] Would place margin limit buy")
        return {"symbol": market_symbol, "side": "BUY", "qty": qty, "price": price, "dry_run": True}

    try:
        order = create_margin_order(
            market_symbol,
            "BUY",
            "LIMIT",
            qty,
            price=price,
            side_effect_type="MARGIN_BUY",
            time_in_force="GTC",
        )
        logger.info(f"  Margin order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        logger.error(f"  Margin order failed: {e}")
        return None


def place_margin_limit_sell(
    sym_cfg: BinanceSymbolConfig,
    price: float,
    qty: float,
    dry_run: bool = True,
) -> Optional[dict]:
    """Place a cross-margin limit sell and auto-repay borrowed quote on fill."""
    market_symbol = _execution_pair(sym_cfg, "margin")
    rules = resolve_symbol_rules(market_symbol)
    qty = quantize_qty(qty, step_size=rules.step_size or 0.00001)
    price = quantize_price(price, tick_size=rules.tick_size or 0.01, side="SELL")

    notional = qty * price
    if notional < MIN_TRADE_USD:
        logger.warning(f"Margin sell too small: {notional:.2f} < {MIN_TRADE_USD}")
        return None
    if rules.min_qty and qty < rules.min_qty:
        logger.warning(f"Qty {qty} below min {rules.min_qty} for {market_symbol}")
        return None

    logger.info(f"MARGIN SELL {market_symbol}: qty={qty}, price={price}, notional={notional:.2f}")
    if dry_run:
        logger.info("  [DRY RUN] Would place margin limit sell")
        return {"symbol": market_symbol, "side": "SELL", "qty": qty, "price": price, "dry_run": True}

    try:
        order = create_margin_order(
            market_symbol,
            "SELL",
            "LIMIT",
            qty,
            price=price,
            side_effect_type="AUTO_REPAY",
            time_in_force="GTC",
        )
        logger.info(f"  Margin order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        logger.error(f"  Margin order failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Hybrid Signal Generation
# ---------------------------------------------------------------------------

def get_hybrid_signal(
    sym_cfg: BinanceSymbolConfig,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    rl_checkpoint: Optional[str] = None,
    position_qty: float = 0.0,
    position_entry_price: float = 0.0,
    position_open_time: Optional[datetime] = None,
    execution_mode: str = "spot",
) -> TradePlan:
    """Get RL+LLM hybrid trading signal for a symbol."""
    import torch
    import torch.nn as nn

    # Get current market data
    market_symbol = _execution_pair(sym_cfg, execution_mode)
    price = binance_wrapper.get_symbol_price(market_symbol)
    if not price:
        return TradePlan("hold", 0, 0, 0, "no price data")

    current_price = float(price)

    # Load recent hourly bars from Binance
    from src.binan import binance_wrapper as bw
    try:
        # Try primary pair, fall back to USDT if FDUSD not available
        pair = market_symbol
        try:
            klines = bw.get_client().get_klines(symbol=pair, interval="1h", limit=72)
        except Exception:
            if execution_mode == "spot" and sym_cfg.quote_asset == "FDUSD":
                pair = sym_cfg.base_asset + "USDT"
                logger.info(f"  Falling back to {pair} (FDUSD not available)")
                klines = bw.get_client().get_klines(symbol=pair, interval="1h", limit=72)
            else:
                raise
        history_rows = []
        for k in klines:
            history_rows.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").isoformat(),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
    except Exception as e:
        logger.error(f"Failed to get klines for {market_symbol}: {e}")
        return TradePlan("hold", 0, 0, 0, f"klines error: {e}")

    if len(history_rows) < 12:
        return TradePlan("hold", 0, 0, 0, "insufficient history")

    # Load Chronos2 forecasts
    from rl_trading_agent_binance_prompt import build_live_prompt, load_latest_forecast
    fc_1h = load_latest_forecast(sym_cfg.symbol, 1)
    fc_24h = load_latest_forecast(sym_cfg.symbol, 24)

    # Build position context
    pos_info = None
    if position_qty > 0 and position_entry_price > 0:
        held_hours = 0.0
        if position_open_time:
            held_hours = (datetime.now(timezone.utc) - position_open_time).total_seconds() / 3600.0
        pos_info = {
            "qty": position_qty,
            "entry_price": position_entry_price,
            "held_hours": held_hours,
        }

    fee_bps = _execution_fee_bps(sym_cfg, execution_mode)
    prompt = build_live_prompt(
        sym_cfg.symbol, history_rows, current_price, fc_1h, fc_24h,
        position_info=pos_info, fee_bps=fee_bps,
    )

    plan = call_llm(prompt, model=model, thinking_level=thinking_level)
    return plan


# ---------------------------------------------------------------------------
# Main Trading Loop
# ---------------------------------------------------------------------------

def run_trading_cycle(
    symbols: list[str],
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    max_position_pct: float = 0.20,
    dry_run: bool = True,
    rl_checkpoint: Optional[str] = None,
    leverage: float = 1.0,
    execution_mode: str = "auto",
):
    """Run one trading cycle across all symbols."""
    resolved_execution_mode = _resolve_execution_mode(execution_mode, leverage)
    effective_leverage = _effective_leverage(resolved_execution_mode, leverage)
    logger.info(f"\n{'='*60}")
    logger.info(f"Trading Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(
        f"Model: {model} (thinking={thinking_level}) | "
        f"Execution: {resolved_execution_mode} | "
        f"Requested leverage: {leverage:.2f}x | Effective leverage: {effective_leverage:.2f}x"
    )
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"{'='*60}")

    # 1. Prepare margin capital, then get portfolio state
    if resolved_execution_mode == "margin":
        _sync_margin_capital(dry_run)
    state = get_portfolio_state(resolved_execution_mode)
    logger.info(
        f"Portfolio: FDUSD={state.fdusd_balance:.2f}, USDT={state.usdt_balance:.2f}, "
        f"borrowable_usdt={state.borrowable_quotes.get('USDT', 0.0):.2f}, total={state.total_value_usd:.2f}"
    )
    for asset, qty in state.positions.items():
        logger.info(f"  Position: {asset} = {qty}")
    _repay_margin_debt_if_flat(state, dry_run, resolved_execution_mode)

    # 2. Cancel stale open orders
    open_orders = _cancel_stale_open_orders(resolved_execution_mode, dry_run)

    # 3. Get signals for each symbol
    orders_placed = []
    for i, sym_name in enumerate(symbols):
        sym_cfg = TRADING_SYMBOLS.get(sym_name)
        if not sym_cfg:
            logger.warning(f"Unknown symbol: {sym_name}")
            continue

        market_symbol = _execution_pair(sym_cfg, resolved_execution_mode)
        trade_quote = _execution_quote_asset(sym_cfg, resolved_execution_mode)
        logger.info(f"\n--- {sym_cfg.symbol} ({market_symbol}) ---")

        # Get current price
        try:
            current_price = _get_market_price(sym_cfg, resolved_execution_mode)
        except Exception:
            logger.error(f"  Cannot get price for {market_symbol}")
            continue

        # Check existing position (ignore dust < $5)
        position_qty = state.positions.get(sym_cfg.base_asset, 0.0)
        position_value = position_qty * current_price
        if position_value < 5.0:
            position_qty = 0.0
            position_value = 0.0
        position_pct = position_value / max(state.total_value_usd, 1.0)

        logger.info(f"  Price: ${current_price:.2f}, Position: {position_qty} ({position_pct:.1%} of portfolio)")

        # Get position entry info for context
        entry_price, open_time = 0.0, None
        if position_qty > 0:
            try:
                entry_price, open_time = get_position_entry(sym_cfg, execution_mode=resolved_execution_mode)
            except Exception:
                pass

        # Get hybrid signal
        try:
            plan = get_hybrid_signal(
                sym_cfg, model, thinking_level, rl_checkpoint,
                position_qty=position_qty,
                position_entry_price=entry_price,
                position_open_time=open_time,
                execution_mode=resolved_execution_mode,
            )
        except Exception as e:
            logger.error(f"  Signal generation failed: {e}")
            continue

        logger.info(f"  Signal: {plan.direction} (conf={plan.confidence:.2f})")
        logger.info(f"  Buy: ${plan.buy_price:.2f}, Sell: ${plan.sell_price:.2f}")
        logger.info(f"  Reasoning: {plan.reasoning[:100]}")

        # 4. Always place take-profit for existing positions
        if position_value >= MIN_TRADE_USD and position_qty > 0:
            sell_price = plan.sell_price if plan.sell_price > current_price else current_price * 1.01
            open_orders, skip_sell = _dedupe_side_orders(
                open_orders,
                symbol=market_symbol,
                side="SELL",
                execution_mode=resolved_execution_mode,
                dry_run=dry_run,
                desired_qty=position_qty,
            )
            if skip_sell:
                logger.info(f"  Existing take-profit sell already working for {market_symbol}")
                sell_price = 0.0
            if sell_price <= 0.0:
                pass
            else:
                order = place_limit_sell(sym_cfg, sell_price, position_qty, execution_mode=resolved_execution_mode, dry_run=dry_run)
                if order:
                    orders_placed.append(order)
                    open_orders.append(order)
                    logger.info(f"  Take-profit sell @ ${sell_price:.2f}")

        # 5. Execute based on signal
        if plan.direction == "long" and plan.confidence >= 0.4:
            target_position_pct = sym_cfg.max_position_pct * effective_leverage
            if position_pct >= target_position_pct:
                logger.info(f"  Skip buy: already at max position ({position_pct:.1%})")
                continue

            # Calculate order size: max_position_pct of portfolio * leverage minus current
            trade_size = state.total_value_usd * sym_cfg.max_position_pct * effective_leverage - position_value
            available = _quote_buying_power(
                state,
                trade_quote,
                execution_mode=resolved_execution_mode,
                effective_leverage=effective_leverage,
            )
            if resolved_execution_mode == "spot" and available < trade_size:
                total_stables = state.fdusd_balance + state.usdt_balance
                trade_size = min(trade_size, total_stables * sym_cfg.max_position_pct * effective_leverage)
            else:
                trade_size = min(trade_size, available * 0.95)
            trade_size = max(trade_size, 0)

            if trade_size < MIN_TRADE_USD:
                logger.info(f"  Skip buy: trade too small ({trade_size:.2f})")
                continue

            # Ensure we have the right stablecoin (converts if needed)
            if resolved_execution_mode == "spot":
                if not ensure_quote_balance(trade_quote, trade_size, state, dry_run):
                    logger.warning(f"  Skip buy: insufficient {trade_quote}")
                    continue
                available_after = _quote_buying_power(
                    state,
                    trade_quote,
                    execution_mode=resolved_execution_mode,
                    effective_leverage=effective_leverage,
                )
            else:
                available_after = _quote_buying_power(
                    state,
                    trade_quote,
                    execution_mode=resolved_execution_mode,
                    effective_leverage=effective_leverage,
                )
            trade_size = min(trade_size, available_after * 0.95)
            if trade_size < MIN_TRADE_USD:
                logger.info(f"  Skip buy: post-conversion too small ({trade_size:.2f})")
                continue

            buy_price = plan.buy_price if plan.buy_price > 0 else current_price * 0.998
            open_orders, skip_buy = _dedupe_side_orders(
                open_orders,
                symbol=market_symbol,
                side="BUY",
                execution_mode=resolved_execution_mode,
                dry_run=dry_run,
                desired_notional=trade_size,
            )
            if skip_buy:
                logger.info(f"  Existing buy order already working for {market_symbol}")
                continue
            order = place_limit_buy(sym_cfg, buy_price, trade_size, execution_mode=resolved_execution_mode, dry_run=dry_run)
            if order:
                orders_placed.append(order)
                open_orders.append(order)
                _reserve_buying_power(
                    state,
                    trade_quote,
                    _estimate_order_notional(order, fallback=trade_size),
                    execution_mode=resolved_execution_mode,
                )

        elif plan.direction == "hold" and position_qty == 0:
            logger.info(f"  Holding (no position, no action)")

    logger.info(f"\n{'='*60}")
    logger.info(f"Cycle complete: {len(orders_placed)} orders placed")
    logger.info(f"{'='*60}\n")

    return orders_placed


# ---------------------------------------------------------------------------
# Hybrid RL+LLM Trading Cycle
# ---------------------------------------------------------------------------

RL_BINANCE_PAIRS = {
    "BTCUSD": "BTCFDUSD",
    "ETHUSD": "ETHFDUSD",
    "DOGEUSD": "DOGEUSDT",
    "AAVEUSD": "AAVEUSDT",
}

_prev_plan: Optional[AllocationPlan] = None
_prev_outcome: Optional[PlanOutcome] = None
_prev_portfolio_value: float = 0.0


def _allocation_plan_has_error(plan: AllocationPlan) -> bool:
    reasoning = (plan.reasoning or "").strip().lower()
    if not reasoning:
        return False
    return reasoning.startswith((
        "failed to parse response",
        "no json found in response",
        "api error:",
        "all retries exhausted",
    ))


def _get_current_positions_valued(
    state: PortfolioState,
    execution_mode: str,
) -> dict[str, tuple[float, float]]:
    """Get {symbol: (qty, value_usd)} for all tradeable symbols with positions."""
    result = {}
    for sym, cfg in TRADING_SYMBOLS.items():
        qty = state.positions.get(cfg.base_asset, 0.0)
        if qty <= 0:
            continue
        try:
            price = _get_market_price(cfg, execution_mode)
            result[sym] = (qty, qty * price)
        except Exception:
            pass
    return result


def run_hybrid_trading_cycle(
    rl_gen: RLSignalGenerator,
    gemini_model: str = "gemini-2.5-flash",
    forecast_cache_root: str = "binanceneural/forecast_cache",
    dry_run: bool = True,
    leverage: float = 1.0,
    execution_mode: str = "auto",
):
    """Hybrid RL+Chronos2+Gemini portfolio allocation cycle."""
    global _prev_plan, _prev_outcome, _prev_portfolio_value

    resolved_execution_mode = _resolve_execution_mode(execution_mode, leverage)
    effective_leverage = _effective_leverage(resolved_execution_mode, leverage)

    logger.info(f"\n{'='*60}")
    logger.info(f"Hybrid Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(
        f"Mode: {'DRY RUN' if dry_run else 'LIVE'} | Gemini: {gemini_model} | "
        f"Execution: {resolved_execution_mode} | Requested leverage: {leverage:.2f}x | "
        f"Effective leverage: {effective_leverage:.2f}x"
    )
    logger.info(f"{'='*60}")

    # 1. Prepare margin capital, then load portfolio state
    if resolved_execution_mode == "margin":
        _sync_margin_capital(dry_run)
    state = get_portfolio_state(resolved_execution_mode)
    cash_usd = state.fdusd_balance + state.usdt_balance
    logger.info(
        f"Portfolio: ${state.total_value_usd:.2f} | Cash: ${cash_usd:.2f} | "
        f"borrowable_usdt={state.borrowable_quotes.get('USDT', 0.0):.2f}"
    )
    for asset, qty in state.positions.items():
        logger.info(f"  {asset}: {qty}")
    _repay_margin_debt_if_flat(state, dry_run, resolved_execution_mode)

    # Compute prev plan outcome
    if _prev_plan and _prev_portfolio_value > 0:
        pnl_usd = state.total_value_usd - _prev_portfolio_value
        pnl_pct = pnl_usd / _prev_portfolio_value
        _prev_outcome = PlanOutcome(plan=_prev_plan, pnl_usd=pnl_usd, pnl_pct=pnl_pct)
        logger.info(f"  Prev plan outcome: PnL=${pnl_usd:+.2f} ({pnl_pct:+.2%})")

    # 2. Cancel stale orders
    open_orders = _cancel_stale_open_orders(resolved_execution_mode, dry_run)

    # 3. Gather market context for all 4 symbols
    cache_root = Path(forecast_cache_root)
    try:
        contexts = gather_symbol_contexts(cache_root)
    except Exception as e:
        logger.error(f"Failed to gather market context: {e}")
        return []

    if not contexts:
        logger.error("No symbol contexts available")
        return []

    # 4. Get RL signal
    positions_valued = _get_current_positions_valued(state, resolved_execution_mode)
    largest_pos = max(positions_valued.items(), key=lambda x: x[1][1]) if positions_valued else None
    cur_sym = largest_pos[0] if largest_pos else None

    portfolio_snap = PortfolioSnapshot(
        cash_usd=cash_usd,
        total_value_usd=state.total_value_usd,
        position_symbol=cur_sym,
        position_value_usd=largest_pos[1][1] if largest_pos else 0.0,
        hold_hours=0,
        is_short=False,
    )

    # Build klines map from contexts (avoid re-fetching)
    klines_map = {ctx.symbol: ctx.klines for ctx in contexts}
    try:
        rl_signal = rl_gen.get_signal(portfolio=portfolio_snap, klines_map=klines_map)
    except Exception as e:
        logger.error(f"RL signal error: {e}")
        return []

    # 5. Build prompt and call Gemini
    prompt = build_allocation_prompt(
        contexts=contexts,
        rl_signal=rl_signal,
        portfolio_value=state.total_value_usd,
        cash_usd=cash_usd,
        positions=state.positions,
        prev_plan=_prev_plan,
        prev_outcome=_prev_outcome,
    )

    logger.info(f"Prompt built ({len(prompt)} chars), calling Gemini...")
    try:
        plan = call_gemini_allocation(prompt, model=gemini_model)
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return []

    if _allocation_plan_has_error(plan):
        logger.warning(f"Skipping execution due to invalid allocation plan: {plan.reasoning}")
        return []

    if not plan.allocations and not plan.reasoning:
        logger.warning("Empty allocation plan, skipping execution")
        return []

    # 6. Execute allocation
    orders_placed = []
    target_values = {}
    for sym, pct in plan.allocations.items():
        target_values[sym] = state.total_value_usd * pct / 100.0 * effective_leverage

    logger.info(f"Target allocation: {plan.allocations} | cash={plan.cash_pct:.0f}%")

    # Sell positions that need reducing
    for sym, (qty, cur_value) in positions_valued.items():
        target_val = target_values.get(sym, 0.0)
        if cur_value > target_val + MIN_TRADE_USD:
            cfg = TRADING_SYMBOLS.get(sym)
            if not cfg:
                continue
            try:
                price = _get_market_price(cfg, resolved_execution_mode)
                sell_value = cur_value - target_val
                sell_qty = sell_value / price
                sell_qty = min(sell_qty, qty)
                exit_price = plan.exit_prices.get(sym, price * 1.001)
                if exit_price <= 0:
                    exit_price = price * 1.001
                market_symbol = _execution_pair(cfg, resolved_execution_mode)
                open_orders, skip_sell = _dedupe_side_orders(
                    open_orders,
                    symbol=market_symbol,
                    side="SELL",
                    execution_mode=resolved_execution_mode,
                    dry_run=dry_run,
                    desired_qty=sell_qty,
                )
                if skip_sell:
                    logger.info(f"  Existing sell order already working for {market_symbol}")
                    continue
                order = place_limit_sell(cfg, exit_price, sell_qty, execution_mode=resolved_execution_mode, dry_run=dry_run)
                if order:
                    orders_placed.append(order)
                    open_orders.append(order)
                    logger.info(f"  SELL {sym}: {sell_qty:.6f} @ ${exit_price:.2f} (reduce to {target_val:.0f})")
            except Exception as e:
                logger.error(f"  Failed to sell {sym}: {e}")

    # Buy positions that need increasing
    for sym, target_val in target_values.items():
        if target_val < MIN_TRADE_USD:
            continue
        cur_value = positions_valued.get(sym, (0, 0))[1]
        buy_needed = target_val - cur_value
        if buy_needed < MIN_TRADE_USD:
            continue

        cfg = TRADING_SYMBOLS.get(sym)
        if not cfg:
            continue

        try:
            price = _get_market_price(cfg, resolved_execution_mode)
            trade_quote = _execution_quote_asset(cfg, resolved_execution_mode)
            available = _quote_buying_power(
                state,
                trade_quote,
                execution_mode=resolved_execution_mode,
                effective_leverage=effective_leverage,
            )
            buy_needed = min(buy_needed, available * 0.95)
            if buy_needed < MIN_TRADE_USD and resolved_execution_mode == "spot":
                if not ensure_quote_balance(trade_quote, buy_needed, state, dry_run):
                    logger.info(f"  Skip buy {sym}: insufficient {trade_quote}")
                    continue
                available = _quote_buying_power(
                    state,
                    trade_quote,
                    execution_mode=resolved_execution_mode,
                    effective_leverage=effective_leverage,
                )
                buy_needed = min(buy_needed, available * 0.95)

            if buy_needed < MIN_TRADE_USD:
                continue

            entry_price = plan.entry_prices.get(sym, price * 0.999)
            if entry_price <= 0:
                entry_price = price * 0.999
            market_symbol = _execution_pair(cfg, resolved_execution_mode)
            open_orders, skip_buy = _dedupe_side_orders(
                open_orders,
                symbol=market_symbol,
                side="BUY",
                execution_mode=resolved_execution_mode,
                dry_run=dry_run,
                desired_notional=buy_needed,
            )
            if skip_buy:
                logger.info(f"  Existing buy order already working for {market_symbol}")
                continue
            order = place_limit_buy(cfg, entry_price, buy_needed, execution_mode=resolved_execution_mode, dry_run=dry_run)
            if order:
                orders_placed.append(order)
                open_orders.append(order)
                _reserve_buying_power(
                    state,
                    trade_quote,
                    _estimate_order_notional(order, fallback=buy_needed),
                    execution_mode=resolved_execution_mode,
                )
                logger.info(f"  BUY {sym}: ${buy_needed:.2f} @ ${entry_price:.2f}")
        except Exception as e:
            logger.error(f"  Failed to buy {sym}: {e}")

    # Save plan for next cycle context
    _prev_plan = plan
    _prev_portfolio_value = state.total_value_usd

    logger.info(f"\n{'='*60}")
    logger.info(f"Hybrid cycle complete: {len(orders_placed)} orders")
    logger.info(f"{'='*60}\n")
    return orders_placed


def main():
    parser = argparse.ArgumentParser(description="RL+LLM Hybrid Binance Trader")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSD", "ETHUSD", "SOLUSD"],
                        help="Symbols to trade")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview",
                        help="Gemini model for hybrid mode or LLM-only mode")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (no real orders)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading mode")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between trading cycles (default: 3600)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="(deprecated) Use --rl-checkpoint instead")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL policy checkpoint. Enables hybrid RL+Gemini allocation mode.")
    parser.add_argument("--rl-alloc-pct", type=float, default=0.90,
                        help="Deprecated legacy RL-only allocation cap; ignored in hybrid mode")
    parser.add_argument("--forecast-cache", type=str,
                        default="binanceneural/forecast_cache",
                        help="Forecast cache root for RL features")
    parser.add_argument("--leverage", type=float, default=1.0,
                        help="Requested position multiplier. Auto mode switches to margin above 1x.")
    parser.add_argument("--execution-mode", type=str, default="auto", choices=sorted(SUPPORTED_EXECUTION_MODES),
                        help="Execution account to use: auto, spot, or margin.")
    args = parser.parse_args()

    dry_run = not args.live
    rl_checkpoint = args.rl_checkpoint or args.checkpoint

    if args.live:
        logger.warning("LIVE TRADING MODE - Real orders will be placed!")
        logger.warning("Press Ctrl+C to stop")
        time.sleep(3)

    # Initialize RL generator if checkpoint provided
    rl_gen = None
    if rl_checkpoint:
        rl_gen = RLSignalGenerator(
            checkpoint_path=rl_checkpoint,
            forecast_cache_root=args.forecast_cache,
        )
        logger.info(f"Hybrid mode: RL={rl_checkpoint} + Gemini={args.model}")
        logger.info(f"RL symbols: {', '.join(RL_SYMBOLS)}")

    while True:
        try:
            if rl_gen:
                run_hybrid_trading_cycle(
                    rl_gen=rl_gen,
                    gemini_model=args.model,
                    forecast_cache_root=args.forecast_cache,
                    dry_run=dry_run,
                    leverage=args.leverage,
                    execution_mode=args.execution_mode,
                )
            else:
                run_trading_cycle(
                    symbols=args.symbols,
                    model=args.model,
                    thinking_level="HIGH",
                    dry_run=dry_run,
                    leverage=args.leverage,
                    execution_mode=args.execution_mode,
                )
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

        if args.once:
            break

        logger.info(f"Next cycle in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
