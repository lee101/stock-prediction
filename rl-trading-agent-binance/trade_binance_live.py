"""
Production Binance Trading - RL+LLM Hybrid System.

Manages:
- FDUSD pairs for BTC/ETH (zero fees)
- USDT pairs for altcoins (DOGE, SUI, SOL, AAVE)
- Automatic stablecoin swaps between FDUSD ↔ USDT
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


# ---------------------------------------------------------------------------
# Portfolio State
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Track current portfolio state."""
    fdusd_balance: float = 0.0
    usdt_balance: float = 0.0
    positions: dict = field(default_factory=dict)  # {base_asset: qty}
    total_value_usd: float = 0.0

    def available_quote(self, quote_asset: str) -> float:
        if quote_asset == "FDUSD":
            return self.fdusd_balance
        return self.usdt_balance


def get_portfolio_state() -> PortfolioState:
    """Fetch current portfolio from Binance."""
    state = PortfolioState()
    state.fdusd_balance = binance_wrapper.get_asset_free_balance("FDUSD") or 0.0
    state.usdt_balance = binance_wrapper.get_asset_free_balance("USDT") or 0.0

    for cfg in TRADING_SYMBOLS.values():
        bal = binance_wrapper.get_asset_free_balance(cfg.base_asset) or 0.0
        if bal > 0:
            state.positions[cfg.base_asset] = bal

    # Total value
    account = binance_wrapper.get_account_value_usdt(include_locked=False)
    state.total_value_usd = account.get("total_usdt", 0.0) if isinstance(account, dict) else 0.0

    return state


def get_position_entry(sym_cfg: 'BinanceSymbolConfig') -> tuple[float, Optional[datetime]]:
    """Get entry price and time for current position from recent trades."""
    trades = binance_wrapper.get_my_trades(sym_cfg.binance_pair, limit=20)
    if not trades:
        return 0.0, None
    # Walk backwards to find last buy that opened the position
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
    dry_run: bool = True,
) -> Optional[dict]:
    """Place a limit buy order."""
    rules = resolve_symbol_rules(sym_cfg.binance_pair)
    qty = amount_usd / price
    qty = quantize_qty(qty, step_size=rules.step_size or 0.00001)
    price = quantize_price(price, tick_size=rules.tick_size or 0.01, side="BUY")

    notional = qty * price
    if notional < MIN_TRADE_USD:
        logger.warning(f"Order too small: {notional:.2f} < {MIN_TRADE_USD}")
        return None

    if rules.min_qty and qty < rules.min_qty:
        logger.warning(f"Qty {qty} below min {rules.min_qty} for {sym_cfg.binance_pair}")
        return None

    logger.info(f"BUY {sym_cfg.binance_pair}: qty={qty}, price={price}, notional={notional:.2f}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would place limit buy")
        return {"symbol": sym_cfg.binance_pair, "side": "BUY", "qty": qty, "price": price, "dry_run": True}

    try:
        order = binance_wrapper.create_order(sym_cfg.binance_pair, "BUY", qty, price)
        logger.info(f"  Order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        logger.error(f"  Order failed: {e}")
        return None


def place_limit_sell(
    sym_cfg: BinanceSymbolConfig,
    price: float,
    qty: float,
    dry_run: bool = True,
) -> Optional[dict]:
    """Place a limit sell (take-profit) order."""
    rules = resolve_symbol_rules(sym_cfg.binance_pair)
    qty = quantize_qty(qty, step_size=rules.step_size or 0.00001)
    price = quantize_price(price, tick_size=rules.tick_size or 0.01, side="SELL")

    notional = qty * price
    if notional < MIN_TRADE_USD:
        logger.warning(f"Sell too small: {notional:.2f} < {MIN_TRADE_USD}")
        return None

    logger.info(f"SELL {sym_cfg.binance_pair}: qty={qty}, price={price}, notional={notional:.2f}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would place limit sell")
        return {"symbol": sym_cfg.binance_pair, "side": "SELL", "qty": qty, "price": price, "dry_run": True}

    try:
        order = binance_wrapper.create_order(sym_cfg.binance_pair, "SELL", qty, price)
        logger.info(f"  Order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        logger.error(f"  Order failed: {e}")
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
) -> TradePlan:
    """Get RL+LLM hybrid trading signal for a symbol."""
    import torch
    import torch.nn as nn

    # Get current market data
    price = binance_wrapper.get_symbol_price(sym_cfg.binance_pair)
    if not price:
        return TradePlan("hold", 0, 0, 0, "no price data")

    current_price = float(price)

    # Load recent hourly bars from Binance
    from src.binan import binance_wrapper as bw
    try:
        # Try primary pair, fall back to USDT if FDUSD not available
        pair = sym_cfg.binance_pair
        try:
            klines = bw.get_client().get_klines(symbol=pair, interval="1h", limit=72)
        except Exception:
            if sym_cfg.quote_asset == "FDUSD":
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
        logger.error(f"Failed to get klines for {sym_cfg.binance_pair}: {e}")
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

    fee_bps = 0 if sym_cfg.quote_asset == "FDUSD" else 10
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
):
    """Run one trading cycle across all symbols."""
    effective_leverage = _resolve_spot_leverage(leverage)
    logger.info(f"\n{'='*60}")
    logger.info(f"Trading Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(
        f"Model: {model} (thinking={thinking_level}) | "
        f"Requested leverage: {leverage:.2f}x | Effective spot leverage: {effective_leverage:.2f}x"
    )
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"{'='*60}")

    # 1. Get portfolio state
    state = get_portfolio_state()
    logger.info(f"Portfolio: FDUSD={state.fdusd_balance:.2f}, USDT={state.usdt_balance:.2f}, "
                f"total={state.total_value_usd:.2f}")
    for asset, qty in state.positions.items():
        logger.info(f"  Position: {asset} = {qty}")

    # 2. Cancel stale open orders
    try:
        open_orders = binance_wrapper.get_open_orders()
        if open_orders:
            logger.info(f"  {len(open_orders)} open orders found")
            # Cancel orders older than 2 hours
            now_ms = int(time.time() * 1000)
            for order in open_orders:
                age_hours = (now_ms - order.get("time", now_ms)) / 3600000
                if age_hours > 2:
                    if not dry_run:
                        binance_wrapper.cancel_order(order["symbol"], order["orderId"])
                    logger.info(f"  Cancelled stale order {order['orderId']} ({age_hours:.1f}h old)")
    except Exception as e:
        logger.warning(f"  Failed to check open orders: {e}")

    # 3. Get signals for each symbol
    orders_placed = []
    for i, sym_name in enumerate(symbols):
        sym_cfg = TRADING_SYMBOLS.get(sym_name)
        if not sym_cfg:
            logger.warning(f"Unknown symbol: {sym_name}")
            continue

        logger.info(f"\n--- {sym_cfg.symbol} ({sym_cfg.binance_pair}) ---")

        # Get current price
        try:
            current_price = float(binance_wrapper.get_symbol_price(sym_cfg.binance_pair))
        except Exception:
            logger.error(f"  Cannot get price for {sym_cfg.binance_pair}")
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
                entry_price, open_time = get_position_entry(sym_cfg)
            except Exception:
                pass

        # Get hybrid signal
        try:
            plan = get_hybrid_signal(
                sym_cfg, model, thinking_level, rl_checkpoint,
                position_qty=position_qty,
                position_entry_price=entry_price,
                position_open_time=open_time,
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
            order = place_limit_sell(sym_cfg, sell_price, position_qty, dry_run)
            if order:
                orders_placed.append(order)
                logger.info(f"  Take-profit sell @ ${sell_price:.2f}")

        # 5. Execute based on signal
        if plan.direction == "long" and plan.confidence >= 0.4:
            if position_pct >= sym_cfg.max_position_pct:
                logger.info(f"  Skip buy: already at max position ({position_pct:.1%})")
                continue

            # Calculate order size: max_position_pct of portfolio * leverage minus current
            trade_size = state.total_value_usd * sym_cfg.max_position_pct * effective_leverage - position_value
            available = state.available_quote(sym_cfg.quote_asset)
            if available < trade_size:
                total_stables = state.fdusd_balance + state.usdt_balance
                trade_size = min(trade_size, total_stables * sym_cfg.max_position_pct * effective_leverage)
            else:
                trade_size = min(trade_size, available * 0.95 * effective_leverage)
            trade_size = max(trade_size, 0)

            if trade_size < MIN_TRADE_USD:
                logger.info(f"  Skip buy: trade too small ({trade_size:.2f})")
                continue

            # Ensure we have the right stablecoin (converts if needed)
            if not ensure_quote_balance(sym_cfg.quote_asset, trade_size, state, dry_run):
                logger.warning(f"  Skip buy: insufficient {sym_cfg.quote_asset}")
                continue

            # Cap trade_size at actual available quote after conversion
            available_after = state.available_quote(sym_cfg.quote_asset)
            trade_size = min(trade_size, available_after * 0.95)
            if trade_size < MIN_TRADE_USD:
                logger.info(f"  Skip buy: post-conversion too small ({trade_size:.2f})")
                continue

            buy_price = plan.buy_price if plan.buy_price > 0 else current_price * 0.998
            order = place_limit_buy(sym_cfg, buy_price, trade_size, dry_run)
            if order:
                orders_placed.append(order)
                _reserve_quote_balance(
                    state,
                    sym_cfg.quote_asset,
                    _estimate_order_notional(order, fallback=trade_size),
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


def _get_current_positions_valued(state: PortfolioState) -> dict[str, tuple[float, float]]:
    """Get {symbol: (qty, value_usd)} for all tradeable symbols with positions."""
    result = {}
    for sym, cfg in TRADING_SYMBOLS.items():
        qty = state.positions.get(cfg.base_asset, 0.0)
        if qty <= 0:
            continue
        try:
            price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
            result[sym] = (qty, qty * price)
        except Exception:
            pass
    return result


def run_hybrid_trading_cycle(
    rl_gen: RLSignalGenerator,
    gemini_model: str = "gemini-2.5-flash",
    forecast_cache_root: str = "binanceneural/forecast_cache",
    dry_run: bool = True,
):
    """Hybrid RL+Chronos2+Gemini portfolio allocation cycle."""
    global _prev_plan, _prev_outcome, _prev_portfolio_value

    logger.info(f"\n{'='*60}")
    logger.info(f"Hybrid Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'} | Gemini: {gemini_model}")
    logger.info(f"{'='*60}")

    # 1. Portfolio state
    state = get_portfolio_state()
    cash_usd = state.fdusd_balance + state.usdt_balance
    logger.info(f"Portfolio: ${state.total_value_usd:.2f} | Cash: ${cash_usd:.2f}")
    for asset, qty in state.positions.items():
        logger.info(f"  {asset}: {qty}")

    # Compute prev plan outcome
    if _prev_plan and _prev_portfolio_value > 0:
        pnl_usd = state.total_value_usd - _prev_portfolio_value
        pnl_pct = pnl_usd / _prev_portfolio_value
        _prev_outcome = PlanOutcome(plan=_prev_plan, pnl_usd=pnl_usd, pnl_pct=pnl_pct)
        logger.info(f"  Prev plan outcome: PnL=${pnl_usd:+.2f} ({pnl_pct:+.2%})")

    # 2. Cancel stale orders
    try:
        open_orders = binance_wrapper.get_open_orders()
        if open_orders:
            now_ms = int(time.time() * 1000)
            for order in open_orders:
                age_hours = (now_ms - order.get("time", now_ms)) / 3600000
                if age_hours > 2:
                    if not dry_run:
                        binance_wrapper.cancel_order(order["symbol"], order["orderId"])
                    logger.info(f"  Cancelled stale order {order['orderId']} ({age_hours:.1f}h old)")
    except Exception as e:
        logger.warning(f"  Failed to check open orders: {e}")

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
    positions_valued = _get_current_positions_valued(state)
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
        target_values[sym] = state.total_value_usd * pct / 100.0

    logger.info(f"Target allocation: {plan.allocations} | cash={plan.cash_pct:.0f}%")

    # Sell positions that need reducing
    for sym, (qty, cur_value) in positions_valued.items():
        target_val = target_values.get(sym, 0.0)
        if cur_value > target_val + MIN_TRADE_USD:
            cfg = TRADING_SYMBOLS.get(sym)
            if not cfg:
                continue
            try:
                price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
                sell_value = cur_value - target_val
                sell_qty = sell_value / price
                sell_qty = min(sell_qty, qty)
                exit_price = plan.exit_prices.get(sym, price * 1.001)
                if exit_price <= 0:
                    exit_price = price * 1.001
                order = place_limit_sell(cfg, exit_price, sell_qty, dry_run)
                if order:
                    orders_placed.append(order)
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
            price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
            # Cap at available cash
            available = state.available_quote(cfg.quote_asset)
            buy_needed = min(buy_needed, available * 0.95)
            if buy_needed < MIN_TRADE_USD:
                # Try stablecoin conversion
                if not ensure_quote_balance(cfg.quote_asset, buy_needed, state, dry_run):
                    logger.info(f"  Skip buy {sym}: insufficient {cfg.quote_asset}")
                    continue
                available = state.available_quote(cfg.quote_asset)
                buy_needed = min(buy_needed, available * 0.95)

            if buy_needed < MIN_TRADE_USD:
                continue

            entry_price = plan.entry_prices.get(sym, price * 0.999)
            if entry_price <= 0:
                entry_price = price * 0.999
            order = place_limit_buy(cfg, entry_price, buy_needed, dry_run)
            if order:
                orders_placed.append(order)
                _reserve_quote_balance(
                    state,
                    cfg.quote_asset,
                    _estimate_order_notional(order, fallback=buy_needed),
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
                        help="Requested position multiplier; spot execution clamps anything above 1x")
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
                )
            else:
                run_trading_cycle(
                    symbols=args.symbols,
                    model=args.model,
                    thinking_level="HIGH",
                    dry_run=dry_run,
                    leverage=args.leverage,
                )
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

        if args.once:
            break

        logger.info(f"Next cycle in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
