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

    # Build prompt (simplified version for production)
    from rl_trading_agent_binance_prompt import build_live_prompt
    prompt = build_live_prompt(sym_cfg.symbol, history_rows, current_price)

    # Call LLM
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
):
    """Run one trading cycle across all symbols."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Trading Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Model: {model} (thinking={thinking_level})")
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
    for sym_name in symbols:
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

        # Check existing position
        position_qty = state.positions.get(sym_cfg.base_asset, 0.0)
        position_value = position_qty * current_price
        position_pct = position_value / max(state.total_value_usd, 1.0)

        logger.info(f"  Price: ${current_price:.2f}, Position: {position_qty} ({position_pct:.1%} of portfolio)")

        # Get hybrid signal
        try:
            plan = get_hybrid_signal(sym_cfg, model, thinking_level, rl_checkpoint)
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

            # Calculate order size using available quote for this specific stablecoin
            trade_size = state.total_value_usd * sym_cfg.max_position_pct - position_value
            available = state.available_quote(sym_cfg.quote_asset)
            # If needed stablecoin is short, check if conversion would help
            if available < trade_size:
                total_stables = state.fdusd_balance + state.usdt_balance
                trade_size = min(trade_size, total_stables * 0.80)
            else:
                trade_size = min(trade_size, available * 0.95)
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

        elif plan.direction == "hold" and position_qty == 0:
            logger.info(f"  Holding (no position, no action)")

    logger.info(f"\n{'='*60}")
    logger.info(f"Cycle complete: {len(orders_placed)} orders placed")
    logger.info(f"{'='*60}\n")

    return orders_placed


# ---------------------------------------------------------------------------
# RL-Only Trading Cycle
# ---------------------------------------------------------------------------

# Map RL symbol names to Binance pairs for kline fetching
RL_BINANCE_PAIRS = {
    "BTCUSD": "BTCFDUSD",
    "ETHUSD": "ETHFDUSD",
    "DOGEUSD": "DOGEUSDT",
    "AAVEUSD": "AAVEUSDT",
}

# Persistent state for RL hold tracking
_rl_hold_start: Optional[datetime] = None
_rl_current_symbol: Optional[str] = None


def _detect_current_rl_position(state: PortfolioState) -> tuple[Optional[str], float, float]:
    """Find largest RL-managed position. Returns (symbol, qty, value)."""
    best_sym, best_qty, best_val = None, 0.0, 0.0
    for rl_sym in RL_SYMBOLS:
        cfg = TRADING_SYMBOLS.get(rl_sym)
        if not cfg:
            continue
        qty = state.positions.get(cfg.base_asset, 0.0)
        if qty <= 0:
            continue
        try:
            price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
        except Exception:
            continue
        val = qty * price
        if val > best_val:
            best_sym, best_qty, best_val = rl_sym, qty, val
    return best_sym, best_qty, best_val


def run_rl_trading_cycle(
    rl_gen: RLSignalGenerator,
    alloc_pct: float = 0.90,
    dry_run: bool = True,
):
    """Run one RL trading cycle. The RL policy picks ONE asset to be in (or flat)."""
    global _rl_hold_start, _rl_current_symbol

    logger.info(f"\n{'='*60}")
    logger.info(f"RL Trading Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"{'='*60}")

    state = get_portfolio_state()
    logger.info(f"Portfolio: FDUSD={state.fdusd_balance:.2f}, USDT={state.usdt_balance:.2f}, "
                f"total={state.total_value_usd:.2f}")
    for asset, qty in state.positions.items():
        logger.info(f"  Position: {asset} = {qty}")

    # Cancel stale orders
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

    # Detect current position
    cur_sym, cur_qty, cur_val = _detect_current_rl_position(state)
    hold_hours = 0
    if cur_sym and _rl_current_symbol == cur_sym and _rl_hold_start:
        hold_hours = int((datetime.now(timezone.utc) - _rl_hold_start).total_seconds() / 3600)

    # Build portfolio snapshot for RL obs
    cash_usd = state.fdusd_balance + state.usdt_balance
    portfolio = PortfolioSnapshot(
        cash_usd=cash_usd,
        total_value_usd=state.total_value_usd,
        position_symbol=cur_sym,
        position_value_usd=cur_val,
        unrealized_pnl_usd=0.0,  # would need entry price tracking for exact pnl
        hold_hours=hold_hours,
        is_short=False,
    )

    # Get RL signal
    try:
        signal = rl_gen.get_signal(portfolio=portfolio, binance_pairs=RL_BINANCE_PAIRS)
    except Exception as e:
        logger.error(f"RL signal error: {e}")
        return []

    target_sym = signal.target_symbol
    orders_placed = []

    logger.info(f"  Current: {cur_sym or 'FLAT'} (${cur_val:.2f})")
    logger.info(f"  RL says: {signal.action_name} -> target={target_sym or 'FLAT'}")

    # If target changed, execute the rotation
    if target_sym != cur_sym:
        # Step 1: Sell current position
        if cur_sym and cur_qty > 0:
            cfg = TRADING_SYMBOLS[cur_sym]
            try:
                price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
                sell_price = price * 0.999  # slightly below market for quick fill
                order = place_limit_sell(cfg, sell_price, cur_qty, dry_run)
                if order:
                    orders_placed.append(order)
                    logger.info(f"  SELL {cur_sym}: qty={cur_qty} @ ${sell_price:.2f}")
            except Exception as e:
                logger.error(f"  Failed to sell {cur_sym}: {e}")

        # Step 2: Buy new target
        if target_sym:
            cfg = TRADING_SYMBOLS.get(target_sym)
            if cfg:
                try:
                    price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
                    trade_size = state.total_value_usd * alloc_pct
                    # Use available cash (after sell, estimate conservatively)
                    available = cash_usd + (cur_val * 0.95 if cur_sym else 0)
                    trade_size = min(trade_size, available * 0.95)

                    if trade_size >= MIN_TRADE_USD:
                        if not ensure_quote_balance(cfg.quote_asset, trade_size, state, dry_run):
                            logger.warning(f"  Skip buy: insufficient {cfg.quote_asset}")
                        else:
                            buy_price = price * 1.001  # slightly above market for quick fill
                            order = place_limit_buy(cfg, buy_price, trade_size, dry_run)
                            if order:
                                orders_placed.append(order)
                                logger.info(f"  BUY {target_sym}: ${trade_size:.2f} @ ${buy_price:.2f}")
                    else:
                        logger.info(f"  Skip buy: trade too small ({trade_size:.2f})")
                except Exception as e:
                    logger.error(f"  Failed to buy {target_sym}: {e}")

        # Update hold tracking
        _rl_current_symbol = target_sym
        _rl_hold_start = datetime.now(timezone.utc) if target_sym else None
    else:
        logger.info(f"  No rotation needed, holding {cur_sym or 'FLAT'}")
        # Place take-profit for existing position
        if cur_sym and cur_qty > 0:
            cfg = TRADING_SYMBOLS[cur_sym]
            try:
                price = float(binance_wrapper.get_symbol_price(cfg.binance_pair))
                tp_price = price * 1.015  # 1.5% take-profit
                order = place_limit_sell(cfg, tp_price, cur_qty, dry_run)
                if order:
                    orders_placed.append(order)
                    logger.info(f"  Take-profit {cur_sym} @ ${tp_price:.2f}")
            except Exception as e:
                logger.error(f"  Failed to place TP for {cur_sym}: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"RL cycle complete: {len(orders_placed)} orders placed")
    logger.info(f"{'='*60}\n")
    return orders_placed


def main():
    parser = argparse.ArgumentParser(description="RL+LLM Hybrid Binance Trader")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSD", "ETHUSD", "SOLUSD"],
                        help="Symbols to trade")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--thinking-level", type=str, default="HIGH")
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
                        help="Path to RL policy checkpoint. Enables RL-only mode.")
    parser.add_argument("--rl-alloc-pct", type=float, default=0.90,
                        help="Fraction of portfolio to allocate to RL position")
    parser.add_argument("--forecast-cache", type=str,
                        default="binanceneural/forecast_cache",
                        help="Forecast cache root for RL features")
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
        logger.info(f"RL mode: checkpoint={rl_checkpoint}, alloc={args.rl_alloc_pct:.0%}")
        logger.info(f"RL symbols: {', '.join(RL_SYMBOLS)}")

    while True:
        try:
            if rl_gen:
                run_rl_trading_cycle(
                    rl_gen=rl_gen,
                    alloc_pct=args.rl_alloc_pct,
                    dry_run=dry_run,
                )
            else:
                run_trading_cycle(
                    symbols=args.symbols,
                    model=args.model,
                    thinking_level=args.thinking_level,
                    dry_run=dry_run,
                )
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

        if args.once:
            break

        logger.info(f"Next cycle in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
