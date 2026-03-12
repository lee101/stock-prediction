"""Unified Stock+Crypto Trading Orchestrator.

24/7 hourly loop that coordinates Alpaca (stocks and crypto).
Runs the right system at the right time with cross-asset awareness.

Usage:
  python -m unified_orchestrator.orchestrator --dry-run --once
  python -m unified_orchestrator.orchestrator --live
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-trading-agent-binance"))

from loguru import logger

from unified_orchestrator.state import (
    build_snapshot,
    save_snapshot,
    UnifiedPortfolioSnapshot,
)
from unified_orchestrator.prompt_builder import build_unified_prompt
from unified_orchestrator.backout import select_backout_candidates, execute_backout
from unified_orchestrator.conditional_orders import (
    execute_plan,
    read_pending_fills,
    TradingPlan,
    TradingStep,
)

from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.gemini_wrapper import TradePlan

from unified_orchestrator.rl_gemini_bridge import RLGeminiBridge


# ---------------------------------------------------------------------------
# Crypto signal generation
# ---------------------------------------------------------------------------

# RL+Gemini hybrid bridges (lazy-loaded, separate for stocks vs crypto)
_rl_bridge: RLGeminiBridge | None = None
_rl_bridge_stock: RLGeminiBridge | None = None

# Validated checkpoints (confirmed on held-out data):
#   stocks: +50% median / 30-day window, Sortino=25 (featlag1, fee=5bps, long-only)
#   crypto: crypto11_ppo_v1 is unprofitable on val; use Gemini-only until better model
STOCK_CHECKPOINT_CANDIDATES = [
    REPO / "pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt",
    REPO / "pufferlib_market/checkpoints/stocks13_issuedat_featlag1_fee5bps_longonly_run5/best.pt",
]
CRYPTO_CHECKPOINT_CANDIDATES = [
    REPO / "pufferlib_market/checkpoints/crypto12_ppo_v8_h1024_300M_annealLR/best.pt",
    REPO / "pufferlib_market/checkpoints/crypto11_ppo_v1_h1024_300M_annealLR/best.pt",
]


def get_rl_bridge(checkpoint_path: str = "", hidden_size: int = 1024) -> RLGeminiBridge | None:
    """Get or create the crypto RL+Gemini bridge (lazy singleton)."""
    global _rl_bridge
    if _rl_bridge is not None:
        return _rl_bridge
    if not checkpoint_path:
        for c in CRYPTO_CHECKPOINT_CANDIDATES:
            if c.exists():
                checkpoint_path = str(c)
                break
    if not checkpoint_path:
        return None
    _rl_bridge = RLGeminiBridge(checkpoint_path=checkpoint_path, hidden_size=hidden_size)
    logger.info(f"  Crypto RL bridge loaded: {checkpoint_path}")
    return _rl_bridge


def get_rl_bridge_stock(checkpoint_path: str = "", hidden_size: int = 1024) -> RLGeminiBridge | None:
    """Get or create the stock RL+Gemini bridge (lazy singleton)."""
    global _rl_bridge_stock
    if _rl_bridge_stock is not None:
        return _rl_bridge_stock
    if not checkpoint_path:
        for c in STOCK_CHECKPOINT_CANDIDATES:
            if c.exists():
                checkpoint_path = str(c)
                break
    if not checkpoint_path:
        logger.warning("  No stock RL checkpoint found — falling back to Gemini-only signals")
        return None
    _rl_bridge_stock = RLGeminiBridge(checkpoint_path=checkpoint_path, hidden_size=hidden_size)
    logger.info(f"  Stock RL bridge loaded: {checkpoint_path}")
    return _rl_bridge_stock

CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
CRYPTO_PAIRS = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT",
                "DOGEUSD": "DOGEUSDT", "SUIUSD": "SUIUSDT", "AAVEUSD": "AAVEUSDT"}

STOCK_SYMBOLS = ["NVDA", "PLTR", "META", "MSFT", "NET"]


def get_crypto_signals(
    symbols: list[str],
    snapshot: UnifiedPortfolioSnapshot,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    dry_run: bool = True,
) -> dict[str, TradePlan]:
    """Generate LLM trading signals for crypto symbols using Alpaca historical data."""
    import pandas as pd
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import timedelta
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    signals = {}
    data_client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    now = datetime.now(timezone.utc)

    # Alpaca crypto historical API uses "BTC/USD" format
    def _alpaca_crypto_sym(s: str) -> str:
        if "/" not in s and s.endswith("USD"):
            return s[:-3] + "/USD"
        return s

    for sym in symbols:
        alpaca_sym = _alpaca_crypto_sym(sym)
        try:
            req = CryptoBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=TimeFrame.Hour,
                start=now - timedelta(hours=78),
                end=now,
                limit=72,
            )
            bars = data_client.get_crypto_bars(req)
            df = bars.df
            if df is None or len(df) < 12:
                logger.warning(f"  {sym}: insufficient bars ({0 if df is None else len(df)})")
                continue

            # Flatten multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(alpaca_sym, level="symbol")
            df = df.reset_index()
            if "timestamp" not in df.columns and "index" in df.columns:
                df.rename(columns={"index": "timestamp"}, inplace=True)

            history = []
            for _, row in df.iterrows():
                history.append({
                    "timestamp": str(row.get("timestamp", ""))[:16],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })

            if len(history) < 12:
                continue

            current_price = history[-1]["close"]

            prompt = build_unified_prompt(
                symbol=sym,  # Use original symbol (ETHUSD) for the prompt
                history_rows=history,
                current_price=current_price,
                snapshot=snapshot,
                asset_class="crypto",
            )

            try:
                plan = call_llm(prompt, model=model, thinking_level=thinking_level)
                signals[sym] = plan
                logger.info(f"  {sym}: {plan.direction} (conf={plan.confidence:.2f}, "
                            f"buy=${plan.buy_price:.2f}, sell=${plan.sell_price:.2f})")
            except Exception as e:
                logger.error(f"  {sym}: LLM error: {e}")

        except Exception as e:
            logger.error(f"  {sym}: bars error: {e}")

    return signals


# ---------------------------------------------------------------------------
# Crypto execution
# ---------------------------------------------------------------------------

def validate_plan_safety(plan: TradePlan, current_price: float,
                         fee_bps: float = 20.0) -> tuple[bool, str]:
    """Validate a trade plan is safe to execute in production.

    Checks:
    1. sell_price > buy_price + fees (must be profitable if both fill)
    2. Prices within reasonable range of current (not >5% away)
    3. Confidence > 0 for non-hold plans
    """
    if plan.direction == "hold":
        return True, "hold"

    if plan.buy_price <= 0 and plan.sell_price <= 0:
        return False, "no prices set"

    # For longs: sell must be above buy + fees
    if plan.direction == "long" and plan.buy_price > 0 and plan.sell_price > 0:
        fee_cost = plan.buy_price * fee_bps / 10000 * 2  # round-trip
        if plan.sell_price <= plan.buy_price + fee_cost:
            return False, (f"sell ${plan.sell_price:.2f} <= buy ${plan.buy_price:.2f} "
                          f"+ fees ${fee_cost:.2f}")

    # Prices should be within 5% of current
    if plan.buy_price > 0:
        pct_diff = abs(plan.buy_price - current_price) / current_price
        if pct_diff > 0.05:
            return False, f"buy_price ${plan.buy_price:.2f} is {pct_diff:.1%} from current ${current_price:.2f}"

    if plan.sell_price > 0:
        pct_diff = abs(plan.sell_price - current_price) / current_price
        if pct_diff > 0.05:
            return False, f"sell_price ${plan.sell_price:.2f} is {pct_diff:.1%} from current ${current_price:.2f}"

    return True, "ok"


def execute_crypto_signals(
    signals: dict[str, TradePlan],
    snapshot: UnifiedPortfolioSnapshot,
    dry_run: bool = True,
) -> list[dict]:
    """Execute crypto trading signals on Alpaca."""
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []
    alpaca = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

    # Normalize Alpaca's "ETH/USD" symbol format back to "ETHUSD" for lookup
    def _norm(s: str) -> str:
        return s.replace("/", "")

    # Cancel stale pending buy orders for crypto symbols (GTC orders from last cycle
    # have stale prices — replace with fresh signals each hour)
    crypto_sym_set = {sym for sym in signals}
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    open_orders = alpaca.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    canceled_for: set[str] = set()
    for order in open_orders:
        sym_norm = _norm(str(order.symbol))
        if sym_norm in crypto_sym_set and order.side.value.lower() == "buy":
            try:
                alpaca.cancel_order_by_id(str(order.id))
                canceled_for.add(sym_norm)
                logger.info(f"  {sym_norm}: canceled stale buy order {order.id} @ {order.limit_price}")
            except Exception as e:
                logger.warning(f"  {sym_norm}: failed to cancel stale order: {e}")

    for sym, plan in signals.items():
        try:
            if plan.direction != "long" or plan.confidence < 0.4:
                continue

            # Safety validation
            pos = snapshot.alpaca_positions.get(sym)
            current_price = pos.current_price if pos else plan.buy_price
            safe, reason = validate_plan_safety(plan, current_price, fee_bps=16.0)
            if not safe:
                logger.warning(f"  {sym}: SAFETY BLOCKED - {reason}")
                continue

            # Check existing position
            pos_value = pos.market_value if pos else 0.0
            # Use alpaca_cash (not buying_power) — crypto is not margined
            # Cap per-symbol crypto at 10% of account cash
            max_position = snapshot.alpaca_cash * 0.10

            if pos_value > 0 and pos_value >= max_position:
                logger.info(f"  {sym}: already at max position (${pos_value:.0f})")
                # Place take-profit sell if we have a position
                if pos and plan.sell_price > 0 and plan.sell_price > pos.current_price:
                    logger.info(f"  {sym}: updating TP sell @ ${plan.sell_price:.2f}")
                    if not dry_run:
                        req = LimitOrderRequest(
                            symbol=sym,
                            qty=round(pos.qty, 8),
                            side=OrderSide.SELL,
                            type="limit",
                            time_in_force=TimeInForce.GTC,
                            limit_price=round(plan.sell_price, 2),
                        )
                        result = alpaca.submit_order(req)
                        orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price,
                                       "qty": pos.qty, "order_id": str(result.id)})
                    else:
                        logger.info(f"    [DRY RUN] Would TP sell {pos.qty:.6f} @ ${plan.sell_price:.2f}")
                        orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price,
                                       "qty": pos.qty, "dry_run": True})
                continue

            # Calculate order size (cash only, no margin for crypto)
            available = snapshot.alpaca_cash
            trade_size = min(max_position - pos_value, available * 0.45)

            if trade_size < 12:  # Min notional ~$10
                logger.info(f"  {sym}: trade too small (${trade_size:.2f})")
                continue

            buy_price = plan.buy_price if plan.buy_price > 0 else 0.0
            if buy_price <= 0:
                continue

            qty = trade_size / buy_price

            logger.info(f"  {sym}: BUY {qty:.6f} @ ${buy_price:.2f} (${trade_size:.0f})")
            if not dry_run:
                req = LimitOrderRequest(
                    symbol=sym,
                    qty=round(qty, 8),
                    side=OrderSide.BUY,
                    type="limit",
                    time_in_force=TimeInForce.GTC,
                    limit_price=round(buy_price, 2),
                )
                result = alpaca.submit_order(req)
                orders.append({"symbol": sym, "action": "buy", "price": buy_price,
                                "qty": qty, "order_id": str(result.id)})
            else:
                logger.info(f"    [DRY RUN]")
                orders.append({"symbol": sym, "action": "buy", "price": buy_price,
                                "qty": qty, "dry_run": True})

        except Exception as e:
            logger.error(f"  {sym}: execution error: {e}")

    return orders


# ---------------------------------------------------------------------------
# Stock signal generation (Alpaca + Gemini)
# ---------------------------------------------------------------------------

def get_stock_signals(
    symbols: list[str],
    snapshot: UnifiedPortfolioSnapshot,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    dry_run: bool = True,
) -> dict[str, TradePlan]:
    """Generate LLM trading signals for stock symbols using Alpaca OHLCV data."""
    import pandas as pd
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from datetime import timedelta
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    data_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    signals = {}
    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Hour,
                start=now - timedelta(hours=78),
                end=now,
                limit=72,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            df = bars.df
            if df is None or len(df) < 12:
                logger.warning(f"  {sym}: insufficient bars ({0 if df is None else len(df)})")
                continue

            # Flatten multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level="symbol")
            df = df.reset_index()
            if "timestamp" not in df.columns and "index" in df.columns:
                df.rename(columns={"index": "timestamp"}, inplace=True)
            if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

            history = []
            for _, row in df.iterrows():
                history.append({
                    "timestamp": str(row.get("timestamp", ""))[:16],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })

            current_price = history[-1]["close"]

            # RL signal context (if stock bridge loaded)
            rl_hint = ""
            rl_bridge = get_rl_bridge_stock()
            if rl_bridge is not None:
                try:
                    rl_plans = rl_bridge.generate_plans([sym], {sym: history})
                    if sym in rl_plans:
                        rp = rl_plans[sym]
                        rl_hint = (f"\nRL MODEL SIGNAL: direction={rp.direction} "
                                   f"confidence={rp.confidence:.2f} "
                                   f"(from stocks13_featlag1 validated model)")
                except Exception as e:
                    logger.debug(f"  {sym}: RL signal error (non-fatal): {e}")

            prompt = build_unified_prompt(
                symbol=sym,
                history_rows=history,
                current_price=current_price,
                snapshot=snapshot,
                asset_class="stock",
            ) + rl_hint

            plan = call_llm(prompt, model=model, thinking_level=thinking_level)

            # Safety gate before accepting the plan
            ok, reason = validate_plan_safety(plan, current_price, fee_bps=10.0)
            if not ok:
                logger.warning(f"  {sym}: plan REJECTED by safety check — {reason}")
                continue

            signals[sym] = plan
            logger.info(f"  {sym}: {plan.direction} conf={plan.confidence:.2f} "
                        f"buy=${plan.buy_price:.2f} sell=${plan.sell_price:.2f} | {plan.reasoning[:60]}")

        except Exception as e:
            logger.error(f"  {sym}: error: {e}")

    return signals


_STOCK_FEE_BPS = 5.0  # 5 bps round-trip; safety margin for limit price checks


def _validate_trade_plan(plan: TradePlan, symbol: str) -> tuple[bool, str]:
    """Safety gate: ensure the plan is safe to execute live.

    Rules (must ALL pass):
      1. direction must be long, short, or hold
      2. buy_price must be positive when direction == long
      3. sell_price > buy_price * (1 + 2*fee) — ensures the round-trip is profitable
      4. confidence >= 0.4
      5. sell_price > 0 when direction == long
    """
    fee_factor = 1 + 2 * _STOCK_FEE_BPS / 10_000
    if plan.direction not in ("long", "short", "hold"):
        return False, f"unknown direction: {plan.direction!r}"
    if plan.direction == "long":
        if plan.buy_price <= 0:
            return False, "buy_price must be > 0 for long"
        if plan.sell_price <= 0:
            return False, "sell_price must be > 0 for long"
        if plan.sell_price <= plan.buy_price * fee_factor:
            return False, (f"sell_price ${plan.sell_price:.2f} <= buy_price ${plan.buy_price:.2f} "
                           f"* fee_factor {fee_factor:.5f} — not profitable after fees")
    if plan.confidence < 0.4:
        return False, f"confidence {plan.confidence:.2f} < threshold 0.40"
    return True, "ok"


def execute_stock_signals(
    signals: dict[str, TradePlan],
    snapshot: UnifiedPortfolioSnapshot,
    dry_run: bool = True,
) -> list[dict]:
    """Execute stock trading signals on Alpaca with safety validation."""
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []
    alpaca = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

    for sym, plan in signals.items():
        try:
            # Safety validation — applied to ALL orders
            ok, reason = _validate_trade_plan(plan, sym)
            if not ok:
                logger.warning(f"  {sym}: REJECTED — {reason}")
                continue

            # Take-profit on existing position (set sell limit regardless of direction)
            pos = snapshot.alpaca_positions.get(sym)
            if pos and pos.qty > 0 and plan.sell_price > 0 and plan.sell_price > pos.current_price:
                logger.info(f"  {sym}: updating take-profit sell @ ${plan.sell_price:.2f} ({pos.qty:.2f} shares)")
                if not dry_run:
                    req = LimitOrderRequest(
                        symbol=sym,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                        type="limit",
                        time_in_force=TimeInForce.DAY,
                        limit_price=round(plan.sell_price, 2),
                    )
                    result = alpaca.submit_order(req)
                    orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price,
                                   "qty": pos.qty, "order_id": str(result.id)})
                else:
                    orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price,
                                   "qty": pos.qty, "dry_run": True})
                continue

            # New long entry
            if plan.direction != "long" or plan.confidence < 0.5 or plan.buy_price <= 0:
                continue

            # Don't add to a position we already hold
            if pos and pos.qty > 0:
                continue

            total_stock = max(snapshot.total_stock_value, 1.0)
            max_position = total_stock * 0.20  # 20% per stock, 5 max
            available = snapshot.alpaca_buying_power * 0.45
            trade_usd = min(max_position, available)

            if trade_usd < 50:
                logger.info(f"  {sym}: insufficient buying power (${trade_usd:.0f})")
                continue

            qty = trade_usd / plan.buy_price
            qty = round(qty, 2)
            if qty < 0.01:
                continue

            logger.info(f"  {sym}: BUY {qty:.2f} @ ${plan.buy_price:.2f} (${trade_usd:.0f})")
            if not dry_run:
                req = LimitOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=OrderSide.BUY,
                    type="limit",
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(plan.buy_price, 2),
                )
                result = alpaca.submit_order(req)
                orders.append({"symbol": sym, "action": "buy", "price": plan.buy_price,
                                "qty": qty, "order_id": str(result.id)})
            else:
                logger.info(f"    [DRY RUN]")
                orders.append({"symbol": sym, "action": "buy", "price": plan.buy_price,
                                "qty": qty, "dry_run": True})

        except Exception as e:
            logger.error(f"  {sym}: execution error: {e}")

    return orders


# ---------------------------------------------------------------------------
# Main trading cycle
# ---------------------------------------------------------------------------

def run_cycle(
    crypto_symbols: list[str],
    stock_symbols: list[str] | None = None,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    dry_run: bool = True,
) -> dict:
    """Run one unified trading cycle."""
    now = datetime.now(timezone.utc)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"UNIFIED TRADING CYCLE: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # 1. Build snapshot
    snapshot = build_snapshot(now)
    logger.info(f"Regime: {snapshot.regime}")
    logger.info(f"Stocks: ${snapshot.total_stock_value:,.0f} | Crypto: ${snapshot.total_crypto_value:,.0f}")
    if snapshot.alpaca_positions:
        crypto_syms = set(CRYPTO_SYMBOLS)
        for sym, pos in snapshot.alpaca_positions.items():
            if sym in crypto_syms:
                logger.info(f"  Crypto: {sym} {pos.qty:.6f} @ ${pos.current_price:.2f} (${pos.market_value:.0f})")
            else:
                logger.info(f"  Stock: {sym} {pos.qty} @ ${pos.avg_price:.2f} (${pos.market_value:.0f})")
    logger.info(f"{'=' * 70}")

    results = {"regime": snapshot.regime, "orders": []}

    # 2. Handle regime-specific logic
    if snapshot.regime == "PRE_MARKET":
        logger.info("\n--- PRE-MARKET: Checking crypto backout opportunities ---")
        # TODO: Load best stock edges from meta-selector
        best_stock_edges = {}  # Will integrate with meta_live_runtime
        candidates = select_backout_candidates(snapshot, best_stock_edges)
        if candidates:
            backout_results = execute_backout(candidates, dry_run=dry_run)
            results["backout"] = backout_results
        else:
            logger.info("  No backout candidates")

    # 3. Generate and execute crypto signals (always, except pure stock hours)
    if snapshot.regime in ("CRYPTO_ONLY", "PRE_MARKET", "POST_MARKET", "STOCK_HOURS"):
        logger.info(f"\n--- CRYPTO SIGNALS ({len(crypto_symbols)} symbols) ---")
        crypto_signals = get_crypto_signals(
            crypto_symbols, snapshot, model, thinking_level, dry_run
        )
        if crypto_signals:
            crypto_orders = execute_crypto_signals(crypto_signals, snapshot, dry_run)
            results["orders"].extend(crypto_orders)

    # 4. Stock signals during market hours (Gemini-driven)
    if snapshot.regime == "STOCK_HOURS":
        syms = stock_symbols or STOCK_SYMBOLS
        logger.info(f"\n--- STOCK SIGNALS ({len(syms)} symbols) ---")
        stock_signals = get_stock_signals(syms, snapshot, model, thinking_level, dry_run)
        if stock_signals:
            stock_orders = execute_stock_signals(stock_signals, snapshot, dry_run)
            results["orders"].extend(stock_orders)
            results["stock_signals"] = {s: {"direction": p.direction, "confidence": p.confidence}
                                         for s, p in stock_signals.items()}

    # 5. Check conditional order triggers
    pending_fills = read_pending_fills(since_minutes=65)
    if pending_fills:
        logger.info(f"\n--- CONDITIONAL TRIGGERS: {len(pending_fills)} recent fills ---")
        for fill in pending_fills:
            logger.info(f"  Fill: {fill['symbol']} {fill['action']} @ ${fill.get('fill_price', '?')}")

    # 6. Persist state
    save_snapshot(snapshot)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Cycle complete: {len(results.get('orders', []))} orders")
    logger.info(f"{'=' * 70}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Stock+Crypto Trading Orchestrator")
    parser.add_argument("--crypto-symbols", nargs="+", default=CRYPTO_SYMBOLS)
    parser.add_argument("--stock-symbols", nargs="+", default=STOCK_SYMBOLS)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--thinking-level", default="HIGH")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()

    dry_run = not args.live
    if args.live:
        logger.warning("LIVE TRADING MODE")
        time.sleep(3)

    while True:
        try:
            run_cycle(
                crypto_symbols=args.crypto_symbols,
                stock_symbols=args.stock_symbols,
                model=args.model,
                thinking_level=args.thinking_level,
                dry_run=dry_run,
            )
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            import traceback
            traceback.print_exc()

        if args.once:
            break

        # Sleep until :01 of next hour
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=1, second=0, microsecond=0)
        if next_hour <= now:
            from datetime import timedelta
            next_hour += timedelta(hours=1)
        sleep_secs = (next_hour - now).total_seconds()
        logger.info(f"Next cycle at {next_hour.strftime('%H:%M')} UTC ({sleep_secs:.0f}s)")
        time.sleep(sleep_secs)


if __name__ == "__main__":
    main()
