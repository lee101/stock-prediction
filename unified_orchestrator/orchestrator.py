"""Unified Stock+Crypto Trading Orchestrator.

24/7 hourly loop that coordinates Alpaca (stocks) and Binance (crypto).
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


# ---------------------------------------------------------------------------
# Crypto signal generation
# ---------------------------------------------------------------------------

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
    """Generate LLM trading signals for crypto symbols."""
    import pandas as pd
    from src.binan import binance_wrapper as bw

    signals = {}
    client = bw.get_client()

    for sym in symbols:
        pair = CRYPTO_PAIRS.get(sym, sym.replace("USD", "USDT"))
        try:
            klines = client.get_klines(symbol=pair, interval="1h", limit=72)
        except Exception as e:
            logger.error(f"  {sym}: klines error: {e}")
            continue

        history = [{
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").isoformat(),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
        } for k in klines]

        if len(history) < 12:
            continue

        current_price = history[-1]["close"]

        prompt = build_unified_prompt(
            symbol=sym,
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

    return signals


# ---------------------------------------------------------------------------
# Crypto execution
# ---------------------------------------------------------------------------

def execute_crypto_signals(
    signals: dict[str, TradePlan],
    snapshot: UnifiedPortfolioSnapshot,
    dry_run: bool = True,
) -> list[dict]:
    """Execute crypto trading signals on Binance."""
    from rl_trading_agent_binance_prompt import build_live_prompt  # noqa: unused but validates import

    orders = []
    for sym, plan in signals.items():
        if plan.direction != "long" or plan.confidence < 0.4:
            continue

        pair = CRYPTO_PAIRS.get(sym, sym.replace("USD", "USDT"))
        base_asset = sym.replace("USD", "")

        # Check existing position
        pos = snapshot.binance_positions.get(base_asset)
        pos_value = pos.market_value if pos else 0.0
        total_crypto = max(snapshot.total_crypto_value, 1.0)
        max_position = total_crypto * 0.25

        if pos_value > 0 and pos_value >= max_position:
            logger.info(f"  {sym}: already at max position (${pos_value:.0f})")
            # Place take-profit if we have position
            if pos and plan.sell_price > pos.current_price:
                step = TradingStep(
                    step_id=f"tp_{sym}_{int(time.time())}",
                    broker="binance",
                    action="sell",
                    symbol=sym,
                    binance_pair=pair,
                    limit_price=plan.sell_price,
                    qty=pos.qty,
                )
                if dry_run:
                    logger.info(f"    [DRY RUN] Would place TP sell @ ${plan.sell_price:.2f}")
                else:
                    from unified_orchestrator.conditional_orders import execute_step_binance
                    execute_step_binance(step, dry_run=False)
                orders.append({"symbol": sym, "action": "sell_tp", "price": plan.sell_price})
            continue

        # Calculate order size
        available = snapshot.binance_usdt + snapshot.binance_fdusd
        trade_size = min(max_position - pos_value, available * 0.45)

        if trade_size < 12:  # Min notional
            logger.info(f"  {sym}: trade too small (${trade_size:.2f})")
            continue

        buy_price = plan.buy_price if plan.buy_price > 0 else 0.0
        if buy_price <= 0:
            continue

        qty = trade_size / buy_price

        logger.info(f"  {sym}: BUY {qty:.6f} @ ${buy_price:.2f} (${trade_size:.0f})")
        if dry_run:
            logger.info(f"    [DRY RUN]")
        else:
            step = TradingStep(
                step_id=f"buy_{sym}_{int(time.time())}",
                broker="binance",
                action="buy",
                symbol=sym,
                binance_pair=pair,
                limit_price=buy_price,
                qty=qty,
            )
            from unified_orchestrator.conditional_orders import execute_step_binance
            execute_step_binance(step, dry_run=False)

        orders.append({"symbol": sym, "action": "buy", "price": buy_price, "qty": qty})

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

            prompt = build_unified_prompt(
                symbol=sym,
                history_rows=history,
                current_price=current_price,
                snapshot=snapshot,
                asset_class="stock",
            )

            plan = call_llm(prompt, model=model, thinking_level=thinking_level)
            signals[sym] = plan
            logger.info(f"  {sym}: {plan.direction} conf={plan.confidence:.2f} "
                        f"buy=${plan.buy_price:.2f} sell=${plan.sell_price:.2f} | {plan.reasoning[:60]}")

        except Exception as e:
            logger.error(f"  {sym}: error: {e}")

    return signals


def execute_stock_signals(
    signals: dict[str, TradePlan],
    snapshot: UnifiedPortfolioSnapshot,
    dry_run: bool = True,
) -> list[dict]:
    """Execute stock trading signals on Alpaca."""
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []
    alpaca = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

    for sym, plan in signals.items():
        try:
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
        for sym, pos in snapshot.alpaca_positions.items():
            logger.info(f"  Stock: {sym} {pos.qty} @ ${pos.avg_price:.2f} (${pos.market_value:.0f})")
    if snapshot.binance_positions:
        for asset, pos in snapshot.binance_positions.items():
            logger.info(f"  Crypto: {asset} {pos.qty:.6f} (${pos.market_value:.0f})")
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
