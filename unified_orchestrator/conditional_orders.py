"""Multi-step conditional order execution.

Supports chained orders like: sell ETH → if filled, buy SOL.
Uses fill events file for cross-broker coordination.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

FILL_EVENTS_FILE = Path("strategy_state/fill_events.jsonl")


@dataclass
class TradingStep:
    """A single step in a trading plan."""
    step_id: str
    broker: str  # "binance" or "alpaca"
    action: str  # "buy" or "sell"
    symbol: str  # e.g. "BTCUSD" or "NVDA"
    binance_pair: str = ""  # e.g. "BTCFDUSD" for Binance orders
    limit_price: float = 0.0
    qty: float = 0.0
    amount_usd: float = 0.0  # Alternative to qty: specify USD amount
    condition: str = "immediate"  # "immediate" | "on_fill:<step_id>" | "on_price:<symbol>:<price>"
    expiry_minutes: int = 55  # within the hour
    status: str = "pending"  # pending | submitted | filled | expired | failed


@dataclass
class TradingPlan:
    """Multi-step trading plan for one hourly cycle."""
    plan_id: str
    created_at: str
    steps: list[TradingStep]
    description: str = ""

    def immediate_steps(self) -> list[TradingStep]:
        return [s for s in self.steps if s.condition == "immediate" and s.status == "pending"]

    def conditional_steps(self, filled_step_id: str) -> list[TradingStep]:
        return [s for s in self.steps
                if s.condition == f"on_fill:{filled_step_id}" and s.status == "pending"]


# ---------------------------------------------------------------------------
# Fill event tracking
# ---------------------------------------------------------------------------

def record_fill_event(
    step: TradingStep,
    fill_price: float,
    fill_qty: float,
    plan_id: str = "",
) -> None:
    """Record a fill event for cross-broker chain triggers."""
    FILL_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan_id,
        "step_id": step.step_id,
        "broker": step.broker,
        "action": step.action,
        "symbol": step.symbol,
        "fill_price": fill_price,
        "fill_qty": fill_qty,
        "status": "filled",
    }
    with open(FILL_EVENTS_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    logger.info(f"Fill event recorded: {step.symbol} {step.action} @ {fill_price}")


def read_pending_fills(since_minutes: int = 60) -> list[dict]:
    """Read recent fill events for conditional order triggering."""
    if not FILL_EVENTS_FILE.exists():
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - since_minutes * 60
    events = []
    for line in FILL_EVENTS_FILE.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            event = json.loads(line)
            event_ts = datetime.fromisoformat(event["timestamp"]).timestamp()
            if event_ts > cutoff:
                events.append(event)
        except (json.JSONDecodeError, KeyError):
            continue
    return events


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_step_binance(step: TradingStep, dry_run: bool = True) -> bool:
    """Execute a trading step on Binance."""
    logger.info(f"  Binance {step.action} {step.symbol}: qty={step.qty:.6f} @ ${step.limit_price:.2f}")

    if dry_run:
        logger.info(f"    [DRY RUN]")
        step.status = "submitted"
        return True

    try:
        from src.binan import binance_wrapper
        from binanceneural.execution import resolve_symbol_rules, quantize_qty, quantize_price

        pair = step.binance_pair or step.symbol.replace("USD", "USDT")
        rules = resolve_symbol_rules(pair)

        qty = quantize_qty(step.qty, rules.step_size or 0.00001)
        price = quantize_price(step.limit_price, rules.tick_size or 0.01,
                               side="BUY" if step.action == "buy" else "SELL")

        side = "BUY" if step.action == "buy" else "SELL"
        order = binance_wrapper.create_order(pair, side, qty, price)
        logger.info(f"    Order placed: {order.get('orderId')}")
        step.status = "submitted"
        return True
    except Exception as e:
        logger.error(f"    Binance order failed: {e}")
        step.status = "failed"
        return False


def execute_step_alpaca(step: TradingStep, dry_run: bool = True) -> bool:
    """Execute a trading step on Alpaca."""
    logger.info(f"  Alpaca {step.action} {step.symbol}: qty={step.qty:.1f} @ ${step.limit_price:.2f}")

    if dry_run:
        logger.info(f"    [DRY RUN]")
        step.status = "submitted"
        return True

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

        client = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

        order_data = LimitOrderRequest(
            symbol=step.symbol,
            qty=step.qty,
            side=OrderSide.BUY if step.action == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=step.limit_price,
        )
        order = client.submit_order(order_data)
        logger.info(f"    Order placed: {order.id}")
        step.status = "submitted"
        return True
    except Exception as e:
        logger.error(f"    Alpaca order failed: {e}")
        step.status = "failed"
        return False


def execute_plan(plan: TradingPlan, dry_run: bool = True) -> None:
    """Execute a multi-step trading plan.

    1. Execute all immediate steps
    2. Check for fills from previous steps
    3. Trigger conditional steps when their conditions are met
    """
    logger.info(f"Executing plan: {plan.description} ({len(plan.steps)} steps)")

    # Phase 1: Execute immediate steps
    for step in plan.immediate_steps():
        if step.broker == "binance":
            execute_step_binance(step, dry_run)
        elif step.broker == "alpaca":
            execute_step_alpaca(step, dry_run)

    # Phase 2: Check for conditional triggers from recent fills
    recent_fills = read_pending_fills(since_minutes=60)
    filled_step_ids = {e["step_id"] for e in recent_fills}

    for filled_id in filled_step_ids:
        for step in plan.conditional_steps(filled_id):
            logger.info(f"  Triggering conditional step {step.step_id} (on fill of {filled_id})")
            if step.broker == "binance":
                execute_step_binance(step, dry_run)
            elif step.broker == "alpaca":
                execute_step_alpaca(step, dry_run)


def build_rebalance_plan(
    sell_symbol: str,
    sell_broker: str,
    sell_qty: float,
    sell_price: float,
    buy_symbol: str,
    buy_broker: str,
    buy_price: float,
    buy_amount_usd: float,
) -> TradingPlan:
    """Build a sell-then-buy rebalancing plan."""
    now = datetime.now(timezone.utc)
    plan_id = f"rebalance_{sell_symbol}_{buy_symbol}_{int(now.timestamp())}"

    sell_step = TradingStep(
        step_id=f"{plan_id}_sell",
        broker=sell_broker,
        action="sell",
        symbol=sell_symbol,
        limit_price=sell_price,
        qty=sell_qty,
        condition="immediate",
        expiry_minutes=30,
    )

    buy_qty = buy_amount_usd / buy_price if buy_price > 0 else 0
    buy_step = TradingStep(
        step_id=f"{plan_id}_buy",
        broker=buy_broker,
        action="buy",
        symbol=buy_symbol,
        limit_price=buy_price,
        qty=buy_qty,
        amount_usd=buy_amount_usd,
        condition=f"on_fill:{sell_step.step_id}",
        expiry_minutes=25,
    )

    return TradingPlan(
        plan_id=plan_id,
        created_at=now.isoformat(),
        steps=[sell_step, buy_step],
        description=f"Rebalance: sell {sell_symbol} → buy {buy_symbol}",
    )
