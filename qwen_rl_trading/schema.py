"""Pydantic models for structured trading plan output and simulator bridge."""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class SymbolPlan(BaseModel):
    symbol: str
    action: TradeAction
    allocation_pct: float = Field(ge=0, le=100)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    max_hold_hours: int = Field(ge=1, le=24, default=6)
    confidence: float = Field(ge=0, le=1, default=0.5)
    reasoning: str = Field(default="", max_length=300)


class TradingPlan(BaseModel):
    plans: list[SymbolPlan] = Field(default_factory=list)
    cash_reserve_pct: float = Field(ge=0, le=100, default=20.0)
    market_regime: MarketRegime = MarketRegime.RANGING


def validate_plan(text: str) -> Optional[TradingPlan]:
    """Parse text into TradingPlan. Returns None on failure."""
    text = text.strip()
    # try direct JSON parse
    try:
        return TradingPlan.model_validate_json(text)
    except Exception:
        pass
    # try extracting JSON from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return TradingPlan.model_validate_json(m.group(1))
        except Exception:
            pass
    # try finding first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return TradingPlan.model_validate_json(m.group(0))
        except Exception:
            pass
    return None


def plan_to_sim_actions(
    plan: TradingPlan,
    forward_bars: pd.DataFrame,
    initial_cash: float = 10_000.0,
) -> pd.DataFrame:
    """Convert a TradingPlan to simulator-compatible actions DataFrame.

    forward_bars must have columns: timestamp, symbol, open, high, low, close, volume.
    Returns actions-only DataFrame (timestamp, symbol, buy_price, sell_price,
    buy_amount, sell_amount). OHLCV stays in forward_bars -- the simulator
    merges the two on (timestamp, symbol).
    """
    if plan.plans is None or len(plan.plans) == 0:
        return pd.DataFrame()

    available_cash = initial_cash * (1 - plan.cash_reserve_pct / 100.0)
    total_alloc = sum(p.allocation_pct for p in plan.plans if p.action != TradeAction.FLAT)
    if total_alloc <= 0:
        return pd.DataFrame()

    rows = []
    for sp in plan.plans:
        if sp.action == TradeAction.FLAT:
            continue

        sym_upper = sp.symbol.upper()
        sym_bars = forward_bars[forward_bars["symbol"].str.upper() == sym_upper].copy()
        if sym_bars.empty:
            continue

        sym_bars = sym_bars.sort_values("timestamp").head(sp.max_hold_hours)
        alloc_cash = available_cash * (sp.allocation_pct / total_alloc) * (total_alloc / 100.0)

        for _, bar in sym_bars.iterrows():
            row = {
                "timestamp": bar["timestamp"],
                "symbol": sym_upper,
                "buy_price": 0.0,
                "sell_price": 0.0,
                "buy_amount": 0.0,
                "sell_amount": 0.0,
            }

            if sp.action == TradeAction.LONG:
                row["buy_price"] = sp.entry_price
                row["buy_amount"] = alloc_cash / sp.entry_price if sp.entry_price > 0 else 0.0
                row["sell_price"] = sp.take_profit
                row["sell_amount"] = row["buy_amount"]
            elif sp.action == TradeAction.SHORT:
                row["sell_price"] = sp.entry_price
                row["sell_amount"] = alloc_cash / sp.entry_price if sp.entry_price > 0 else 0.0
                row["buy_price"] = sp.take_profit
                row["buy_amount"] = row["sell_amount"]

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
