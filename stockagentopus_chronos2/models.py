"""Pydantic models for structured trading plan outputs."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    buy = "buy"
    exit = "exit"
    hold = "hold"


class ExecutionSession(str, Enum):
    market_open = "market_open"
    market_close = "market_close"


class ExitReason(str, Enum):
    profit_target = "profit_target"
    stop_loss = "stop_loss"
    time_exit = "time_exit"
    signal_reversal = "signal_reversal"


class TradingInstruction(BaseModel):
    """A single trading instruction."""
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL)")
    action: ActionType = Field(description="Trade action: buy, exit, or hold")
    quantity: int = Field(ge=0, description="Number of shares (integer)")
    execution_session: ExecutionSession = Field(description="When to execute: market_open or market_close")
    entry_price: float = Field(ge=0, description="Limit price for entry (buy near 10th percentile)")
    exit_price: float = Field(ge=0, description="Target exit price (sell near 90th percentile)")
    exit_reason: ExitReason = Field(description="Reason for exit strategy")
    notes: str = Field(description="Brief explanation of trade rationale")


class TradingPlanMetadata(BaseModel):
    """Metadata about the trading plan."""
    capital_allocation_plan: str = Field(description="Explanation of how capital was allocated across stocks")


class TradingPlanOutput(BaseModel):
    """Complete trading plan output."""
    target_date: str = Field(description="Target trading date in YYYY-MM-DD format")
    instructions: List[TradingInstruction] = Field(
        default_factory=list,
        description="List of trading instructions for the day"
    )
    risk_notes: str = Field(description="Brief summary of risk considerations (1-2 sentences)")
    metadata: TradingPlanMetadata = Field(description="Additional plan metadata")


def get_trading_plan_schema() -> dict:
    """Return JSON schema for the trading plan."""
    return TradingPlanOutput.model_json_schema()


__all__ = [
    "ActionType",
    "ExecutionSession",
    "ExitReason",
    "TradingInstruction",
    "TradingPlanMetadata",
    "TradingPlanOutput",
    "get_trading_plan_schema",
]
