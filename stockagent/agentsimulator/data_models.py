"""Dataclasses describing simulator contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class ExecutionSession(str, Enum):
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"

    @classmethod
    def from_value(cls, value: str) -> "ExecutionSession":
        value = (value or cls.MARKET_OPEN.value).strip().lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unsupported execution session: {value!r}")


class PlanActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    EXIT = "exit"
    HOLD = "hold"

    @classmethod
    def from_value(cls, value: str) -> "PlanActionType":
        value = (value or cls.HOLD.value).strip().lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unsupported action type: {value!r}")


@dataclass
class TradingInstruction:
    symbol: str
    action: PlanActionType
    quantity: float
    execution_session: ExecutionSession = ExecutionSession.MARKET_OPEN
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["execution_session"] = self.execution_session.value
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingInstruction":
        symbol = str(data.get("symbol", "")).upper()
        if not symbol:
            raise ValueError("Instruction missing symbol")
        action = PlanActionType.from_value(data.get("action"))
        execution_session = ExecutionSession.from_value(data.get("execution_session"))
        quantity = float(data.get("quantity", 0) or 0)
        entry_price = cls._maybe_float(data.get("entry_price"))
        exit_price = cls._maybe_float(data.get("exit_price"))
        exit_reason = data.get("exit_reason")
        notes = data.get("notes")
        return cls(
            symbol=symbol,
            action=action,
            quantity=quantity,
            execution_session=execution_session,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            notes=notes,
        )

    @staticmethod
    def _maybe_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass
class TradingPlan:
    target_date: date
    instructions: List[TradingInstruction] = field(default_factory=list)
    risk_notes: Optional[str] = None
    focus_symbols: List[str] = field(default_factory=list)
    stop_trading_symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_window: ExecutionSession = ExecutionSession.MARKET_OPEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_date": self.target_date.isoformat(),
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "risk_notes": self.risk_notes,
            "focus_symbols": self.focus_symbols,
            "stop_trading_symbols": self.stop_trading_symbols,
            "metadata": self.metadata,
            "execution_window": self.execution_window.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingPlan":
        raw_date = data.get("target_date")
        if not raw_date:
            raise ValueError("Trading plan missing target_date")
        try:
            target_date = (
                raw_date
                if isinstance(raw_date, date)
                else datetime.fromisoformat(raw_date).date()
            )
        except ValueError as exc:
            raise ValueError(f"Invalid target_date {raw_date!r}") from exc

        instructions_raw = data.get("instructions", [])
        if not isinstance(instructions_raw, Iterable):
            raise ValueError("Plan instructions must be iterable")
        instructions = [
            TradingInstruction.from_dict(item) for item in instructions_raw
        ]
        risk_notes = data.get("risk_notes")
        focus_symbols = [str(sym).upper() for sym in data.get("focus_symbols", [])]
        stop_trading_symbols = [
            str(sym).upper() for sym in data.get("stop_trading_symbols", [])
        ]
        metadata = data.get("metadata") or {}
        execution_window = ExecutionSession.from_value(
            data.get("execution_window", ExecutionSession.MARKET_OPEN.value)
        )
        return cls(
            target_date=target_date,
            instructions=instructions,
            risk_notes=risk_notes,
            focus_symbols=focus_symbols,
            stop_trading_symbols=stop_trading_symbols,
            metadata=metadata,
            execution_window=execution_window,
        )


@dataclass
class TradingPlanEnvelope:
    plan: TradingPlan
    commentary: Optional[str] = None

    def to_json(self) -> str:
        payload = {
            "plan": self.plan.to_dict(),
            "commentary": self.commentary,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "TradingPlanEnvelope":
        data = json.loads(raw)
        if "plan" not in data:
            raise ValueError("GPT response missing plan key")
        plan = TradingPlan.from_dict(data["plan"])
        commentary = data.get("commentary")
        return cls(plan=plan, commentary=commentary)


@dataclass
class AccountPosition:
    symbol: str
    quantity: float
    side: str
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float

    @classmethod
    def from_alpaca(cls, position_obj: Any) -> "AccountPosition":
        return cls(
            symbol=str(position_obj.symbol).upper(),
            quantity=float(position_obj.qty),
            side=str(position_obj.side),
            market_value=float(position_obj.market_value),
            avg_entry_price=float(position_obj.avg_entry_price),
            unrealized_pl=float(getattr(position_obj, "unrealized_pl", 0.0)),
            unrealized_plpc=float(getattr(position_obj, "unrealized_plpc", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: Optional[float]
    timestamp: datetime
    positions: List[AccountPosition] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "timestamp": self.timestamp.isoformat(),
            "positions": [position.to_dict() for position in self.positions],
        }

    def has_position(self, symbol: str) -> bool:
        symbol = symbol.upper()
        return any(position.symbol == symbol for position in self.positions)
