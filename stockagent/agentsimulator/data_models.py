"""Dataclasses describing simulator contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from collections.abc import Mapping, Sequence


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
    entry_price: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["action"] = self.action.value
        payload["execution_session"] = self.execution_session.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TradingInstruction":
        symbol_raw = data.get("symbol", "")
        symbol = str(symbol_raw).upper()
        if not symbol:
            raise ValueError("Instruction missing symbol")
        action_raw = str(data.get("action", ""))
        action = PlanActionType.from_value(action_raw)
        execution_session_raw = str(data.get("execution_session", ""))
        execution_session = ExecutionSession.from_value(execution_session_raw)
        quantity = cls._coerce_float(data.get("quantity"), default=0.0)
        entry_price = cls._maybe_float(data.get("entry_price"))
        exit_price = cls._maybe_float(data.get("exit_price"))
        exit_reason_raw = data.get("exit_reason")
        exit_reason = exit_reason_raw if isinstance(exit_reason_raw, str) else None
        notes_raw = data.get("notes")
        notes = notes_raw if isinstance(notes_raw, str) else None
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
    def _maybe_float(value: object) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_float(value: object, *, default: float) -> float:
        maybe = TradingInstruction._maybe_float(value)
        if maybe is None:
            return default
        return maybe


@dataclass
class TradingPlan:
    target_date: date
    instructions: list[TradingInstruction] = field(default_factory=list)
    risk_notes: str | None = None
    focus_symbols: list[str] = field(default_factory=list)
    stop_trading_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    execution_window: ExecutionSession = ExecutionSession.MARKET_OPEN

    def to_dict(self) -> dict[str, object]:
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
    def from_dict(cls, data: Mapping[str, object]) -> "TradingPlan":
        raw_date = data.get("target_date")
        if raw_date is None:
            raise ValueError("Trading plan missing target_date")
        if isinstance(raw_date, date):
            target_date = raw_date
        elif isinstance(raw_date, str):
            try:
                target_date = datetime.fromisoformat(raw_date).date()
            except ValueError as exc:
                raise ValueError(f"Invalid target_date {raw_date!r}") from exc
        else:
            raise ValueError(f"Unsupported target_date type: {type(raw_date)!r}")

        instructions_obj = data.get("instructions", [])
        if not isinstance(instructions_obj, Sequence):
            raise ValueError("Plan instructions must be a sequence")
        instructions: list[TradingInstruction] = []
        for item in instructions_obj:
            if not isinstance(item, Mapping):
                raise ValueError("Plan instruction entries must be mappings")
            normalized_item: dict[str, object] = {str(key): value for key, value in item.items()}
            instructions.append(TradingInstruction.from_dict(normalized_item))

        risk_notes_raw = data.get("risk_notes")
        risk_notes = risk_notes_raw if isinstance(risk_notes_raw, str) else None
        focus_symbols_raw = data.get("focus_symbols", [])
        focus_symbols = [sym.upper() for sym in focus_symbols_raw if isinstance(sym, str)] if isinstance(focus_symbols_raw, Sequence) else []

        stop_symbols_raw = data.get("stop_trading_symbols", [])
        stop_trading_symbols = [sym.upper() for sym in stop_symbols_raw if isinstance(sym, str)] if isinstance(stop_symbols_raw, Sequence) else []

        metadata_obj = data.get("metadata")
        metadata: dict[str, object] = {}
        if isinstance(metadata_obj, Mapping):
            for key, value in metadata_obj.items():
                metadata[str(key)] = value

        execution_window_raw = data.get("execution_window")
        execution_window = (
            ExecutionSession.from_value(execution_window_raw)
            if isinstance(execution_window_raw, str)
            else ExecutionSession.MARKET_OPEN
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
    commentary: str | None = None

    def to_json(self) -> str:
        payload = {
            "plan": self.plan.to_dict(),
            "commentary": self.commentary,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "TradingPlanEnvelope":
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("GPT response payload must be an object")
        if "plan" not in payload:
            raise ValueError("GPT response missing plan key")
        plan_data = payload["plan"]
        if not isinstance(plan_data, Mapping):
            raise ValueError("Plan payload must be a mapping")
        plan = TradingPlan.from_dict(plan_data)
        commentary_raw = payload.get("commentary")
        commentary = commentary_raw if isinstance(commentary_raw, str) else None
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
    def from_alpaca(cls, position_obj: object) -> "AccountPosition":
        def _float_attr(name: str, default: float = 0.0) -> float:
            raw = getattr(position_obj, name, default)
            if raw in (None, ""):
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        symbol = str(getattr(position_obj, "symbol", "")).upper()
        side = str(getattr(position_obj, "side", ""))
        return cls(
            symbol=symbol,
            quantity=_float_attr("qty"),
            side=side,
            market_value=_float_attr("market_value"),
            avg_entry_price=_float_attr("avg_entry_price"),
            unrealized_pl=_float_attr("unrealized_pl"),
            unrealized_plpc=_float_attr("unrealized_plpc"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float | None
    timestamp: datetime
    positions: list[AccountPosition] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
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
