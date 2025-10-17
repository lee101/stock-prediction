"""Utilities for summarising stockagent simulation outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from stock.state import get_state_dir, resolve_state_suffix
from stock.state_utils import StateLoadError, load_all_state


@dataclass
class TradeRecord:
    symbol: str
    side: str
    pnl: float
    qty: float
    mode: str
    reason: Optional[str]
    entry_strategy: Optional[str]
    closed_at: Optional[datetime]


@dataclass
class SymbolAggregate:
    symbol: str
    trades: int
    total_pnl: float
    wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


@dataclass
class ModeAggregate:
    mode: str
    trades: int
    total_pnl: float
    wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


@dataclass
class ActivePosition:
    symbol: str
    side: str
    qty: float
    mode: str
    opened_at: Optional[datetime]


@dataclass
class SimulationSummary:
    directory: Path
    suffix: str
    trades: List[TradeRecord]
    total_pnl: float
    total_trades: int
    win_rate: float
    avg_pnl: float
    profit_factor: float
    max_drawdown: float
    start_at: Optional[datetime]
    end_at: Optional[datetime]
    symbol_stats: List[SymbolAggregate]
    mode_stats: List[ModeAggregate]
    best_trades: List[TradeRecord]
    worst_trades: List[TradeRecord]
    active_positions: List[ActivePosition]


class SummaryError(RuntimeError):
    """Raised when a summary cannot be generated."""


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - data corruption
        raise SummaryError(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError(f"Expected object root in {path}, found {type(data).__name__}")
    return data


def _parse_state_key(key: str) -> tuple[str, str]:
    if "|" in key:
        symbol, side = key.split("|", 1)
        return symbol.upper(), side.lower()
    return key.upper(), "buy"


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_state_snapshot(
    *,
    state_dir: Optional[Path] = None,
    state_suffix: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    suffix_raw = state_suffix
    suffix_resolved = resolve_state_suffix(state_suffix)
    directory = Path(state_dir) if state_dir is not None else get_state_dir()

    if not directory.exists():
        raise SummaryError(f"State directory {directory} does not exist")

    if state_dir is None:
        try:
            snapshot = load_all_state(suffix_raw)
        except StateLoadError as exc:
            raise SummaryError(str(exc)) from exc
        snapshot["__directory__"] = str(directory)
        return snapshot

    files = {
        "trade_outcomes": directory / f"trade_outcomes{suffix_resolved}.json",
        "trade_learning": directory / f"trade_learning{suffix_resolved}.json",
        "active_trades": directory / f"active_trades{suffix_resolved}.json",
        "trade_history": directory / f"trade_history{suffix_resolved}.json",
    }

    snapshot = {name: _load_json_file(path) for name, path in files.items()}
    snapshot["__directory__"] = str(directory)
    return snapshot


def _collect_trades(trade_history: Dict[str, Any]) -> List[TradeRecord]:
    trades: List[TradeRecord] = []
    for key, entries in trade_history.items():
        if not isinstance(entries, Iterable):
            continue
        symbol, side = _parse_state_key(key)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                pnl = float(entry.get("pnl", 0.0) or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            try:
                qty = float(entry.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            trades.append(
                TradeRecord(
                    symbol=symbol,
                    side=side,
                    pnl=pnl,
                    qty=qty,
                    mode=str(entry.get("mode", "unknown")),
                    reason=entry.get("reason"),
                    entry_strategy=entry.get("entry_strategy"),
                    closed_at=_parse_timestamp(entry.get("closed_at")),
                )
            )
    return trades


def _collect_active_positions(active: Dict[str, Any]) -> List[ActivePosition]:
    positions: List[ActivePosition] = []
    for key, payload in active.items():
        if not isinstance(payload, dict):
            continue
        symbol, side = _parse_state_key(key)
        try:
            qty = float(payload.get("qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        positions.append(
            ActivePosition(
                symbol=symbol,
                side=side,
                qty=qty,
                mode=str(payload.get("mode", "unknown")),
                opened_at=_parse_timestamp(payload.get("opened_at")),
            )
        )
    positions.sort(key=lambda item: item.opened_at or datetime.min)
    return positions


def summarize_trades(
    *,
    snapshot: Dict[str, Dict[str, Any]],
    directory: Path,
    suffix: Optional[str],
) -> SimulationSummary:
    trade_history = snapshot.get("trade_history", {})
    trades = _collect_trades(trade_history if isinstance(trade_history, dict) else {})
    trades.sort(key=lambda record: record.closed_at or datetime.min)

    total_trades = len(trades)
    total_pnl = sum(trade.pnl for trade in trades)
    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl < 0)
    win_rate = wins / total_trades if total_trades else 0.0
    avg_pnl = total_pnl / total_trades if total_trades else 0.0

    positive_sum = sum(trade.pnl for trade in trades if trade.pnl > 0)
    negative_sum = sum(trade.pnl for trade in trades if trade.pnl < 0)
    if negative_sum < 0:
        profit_factor = positive_sum / abs(negative_sum) if positive_sum > 0 else 0.0
    else:
        profit_factor = float("inf") if positive_sum > 0 else 0.0

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        cumulative += trade.pnl
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_drawdown = max(max_drawdown, drawdown)

    start_at = trades[0].closed_at if trades else None
    end_at = trades[-1].closed_at if trades else None

    symbol_stats_map: Dict[str, SymbolAggregate] = {}
    for trade in trades:
        stats = symbol_stats_map.setdefault(
            trade.symbol,
            SymbolAggregate(symbol=trade.symbol, trades=0, total_pnl=0.0, wins=0),
        )
        stats.trades += 1
        stats.total_pnl += trade.pnl
        if trade.pnl > 0:
            stats.wins += 1

    mode_stats_map: Dict[str, ModeAggregate] = {}
    for trade in trades:
        stats = mode_stats_map.setdefault(
            trade.mode,
            ModeAggregate(mode=trade.mode, trades=0, total_pnl=0.0, wins=0),
        )
        stats.trades += 1
        stats.total_pnl += trade.pnl
        if trade.pnl > 0:
            stats.wins += 1

    symbol_stats = sorted(symbol_stats_map.values(), key=lambda item: item.total_pnl)
    mode_stats = sorted(mode_stats_map.values(), key=lambda item: item.mode)

    best_trades = sorted(trades, key=lambda record: record.pnl, reverse=True)[:3]
    worst_trades = sorted(trades, key=lambda record: record.pnl)[:3]

    active_positions = _collect_active_positions(snapshot.get("active_trades", {}))

    return SimulationSummary(
        directory=directory,
        suffix=resolve_state_suffix(suffix),
        trades=trades,
        total_pnl=total_pnl,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_pnl=avg_pnl,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        start_at=start_at,
        end_at=end_at,
        symbol_stats=symbol_stats,
        mode_stats=mode_stats,
        best_trades=best_trades,
        worst_trades=worst_trades,
        active_positions=active_positions,
    )


def format_summary(summary: SimulationSummary, label: str) -> str:
    def fmt_currency(value: float) -> str:
        return f"${value:,.2f}"

    def fmt_dt(value: Optional[datetime]) -> str:
        return value.isoformat() if value else "n/a"

    lines: List[str] = []
    suffix_display = summary.suffix or "<default>"
    lines.append(f"[{label}] State: {summary.directory} (suffix {suffix_display})")

    if summary.total_trades == 0:
        lines.append("  No closed trades recorded.")
    else:
        lines.append(
            f"  Closed trades: {summary.total_trades} | Realized PnL: {fmt_currency(summary.total_pnl)} "
            f"| Avg/trade: {fmt_currency(summary.avg_pnl)} | Win rate: {summary.win_rate:.1%}"
        )
        lines.append(
            f"  Period: {fmt_dt(summary.start_at)} → {fmt_dt(summary.end_at)} | "
            f"Max drawdown: {fmt_currency(-summary.max_drawdown)} | "
            f"Profit factor: {'∞' if summary.profit_factor == float('inf') else f'{summary.profit_factor:.2f}'}"
        )

        worst_symbols = [stat for stat in summary.symbol_stats if stat.total_pnl < 0][:3]
        best_symbols = [stat for stat in reversed(summary.symbol_stats) if stat.total_pnl > 0][:3]

        if worst_symbols:
            lines.append("  Worst symbols:")
            for stat in worst_symbols:
                lines.append(
                    f"    - {stat.symbol}: {fmt_currency(stat.total_pnl)} over {stat.trades} trades "
                    f"(win {stat.win_rate:.1%})"
                )
        if best_symbols:
            lines.append("  Best symbols:")
            for stat in best_symbols:
                lines.append(
                    f"    - {stat.symbol}: {fmt_currency(stat.total_pnl)} over {stat.trades} trades "
                    f"(win {stat.win_rate:.1%})"
                )

        if summary.best_trades:
            lines.append("  Top trades:")
            for trade in summary.best_trades:
                lines.append(
                    f"    - {trade.symbol} {trade.side} {trade.mode} "
                    f"{fmt_currency(trade.pnl)} qty={trade.qty:.3f} closed={fmt_dt(trade.closed_at)}"
                )

        if summary.worst_trades:
            lines.append("  Bottom trades:")
            for trade in summary.worst_trades:
                lines.append(
                    f"    - {trade.symbol} {trade.side} {trade.mode} "
                    f"{fmt_currency(trade.pnl)} qty={trade.qty:.3f} closed={fmt_dt(trade.closed_at)}"
                )

    if summary.active_positions:
        lines.append("  Active positions:")
        for position in summary.active_positions:
            lines.append(
                f"    - {position.symbol} {position.side} mode={position.mode} "
                f"qty={position.qty:.4f} opened={fmt_dt(position.opened_at)}"
            )

    return "\n".join(lines)
