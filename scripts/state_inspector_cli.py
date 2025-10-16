from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer

LOSS_BLOCK_COOLDOWN = timedelta(days=3)
POSITIONS_SHELF_PATH = Path(__file__).resolve().parents[1] / "positions_shelf.json"

app = typer.Typer(help="Inspect persisted trade state for the live trading agent.")


def _resolve_state_dir(state_dir: Optional[Path]) -> Path:
    if state_dir is not None:
        return state_dir
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "strategy_state"


def _compute_state_suffix(explicit_suffix: Optional[str]) -> str:
    suffix = explicit_suffix if explicit_suffix is not None else os.getenv("TRADE_STATE_SUFFIX", "")
    suffix = suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return suffix


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        typer.secho(f"[error] Failed to parse {path}: {exc}", fg=typer.colors.RED)
        return {}
    if not isinstance(loaded, dict):
        typer.secho(f"[warning] Expected object root in {path}, got {type(loaded).__name__}", fg=typer.colors.YELLOW)
        return {}
    return loaded


def _parse_state_key(key: str) -> Tuple[str, str]:
    if "|" in key:
        symbol, side = key.split("|", 1)
    else:
        symbol, side = key, "buy"
    return symbol, side


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    candidates = (raw, raw.replace("Z", "+00:00"))
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            break
        except ValueError:
            continue
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(ts: Optional[datetime], now: datetime) -> str:
    if ts is None:
        return "never"
    delta = now - ts
    suffix = ""
    if delta.total_seconds() >= 0:
        suffix = f"{_format_timedelta(delta)} ago"
    else:
        suffix = f"in {_format_timedelta(-delta)}"
    return f"{ts.isoformat()} ({suffix})"


def _format_timedelta(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class SymbolState:
    key: str
    symbol: str
    side: str
    outcome: Dict[str, Any]
    learning: Dict[str, Any]
    active: Dict[str, Any]
    history: List[Dict[str, Any]]

    def last_trade_at(self) -> Optional[datetime]:
        return _parse_timestamp(self.outcome.get("closed_at") if self.outcome else None)

    def last_trade_pnl(self) -> Optional[float]:
        if not self.outcome:
            return None
        return _safe_float(self.outcome.get("pnl"))

    def status(self, now: datetime) -> Tuple[str, Optional[datetime]]:
        if self.active:
            return "active", _parse_timestamp(self.active.get("opened_at"))

        probe_active = bool(self.learning.get("probe_active")) if self.learning else False
        if probe_active:
            started_at = _parse_timestamp(self.learning.get("probe_started_at"))
            return "probe-active", started_at

        pending_probe = bool(self.learning.get("pending_probe")) if self.learning else False
        if pending_probe:
            updated_at = _parse_timestamp(self.learning.get("updated_at"))
            return "pending-probe", updated_at

        pnl = self.last_trade_pnl()
        closed_at = self.last_trade_at()
        if pnl is not None and pnl < 0 and closed_at is not None:
            cooldown_expires = closed_at + LOSS_BLOCK_COOLDOWN
            if cooldown_expires > now:
                return "cooldown", cooldown_expires

        return "idle", closed_at


@dataclass
class AgentState:
    suffix: str
    directory: Path
    trade_outcomes: Dict[str, Any]
    trade_learning: Dict[str, Any]
    active_trades: Dict[str, Any]
    trade_history: Dict[str, Any]
    files: Dict[str, Path]

    @property
    def keys(self) -> Iterable[str]:
        all_keys = set(self.trade_outcomes) | set(self.trade_learning) | set(self.active_trades) | set(self.trade_history)
        return sorted(all_keys)

    def symbol_states(self) -> List[SymbolState]:
        states: List[SymbolState] = []
        for key in self.keys:
            symbol, side = _parse_state_key(key)
            states.append(
                SymbolState(
                    key=key,
                    symbol=symbol,
                    side=side,
                    outcome=self.trade_outcomes.get(key, {}),
                    learning=self.trade_learning.get(key, {}),
                    active=self.active_trades.get(key, {}),
                    history=self.trade_history.get(key, []),
                )
            )
        return states


def _load_agent_state(state_dir: Optional[Path], state_suffix: Optional[str]) -> AgentState:
    directory = _resolve_state_dir(state_dir)
    suffix = _compute_state_suffix(state_suffix)
    files = {
        "trade_outcomes": directory / f"trade_outcomes{suffix}.json",
        "trade_learning": directory / f"trade_learning{suffix}.json",
        "active_trades": directory / f"active_trades{suffix}.json",
        "trade_history": directory / f"trade_history{suffix}.json",
    }
    trade_outcomes = _load_json_file(files["trade_outcomes"])
    trade_learning = _load_json_file(files["trade_learning"])
    active_trades = _load_json_file(files["active_trades"])
    trade_history = _load_json_file(files["trade_history"])
    return AgentState(
        suffix=suffix,
        directory=directory,
        trade_outcomes=trade_outcomes,
        trade_learning=trade_learning,
        active_trades=active_trades,
        trade_history=trade_history,
        files=files,
    )


def _print_store_summary(agent_state: AgentState) -> None:
    typer.echo(
        f"Using state directory: {agent_state.directory} "
        f"(suffix: {agent_state.suffix or 'default'})"
    )
    lines = []
    now = datetime.now(timezone.utc)
    for store_name, data in (
        ("trade_outcomes", agent_state.trade_outcomes),
        ("trade_learning", agent_state.trade_learning),
        ("active_trades", agent_state.active_trades),
        ("trade_history", agent_state.trade_history),
    ):
        path = agent_state.files.get(store_name)
        if path and path.exists():
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age = _format_timestamp(modified, now)
        else:
            age = "missing"
        lines.append(f"{store_name}: {len(data)} (updated {age})")
    typer.echo("Stores -> " + " | ".join(lines))


def _discover_suffix_metrics(directory: Path) -> Dict[str, Dict[str, Any]]:
    suffixes = set()
    for prefix in ("trade_outcomes", "trade_learning", "active_trades", "trade_history"):
        for path in directory.glob(f"{prefix}*.json"):
            suffix = path.stem[len(prefix):]
            suffixes.add(suffix)

    metrics: Dict[str, Dict[str, Any]] = {}
    for suffix in sorted(suffixes):
        agent = _load_agent_state(directory, suffix if suffix else None)
        metrics[suffix] = {
            "counts": {
                "trade_outcomes": len(agent.trade_outcomes),
                "trade_learning": len(agent.trade_learning),
                "active_trades": len(agent.active_trades),
                "trade_history": len(agent.trade_history),
            },
            "files": agent.files,
        }
    return metrics


def _suggest_alternative_suffixes(
    directory: Path, current_suffix: str, have_state: bool
) -> None:
    metrics = _discover_suffix_metrics(directory)
    if not metrics:
        typer.echo(
            "No state files found in strategy_state. Has the trading bot persisted any state yet?"
        )
        return

    if have_state:
        return

    alternatives = [
        (suffix, data)
        for suffix, data in metrics.items()
        if suffix != current_suffix and sum(data["counts"].values()) > 0
    ]
    if not alternatives:
        typer.echo(
            "State files exist but contain no entries yet. The bot may not have recorded any trades."
        )
        return

    typer.echo("Other suffixes with data detected:")
    for suffix, data in alternatives:
        label = suffix or "default"
        counts = ", ".join(f"{store}={count}" for store, count in data["counts"].items())
        typer.echo(f"  --state-suffix {label} -> {counts}")


def _load_positions_shelf() -> Dict[str, Any]:
    if not POSITIONS_SHELF_PATH.exists():
        return {}
    return _load_json_file(POSITIONS_SHELF_PATH)


def _sorted_states(states: List[SymbolState], now: datetime) -> List[SymbolState]:
    priority = {"active": 0, "probe-active": 1, "pending-probe": 2, "cooldown": 3, "idle": 4}

    def sort_key(state: SymbolState):
        status, reference = state.status(now)
        ts = reference or datetime.fromtimestamp(0, tz=timezone.utc)
        return (priority.get(status, 99), -ts.timestamp(), state.symbol, state.side)

    return sorted(states, key=sort_key)


def _render_symbol_summary(state: SymbolState, now: datetime) -> str:
    status, reference = state.status(now)
    pieces = [
        f"{state.symbol:<8}",
        f"{state.side:<4}",
        f"{status:<13}",
    ]

    if state.active:
        qty = _safe_float(state.active.get("qty"))
        qty_display = f"{qty:.4f}" if qty is not None else "?"
        mode = state.active.get("mode", "unknown")
        opened = _format_timestamp(_parse_timestamp(state.active.get("opened_at")), now)
        pieces.append(f"qty={qty_display}")
        pieces.append(f"mode={mode}")
        pieces.append(f"opened={opened}")

    last_pnl = state.last_trade_pnl()
    if last_pnl is not None:
        pieces.append(f"last_pnl={last_pnl:.2f}")
    closed_at = _format_timestamp(state.last_trade_at(), now)
    pieces.append(f"last_close={closed_at}")

    if state.outcome:
        reason = state.outcome.get("reason", "n/a")
        mode = state.outcome.get("mode", "n/a")
        pieces.append(f"reason={reason}")
        pieces.append(f"mode={mode}")

    if status == "cooldown" and reference is not None:
        pieces.append(f"cooldown_until={_format_timestamp(reference, now)}")

    if state.learning:
        pending_probe = bool(state.learning.get("pending_probe"))
        probe_active = bool(state.learning.get("probe_active"))
        if pending_probe or probe_active:
            pieces.append(f"pending_probe={pending_probe}")
            pieces.append(f"probe_active={probe_active}")
        last_positive = _parse_timestamp(state.learning.get("last_positive_at"))
        if last_positive:
            pieces.append(f"last_positive={_format_timestamp(last_positive, now)}")

    return " | ".join(pieces)


def _render_history_entries(state: SymbolState, now: datetime, limit: int) -> List[str]:
    history = state.history[-limit:] if limit > 0 else state.history
    lines = []
    for entry in history:
        closed_at = _format_timestamp(_parse_timestamp(entry.get("closed_at")), now)
        pnl = _safe_float(entry.get("pnl"))
        pnl_text = f"{pnl:.2f}" if pnl is not None else "?"
        mode = entry.get("mode", "n/a")
        reason = entry.get("reason", "n/a")
        qty = _safe_float(entry.get("qty"))
        qty_text = f"{qty:.4f}" if qty is not None else "?"
        lines.append(
            f"- closed_at={closed_at} | pnl={pnl_text} | qty={qty_text} | mode={mode} | reason={reason}"
        )
    return lines


@app.callback()
def main(
    ctx: typer.Context,
    state_suffix: Optional[str] = typer.Option(
        None,
        "--state-suffix",
        help="State suffix override. Defaults to TRADE_STATE_SUFFIX env var.",
    ),
    state_dir: Optional[Path] = typer.Option(
        None,
        "--state-dir",
        help="Override the directory containing trade state JSON files.",
    ),
) -> None:
    ctx.obj = {
        "state_suffix": state_suffix,
        "state_dir": state_dir,
    }


@app.command()
def overview(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum symbols to display."),
) -> None:
    """Show a high-level summary of the trading agent state."""
    state_dir = ctx.obj.get("state_dir")
    state_suffix = ctx.obj.get("state_suffix")
    agent_state = _load_agent_state(state_dir, state_suffix)
    now = datetime.now(timezone.utc)
    states = agent_state.symbol_states()
    _print_store_summary(agent_state)

    if not states:
        typer.echo("No symbol state recorded yet.")
        directory = _resolve_state_dir(state_dir)
        current_suffix = _compute_state_suffix(state_suffix)
        _suggest_alternative_suffixes(directory, current_suffix, have_state=False)
        return

    status_counts: Dict[str, int] = {}
    for state in states:
        status, _ = state.status(now)
        status_counts[status] = status_counts.get(status, 0) + 1

    typer.echo("Status counts -> " + ", ".join(f"{status}: {count}" for status, count in sorted(status_counts.items())))

    typer.echo("")
    typer.echo("Symbols:")
    for state in _sorted_states(states, now)[:limit]:
        typer.echo(_render_symbol_summary(state, now))


@app.command()
def symbol(
    ctx: typer.Context,
    symbol: str,
    side: Optional[str] = typer.Option(None, help="Filter to a side: buy or sell."),
) -> None:
    """Display detailed state for a specific symbol."""
    agent_state = _load_agent_state(ctx.obj.get("state_dir"), ctx.obj.get("state_suffix"))
    now = datetime.now(timezone.utc)
    side_filter = side.lower() if side else None
    matches = [
        state
        for state in agent_state.symbol_states()
        if state.symbol.upper() == symbol.upper() and (side_filter is None or state.side.lower() == side_filter)
    ]

    if not matches:
        typer.echo(f"No state found for {symbol} (side={side_filter or 'any'}).")
        available = {s.symbol.upper() for s in agent_state.symbol_states()}
        if available:
            typer.echo("Available symbols: " + ", ".join(sorted(available)))
        return

    for state in matches:
        typer.echo(_render_symbol_summary(state, now))
        history_lines = _render_history_entries(state, now, limit=5)
        if history_lines:
            typer.echo("  Recent history:")
            for line in history_lines:
                typer.echo("   " + line)
        else:
            typer.echo("  No recorded history entries.")
        typer.echo("")


@app.command()
def history(
    ctx: typer.Context,
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter to a specific symbol."),
    side: Optional[str] = typer.Option(None, help="Filter to a side for the selected symbol."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum history entries per key."),
) -> None:
    """Dump trade history for all keys (or a specific symbol)."""
    agent_state = _load_agent_state(ctx.obj.get("state_dir"), ctx.obj.get("state_suffix"))
    now = datetime.now(timezone.utc)
    entries = agent_state.symbol_states()
    if symbol:
        entries = [e for e in entries if e.symbol.upper() == symbol.upper()]
    if side:
        side_lower = side.lower()
        entries = [e for e in entries if e.side.lower() == side_lower]

    if not entries:
        typer.echo("No matching history entries.")
        return

    for state in entries:
        typer.echo(f"{state.symbol} {state.side}:")
        lines = _render_history_entries(state, now, limit=limit)
        if lines:
            for line in lines[-limit:]:
                typer.echo("  " + line)
        else:
            typer.echo("  No history recorded.")
        typer.echo("")


@app.command()
def strategies(
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Limit output to a specific YYYY-MM-DD."),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol."),
    days: int = typer.Option(3, "--days", help="Show this many most recent days when no date is specified."),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum entries per day."),
) -> None:
    """Inspect the strategy assignments recorded in positions_shelf.json."""
    shelf = _load_positions_shelf()
    if not shelf:
        typer.echo("positions_shelf.json is empty or missing.")
        return

    entries: List[Tuple[str, str, str]] = []
    for key, strategy in shelf.items():
        parts = str(key).split("-")
        if len(parts) < 4:
            continue
        day = "-".join(parts[-3:])
        symbol_key = "-".join(parts[:-3])
        if date and day != date:
            continue
        if symbol and symbol_key.upper() != symbol.upper():
            continue
        entries.append((day, symbol_key, str(strategy)))

    if not entries:
        typer.echo("No matching strategy assignments found.")
        return

    entries.sort(key=lambda item: (item[0], item[1]))
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for day, sym, strat in entries:
        grouped.setdefault(day, []).append((sym, strat))

    if date:
        days_to_show = [date]
    else:
        days_to_show = sorted(grouped.keys(), reverse=True)[:days]

    for day in days_to_show:
        day_entries = grouped.get(day, [])
        if not day_entries:
            continue
        typer.echo(f"{day}:")
        for sym, strat in day_entries[:limit]:
            typer.echo(f"  {sym:<8} -> {strat}")
        remaining = max(len(day_entries) - limit, 0)
        if remaining > 0:
            typer.echo(f"  ... {remaining} more")
        typer.echo("")


if __name__ == "__main__":
    app()
