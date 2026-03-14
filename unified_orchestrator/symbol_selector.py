"""Meta-strategy symbol selector based on recent trading performance.

Tracks per-symbol rolling P&L and filters out underperforming symbols
to avoid trading in low-liquidity or losing environments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TradeRecord:
    symbol: str
    pnl_pct: float
    timestamp: str  # ISO format


@dataclass
class SymbolStats:
    symbol: str
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        return wins / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    @property
    def mean_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def sortino(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl_pct for t in self.trades]
        mean_ret = sum(returns) / len(returns)
        neg_sq = [r ** 2 for r in returns if r < 0]
        if not neg_sq:
            return 10.0  # all positive, cap at 10
        downside_std = math.sqrt(sum(neg_sq) / len(neg_sq))
        if downside_std < 1e-8:
            return 10.0
        return mean_ret / downside_std


class SymbolSelector:
    """Track per-symbol rolling performance and filter underperformers.

    Usage:
        selector = SymbolSelector(["BTCUSD", "ETHUSD", ...])
        selector.record_trade("BTCUSD", pnl_pct=0.5, timestamp=now)
        allowed = selector.get_allowed_symbols()
    """

    def __init__(
        self,
        symbols: list[str],
        lookback_hours: int = 168,  # 7 days
        min_trades: int = 3,
        min_win_rate: float = 0.3,
        min_sortino: float = -1.0,
        persist_path: Optional[str] = None,
    ):
        self.symbols = list(symbols)
        self.lookback_hours = lookback_hours
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_sortino = min_sortino
        self.persist_path = persist_path
        self._stats: dict[str, SymbolStats] = {s: SymbolStats(symbol=s) for s in symbols}

        if persist_path and Path(persist_path).exists():
            self._load()

    def record_trade(
        self,
        symbol: str,
        pnl_pct: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a completed trade's P&L."""
        if symbol not in self._stats:
            self._stats[symbol] = SymbolStats(symbol=symbol)
        ts = timestamp or datetime.now(timezone.utc)
        record = TradeRecord(
            symbol=symbol,
            pnl_pct=pnl_pct,
            timestamp=ts.isoformat(),
        )
        self._stats[symbol].trades.append(record)
        self._prune(symbol)
        if self.persist_path:
            self._save()

    def _prune(self, symbol: str) -> None:
        """Remove trades older than lookback window."""
        cutoff = datetime.now(timezone.utc).timestamp() - self.lookback_hours * 3600
        stats = self._stats[symbol]
        stats.trades = [
            t for t in stats.trades
            if datetime.fromisoformat(t.timestamp).timestamp() >= cutoff
        ]

    def get_allowed_symbols(self) -> list[str]:
        """Return symbols that pass the performance filter."""
        allowed = []
        for sym in self.symbols:
            if self.should_trade(sym):
                allowed.append(sym)
        return allowed

    def should_trade(self, symbol: str) -> bool:
        """Check if a symbol is currently allowed for trading."""
        stats = self._stats.get(symbol)
        if stats is None:
            return True  # unknown symbol, allow by default

        # Not enough data — allow trading (don't filter without evidence)
        if stats.num_trades < self.min_trades:
            return True

        # Prune old trades first
        self._prune(symbol)
        if stats.num_trades < self.min_trades:
            return True

        # Filter by win rate
        if stats.win_rate < self.min_win_rate:
            return False

        # Filter by sortino
        if stats.sortino < self.min_sortino:
            return False

        return True

    def get_symbol_weights(self) -> dict[str, float]:
        """Return confidence weights per symbol based on recent performance.

        Weights are normalized to sum to 1.0 for allowed symbols.
        Symbols with more trades and better sortino get higher weights.
        """
        weights = {}
        for sym in self.symbols:
            if not self.should_trade(sym):
                weights[sym] = 0.0
                continue
            stats = self._stats[sym]
            if stats.num_trades < self.min_trades:
                weights[sym] = 1.0  # default weight for unknown
            else:
                # Weight by sortino, clamped to [0, 5]
                w = max(0.0, min(5.0, stats.sortino + 1.0))
                weights[sym] = w

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def summary(self) -> str:
        """Return a human-readable summary of symbol performance."""
        lines = ["Symbol Performance (rolling {}h):".format(self.lookback_hours)]
        for sym in self.symbols:
            stats = self._stats.get(sym)
            if not stats or stats.num_trades == 0:
                lines.append(f"  {sym}: no trades")
                continue
            allowed = "OK" if self.should_trade(sym) else "BLOCKED"
            lines.append(
                f"  {sym}: {stats.num_trades} trades, "
                f"WR={stats.win_rate:.0%}, "
                f"PnL={stats.total_pnl:+.2f}%, "
                f"Sortino={stats.sortino:.2f} "
                f"[{allowed}]"
            )
        return "\n".join(lines)

    def _save(self) -> None:
        """Persist state to JSON."""
        if not self.persist_path:
            return
        data = {}
        for sym, stats in self._stats.items():
            data[sym] = [
                {"pnl_pct": t.pnl_pct, "timestamp": t.timestamp}
                for t in stats.trades
            ]
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.persist_path).write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load state from JSON."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        try:
            raw = json.loads(Path(self.persist_path).read_text())
            for sym, trades in raw.items():
                if sym not in self._stats:
                    self._stats[sym] = SymbolStats(symbol=sym)
                self._stats[sym].trades = [
                    TradeRecord(symbol=sym, pnl_pct=t["pnl_pct"], timestamp=t["timestamp"])
                    for t in trades
                ]
                self._prune(sym)
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, start fresh
