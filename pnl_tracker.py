"""
PnL Tracker - Live trading performance tracking and logging.

Tracks account net worth, trades, and performance metrics over time.
Logs to JSON files for persistence and analysis.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: str
    action: str  # 'buy' or 'sell'
    token: str
    amount: float  # amount of token
    price: float  # price per token in SOL
    sol_amount: float  # SOL spent/received
    signal_strength: float  # model confidence
    position_size: float  # suggested position size


@dataclass
class SnapshotRecord:
    """Snapshot of account state at a point in time."""
    timestamp: str
    sol_balance: float
    token_balance: float
    token_price: float
    net_worth_sol: float  # total value in SOL
    net_worth_usd: Optional[float] = None  # optional USD value


class PnLTracker:
    """
    Live trading performance tracker.

    Tracks:
    - Account net worth over time
    - Individual trades
    - Running PnL
    - Performance metrics (returns, drawdown, etc.)
    """

    def __init__(
        self,
        token: str,
        log_dir: str = "logs/pnl",
        snapshot_interval: int = 60,  # seconds between snapshots
        max_trades_in_memory: Optional[int] = None,
        max_snapshots_in_memory: Optional[int] = None,
    ):
        self.token = token
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_interval = snapshot_interval
        self.last_snapshot_time: Optional[datetime] = None
        self.max_trades_in_memory = max_trades_in_memory
        self.max_snapshots_in_memory = max_snapshots_in_memory

        if self.max_trades_in_memory is not None and self.max_trades_in_memory < 0:
            self.max_trades_in_memory = 0
        if self.max_snapshots_in_memory is not None and self.max_snapshots_in_memory < 0:
            self.max_snapshots_in_memory = 0

        # File paths
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.trades_file = self.log_dir / f"trades_{token}_{date_str}.jsonl"
        self.snapshots_file = self.log_dir / f"snapshots_{token}_{date_str}.jsonl"
        self.summary_file = self.log_dir / f"summary_{token}.json"

        # In-memory state
        self.trades: list[TradeRecord] = []
        self.snapshots: list[SnapshotRecord] = []
        self.initial_net_worth: Optional[float] = None
        self.peak_net_worth: float = 0.0
        self._last_snapshot: Optional[SnapshotRecord] = None
        self.total_trades: int = 0
        self.total_buy_trades: int = 0
        self.total_sell_trades: int = 0
        self.total_snapshots: int = 0

        # Load existing data if any
        self._load_existing()

        logger.info(f"PnLTracker initialized for {token}")
        logger.info(f"  Trades log: {self.trades_file}")
        logger.info(f"  Snapshots log: {self.snapshots_file}")

    def _load_existing(self):
        """Load existing trades and snapshots from today's files."""
        # Load trades
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = TradeRecord(**data)
                        self.total_trades += 1
                        if record.action == "buy":
                            self.total_buy_trades += 1
                        elif record.action == "sell":
                            self.total_sell_trades += 1

                        if self.max_trades_in_memory is None:
                            self.trades.append(record)
                        elif self.max_trades_in_memory > 0:
                            if len(self.trades) >= self.max_trades_in_memory:
                                self.trades.pop(0)
                            self.trades.append(record)

            logger.info(
                "Loaded %d existing trades (kept %d in memory)",
                self.total_trades,
                len(self.trades),
            )

        # Load snapshots
        if self.snapshots_file.exists():
            with open(self.snapshots_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = SnapshotRecord(**data)
                        self.total_snapshots += 1
                        if self.initial_net_worth is None:
                            self.initial_net_worth = record.net_worth_sol
                        self.peak_net_worth = max(self.peak_net_worth, record.net_worth_sol)
                        self._last_snapshot = record

                        if self.max_snapshots_in_memory is None:
                            self.snapshots.append(record)
                        elif self.max_snapshots_in_memory > 0:
                            if len(self.snapshots) >= self.max_snapshots_in_memory:
                                self.snapshots.pop(0)
                            self.snapshots.append(record)

            logger.info(
                "Loaded %d existing snapshots (kept %d in memory)",
                self.total_snapshots,
                len(self.snapshots),
            )

    def log_trade(
        self,
        action: str,
        amount: float,
        price: float,
        sol_amount: float,
        signal_strength: float,
        position_size: float,
    ):
        """Log a trade execution."""
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            token=self.token,
            amount=amount,
            price=price,
            sol_amount=sol_amount,
            signal_strength=signal_strength,
            position_size=position_size,
        )

        self.trades.append(record)
        if self.max_trades_in_memory is not None:
            if self.max_trades_in_memory <= 0:
                self.trades.clear()
            elif len(self.trades) > self.max_trades_in_memory:
                self.trades.pop(0)

        self.total_trades += 1
        if action == "buy":
            self.total_buy_trades += 1
        elif action == "sell":
            self.total_sell_trades += 1

        # Append to file
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(asdict(record)) + '\n')

        logger.info(
            f"TRADE: {action.upper()} {amount:.4f} {self.token} @ {price:.8f} SOL "
            f"(total: {sol_amount:.4f} SOL, signal: {signal_strength:.3f})"
        )

    def log_snapshot(
        self,
        sol_balance: float,
        token_balance: float,
        token_price: float,
        sol_price_usd: Optional[float] = None,
        force: bool = False,
    ) -> Optional[SnapshotRecord]:
        """
        Log account snapshot if interval has passed.

        Args:
            sol_balance: Current SOL balance
            token_balance: Current token balance
            token_price: Current token price in SOL
            sol_price_usd: Optional SOL price in USD
            force: Force snapshot even if interval hasn't passed
        """
        now = datetime.now(timezone.utc)

        # Check if we should snapshot
        if not force and self.last_snapshot_time:
            elapsed = (now - self.last_snapshot_time).total_seconds()
            if elapsed < self.snapshot_interval:
                return None

        # Calculate net worth
        net_worth_sol = sol_balance + (token_balance * token_price)
        net_worth_usd = net_worth_sol * sol_price_usd if sol_price_usd else None

        record = SnapshotRecord(
            timestamp=now.isoformat(),
            sol_balance=sol_balance,
            token_balance=token_balance,
            token_price=token_price,
            net_worth_sol=net_worth_sol,
            net_worth_usd=net_worth_usd,
        )

        self._last_snapshot = record
        self.total_snapshots += 1
        if self.max_snapshots_in_memory is None:
            self.snapshots.append(record)
        elif self.max_snapshots_in_memory > 0:
            self.snapshots.append(record)
            if len(self.snapshots) > self.max_snapshots_in_memory:
                self.snapshots.pop(0)
        self.last_snapshot_time = now

        # Track initial and peak
        if self.initial_net_worth is None:
            self.initial_net_worth = net_worth_sol
        self.peak_net_worth = max(self.peak_net_worth, net_worth_sol)

        # Append to file
        with open(self.snapshots_file, 'a') as f:
            f.write(json.dumps(asdict(record)) + '\n')

        # Update summary
        self._update_summary()

        return record

    def _update_summary(self):
        """Update the summary file with current stats."""
        current = self._last_snapshot or (self.snapshots[-1] if self.snapshots else None)
        if current is None or self.initial_net_worth is None:
            return

        # Calculate metrics
        total_return = (current.net_worth_sol - self.initial_net_worth) / self.initial_net_worth
        drawdown = (self.peak_net_worth - current.net_worth_sol) / self.peak_net_worth if self.peak_net_worth > 0 else 0

        # Count trades
        buy_count = self.total_buy_trades
        sell_count = self.total_sell_trades

        summary = {
            "token": self.token,
            "last_updated": current.timestamp,
            "initial_net_worth_sol": self.initial_net_worth,
            "current_net_worth_sol": current.net_worth_sol,
            "peak_net_worth_sol": self.peak_net_worth,
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": drawdown * 100,
            "total_trades": self.total_trades,
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "snapshots_count": self.total_snapshots,
            "sol_balance": current.sol_balance,
            "token_balance": current.token_balance,
            "token_price": current.token_price,
        }

        if current.net_worth_usd:
            summary["current_net_worth_usd"] = current.net_worth_usd

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def get_stats(self) -> dict:
        """Get current performance statistics."""
        current = self._last_snapshot or (self.snapshots[-1] if self.snapshots else None)
        if current is None or self.initial_net_worth is None:
            return {"status": "no data"}

        total_return = (current.net_worth_sol - self.initial_net_worth) / self.initial_net_worth
        drawdown = (self.peak_net_worth - current.net_worth_sol) / self.peak_net_worth if self.peak_net_worth > 0 else 0

        return {
            "net_worth_sol": current.net_worth_sol,
            "initial_net_worth_sol": self.initial_net_worth,
            "total_return_pct": total_return * 100,
            "drawdown_pct": drawdown * 100,
            "trades": self.total_trades,
            "snapshots": self.total_snapshots,
        }

    def print_status(self):
        """Print current status to console."""
        stats = self.get_stats()

        if stats.get("status") == "no data":
            print("PnLTracker: No data yet")
            return

        print("\n" + "=" * 50)
        print(f"PnL Tracker - {self.token}")
        print("=" * 50)
        print(f"Initial:      {stats['initial_net_worth_sol']:.4f} SOL")
        print(f"Current:      {stats['net_worth_sol']:.4f} SOL")
        print(f"Total Return: {stats['total_return_pct']:+.2f}%")
        print(f"Drawdown:     {stats['drawdown_pct']:.2f}%")
        print(f"Trades:       {stats['trades']}")
        print("=" * 50 + "\n")


def load_daily_snapshots(token: str, date: str, log_dir: str = "logs/pnl") -> list[SnapshotRecord]:
    """Load snapshots for a specific date."""
    filepath = Path(log_dir) / f"snapshots_{token}_{date}.jsonl"
    snapshots = []

    if filepath.exists():
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    snapshots.append(SnapshotRecord(**data))

    return snapshots


def load_daily_trades(token: str, date: str, log_dir: str = "logs/pnl") -> list[TradeRecord]:
    """Load trades for a specific date."""
    filepath = Path(log_dir) / f"trades_{token}_{date}.jsonl"
    trades = []

    if filepath.exists():
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    trades.append(TradeRecord(**data))

    return trades


if __name__ == "__main__":
    # Demo/test
    tracker = PnLTracker("TEST_TOKEN")

    # Simulate some activity
    sol = 100.0
    tokens = 0.0
    price = 0.001

    tracker.log_snapshot(sol, tokens, price, force=True)

    # Simulate buy
    buy_amount = 1000
    buy_cost = buy_amount * price
    sol -= buy_cost
    tokens += buy_amount
    tracker.log_trade("buy", buy_amount, price, buy_cost, 0.75, 0.5)
    tracker.log_snapshot(sol, tokens, price, force=True)

    # Price goes up
    price = 0.0012
    tracker.log_snapshot(sol, tokens, price, force=True)

    # Simulate sell
    sell_amount = 500
    sell_revenue = sell_amount * price
    sol += sell_revenue
    tokens -= sell_amount
    tracker.log_trade("sell", sell_amount, price, sell_revenue, 0.35, 0.3)
    tracker.log_snapshot(sol, tokens, price, force=True)

    tracker.print_status()
    print(f"\nSummary saved to: {tracker.summary_file}")
