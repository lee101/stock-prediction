from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sqlite3


DEFAULT_DB_PATH = Path("strategy_state") / "binance_holdings_snapshots.db"


@dataclass
class AssetSnapshot:
    asset: str
    free: float
    locked: float
    amount: float
    price_usdt: float
    value_usdt: float


@dataclass
class HoldingsSnapshot:
    snapshot_id: int
    ts_ms: int
    ts_iso: str
    total_usdt: float
    assets: List[AssetSnapshot]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            total_usdt REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings_asset (
            snapshot_id INTEGER NOT NULL,
            asset TEXT NOT NULL,
            free REAL NOT NULL,
            locked REAL NOT NULL,
            amount REAL NOT NULL,
            price_usdt REAL NOT NULL,
            value_usdt REAL NOT NULL,
            FOREIGN KEY(snapshot_id) REFERENCES holdings_snapshot(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_holdings_snapshot_ts ON holdings_snapshot(ts_ms)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_holdings_asset_snap ON holdings_asset(snapshot_id)"
    )


def _now_timestamp() -> Tuple[int, str]:
    now = datetime.now(timezone.utc)
    ts_ms = int(now.timestamp() * 1000)
    return ts_ms, now.isoformat()


def record_snapshot(
    *,
    total_usdt: float,
    assets: Iterable[Dict[str, float]],
    db_path: Path = DEFAULT_DB_PATH,
) -> HoldingsSnapshot:
    ts_ms, ts_iso = _now_timestamp()
    conn = _connect(db_path)
    try:
        _init_db(conn)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO holdings_snapshot (ts_ms, ts_iso, total_usdt) VALUES (?, ?, ?)",
            (ts_ms, ts_iso, float(total_usdt)),
        )
        snapshot_id = int(cursor.lastrowid)
        asset_rows: List[AssetSnapshot] = []
        for entry in assets:
            asset = str(entry.get("asset", "")).upper()
            free = float(entry.get("free", 0.0))
            locked = float(entry.get("locked", 0.0))
            amount = float(entry.get("amount", free + locked))
            price_usdt = float(entry.get("price_usdt", 0.0))
            value_usdt = float(entry.get("value_usdt", 0.0))
            cursor.execute(
                """
                INSERT INTO holdings_asset (
                    snapshot_id, asset, free, locked, amount, price_usdt, value_usdt
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, asset, free, locked, amount, price_usdt, value_usdt),
            )
            asset_rows.append(
                AssetSnapshot(
                    asset=asset,
                    free=free,
                    locked=locked,
                    amount=amount,
                    price_usdt=price_usdt,
                    value_usdt=value_usdt,
                )
            )
        conn.commit()
        return HoldingsSnapshot(
            snapshot_id=snapshot_id,
            ts_ms=ts_ms,
            ts_iso=ts_iso,
            total_usdt=float(total_usdt),
            assets=asset_rows,
        )
    finally:
        conn.close()


def load_latest_snapshot(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    before_ts_ms: Optional[int] = None,
) -> Optional[HoldingsSnapshot]:
    if not db_path.exists():
        return None
    conn = _connect(db_path)
    try:
        _init_db(conn)
        cursor = conn.cursor()
        if before_ts_ms is None:
            cursor.execute(
                "SELECT id, ts_ms, ts_iso, total_usdt FROM holdings_snapshot ORDER BY ts_ms DESC LIMIT 1"
            )
        else:
            cursor.execute(
                """
                SELECT id, ts_ms, ts_iso, total_usdt
                FROM holdings_snapshot
                WHERE ts_ms < ?
                ORDER BY ts_ms DESC
                LIMIT 1
                """,
                (int(before_ts_ms),),
            )
        row = cursor.fetchone()
        if row is None:
            return None
        snapshot_id, ts_ms, ts_iso, total_usdt = row
        cursor.execute(
            """
            SELECT asset, free, locked, amount, price_usdt, value_usdt
            FROM holdings_asset
            WHERE snapshot_id = ?
            ORDER BY value_usdt DESC
            """,
            (snapshot_id,),
        )
        assets = [
            AssetSnapshot(
                asset=str(asset),
                free=float(free),
                locked=float(locked),
                amount=float(amount),
                price_usdt=float(price_usdt),
                value_usdt=float(value_usdt),
            )
            for asset, free, locked, amount, price_usdt, value_usdt in cursor.fetchall()
        ]
        return HoldingsSnapshot(
            snapshot_id=int(snapshot_id),
            ts_ms=int(ts_ms),
            ts_iso=str(ts_iso),
            total_usdt=float(total_usdt),
            assets=assets,
        )
    finally:
        conn.close()
