from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .config import DashboardConfig

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class ShelfSnapshot:
    recorded_at: datetime
    file_path: Path
    data: str
    sha256: str
    bytes: int


@dataclass
class SpreadObservation:
    recorded_at: datetime
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    spread_ratio: float

    @property
    def spread_bps(self) -> float:
        return (self.spread_ratio - 1.0) * 10_000

    @property
    def spread_absolute(self) -> Optional[float]:
        if self.ask is None or self.bid is None:
            return None
        return self.ask - self.bid


@dataclass
class MetricEntry:
    recorded_at: datetime
    source: str
    metric: str
    value: Optional[float]
    symbol: Optional[str] = None
    message: Optional[str] = None


class DashboardDatabase:
    """Thin wrapper around sqlite3 for the dashboards module."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.path = config.db_path
        self._conn = sqlite3.connect(
            str(self.path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._setup_connection()
        self.initialize()

    def _setup_connection(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DashboardDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS shelf_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                file_path TEXT NOT NULL,
                data TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                bytes INTEGER NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shelf_snapshots_path_time ON shelf_snapshots(file_path, recorded_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shelf_snapshots_hash ON shelf_snapshots(file_path, sha256)")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spread_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bid REAL,
                ask REAL,
                spread_ratio REAL NOT NULL,
                spread_absolute REAL,
                spread_bps REAL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_spread_symbol_time ON spread_observations(symbol, recorded_at)")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                source TEXT NOT NULL,
                symbol TEXT,
                metric TEXT NOT NULL,
                value REAL,
                message TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_metric_time ON metrics(metric, recorded_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_symbol_metric_time ON metrics(symbol, metric, recorded_at)")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS log_offsets (
                file_path TEXT PRIMARY KEY,
                offset INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()
        cursor.close()

    def _fetch_last_snapshot_hash(self, file_path: Path) -> Optional[str]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT sha256
            FROM shelf_snapshots
            WHERE file_path = ?
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            (str(file_path),),
        )
        row = cursor.fetchone()
        cursor.close()
        return row["sha256"] if row else None

    def record_shelf_snapshot(self, file_path: Path, data: str) -> Optional[ShelfSnapshot]:
        sha = hashlib.sha256(data.encode("utf-8")).hexdigest()
        last_sha = self._fetch_last_snapshot_hash(file_path)
        if last_sha == sha:
            return None
        recorded_at = utc_now()
        snapshot = ShelfSnapshot(
            recorded_at=recorded_at,
            file_path=file_path,
            data=data,
            sha256=sha,
            bytes=len(data.encode("utf-8")),
        )
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO shelf_snapshots (recorded_at, file_path, data, sha256, bytes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.recorded_at.strftime(ISO_FORMAT),
                str(snapshot.file_path),
                snapshot.data,
                snapshot.sha256,
                snapshot.bytes,
            ),
        )
        self._conn.commit()
        cursor.close()
        return snapshot

    def record_spread(self, observation: SpreadObservation) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO spread_observations (
                recorded_at, symbol, bid, ask, spread_ratio, spread_absolute, spread_bps
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.recorded_at.strftime(ISO_FORMAT),
                observation.symbol.upper(),
                observation.bid,
                observation.ask,
                observation.spread_ratio,
                observation.spread_absolute,
                observation.spread_bps,
            ),
        )
        self._conn.commit()
        cursor.close()

    def record_metric(self, entry: MetricEntry) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO metrics (recorded_at, source, symbol, metric, value, message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.recorded_at.strftime(ISO_FORMAT),
                entry.source,
                entry.symbol.upper() if entry.symbol else None,
                entry.metric,
                entry.value,
                entry.message,
            ),
        )
        self._conn.commit()
        cursor.close()

    def iter_spreads(
        self,
        symbol: str,
        limit: Optional[int] = None,
    ) -> Iterator[SpreadObservation]:
        cursor = self._conn.cursor()
        query = """
            SELECT recorded_at, symbol, bid, ask, spread_ratio
            FROM spread_observations
            WHERE symbol = ?
            ORDER BY recorded_at DESC
        """
        if limit:
            query += " LIMIT ?"
            cursor.execute(query, (symbol.upper(), limit))
        else:
            cursor.execute(query, (symbol.upper(),))
        rows = cursor.fetchall()
        cursor.close()
        for row in rows:
            recorded_at = datetime.strptime(row["recorded_at"], ISO_FORMAT)
            yield SpreadObservation(
                recorded_at=recorded_at,
                symbol=row["symbol"],
                bid=row["bid"],
                ask=row["ask"],
                spread_ratio=row["spread_ratio"],
            )

    def iter_metrics(
        self,
        metric: str,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[MetricEntry]:
        cursor = self._conn.cursor()
        query = """
            SELECT recorded_at, source, symbol, metric, value, message
            FROM metrics
            WHERE metric = ?
        """
        params: list = [metric]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY recorded_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        for row in rows:
            recorded_at = datetime.strptime(row["recorded_at"], ISO_FORMAT)
            yield MetricEntry(
                recorded_at=recorded_at,
                source=row["source"],
                metric=row["metric"],
                value=row["value"],
                symbol=row["symbol"],
                message=row["message"],
            )

    def iter_latest_snapshots(self, file_path: Path, limit: Optional[int] = None) -> Iterator[ShelfSnapshot]:
        cursor = self._conn.cursor()
        query = """
            SELECT recorded_at, file_path, data, sha256, bytes
            FROM shelf_snapshots
            WHERE file_path = ?
            ORDER BY recorded_at DESC
        """
        params: list = [str(file_path)]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        for row in rows:
            recorded_at = datetime.strptime(row["recorded_at"], ISO_FORMAT)
            yield ShelfSnapshot(
                recorded_at=recorded_at,
                file_path=Path(row["file_path"]),
                data=row["data"],
                sha256=row["sha256"],
                bytes=row["bytes"],
            )

    def get_log_offset(self, file_path: Path) -> int:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT offset
            FROM log_offsets
            WHERE file_path = ?
            """,
            (str(file_path),),
        )
        row = cursor.fetchone()
        cursor.close()
        return int(row["offset"]) if row else 0

    def update_log_offset(self, file_path: Path, offset: int) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO log_offsets (file_path, offset)
            VALUES (?, ?)
            ON CONFLICT(file_path) DO UPDATE SET offset = excluded.offset
            """,
            (str(file_path), offset),
        )
        self._conn.commit()
        cursor.close()


@contextmanager
def open_database(config: DashboardConfig) -> Iterator[DashboardDatabase]:
    db = DashboardDatabase(config)
    try:
        yield db
    finally:
        db.close()


__all__ = [
    "DashboardDatabase",
    "open_database",
    "ShelfSnapshot",
    "SpreadObservation",
    "MetricEntry",
]
