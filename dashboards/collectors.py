from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import DashboardConfig
from .db import DashboardDatabase, SpreadObservation, utc_now
from .log_ingestor import collect_log_metrics as ingest_log_metrics
from .spread_fetcher import QuoteResult, SpreadFetcher

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CollectionStats:
    shelf_snapshots: int = 0
    spread_observations: int = 0
    metrics: int = 0

    def __iadd__(self, other: "CollectionStats") -> "CollectionStats":
        self.shelf_snapshots += other.shelf_snapshots
        self.spread_observations += other.spread_observations
        self.metrics += other.metrics
        return self


def collect_shelf_snapshots(config: DashboardConfig, db: DashboardDatabase) -> CollectionStats:
    stats = CollectionStats()
    for shelf_path in config.shelf_files:
        if not shelf_path.exists():
            logger.debug("Shelf path %s not found; skipping", shelf_path)
            continue
        try:
            data = shelf_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - I/O failure path
            logger.exception("Failed to read shelf file %s", shelf_path)
            continue

        if 0 < config.snapshot_chunk_size < len(data.encode("utf-8")):
            truncated_data = data.encode("utf-8")[: config.snapshot_chunk_size].decode("utf-8", errors="ignore")
            logger.warning(
                "Shelf snapshot for %s exceeded %d bytes; truncated output",
                shelf_path,
                config.snapshot_chunk_size,
            )
            data = truncated_data

        snapshot = db.record_shelf_snapshot(shelf_path, data)
        if snapshot:
            stats.shelf_snapshots += 1
            logger.info(
                "Captured shelf snapshot for %s @ %s (%d bytes)",
                shelf_path,
                snapshot.recorded_at.isoformat(),
                snapshot.bytes,
            )
    return stats


def _sanitize_quote(symbol: str, result: QuoteResult) -> SpreadObservation:
    bid = result.bid if result.bid and result.bid > 0 else None
    ask = result.ask if result.ask and result.ask > 0 else None
    spread_ratio = result.spread_ratio
    if bid and ask:
        spread_ratio = ask / bid if bid else 1.0
    return SpreadObservation(
        recorded_at=utc_now(),
        symbol=symbol,
        bid=bid,
        ask=ask,
        spread_ratio=spread_ratio,
    )


def collect_spreads(
    config: DashboardConfig,
    db: DashboardDatabase,
    fetcher: SpreadFetcher,
) -> CollectionStats:
    stats = CollectionStats()
    for symbol in config.spread_symbols:
        try:
            quote = fetcher.fetch(symbol)
        except Exception:
            logger.exception("Failed to fetch spread for %s", symbol)
            continue

        observation = _sanitize_quote(symbol, quote)
        db.record_spread(observation)
        stats.spread_observations += 1
        bid_display = f"{observation.bid:.4f}" if observation.bid is not None else "None"
        ask_display = f"{observation.ask:.4f}" if observation.ask is not None else "None"
        logger.info(
            "Recorded %s spread %.2fbps (bid=%s ask=%s)",
            symbol,
            observation.spread_bps,
            bid_display,
            ask_display,
        )
    return stats


def collect_log_metrics(config: DashboardConfig, db: DashboardDatabase) -> CollectionStats:
    stats = CollectionStats()
    stats.metrics = ingest_log_metrics(config, db)
    if stats.metrics:
        logger.info("Recorded %d metrics from log ingestion", stats.metrics)
    return stats


__all__ = ["collect_spreads", "collect_shelf_snapshots", "collect_log_metrics", "CollectionStats"]
