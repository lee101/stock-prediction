from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pytest

import src.trade_stock_state_utils as state_utils


@dataclass
class DummyStore:
    data: Dict[str, Dict] | None = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = {}

    def load(self) -> None:
        # FlatShelf.load() populates internal state; no-op for dummy.
        return None

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def pop(self, key, default=None):
        return self.data.pop(key, default)


class ListLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warning(self, msg, *args) -> None:
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args) -> None:
        self.errors.append(msg % args if args else msg)


@pytest.fixture
def dummy_store():
    store = DummyStore()

    def loader():
        return store

    return store, loader


def test_normalize_and_state_key():
    assert state_utils.normalize_side_for_key("Short") == "sell"
    assert state_utils.normalize_side_for_key("BUY") == "buy"
    assert state_utils.state_key("AAPL", "Short") == "AAPL|sell"


def test_parse_timestamp_handles_invalid_input():
    logger = ListLogger()
    ts = state_utils.parse_timestamp("not-a-time", logger=logger)
    assert ts is None
    assert logger.warnings  # warning recorded


def test_update_learning_state_sets_updated_at(dummy_store):
    store, loader = dummy_store
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    state = state_utils.update_learning_state(
        loader,
        "AAPL",
        "buy",
        {"pending_probe": True},
        logger=None,
        now=now,
    )

    assert state["pending_probe"] is True
    assert state["updated_at"] == now.isoformat()
    key = state_utils.state_key("AAPL", "buy")
    assert store.data[key]["pending_probe"] is True


def test_probe_state_helpers(dummy_store):
    _, loader = dummy_store
    now = datetime(2025, 1, 2, 15, tzinfo=timezone.utc)
    started = now - timedelta(hours=1)

    state_utils.mark_probe_active(
        loader,
        "MSFT",
        "sell",
        qty=5.0,
        logger=None,
        now=started,
    )

    summary = state_utils.describe_probe_state(
        {
            "probe_active": True,
            "probe_started_at": started.isoformat(),
        },
        now=now,
        probe_max_duration=timedelta(hours=2),
    )

    assert summary["probe_active"] is True
    assert 3500 < summary["probe_age_seconds"] < 3700  # ~1 hour
    assert summary["probe_expired"] is False
    assert summary["probe_transition_ready"] is False

    state_utils.mark_probe_completed(
        loader,
        "MSFT",
        "sell",
        successful=True,
        logger=None,
        now=now,
    )

    completed = state_utils.load_store_entry(
        loader,
        "MSFT",
        "sell",
        store_name="trade learning",
    )
    assert completed["pending_probe"] is False
    assert completed["probe_active"] is False
    assert completed["last_probe_successful"] is True


def test_active_trade_record_round_trip(dummy_store):
    store, loader = dummy_store
    now = datetime(2025, 3, 4, tzinfo=timezone.utc)

    state_utils.update_active_trade_record(
        loader,
        "NVDA",
        "buy",
        mode="probe",
        qty=1.5,
        strategy="maxdiff",
        opened_at_sim="2025-03-04T10:00:00+00:00",
        logger=None,
        now=now,
    )

    key = state_utils.state_key("NVDA", "buy")
    assert key in store.data
    record = store.data[key]
    assert record["mode"] == "probe"
    assert record["qty"] == 1.5
    assert record["entry_strategy"] == "maxdiff"

    fetched = state_utils.get_active_trade_record(loader, "NVDA", "buy")
    assert fetched == record

    state_utils.tag_active_trade_strategy(loader, "NVDA", "buy", "ci_guard")
    assert store.data[key]["entry_strategy"] == "ci_guard"

    removed = state_utils.pop_active_trade_record(loader, "NVDA", "buy")
    assert removed["mode"] == "probe"
    assert key not in store.data
