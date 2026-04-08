"""Tests for watcher refresh utilities."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread

import src.process_utils as process_utils
import src.watcher_refresh_utils as watcher_refresh_utils
from src.watcher_refresh_utils import (
    find_existing_watcher_price,
    should_use_existing_watcher_prices,
    should_spawn_watcher,
)


class TestShouldUseExistingWatcherPrices:
    """Test the logic for deciding whether to reuse existing watcher prices."""

    def test_no_metadata(self):
        """Empty metadata should not use existing prices."""
        should_use, reason = should_use_existing_watcher_prices({}, is_crypto=True)
        assert should_use is False
        assert reason == "no_metadata"

    def test_missing_timestamps(self):
        """Metadata without timestamps should not use existing prices."""
        metadata = {"limit_price": 100.0}
        should_use, reason = should_use_existing_watcher_prices(metadata, is_crypto=True)
        assert should_use is False
        assert reason == "missing_timestamps"

    def test_invalid_timestamps(self):
        """Metadata with invalid timestamps should not use existing prices."""
        metadata = {
            "started_at": "not a timestamp",
            "expiry_at": "also not a timestamp",
            "limit_price": 100.0,
        }
        should_use, reason = should_use_existing_watcher_prices(metadata, is_crypto=True)
        assert should_use is False
        assert reason == "invalid_timestamps"

    def test_expired_watcher(self):
        """Expired watchers should not use existing prices."""
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(hours=12)
        expiry_at = now - timedelta(hours=1)  # Expired 1 hour ago

        metadata = {
            "started_at": started_at.isoformat(),
            "expiry_at": expiry_at.isoformat(),
            "limit_price": 100.0,
        }
        should_use, reason = should_use_existing_watcher_prices(metadata, is_crypto=True)
        assert should_use is False
        assert "expired" in reason

    def test_crypto_within_24hrs_not_expired(self):
        """Crypto watchers within 24hrs and not expired should use existing prices."""
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(hours=6)  # 6 hours old
        expiry_at = now + timedelta(hours=18)  # Still valid

        metadata = {
            "started_at": started_at.isoformat(),
            "expiry_at": expiry_at.isoformat(),
            "limit_price": 100.0,
        }
        should_use, reason = should_use_existing_watcher_prices(metadata, is_crypto=True)
        assert should_use is True
        assert "within" in reason
        assert "keeping_original_plan" in reason

    def test_crypto_exceeds_24hrs(self):
        """Crypto watchers older than 24hrs should use new prices."""
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(hours=25)  # 25 hours old
        expiry_at = now + timedelta(hours=1)  # Still valid but old

        metadata = {
            "started_at": started_at.isoformat(),
            "expiry_at": expiry_at.isoformat(),
            "limit_price": 100.0,
        }
        should_use, reason = should_use_existing_watcher_prices(metadata, is_crypto=True)
        assert should_use is False
        assert "age_exceeded" in reason

    def test_stock_always_uses_new_prices(self):
        """Stocks should always use new prices (market conditions change)."""
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(hours=1)  # Fresh watcher
        expiry_at = now + timedelta(hours=23)  # Still valid

        metadata = {
            "started_at": started_at.isoformat(),
            "expiry_at": expiry_at.isoformat(),
            "limit_price": 100.0,
        }
        should_use, reason = should_use_existing_watcher_prices(metadata, is_crypto=False)
        assert should_use is False
        assert "stock_market_conditions_changed" in reason

    def test_custom_max_age(self):
        """Custom max_age_hours should be respected."""
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(hours=6)  # 6 hours old
        expiry_at = now + timedelta(hours=18)  # Still valid

        metadata = {
            "started_at": started_at.isoformat(),
            "expiry_at": expiry_at.isoformat(),
            "limit_price": 100.0,
        }

        # With default 24hrs - should use existing
        should_use, reason = should_use_existing_watcher_prices(
            metadata, is_crypto=True, max_age_hours=24.0
        )
        assert should_use is True

        # With 4hrs max - should not use existing
        should_use, reason = should_use_existing_watcher_prices(
            metadata, is_crypto=True, max_age_hours=4.0
        )
        assert should_use is False
        assert "age_exceeded" in reason


class TestShouldSpawnWatcher:
    """Test the logic for deciding whether to spawn a watcher."""

    def test_has_valid_existing_price(self):
        """If existing price is valid, don't spawn."""
        should_spawn, price, reason = should_spawn_watcher(
            existing_price=100.0,
            new_price=105.0,
            mode="entry",
        )
        assert should_spawn is False
        assert price == 100.0
        assert reason == "existing_watcher_valid"

    def test_no_existing_valid_new(self):
        """If no existing but new price valid, spawn with new price."""
        should_spawn, price, reason = should_spawn_watcher(
            existing_price=None,
            new_price=105.0,
            mode="entry",
        )
        assert should_spawn is True
        assert price == 105.0
        assert reason == "spawning_with_new_forecast"

    def test_no_existing_invalid_new(self):
        """If no existing and new price invalid, don't spawn."""
        should_spawn, price, reason = should_spawn_watcher(
            existing_price=None,
            new_price=None,
            mode="entry",
        )
        assert should_spawn is False
        assert price is None
        assert reason == "invalid_new_price"

    def test_no_existing_zero_new_price(self):
        """Zero new price should be treated as invalid."""
        should_spawn, price, reason = should_spawn_watcher(
            existing_price=None,
            new_price=0.0,
            mode="entry",
        )
        assert should_spawn is False
        assert price is None
        assert reason == "invalid_new_price"

    def test_no_existing_negative_new_price(self):
        """Negative new price should be treated as invalid."""
        should_spawn, price, reason = should_spawn_watcher(
            existing_price=None,
            new_price=-10.0,
            mode="entry",
        )
        assert should_spawn is False
        assert price is None
        assert reason == "invalid_new_price"


def test_find_existing_watcher_price_uses_public_process_utils_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_file = tmp_path / "BTCUSD_buy_entry_test.json"
    watcher_file.write_text("{not valid json", encoding="utf-8")

    metadata = {
        "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "limit_price": 101.25,
    }

    monkeypatch.setattr(process_utils, "load_watcher_metadata", lambda _path: dict(metadata))

    price, reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "entry",
        is_crypto=True,
    )

    assert price == 101.25
    assert reason is not None and "keeping_original_plan" in reason


def test_find_existing_watcher_price_uses_exit_takeprofit_field(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_file = tmp_path / "BTCUSD_sell_exit_test.json"
    watcher_file.write_text("{not valid json", encoding="utf-8")

    metadata = {
        "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "takeprofit_price": 202.5,
    }

    monkeypatch.setattr(process_utils, "load_watcher_metadata", lambda _path: dict(metadata))

    price, reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "sell",
        "exit",
        is_crypto=True,
    )

    assert price == 202.5
    assert reason is not None and "keeping_original_plan" in reason


def test_find_existing_watcher_price_rejects_unknown_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_file = tmp_path / "BTCUSD_buy_weird_test.json"
    watcher_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        process_utils,
        "load_watcher_metadata",
        lambda _path: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    price, reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "weird",
        is_crypto=True,
    )

    assert price is None
    assert reason == "unknown_watcher_mode"


def test_find_existing_watcher_price_reuses_cached_directory_index(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_refresh_utils._WATCHER_FILE_INDEX_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE.clear()
    (tmp_path / "BTCUSD_buy_entry_test.json").write_text("{}", encoding="utf-8")
    metadata = {
        "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "limit_price": 101.25,
    }
    monkeypatch.setattr(process_utils, "load_watcher_metadata", lambda _path: dict(metadata))

    build_calls = 0
    original_builder = watcher_refresh_utils._build_watcher_file_index

    def counting_builder(watcher_dir: Path) -> dict[tuple[str, str, watcher_refresh_utils.WatcherMode], tuple[Path, ...]]:
        nonlocal build_calls
        build_calls += 1
        return original_builder(watcher_dir)

    monkeypatch.setattr(watcher_refresh_utils, "_build_watcher_file_index", counting_builder)

    first_price, first_reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "entry",
        is_crypto=True,
    )
    second_price, second_reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "entry",
        is_crypto=True,
    )

    assert first_price == second_price == 101.25
    assert first_reason == second_reason
    assert build_calls == 1


def test_find_existing_watcher_price_refreshes_cached_directory_index_when_listing_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_refresh_utils._WATCHER_FILE_INDEX_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE.clear()
    (tmp_path / "BTCUSD_buy_entry_test.json").write_text("{}", encoding="utf-8")
    metadata_by_name = {
        "BTCUSD_buy_entry_test.json": {
            "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
            "limit_price": 101.25,
        },
        "ETHUSD_buy_entry_test.json": {
            "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
            "limit_price": 202.5,
        },
    }
    monkeypatch.setattr(process_utils, "load_watcher_metadata", lambda path: dict(metadata_by_name[path.name]))

    build_calls = 0
    original_builder = watcher_refresh_utils._build_watcher_file_index

    def counting_builder(watcher_dir: Path) -> dict[tuple[str, str, watcher_refresh_utils.WatcherMode], tuple[Path, ...]]:
        nonlocal build_calls
        build_calls += 1
        return original_builder(watcher_dir)

    listing_versions = iter((1, 2))
    monkeypatch.setattr(
        watcher_refresh_utils,
        "_watcher_directory_listing_version",
        lambda _watcher_dir: next(listing_versions),
    )
    monkeypatch.setattr(watcher_refresh_utils, "_build_watcher_file_index", counting_builder)

    first_price, _first_reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "entry",
        is_crypto=True,
    )
    (tmp_path / "ETHUSD_buy_entry_test.json").write_text("{}", encoding="utf-8")
    second_price, second_reason = find_existing_watcher_price(
        tmp_path,
        "ETHUSD",
        "buy",
        "entry",
        is_crypto=True,
    )

    assert first_price == 101.25
    assert second_price == 202.5
    assert second_reason is not None and "keeping_original_plan" in second_reason
    assert build_calls == 2


def test_find_existing_watcher_price_coalesces_concurrent_directory_index_cache_misses(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_refresh_utils._WATCHER_FILE_INDEX_CACHE.clear()
    watcher_refresh_utils._WATCHER_FILE_INDEX_CACHE_INFLIGHT.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE_INFLIGHT.clear()
    watcher_file = tmp_path / "BTCUSD_buy_entry_test.json"
    watcher_file.write_text("{}", encoding="utf-8")

    metadata = {
        "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "limit_price": 101.25,
    }

    monkeypatch.setattr(process_utils, "load_watcher_metadata", lambda _path: dict(metadata))

    build_calls = 0
    builder_started = Event()
    release_builder = Event()
    original_builder = watcher_refresh_utils._build_watcher_file_index

    def blocking_builder(
        watcher_dir: Path,
    ) -> dict[tuple[str, str, watcher_refresh_utils.WatcherMode], tuple[Path, ...]]:
        nonlocal build_calls
        build_calls += 1
        builder_started.set()
        release_builder.wait(timeout=5.0)
        return original_builder(watcher_dir)

    monkeypatch.setattr(watcher_refresh_utils, "_build_watcher_file_index", blocking_builder)

    results: list[tuple[float | None, str | None]] = []

    def run_lookup() -> None:
        results.append(
            find_existing_watcher_price(
                tmp_path,
                "BTCUSD",
                "buy",
                "entry",
                is_crypto=True,
            )
        )

    first = Thread(target=run_lookup)
    second = Thread(target=run_lookup)
    first.start()
    assert builder_started.wait(timeout=5.0)
    second.start()
    release_builder.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert len(results) == 2
    assert [price for price, _reason in results] == [101.25, 101.25]
    assert all(reason is not None and "keeping_original_plan" in reason for _, reason in results)
    assert build_calls == 1


def test_find_existing_watcher_price_reuses_cached_watcher_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_refresh_utils._WATCHER_FILE_INDEX_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE_INFLIGHT.clear()
    watcher_file = tmp_path / "BTCUSD_buy_entry_test.json"
    watcher_file.write_text("{}", encoding="utf-8")

    metadata = {
        "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "limit_price": 101.25,
    }

    load_calls = 0

    def counting_loader(_path: Path) -> dict:
        nonlocal load_calls
        load_calls += 1
        return dict(metadata)

    monkeypatch.setattr(process_utils, "load_watcher_metadata", counting_loader)

    first_price, first_reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "entry",
        is_crypto=True,
    )
    second_price, second_reason = find_existing_watcher_price(
        tmp_path,
        "BTCUSD",
        "buy",
        "entry",
        is_crypto=True,
    )

    assert first_price == second_price == 101.25
    assert first_reason == second_reason
    assert load_calls == 1


def test_find_existing_watcher_price_coalesces_concurrent_metadata_cache_misses(
    monkeypatch,
    tmp_path: Path,
) -> None:
    watcher_refresh_utils._WATCHER_FILE_INDEX_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE.clear()
    watcher_refresh_utils._WATCHER_METADATA_CACHE_INFLIGHT.clear()
    watcher_file = tmp_path / "BTCUSD_buy_entry_test.json"
    watcher_file.write_text("{}", encoding="utf-8")

    metadata = {
        "started_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "expiry_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "limit_price": 101.25,
    }

    load_calls = 0
    loader_started = Event()
    release_loader = Event()

    def blocking_loader(_path: Path) -> dict:
        nonlocal load_calls
        load_calls += 1
        loader_started.set()
        release_loader.wait(timeout=5.0)
        return dict(metadata)

    monkeypatch.setattr(process_utils, "load_watcher_metadata", blocking_loader)

    results: list[tuple[float | None, str | None]] = []

    def run_lookup() -> None:
        results.append(
            find_existing_watcher_price(
                tmp_path,
                "BTCUSD",
                "buy",
                "entry",
                is_crypto=True,
            )
        )

    first = Thread(target=run_lookup)
    second = Thread(target=run_lookup)
    first.start()
    assert loader_started.wait(timeout=5.0)
    second.start()
    release_loader.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert len(results) == 2
    assert [price for price, _reason in results] == [101.25, 101.25]
    assert all(reason is not None and "keeping_original_plan" in reason for _, reason in results)
    assert load_calls == 1


def test_watcher_price_field_accepts_enum_and_string_modes() -> None:
    assert (
        watcher_refresh_utils._watcher_price_field(watcher_refresh_utils.WatcherMode.ENTRY)
        == "limit_price"
    )
    assert (
        watcher_refresh_utils._watcher_price_field(watcher_refresh_utils.WatcherMode.EXIT)
        == "takeprofit_price"
    )
    assert watcher_refresh_utils._watcher_price_field(" entry ") == "limit_price"
    assert watcher_refresh_utils._watcher_price_field("bogus") is None
