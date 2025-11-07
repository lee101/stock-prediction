"""Tests for watcher refresh utilities."""

from datetime import datetime, timedelta, timezone

import pytest

from src.watcher_refresh_utils import (
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
