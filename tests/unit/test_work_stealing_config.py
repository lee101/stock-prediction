"""Tests for work stealing configuration."""

from datetime import datetime

from src.work_stealing_config import (
    CRYPTO_NORMAL_TOLERANCE_PCT,
    CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT,
    EST,
    get_entry_tolerance_for_symbol,
    is_crypto_out_of_hours,
    is_nyse_open,
    should_force_immediate_crypto,
)


def _eastern_dt(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    naive = datetime(year, month, day, hour, minute, second)
    localize = getattr(EST, "localize", None)
    if callable(localize):
        return localize(naive)
    return naive.replace(tzinfo=EST)


class TestNYSEHours:
    """Test NYSE market hours detection."""

    def test_weekday_market_hours_is_open(self):
        """Market should be open during trading hours on weekdays."""
        # Wednesday at 10:00 AM EST
        dt = _eastern_dt(2025, 1, 15, 10, 0, 0)
        assert is_nyse_open(dt) is True

    def test_weekday_before_open_is_closed(self):
        """Market should be closed before 9:30 AM."""
        # Wednesday at 9:00 AM EST (before open)
        dt = _eastern_dt(2025, 1, 15, 9, 0, 0)
        assert is_nyse_open(dt) is False

    def test_weekday_after_close_is_closed(self):
        """Market should be closed after 4:00 PM."""
        # Wednesday at 5:00 PM EST (after close)
        dt = _eastern_dt(2025, 1, 15, 17, 0, 0)
        assert is_nyse_open(dt) is False

    def test_saturday_is_closed(self):
        """Market should be closed on Saturday."""
        # Saturday at 10:00 AM EST
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)
        assert is_nyse_open(dt) is False

    def test_sunday_is_closed(self):
        """Market should be closed on Sunday."""
        # Sunday at 10:00 AM EST
        dt = _eastern_dt(2025, 1, 19, 10, 0, 0)
        assert is_nyse_open(dt) is False

    def test_market_open_edge_case(self):
        """Market should be open exactly at 9:30 AM."""
        # Wednesday at 9:30:00 AM EST (exact open)
        dt = _eastern_dt(2025, 1, 15, 9, 30, 0)
        assert is_nyse_open(dt) is True

    def test_market_close_edge_case(self):
        """Market should be closed exactly at 4:00 PM."""
        # Wednesday at 4:00:00 PM EST (exact close)
        dt = _eastern_dt(2025, 1, 15, 16, 0, 0)
        assert is_nyse_open(dt) is False


class TestCryptoOutOfHours:
    """Test crypto out-of-hours detection."""

    def test_weekday_market_hours_not_out_of_hours(self):
        """During market hours, not out-of-hours."""
        dt = _eastern_dt(2025, 1, 15, 10, 0, 0)
        assert is_crypto_out_of_hours(dt) is False

    def test_weekend_is_out_of_hours(self):
        """Weekends are out-of-hours."""
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)
        assert is_crypto_out_of_hours(dt) is True

    def test_after_hours_is_out_of_hours(self):
        """After market close is out-of-hours."""
        dt = _eastern_dt(2025, 1, 15, 18, 0, 0)
        assert is_crypto_out_of_hours(dt) is True


class TestEntryTolerance:
    """Test entry tolerance calculation."""

    def test_stock_always_normal_tolerance(self):
        """Stocks always use normal tolerance."""
        # During market hours
        dt = _eastern_dt(2025, 1, 15, 10, 0, 0)
        tolerance = get_entry_tolerance_for_symbol("AAPL", is_top_crypto=False, dt=dt)
        assert tolerance == CRYPTO_NORMAL_TOLERANCE_PCT

        # Out of hours
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)
        tolerance = get_entry_tolerance_for_symbol("AAPL", is_top_crypto=False, dt=dt)
        assert tolerance == CRYPTO_NORMAL_TOLERANCE_PCT

    def test_crypto_during_market_hours_normal_tolerance(self):
        """Crypto uses normal tolerance during market hours."""
        dt = _eastern_dt(2025, 1, 15, 10, 0, 0)

        tolerance = get_entry_tolerance_for_symbol("BTCUSD", is_top_crypto=False, dt=dt)
        assert tolerance == CRYPTO_NORMAL_TOLERANCE_PCT

    def test_crypto_out_of_hours_aggressive_tolerance(self):
        """Crypto uses aggressive tolerance out of hours."""
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)  # Weekend

        tolerance = get_entry_tolerance_for_symbol("BTCUSD", is_top_crypto=False, dt=dt)
        assert tolerance == CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT
        assert tolerance > CRYPTO_NORMAL_TOLERANCE_PCT  # More aggressive

    def test_top_crypto_out_of_hours_aggressive_tolerance(self):
        """Top crypto uses aggressive tolerance out of hours."""
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)  # Weekend

        tolerance = get_entry_tolerance_for_symbol("BTCUSD", is_top_crypto=True, dt=dt)
        assert tolerance == CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT


class TestForceImmediate:
    """Test force_immediate flag for crypto."""

    def test_top_crypto_out_of_hours_forces_immediate(self):
        """Rank 1 crypto out of hours should force immediate."""
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)  # Weekend

        assert should_force_immediate_crypto(rank=1, dt=dt) is True

    def test_third_crypto_out_of_hours_no_force(self):
        """Rank 3 crypto out of hours should not force immediate (default count=2)."""
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)  # Weekend

        assert should_force_immediate_crypto(rank=3, dt=dt) is False

    def test_crypto_during_market_hours_no_force(self):
        """Crypto during market hours should not force immediate."""
        dt = _eastern_dt(2025, 1, 15, 10, 0, 0)  # Weekday market hours

        assert should_force_immediate_crypto(rank=1, dt=dt) is False

    def test_configurable_force_count(self, monkeypatch):
        """Force immediate count should be configurable."""
        monkeypatch.setenv("CRYPTO_OUT_OF_HOURS_FORCE_COUNT", "2")

        # Need to reload the module to pick up env change
        import importlib

        import src.work_stealing_config as config_module

        importlib.reload(config_module)

        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)  # Weekend

        # Now rank 2 should also force
        assert config_module.should_force_immediate_crypto(rank=2, dt=dt) is True


class TestToleranceEdgeCases:
    """Test tolerance edge cases."""

    def test_unknown_symbol_defaults_to_normal_tolerance(self):
        """Unknown symbols should use normal tolerance."""
        dt = _eastern_dt(2025, 1, 18, 10, 0, 0)

        tolerance = get_entry_tolerance_for_symbol("UNKNOWNXYZ", is_top_crypto=False, dt=dt)
        assert tolerance == CRYPTO_NORMAL_TOLERANCE_PCT

    def test_tolerance_values_are_sensible(self):
        """Tolerance values should be in reasonable ranges."""
        # Normal tolerance around 0.66%
        assert 0.005 <= CRYPTO_NORMAL_TOLERANCE_PCT <= 0.01

        # Aggressive tolerance around 1.6%
        assert 0.01 <= CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT <= 0.03

        # Aggressive should be meaningfully larger
        assert CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT >= CRYPTO_NORMAL_TOLERANCE_PCT * 2
