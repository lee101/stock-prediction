"""Tests for backtest utility modules."""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.backtest_env_utils import read_env_flag, coerce_keepalive_seconds, cpu_fallback_enabled, in_test_mode
from src.backtest_formatting_utils import fmt_number, format_table, log_table
from src.backtest_data_utils import mean_if_exists, to_numpy_array, normalize_series
from src.backtest_path_utils import canonicalize_path
from src.cooldown_utils import record_loss_timestamp, clear_cooldown, can_trade_now


class TestEnvUtils:
    """Tests for environment utility functions."""

    def test_read_env_flag_true(self, monkeypatch):
        monkeypatch.setenv("TEST_FLAG", "1")
        assert read_env_flag(["TEST_FLAG"]) is True

        monkeypatch.setenv("TEST_FLAG", "true")
        assert read_env_flag(["TEST_FLAG"]) is True

    def test_read_env_flag_false(self, monkeypatch):
        monkeypatch.setenv("TEST_FLAG", "0")
        assert read_env_flag(["TEST_FLAG"]) is False

        monkeypatch.setenv("TEST_FLAG", "false")
        assert read_env_flag(["TEST_FLAG"]) is False

    def test_read_env_flag_none(self, monkeypatch):
        monkeypatch.delenv("TEST_FLAG", raising=False)
        assert read_env_flag(["TEST_FLAG"]) is None

    def test_coerce_keepalive_seconds_valid(self, monkeypatch):
        monkeypatch.setenv("TEST_KEEPALIVE", "300")
        assert coerce_keepalive_seconds("TEST_KEEPALIVE", default=100.0) == 300.0

    def test_coerce_keepalive_seconds_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_KEEPALIVE", "invalid")
        assert coerce_keepalive_seconds("TEST_KEEPALIVE", default=100.0) == 100.0

    def test_coerce_keepalive_seconds_negative(self, monkeypatch):
        monkeypatch.setenv("TEST_KEEPALIVE", "-50")
        assert coerce_keepalive_seconds("TEST_KEEPALIVE", default=100.0) == 100.0

    def test_cpu_fallback_enabled(self, monkeypatch):
        monkeypatch.setenv("TEST_ENV", "1")
        assert cpu_fallback_enabled("TEST_ENV") is True

        monkeypatch.setenv("TEST_ENV", "0")
        assert cpu_fallback_enabled("TEST_ENV") is False

    def test_in_test_mode(self, monkeypatch):
        monkeypatch.setenv("TESTING", "1")
        assert in_test_mode() is True

        monkeypatch.delenv("TESTING", raising=False)
        monkeypatch.setenv("MARKETSIM_ALLOW_MOCK_ANALYTICS", "1")
        assert in_test_mode() is True


class TestFormattingUtils:
    """Tests for formatting utility functions."""

    def test_fmt_number_valid(self):
        assert fmt_number(3.14159, precision=2) == "3.14"
        assert fmt_number(100.0, precision=1) == "100.0"

    def test_fmt_number_none(self):
        assert fmt_number(None) == "-"

    def test_format_table(self):
        headers = ["Name", "Value"]
        rows = [["foo", "123"], ["bar", "456"]]
        result = format_table(headers, rows)
        assert "Name" in result
        assert "Value" in result
        assert "foo" in result
        assert "123" in result

    def test_format_table_empty(self):
        assert format_table(["A", "B"], []) == ""


class TestDataUtils:
    """Tests for data utility functions."""

    def test_mean_if_exists_valid(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert mean_if_exists(df, "a") == 2.0
        assert mean_if_exists(df, "b") == 5.0

    def test_mean_if_exists_missing_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert mean_if_exists(df, "missing") is None

    def test_mean_if_exists_empty_series(self):
        df = pd.DataFrame({"a": []})
        assert mean_if_exists(df, "a") is None

    def test_to_numpy_array_from_series(self):
        series = pd.Series([1.0, 2.0, 3.0])
        result = to_numpy_array(series)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_array_from_array(self):
        arr = np.array([1, 2, 3])
        result = to_numpy_array(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_array_scalar(self):
        result = to_numpy_array(np.array(5.0))
        assert result.shape == (1,)
        assert result[0] == 5.0


class TestPathUtils:
    """Tests for path utility functions."""

    def test_canonicalize_path_absolute(self):
        path = Path("/tmp/test")
        result = canonicalize_path(path)
        assert result.is_absolute()
        assert str(result) == "/tmp/test"

    def test_canonicalize_path_relative(self):
        result = canonicalize_path("foo/bar")
        assert result.is_absolute()

    def test_canonicalize_path_expanduser(self):
        result = canonicalize_path("~/test")
        assert result.is_absolute()
        assert "~" not in str(result)


class TestCooldownUtils:
    """Tests for cooldown utility functions."""

    def setup_method(self):
        """Clear cooldown state before each test."""
        # Import the module's state and clear it
        from src.cooldown_utils import _COOLDOWN_STATE
        _COOLDOWN_STATE.clear()

    def test_record_loss_timestamp(self):
        now = datetime.now(timezone.utc)
        iso_time = now.isoformat()
        record_loss_timestamp("AAPL", iso_time)

        # Verify we can't trade immediately
        assert can_trade_now("AAPL", now, min_cooldown_minutes=5) is False

        # Verify we can trade after cooldown
        future = now + timedelta(minutes=10)
        assert can_trade_now("AAPL", future, min_cooldown_minutes=5) is True

    def test_clear_cooldown(self):
        now = datetime.now(timezone.utc)
        iso_time = now.isoformat()
        record_loss_timestamp("AAPL", iso_time)

        # Clear the cooldown
        clear_cooldown("AAPL")

        # Should be able to trade immediately
        assert can_trade_now("AAPL", now, min_cooldown_minutes=5) is True

    def test_can_trade_now_no_cooldown(self):
        now = datetime.now(timezone.utc)
        # No cooldown recorded, should be able to trade
        assert can_trade_now("AAPL", now, min_cooldown_minutes=5) is True

    def test_record_loss_timestamp_none(self):
        # Should handle None gracefully
        record_loss_timestamp("AAPL", None)
        now = datetime.now(timezone.utc)
        assert can_trade_now("AAPL", now, min_cooldown_minutes=5) is True
