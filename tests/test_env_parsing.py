"""Unit tests for src/env_parsing.py"""

import os
import pytest

from src.env_parsing import (
    FALSY_VALUES,
    TRUTHY_VALUES,
    parse_bool_env,
    parse_enum_env,
    parse_float_env,
    parse_int_env,
    parse_positive_float_env,
    parse_positive_int_env,
)


class TestParseBoolEnv:
    """Tests for parse_bool_env function."""

    def test_truthy_values(self, monkeypatch):
        """Test all truthy values."""
        for value in TRUTHY_VALUES:
            monkeypatch.setenv("TEST_FLAG", value)
            assert parse_bool_env("TEST_FLAG") is True

    def test_truthy_values_uppercase(self, monkeypatch):
        """Test truthy values are case-insensitive."""
        for value in TRUTHY_VALUES:
            monkeypatch.setenv("TEST_FLAG", value.upper())
            assert parse_bool_env("TEST_FLAG") is True

    def test_falsy_values(self, monkeypatch):
        """Test falsy values."""
        for value in FALSY_VALUES:
            monkeypatch.setenv("TEST_FLAG", value)
            assert parse_bool_env("TEST_FLAG") is False

    def test_default_when_not_set(self, monkeypatch):
        """Test default value when variable not set."""
        monkeypatch.delenv("TEST_FLAG", raising=False)
        assert parse_bool_env("TEST_FLAG", default=True) is True
        assert parse_bool_env("TEST_FLAG", default=False) is False

    def test_default_when_empty(self, monkeypatch):
        """Test default value when variable is empty."""
        monkeypatch.setenv("TEST_FLAG", "")
        assert parse_bool_env("TEST_FLAG", default=True) is True

    def test_whitespace_stripped(self, monkeypatch):
        """Test that whitespace is properly stripped."""
        monkeypatch.setenv("TEST_FLAG", "  1  ")
        assert parse_bool_env("TEST_FLAG") is True


class TestParseIntEnv:
    """Tests for parse_int_env function."""

    def test_valid_integer(self, monkeypatch):
        """Test parsing valid integers."""
        monkeypatch.setenv("TEST_INT", "42")
        assert parse_int_env("TEST_INT") == 42

    def test_negative_integer(self, monkeypatch):
        """Test parsing negative integers."""
        monkeypatch.setenv("TEST_INT", "-10")
        assert parse_int_env("TEST_INT") == -10

    def test_default_when_not_set(self, monkeypatch):
        """Test default value when not set."""
        monkeypatch.delenv("TEST_INT", raising=False)
        assert parse_int_env("TEST_INT", default=99) == 99

    def test_default_when_invalid(self, monkeypatch):
        """Test default value when invalid."""
        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert parse_int_env("TEST_INT", default=10) == 10

    def test_min_val_clamp(self, monkeypatch):
        """Test minimum value clamping."""
        monkeypatch.setenv("TEST_INT", "5")
        assert parse_int_env("TEST_INT", min_val=10) == 10

    def test_max_val_clamp(self, monkeypatch):
        """Test maximum value clamping."""
        monkeypatch.setenv("TEST_INT", "100")
        assert parse_int_env("TEST_INT", max_val=50) == 50

    def test_both_bounds(self, monkeypatch):
        """Test both min and max bounds."""
        monkeypatch.setenv("TEST_INT", "25")
        result = parse_int_env("TEST_INT", min_val=10, max_val=50)
        assert result == 25

    def test_whitespace_stripped(self, monkeypatch):
        """Test whitespace is stripped."""
        monkeypatch.setenv("TEST_INT", "  42  ")
        assert parse_int_env("TEST_INT") == 42


class TestParseFloatEnv:
    """Tests for parse_float_env function."""

    def test_valid_float(self, monkeypatch):
        """Test parsing valid floats."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert abs(parse_float_env("TEST_FLOAT") - 3.14) < 0.001

    def test_integer_as_float(self, monkeypatch):
        """Test parsing integers as floats."""
        monkeypatch.setenv("TEST_FLOAT", "42")
        assert abs(parse_float_env("TEST_FLOAT") - 42.0) < 0.001

    def test_scientific_notation(self, monkeypatch):
        """Test scientific notation."""
        monkeypatch.setenv("TEST_FLOAT", "1.5e-3")
        assert abs(parse_float_env("TEST_FLOAT") - 0.0015) < 0.00001

    def test_default_when_not_set(self, monkeypatch):
        """Test default value when not set."""
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        assert abs(parse_float_env("TEST_FLOAT", default=1.5) - 1.5) < 0.001

    def test_default_when_invalid(self, monkeypatch):
        """Test default value when invalid."""
        monkeypatch.setenv("TEST_FLOAT", "not_a_number")
        assert abs(parse_float_env("TEST_FLOAT", default=2.5) - 2.5) < 0.001

    def test_min_val_clamp(self, monkeypatch):
        """Test minimum value clamping."""
        monkeypatch.setenv("TEST_FLOAT", "0.5")
        result = parse_float_env("TEST_FLOAT", min_val=1.0)
        assert abs(result - 1.0) < 0.001

    def test_max_val_clamp(self, monkeypatch):
        """Test maximum value clamping."""
        monkeypatch.setenv("TEST_FLOAT", "10.5")
        result = parse_float_env("TEST_FLOAT", max_val=5.0)
        assert abs(result - 5.0) < 0.001


class TestParseEnumEnv:
    """Tests for parse_enum_env function."""

    def test_valid_value(self, monkeypatch):
        """Test valid enum value."""
        monkeypatch.setenv("TEST_ENUM", "INFO")
        result = parse_enum_env("TEST_ENUM", ["DEBUG", "INFO", "WARNING"], "INFO")
        assert result == "info"

    def test_case_insensitive(self, monkeypatch):
        """Test case insensitivity."""
        monkeypatch.setenv("TEST_ENUM", "InFo")
        result = parse_enum_env("TEST_ENUM", ["DEBUG", "INFO", "WARNING"], "INFO")
        assert result == "info"

    def test_invalid_value_returns_default(self, monkeypatch):
        """Test invalid value returns default."""
        monkeypatch.setenv("TEST_ENUM", "INVALID")
        result = parse_enum_env("TEST_ENUM", ["DEBUG", "INFO"], "DEBUG")
        assert result == "debug"

    def test_not_set_returns_default(self, monkeypatch):
        """Test not set returns default."""
        monkeypatch.delenv("TEST_ENUM", raising=False)
        result = parse_enum_env("TEST_ENUM", ["DEBUG", "INFO"], "INFO")
        assert result == "info"


class TestParsePositiveIntEnv:
    """Tests for parse_positive_int_env function."""

    def test_positive_value(self, monkeypatch):
        """Test positive values."""
        monkeypatch.setenv("TEST_POS_INT", "10")
        assert parse_positive_int_env("TEST_POS_INT") == 10

    def test_zero_clamped_to_min(self, monkeypatch):
        """Test zero is clamped to minimum 1."""
        monkeypatch.setenv("TEST_POS_INT", "0")
        assert parse_positive_int_env("TEST_POS_INT", default=1) == 1

    def test_negative_clamped_to_min(self, monkeypatch):
        """Test negative is clamped to minimum 1."""
        monkeypatch.setenv("TEST_POS_INT", "-5")
        assert parse_positive_int_env("TEST_POS_INT", default=1) == 1

    def test_default_positive(self, monkeypatch):
        """Test default is used when not set."""
        monkeypatch.delenv("TEST_POS_INT", raising=False)
        assert parse_positive_int_env("TEST_POS_INT", default=5) == 5


class TestParsePositiveFloatEnv:
    """Tests for parse_positive_float_env function."""

    def test_positive_value(self, monkeypatch):
        """Test positive values."""
        monkeypatch.setenv("TEST_POS_FLOAT", "3.14")
        result = parse_positive_float_env("TEST_POS_FLOAT")
        assert abs(result - 3.14) < 0.001

    def test_zero_with_positive_default(self, monkeypatch):
        """Test zero with positive default."""
        monkeypatch.setenv("TEST_POS_FLOAT", "0.0")
        result = parse_positive_float_env("TEST_POS_FLOAT", default=1.0)
        assert abs(result - 0.0) < 0.001

    def test_negative_clamped(self, monkeypatch):
        """Test negative is clamped to 0."""
        monkeypatch.setenv("TEST_POS_FLOAT", "-1.5")
        result = parse_positive_float_env("TEST_POS_FLOAT", default=1.0)
        assert abs(result - 0.0) < 0.001


class TestConstants:
    """Tests for module constants."""

    def test_truthy_values_defined(self):
        """Test TRUTHY_VALUES is properly defined."""
        assert "1" in TRUTHY_VALUES
        assert "true" in TRUTHY_VALUES
        assert "yes" in TRUTHY_VALUES
        assert "on" in TRUTHY_VALUES

    def test_falsy_values_defined(self):
        """Test FALSY_VALUES is properly defined."""
        assert "0" in FALSY_VALUES
        assert "false" in FALSY_VALUES
        assert "no" in FALSY_VALUES
        assert "off" in FALSY_VALUES
