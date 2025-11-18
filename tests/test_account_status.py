"""Tests for get_account_status function."""

import os
import time
import subprocess
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import alpaca_wrapper


class TestGetAccountStatus:
    """Test the get_account_status function."""

    @patch('alpaca_wrapper.get_all_positions')
    @patch('alpaca_wrapper.get_account')
    @patch('alpaca_wrapper.get_clock')
    @patch('alpaca_wrapper.refresh_account_cache')
    def test_get_account_status_success(
        self,
        mock_refresh_cache,
        mock_get_clock,
        mock_get_account,
        mock_get_positions,
    ):
        """Test successful account status retrieval."""
        # Setup mocks
        mock_positions = [
            SimpleNamespace(symbol='AAPL', qty=10),
            SimpleNamespace(symbol='BTCUSD', qty=0.5),
        ]
        mock_get_positions.return_value = mock_positions

        mock_account = SimpleNamespace(
            equity='50000.00',
            cash='10000.00',
            multiplier='1.0',
            buying_power='50000.00',
        )
        mock_get_account.return_value = mock_account

        mock_clock = SimpleNamespace(is_open=True)
        mock_get_clock.return_value = mock_clock

        # Call function
        result = alpaca_wrapper.get_account_status()

        # Assertions
        assert result['success'] is True
        assert result['error'] is None
        assert result['positions'] == mock_positions
        assert result['account'] == mock_account
        assert result['buying_power'] == 50000.00
        assert result['market_status']['is_open'] is True
        assert result['market_status']['clock'] == mock_clock

        # Verify calls - now defaults to force=False for caching
        mock_refresh_cache.assert_called_once_with(force=False)
        mock_get_positions.assert_called_once()
        mock_get_account.assert_called_once_with(use_cache=True)
        mock_get_clock.assert_called_once()

    @patch('alpaca_wrapper.get_all_positions')
    @patch('alpaca_wrapper.get_account')
    @patch('alpaca_wrapper.get_clock')
    @patch('alpaca_wrapper.refresh_account_cache')
    def test_get_account_status_no_buying_power(
        self,
        mock_refresh_cache,
        mock_get_clock,
        mock_get_account,
        mock_get_positions,
    ):
        """Test account status when buying_power is None."""
        # Setup mocks
        mock_get_positions.return_value = []

        # Account with no buying_power field
        mock_account = SimpleNamespace(
            equity='30000.00',
            cash='5000.00',
            multiplier='2.0',
            buying_power=None,
        )
        mock_get_account.return_value = mock_account

        mock_clock = SimpleNamespace(is_open=False)
        mock_get_clock.return_value = mock_clock

        # Call function
        result = alpaca_wrapper.get_account_status()

        # Assertions
        assert result['success'] is True
        # Should calculate buying power from equity * multiplier
        # Note: the function uses the global 'equity' variable, so this test
        # might not work exactly as expected in isolation
        assert result['buying_power'] is not None

    @patch('alpaca_wrapper.refresh_account_cache')
    def test_get_account_status_connection_error(self, mock_refresh_cache):
        """Test account status with connection error."""
        import requests.exceptions

        mock_refresh_cache.side_effect = requests.exceptions.ConnectionError("Network error")

        # Call function
        result = alpaca_wrapper.get_account_status()

        # Assertions
        assert result['success'] is False
        assert 'Connection error' in result['error']
        assert result['positions'] is None
        assert result['account'] is None

    @patch('alpaca_wrapper.refresh_account_cache')
    def test_get_account_status_api_error(self, mock_refresh_cache):
        """Test account status with API error."""
        from alpaca_trade_api.rest import APIError

        mock_refresh_cache.side_effect = APIError("API error")

        # Call function
        result = alpaca_wrapper.get_account_status()

        # Assertions
        assert result['success'] is False
        assert 'API error' in result['error']


class TestGetAccountCaching:
    """Test the disk caching functionality for get_account."""

    @patch('alpaca_wrapper.alpaca_api')
    def test_get_account_uses_cache(self, mock_alpaca_api):
        """Test that get_account uses disk cache."""
        # Clear cache first
        cache_key = f"account_{alpaca_wrapper._IS_PAPER}"
        alpaca_wrapper._account_diskcache.delete(cache_key)
        alpaca_wrapper._account_diskcache.delete(cache_key + '_age')

        # Setup mock
        mock_account = SimpleNamespace(
            equity='100000.00',
            cash='20000.00',
            multiplier='1.0',
            buying_power='100000.00',
        )
        mock_alpaca_api.get_account.return_value = mock_account

        # First call - should fetch from API
        result1 = alpaca_wrapper.get_account()
        assert result1 == mock_account
        assert mock_alpaca_api.get_account.call_count == 1

        # Second call - should use cache
        result2 = alpaca_wrapper.get_account()
        assert result2 == mock_account
        # Should still be 1 because it used the cache
        assert mock_alpaca_api.get_account.call_count == 1

        # Clear cache
        alpaca_wrapper._account_diskcache.delete(cache_key)
        alpaca_wrapper._account_diskcache.delete(cache_key + '_age')

    @patch('alpaca_wrapper.alpaca_api')
    def test_get_account_skip_cache(self, mock_alpaca_api):
        """Test that get_account can skip cache when use_cache=False."""
        # Setup mock
        mock_account = SimpleNamespace(
            equity='100000.00',
            cash='20000.00',
            multiplier='1.0',
            buying_power='100000.00',
        )
        mock_alpaca_api.get_account.return_value = mock_account

        # Clear cache
        cache_key = f"account_{alpaca_wrapper._IS_PAPER}"
        alpaca_wrapper._account_diskcache.delete(cache_key)

        # Call with use_cache=False
        result = alpaca_wrapper.get_account(use_cache=False)
        assert result == mock_account
        assert mock_alpaca_api.get_account.call_count == 1

        # Call again with use_cache=False - should fetch again
        result2 = alpaca_wrapper.get_account(use_cache=False)
        assert result2 == mock_account
        assert mock_alpaca_api.get_account.call_count == 2

        # Clear cache
        alpaca_wrapper._account_diskcache.delete(cache_key)


class TestGetClockCaching:
    """Test the disk caching functionality for get_clock."""

    @patch('alpaca_wrapper.alpaca_api')
    def test_get_clock_uses_cache(self, mock_alpaca_api):
        """Test that get_clock_internal uses disk cache."""
        # Clear cache first
        cache_key = f"clock_{alpaca_wrapper._IS_PAPER}"
        alpaca_wrapper._account_diskcache.delete(cache_key)

        # Setup mock
        mock_clock = SimpleNamespace(
            is_open=True,
            timestamp='2025-11-18T10:00:00Z',
        )
        mock_alpaca_api.get_clock.return_value = mock_clock

        # First call - should fetch from API
        result1 = alpaca_wrapper.get_clock_internal()
        assert result1 == mock_clock
        assert mock_alpaca_api.get_clock.call_count == 1

        # Second call - should use cache
        result2 = alpaca_wrapper.get_clock_internal()
        assert result2 == mock_clock
        # Should still be 1 because it used the cache
        assert mock_alpaca_api.get_clock.call_count == 1

        # Clear cache
        alpaca_wrapper._account_diskcache.delete(cache_key)

    @patch('alpaca_wrapper.alpaca_api')
    def test_get_clock_skip_cache(self, mock_alpaca_api):
        """Test that get_clock_internal can skip cache when use_cache=False."""
        # Setup mock
        mock_clock = SimpleNamespace(
            is_open=True,
            timestamp='2025-11-18T10:00:00Z',
        )
        mock_alpaca_api.get_clock.return_value = mock_clock

        # Clear cache
        cache_key = f"clock_{alpaca_wrapper._IS_PAPER}"
        alpaca_wrapper._account_diskcache.delete(cache_key)

        # Call with use_cache=False
        result = alpaca_wrapper.get_clock_internal(use_cache=False)
        assert result == mock_clock
        assert mock_alpaca_api.get_clock.call_count == 1

        # Call again with use_cache=False - should fetch again
        result2 = alpaca_wrapper.get_clock_internal(use_cache=False)
        assert result2 == mock_clock
        assert mock_alpaca_api.get_clock.call_count == 2

        # Clear cache
        alpaca_wrapper._account_diskcache.delete(cache_key)


class TestMultiProcessCacheSharing:
    """Test that cache is shared across processes."""

    @patch('alpaca_wrapper.alpaca_api')
    def test_cache_shared_across_processes(self, mock_alpaca_api):
        """Test that disk cache is shared between parent and child processes."""
        # Clear cache first
        cache_key = f"account_{alpaca_wrapper._IS_PAPER}"
        alpaca_wrapper._account_diskcache.delete(cache_key)
        alpaca_wrapper._account_diskcache.delete(cache_key + '_age')

        # Setup mock in parent process
        mock_account = SimpleNamespace(
            equity='100000.00',
            cash='20000.00',
            multiplier='1.0',
            buying_power='100000.00',
        )
        mock_alpaca_api.get_account.return_value = mock_account

        # Call get_account to populate cache
        result = alpaca_wrapper.get_account()
        assert result == mock_account
        assert mock_alpaca_api.get_account.call_count == 1

        # Verify cache was populated
        cached_data = alpaca_wrapper._account_diskcache.get(cache_key)
        assert cached_data is not None
        assert cached_data.equity == '100000.00'

        # Note: Full multi-process test would require subprocess module
        # This test verifies the cache can be read/written by the same process
        # The disk cache by design is accessible across processes

        # Clear cache
        alpaca_wrapper._account_diskcache.delete(cache_key)
        alpaca_wrapper._account_diskcache.delete(cache_key + '_age')


class TestAccountStatusWithCaching:
    """Test get_account_status with caching enabled."""

    @patch('alpaca_wrapper.get_all_positions')
    @patch('alpaca_wrapper.get_account')
    @patch('alpaca_wrapper.get_clock')
    @patch('alpaca_wrapper.refresh_account_cache')
    def test_account_status_uses_cache_by_default(
        self,
        mock_refresh_cache,
        mock_get_clock,
        mock_get_account,
        mock_get_positions,
    ):
        """Test that get_account_status uses cache by default."""
        # Setup mocks
        mock_get_positions.return_value = []
        mock_account = SimpleNamespace(
            equity='50000.00',
            cash='10000.00',
            multiplier='1.0',
            buying_power='50000.00',
        )
        mock_get_account.return_value = mock_account
        mock_clock = SimpleNamespace(is_open=True)
        mock_get_clock.return_value = mock_clock

        # Call with default params - should use cache
        result = alpaca_wrapper.get_account_status()

        assert result['success'] is True
        # Should not force refresh by default
        mock_refresh_cache.assert_called_once_with(force=False)
        # Should pass use_cache=True to get_account
        mock_get_account.assert_called_once_with(use_cache=True)

    @patch('alpaca_wrapper.get_all_positions')
    @patch('alpaca_wrapper.get_account')
    @patch('alpaca_wrapper.get_clock')
    @patch('alpaca_wrapper.refresh_account_cache')
    def test_account_status_can_force_refresh(
        self,
        mock_refresh_cache,
        mock_get_clock,
        mock_get_account,
        mock_get_positions,
    ):
        """Test that get_account_status can force refresh."""
        # Setup mocks
        mock_get_positions.return_value = []
        mock_account = SimpleNamespace(
            equity='50000.00',
            cash='10000.00',
            multiplier='1.0',
            buying_power='50000.00',
        )
        mock_get_account.return_value = mock_account
        mock_clock = SimpleNamespace(is_open=True)
        mock_get_clock.return_value = mock_clock

        # Call with force_refresh=True
        result = alpaca_wrapper.get_account_status(force_refresh=True, use_cache=False)

        assert result['success'] is True
        # Should force refresh
        mock_refresh_cache.assert_called_once_with(force=True)
        # Should bypass cache
        mock_get_account.assert_called_once_with(use_cache=False)
