"""Unit tests for src/strategy_price_lookup.py"""

import pytest

from src.strategy_price_lookup import (
    get_entry_price,
    get_strategy_price_fields,
    get_takeprofit_price,
    is_limit_order_strategy,
)


@pytest.fixture
def sample_data():
    """Sample analysis data with all strategy price fields."""
    return {
        # Maxdiff strategy
        "maxdiffprofit_low_price": 95.0,
        "maxdiffprofit_high_price": 105.0,
        "neuralpricing_low_price": 96.5,
        "neuralpricing_high_price": 103.5,
        # Maxdiff always on strategy
        "maxdiffalwayson_low_price": 94.0,
        "maxdiffalwayson_high_price": 106.0,
        # Pctdiff strategy
        "pctdiff_entry_low_price": 96.0,
        "pctdiff_entry_high_price": 104.0,
        "pctdiff_takeprofit_low_price": 93.0,
        "pctdiff_takeprofit_high_price": 107.0,
        # Highlow strategy
        "predicted_low": 97.0,
        "predicted_high": 103.0,
    }


class TestGetEntryPrice:
    """Tests for get_entry_price function."""

    def test_maxdiff_buy(self, sample_data):
        """Test maxdiff strategy buy entry price."""
        price = get_entry_price(sample_data, "maxdiff", "buy")
        assert price == 96.5

    def test_maxdiff_sell(self, sample_data):
        """Test maxdiff strategy sell entry price."""
        price = get_entry_price(sample_data, "maxdiff", "sell")
        assert price == 103.5

    def test_maxdiff_fallback_without_neural(self, sample_data):
        """Ensure fallback to maxdiffprofit fields when neural pricing absent."""
        sample_data.pop("neuralpricing_low_price", None)
        sample_data.pop("neuralpricing_high_price", None)
        assert get_entry_price(sample_data, "maxdiff", "buy") == 95.0

    def test_maxdiffalwayson_buy(self, sample_data):
        """Test maxdiffalwayson strategy buy entry price."""
        price = get_entry_price(sample_data, "maxdiffalwayson", "buy")
        assert price == 94.0

    def test_maxdiffalwayson_sell(self, sample_data):
        """Test maxdiffalwayson strategy sell entry price."""
        price = get_entry_price(sample_data, "maxdiffalwayson", "sell")
        assert price == 106.0

    def test_pctdiff_buy(self, sample_data):
        """Test pctdiff strategy buy entry price."""
        price = get_entry_price(sample_data, "pctdiff", "buy")
        assert price == 96.0

    def test_pctdiff_sell(self, sample_data):
        """Test pctdiff strategy sell entry price."""
        price = get_entry_price(sample_data, "pctdiff", "sell")
        assert price == 104.0

    def test_highlow_buy(self, sample_data):
        """Test highlow strategy buy entry price."""
        price = get_entry_price(sample_data, "highlow", "buy")
        assert price == 97.0

    def test_highlow_sell(self, sample_data):
        """Test highlow strategy sell entry price."""
        price = get_entry_price(sample_data, "highlow", "sell")
        assert price == 103.0

    def test_case_insensitive(self, sample_data):
        """Test strategy name is case-insensitive."""
        assert get_entry_price(sample_data, "MaxDiff", "buy") == 96.5
        assert get_entry_price(sample_data, "PCTDIFF", "sell") == 104.0

    def test_whitespace_stripped(self, sample_data):
        """Test whitespace is stripped from strategy name."""
        assert get_entry_price(sample_data, "  maxdiff  ", "buy") == 96.5

    def test_unknown_strategy_returns_none(self, sample_data):
        """Test unknown strategy returns None."""
        assert get_entry_price(sample_data, "unknown", "buy") is None

    def test_none_strategy_returns_none(self, sample_data):
        """Test None strategy returns None."""
        assert get_entry_price(sample_data, None, "buy") is None

    def test_empty_strategy_returns_none(self, sample_data):
        """Test empty strategy returns None."""
        assert get_entry_price(sample_data, "", "buy") is None

    def test_missing_field_returns_none(self):
        """Test missing price field returns None."""
        data = {}
        assert get_entry_price(data, "maxdiff", "buy") is None


class TestGetTakeprofitPrice:
    """Tests for get_takeprofit_price function."""

    def test_maxdiff_buy(self, sample_data):
        """Test maxdiff strategy buy take-profit price."""
        price = get_takeprofit_price(sample_data, "maxdiff", "buy")
        assert price == 103.5

    def test_maxdiff_sell(self, sample_data):
        """Test maxdiff strategy sell take-profit price."""
        price = get_takeprofit_price(sample_data, "maxdiff", "sell")
        assert price == 96.5

    def test_pctdiff_buy(self, sample_data):
        """Test pctdiff strategy buy take-profit price."""
        price = get_takeprofit_price(sample_data, "pctdiff", "buy")
        assert price == 107.0

    def test_pctdiff_sell(self, sample_data):
        """Test pctdiff strategy sell take-profit price."""
        price = get_takeprofit_price(sample_data, "pctdiff", "sell")
        assert price == 93.0

    def test_highlow_buy(self, sample_data):
        """Test highlow strategy buy take-profit price."""
        price = get_takeprofit_price(sample_data, "highlow", "buy")
        assert price == 103.0

    def test_highlow_sell(self, sample_data):
        """Test highlow strategy sell take-profit price."""
        price = get_takeprofit_price(sample_data, "highlow", "sell")
        assert price == 97.0

    def test_unknown_strategy_returns_none(self, sample_data):
        """Test unknown strategy returns None."""
        assert get_takeprofit_price(sample_data, "unknown", "buy") is None


class TestGetStrategyPriceFields:
    """Tests for get_strategy_price_fields function."""

    def test_maxdiff_fields(self):
        """Test maxdiff strategy field names."""
        fields = get_strategy_price_fields("maxdiff")
        assert fields["buy_entry"] == "maxdiffprofit_low_price"
        assert fields["buy_takeprofit"] == "maxdiffprofit_high_price"
        assert fields["sell_entry"] == "maxdiffprofit_high_price"
        assert fields["sell_takeprofit"] == "maxdiffprofit_low_price"

    def test_maxdiffalwayson_fields(self):
        """Test maxdiffalwayson strategy field names."""
        fields = get_strategy_price_fields("maxdiffalwayson")
        assert fields["buy_entry"] == "maxdiffalwayson_low_price"
        assert fields["buy_takeprofit"] == "maxdiffalwayson_high_price"
        assert fields["sell_entry"] == "maxdiffalwayson_high_price"
        assert fields["sell_takeprofit"] == "maxdiffalwayson_low_price"

    def test_pctdiff_fields(self):
        """Test pctdiff strategy field names."""
        fields = get_strategy_price_fields("pctdiff")
        assert fields["buy_entry"] == "pctdiff_entry_low_price"
        assert fields["buy_takeprofit"] == "pctdiff_takeprofit_high_price"
        assert fields["sell_entry"] == "pctdiff_entry_high_price"
        assert fields["sell_takeprofit"] == "pctdiff_takeprofit_low_price"

    def test_highlow_fields(self):
        """Test highlow strategy field names."""
        fields = get_strategy_price_fields("highlow")
        assert fields["buy_entry"] == "predicted_low"
        assert fields["buy_takeprofit"] == "predicted_high"
        assert fields["sell_entry"] == "predicted_high"
        assert fields["sell_takeprofit"] == "predicted_low"

    def test_case_insensitive(self):
        """Test strategy name is case-insensitive."""
        fields = get_strategy_price_fields("MaxDiff")
        assert fields["buy_entry"] == "maxdiffprofit_low_price"

    def test_unknown_strategy_returns_empty(self):
        """Test unknown strategy returns empty dict."""
        fields = get_strategy_price_fields("unknown")
        assert fields == {}


class TestIsLimitOrderStrategy:
    """Tests for is_limit_order_strategy function."""

    def test_maxdiff_is_limit_order(self):
        """Test maxdiff is a limit order strategy."""
        assert is_limit_order_strategy("maxdiff") is True

    def test_maxdiffalwayson_is_limit_order(self):
        """Test maxdiffalwayson is a limit order strategy."""
        assert is_limit_order_strategy("maxdiffalwayson") is True

    def test_pctdiff_is_limit_order(self):
        """Test pctdiff is a limit order strategy."""
        assert is_limit_order_strategy("pctdiff") is True

    def test_highlow_is_limit_order(self):
        """Test highlow is a limit order strategy."""
        assert is_limit_order_strategy("highlow") is True

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_limit_order_strategy("MaxDiff") is True
        assert is_limit_order_strategy("PCTDIFF") is True

    def test_unknown_strategy_is_not_limit(self):
        """Test unknown strategy is not limit order."""
        assert is_limit_order_strategy("market") is False
        assert is_limit_order_strategy("unknown") is False

    def test_none_strategy_is_not_limit(self):
        """Test None strategy is not limit order."""
        assert is_limit_order_strategy(None) is False

    def test_empty_strategy_is_not_limit(self):
        """Test empty strategy is not limit order."""
        assert is_limit_order_strategy("") is False

    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        assert is_limit_order_strategy("  maxdiff  ") is True


class TestSymmetry:
    """Tests for symmetry between buy and sell sides."""

    def test_entry_exit_symmetry(self, sample_data):
        """Test that buy entry = sell takeprofit and vice versa."""
        for strategy in ["maxdiff", "maxdiffalwayson", "highlow"]:
            buy_entry = get_entry_price(sample_data, strategy, "buy")
            sell_tp = get_takeprofit_price(sample_data, strategy, "sell")
            assert buy_entry == sell_tp, f"{strategy} buy entry != sell takeprofit"

            sell_entry = get_entry_price(sample_data, strategy, "sell")
            buy_tp = get_takeprofit_price(sample_data, strategy, "buy")
            assert sell_entry == buy_tp, f"{strategy} sell entry != buy takeprofit"
