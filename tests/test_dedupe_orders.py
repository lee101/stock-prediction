"""Tests for _dedupe_side_orders exit-order cancellation logic."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-trading-agent-binance"))

from trade_binance_live import (
    _binance_error_code,
    _dedupe_side_orders,
    _BINANCE_ORDER_GONE_CODES,
)


def _make_order(symbol="BTCUSDT", side="SELL", order_id=100, qty=1.0, executed=0.0, price=50000.0):
    return {
        "symbol": symbol,
        "side": side,
        "orderId": order_id,
        "origQty": str(qty),
        "executedQty": str(executed),
        "price": str(price),
    }


class FakeBinanceAPIException(Exception):
    def __init__(self, code, message=""):
        self.code = code
        self.message = message
        super().__init__(f"APIError(code={code}): {message}")


# --- _binance_error_code tests ---

def test_error_code_from_attribute():
    exc = FakeBinanceAPIException(-2011, "Unknown order sent.")
    assert _binance_error_code(exc) == -2011


def test_error_code_from_string():
    exc = Exception("APIError(code=-2011): Unknown order sent.")
    assert _binance_error_code(exc) == -2011


def test_error_code_from_string_other():
    exc = Exception("APIError(code=-2013): Order does not exist.")
    assert _binance_error_code(exc) == -2013


def test_error_code_none_for_generic():
    exc = Exception("Connection timeout")
    assert _binance_error_code(exc) is None


def test_gone_codes():
    assert -2011 in _BINANCE_ORDER_GONE_CODES
    assert -2013 in _BINANCE_ORDER_GONE_CODES


# --- _dedupe_side_orders tests ---

CANCEL_PATH = "trade_binance_live._cancel_open_order"


def test_no_matching_orders():
    orders = [_make_order(symbol="ETHUSDT")]
    result, skip = _dedupe_side_orders(
        orders, symbol="BTCUSDT", side="SELL", execution_mode="margin", dry_run=False
    )
    assert skip is False
    assert result == orders


def test_existing_order_covers_qty():
    order = _make_order(qty=1.0, executed=0.0)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.9,
    )
    assert skip is True
    assert len(result) == 1


def test_existing_order_covers_notional():
    order = _make_order(qty=1.0, executed=0.0, price=50000.0)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_notional=49000.0,
    )
    assert skip is True


@patch(CANCEL_PATH)
def test_cancel_succeeds(mock_cancel):
    order = _make_order(order_id=101, qty=0.3)
    other = _make_order(symbol="ETHUSDT", order_id=200)
    result, skip = _dedupe_side_orders(
        [order, other], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False
    mock_cancel.assert_called_once_with("margin", "BTCUSDT", 101)
    assert len(result) == 1
    assert result[0]["orderId"] == 200


@patch(CANCEL_PATH, side_effect=FakeBinanceAPIException(-2011, "Unknown order sent."))
def test_cancel_fails_2011_order_gone(mock_cancel):
    order = _make_order(order_id=555, qty=0.3)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False, "should place new order when old order is gone"
    assert order not in result


@patch(CANCEL_PATH, side_effect=FakeBinanceAPIException(-2013, "Order does not exist."))
def test_cancel_fails_2013_order_gone(mock_cancel):
    order = _make_order(order_id=556, qty=0.3)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False
    assert order not in result


@patch(CANCEL_PATH, side_effect=Exception("APIError(code=-2011): Unknown order sent."))
def test_cancel_fails_2011_from_string(mock_cancel):
    order = _make_order(order_id=557, qty=0.3)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False


@patch(CANCEL_PATH, side_effect=Exception("Connection timeout"))
def test_cancel_fails_transient_error(mock_cancel):
    order = _make_order(order_id=558, qty=0.3)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False, "transient error should not prevent placing new order"


@patch(CANCEL_PATH, side_effect=ConnectionError("Network unreachable"))
def test_cancel_fails_network_error(mock_cancel):
    order = _make_order(order_id=559, qty=0.3)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False


def test_order_id_none():
    order = {"symbol": "BTCUSDT", "side": "SELL", "origQty": "0.3", "executedQty": "0.0", "price": "50000"}
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False, "no orderId means order is unverifiable, should place new"
    assert order not in result


def test_dry_run_skips_cancel():
    order = _make_order(order_id=600, qty=0.3)
    result, skip = _dedupe_side_orders(
        [order], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=True, desired_qty=0.5,
    )
    assert skip is False


@patch(CANCEL_PATH)
def test_multiple_orders_all_cancelled(mock_cancel):
    o1 = _make_order(order_id=701, qty=0.2)
    o2 = _make_order(order_id=702, qty=0.2)
    other = _make_order(symbol="ETHUSDT", order_id=800)
    result, skip = _dedupe_side_orders(
        [o1, o2, other], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False
    assert mock_cancel.call_count == 2
    assert len(result) == 1
    assert result[0]["orderId"] == 800


@patch(CANCEL_PATH)
def test_multiple_orders_second_fails_2011(mock_cancel):
    mock_cancel.side_effect = [None, FakeBinanceAPIException(-2011, "Unknown order")]
    o1 = _make_order(order_id=901, qty=0.2)
    o2 = _make_order(order_id=902, qty=0.2)
    result, skip = _dedupe_side_orders(
        [o1, o2], symbol="BTCUSDT", side="SELL", execution_mode="margin",
        dry_run=False, desired_qty=0.5,
    )
    assert skip is False
    assert len(result) == 0
