"""Tests for scripts/unwind_positions.py"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _mock_binance_client():
    with patch("src.binan.binance_wrapper._init_client", return_value=MagicMock()):
        yield


from scripts.unwind_positions import (
    _get_sell_qty,
    display_positions,
    scan_positions,
    unwind,
)


MOCK_MARGIN_ACCOUNT = {
    "userAssets": [
        {"asset": "LINK", "free": "107.789", "locked": "0", "borrowed": "50.0", "interest": "0.01", "netAsset": "57.779"},
        {"asset": "BTC", "free": "0.0000038", "locked": "0", "borrowed": "0", "interest": "0", "netAsset": "0.0000038"},
        {"asset": "USDT", "free": "2109.05", "locked": "0", "borrowed": "0", "interest": "0", "netAsset": "2109.05"},
        {"asset": "ETH", "free": "0.0000350", "locked": "0", "borrowed": "0", "interest": "0", "netAsset": "0.0000350"},
    ],
}


@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_scan_positions_finds_link(mock_account, mock_price):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.side_effect = lambda pair: {
        "LINKUSDT": 9.09,
        "BTCUSDT": 87000.0,
        "ETHUSDT": 2000.0,
    }.get(pair)

    positions = scan_positions()
    assets = [p["asset"] for p in positions]
    assert "LINK" in assets
    link = next(p for p in positions if p["asset"] == "LINK")
    assert link["free"] == 107.789
    assert abs(link["notional"] - 107.789 * 9.09) < 0.01


@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_scan_filters_dust(mock_account, mock_price):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.side_effect = lambda pair: {
        "LINKUSDT": 9.09,
        "BTCUSDT": 87000.0,
        "ETHUSDT": 2000.0,
    }.get(pair)
    positions = scan_positions()
    assets = [p["asset"] for p in positions]
    assert "ETH" not in assets


@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_scan_skips_stables(mock_account, mock_price):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.return_value = 9.09
    positions = scan_positions()
    assets = [p["asset"] for p in positions]
    assert "USDT" not in assets


@patch("scripts.unwind_positions.create_margin_order")
@patch("scripts.unwind_positions.resolve_symbol_rules")
@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_dry_run_no_orders(mock_account, mock_price, mock_rules, mock_order, tmp_path):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.return_value = 9.09
    mock_rules.return_value = MagicMock(step_size=0.01, min_qty=0.01)
    log_path = tmp_path / "unwind.jsonl"

    positions = scan_positions()
    unwind(positions, ["LINK"], False, log_path)
    mock_order.assert_not_called()

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) >= 1
    event = json.loads(lines[0])
    assert event["action"] == "dry_run_sell"
    assert event["asset"] == "LINK"


@patch("scripts.unwind_positions.create_margin_order")
@patch("scripts.unwind_positions.resolve_symbol_rules")
@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_live_places_sell(mock_account, mock_price, mock_rules, mock_order, tmp_path):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.return_value = 9.09
    mock_rules.return_value = MagicMock(step_size=0.01, min_qty=0.01)
    mock_order.return_value = {"orderId": 12345, "status": "FILLED"}
    log_path = tmp_path / "unwind.jsonl"

    positions = scan_positions()
    unwind(positions, ["LINK"], True, log_path)

    mock_order.assert_called_once_with(
        "LINKUSDT", "SELL", "MARKET", 107.78,
        side_effect_type="AUTO_REPAY",
    )
    lines = log_path.read_text().strip().split("\n")
    event = json.loads(lines[0])
    assert event["action"] == "sell"
    assert event["order"]["orderId"] == 12345


@patch("scripts.unwind_positions.create_margin_order")
@patch("scripts.unwind_positions.resolve_symbol_rules")
@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_live_handles_error(mock_account, mock_price, mock_rules, mock_order, tmp_path):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.return_value = 9.09
    mock_rules.return_value = MagicMock(step_size=0.01, min_qty=0.01)
    mock_order.side_effect = Exception("API error")
    log_path = tmp_path / "unwind.jsonl"

    positions = scan_positions()
    unwind(positions, ["LINK"], True, log_path)

    lines = log_path.read_text().strip().split("\n")
    event = json.loads(lines[0])
    assert event["action"] == "sell_error"
    assert "API error" in event["error"]


@patch("scripts.unwind_positions.create_margin_order")
@patch("scripts.unwind_positions.resolve_symbol_rules")
@patch("scripts.unwind_positions.bw.get_symbol_price")
@patch("scripts.unwind_positions.get_margin_account")
def test_symbol_filter(mock_account, mock_price, mock_rules, mock_order, tmp_path):
    mock_account.return_value = MOCK_MARGIN_ACCOUNT
    mock_price.side_effect = lambda pair: {"LINKUSDT": 9.09, "BTCUSDT": 87000.0}.get(pair)
    mock_rules.return_value = MagicMock(step_size=0.01, min_qty=0.01)
    mock_order.return_value = {"orderId": 99}
    log_path = tmp_path / "unwind.jsonl"

    positions = scan_positions()
    unwind(positions, ["LINK"], True, log_path)
    assert mock_order.call_count == 1
    assert mock_order.call_args[0][0] == "LINKUSDT"


def test_display_positions_empty(capsys):
    display_positions([])
    out = capsys.readouterr().out
    assert "No significant" in out


def test_display_positions_prints_table(capsys):
    positions = [{
        "asset": "LINK", "free": 107.789, "locked": 0.0,
        "borrowed": 50.0, "interest": 0.01, "net": 57.779,
        "price": 9.09, "notional": 979.80,
    }]
    display_positions(positions)
    out = capsys.readouterr().out
    assert "LINK" in out
    assert "979" in out
