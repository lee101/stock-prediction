from __future__ import annotations

import binance_status


def test_normalize_margin_order_symbol_maps_usd_pairs_to_usdt() -> None:
    assert binance_status._normalize_margin_order_symbol("BTCUSD") == "BTCUSDT"
    assert binance_status._normalize_margin_order_symbol("DOGEUSDT") == "DOGEUSDT"
    assert binance_status._normalize_margin_order_symbol("ETHFDUSD") == "ETHFDUSD"
    assert binance_status._normalize_margin_order_symbol("") is None


def test_margin_recent_order_symbols_merges_open_orders_positions_and_pnl_state() -> None:
    margin_rows = [
        {"asset": "USDT"},
        {"asset": "DOGE"},
        {"asset": "AAVE"},
    ]
    pnl_state = {
        "BTCUSD": {"realized_pnl": 1.0},
        "DOGEUSDT": {"realized_pnl": 2.0},
    }
    open_margin_orders = [
        {"symbol": "LINKUSDT"},
        {"symbol": "DOGEUSDT"},
    ]

    symbols = binance_status._margin_recent_order_symbols(
        margin_rows,
        pnl_state,
        open_margin_orders,
    )

    assert symbols == ["AAVEUSDT", "BTCUSDT", "DOGEUSDT", "LINKUSDT"]


def test_margin_exit_coverage_rows_split_actionable_and_dust() -> None:
    margin_rows = [
        {"asset": "USDT", "net": 100.0, "value_usdt": 100.0},
        {"asset": "BTC", "net": 0.01, "value_usdt": 800.0},
        {"asset": "ETH", "net": 0.2, "value_usdt": 500.0},
        {"asset": "AAVE", "net": 3.0, "value_usdt": 300.0},
        {"asset": "ADA", "net": 0.05, "value_usdt": 0.02},
    ]
    open_margin_orders = [
        {"symbol": "BTCFDUSD", "side": "SELL", "origQty": "0.01", "executedQty": "0"},
        {"symbol": "ETHUSDT", "side": "SELL", "origQty": "0.1", "executedQty": "0"},
        {"symbol": "ADAUSDT", "side": "BUY", "origQty": "1.0", "executedQty": "0"},
    ]

    rows = binance_status._margin_exit_coverage_rows(margin_rows, open_margin_orders)
    by_asset = {row["asset"]: row for row in rows}

    assert by_asset["BTC"]["status"] == "covered"
    assert by_asset["BTC"]["sell_qty"] == 0.01
    assert by_asset["ETH"]["status"] == "partial"
    assert by_asset["AAVE"]["status"] == "missing"
    assert by_asset["ADA"]["status"] == "dust"
    assert by_asset["ADA"]["buy_qty"] == 1.0
    assert "USDT" not in by_asset


def test_load_recent_margin_orders_sorts_descending_and_ignores_failures(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_get_all_margin_orders(symbol: str, limit: int = 10):
        calls.append((symbol, limit))
        if symbol == "DOGEUSDT":
            return [
                {"symbol": "DOGEUSDT", "time": 10, "status": "OLD"},
                {"symbol": "DOGEUSDT", "time": 30, "status": "NEWEST"},
            ]
        if symbol == "AAVEUSDT":
            return [{"symbol": "AAVEUSDT", "time": 20, "status": "MID"}]
        raise RuntimeError("bad symbol")

    monkeypatch.setattr(binance_status, "get_all_margin_orders", fake_get_all_margin_orders)

    rows = binance_status._load_recent_margin_orders(
        ["DOGEUSDT", "BADUSDT", "AAVEUSDT"]
    )

    assert calls == [
        ("DOGEUSDT", 10),
        ("BADUSDT", 10),
        ("AAVEUSDT", 10),
    ]
    assert [(row["symbol"], row["time"], row["status"]) for row in rows] == [
        ("DOGEUSDT", 30, "NEWEST"),
        ("AAVEUSDT", 20, "MID"),
        ("DOGEUSDT", 10, "OLD"),
    ]
