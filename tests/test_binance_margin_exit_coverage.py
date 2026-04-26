from __future__ import annotations

import scripts.binance_margin_exit_coverage as cov
import pytest


def _asset(asset: str, net: float) -> dict[str, str]:
    return {"asset": asset, "netAsset": str(net)}


def _order(symbol: str, side: str, orig: float, executed: float = 0.0) -> dict[str, str]:
    return {
        "symbol": symbol,
        "side": side,
        "origQty": str(orig),
        "executedQty": str(executed),
    }


def test_load_coverage_flags_covered_partial_missing_and_ignores_dust(monkeypatch) -> None:
    monkeypatch.setattr(
        cov,
        "get_margin_account",
        lambda: {
            "userAssets": [
                _asset("USDT", 100.0),
                _asset("LINK", 10.0),
                _asset("BTC", 0.01),
                _asset("ETH", 2.0),
                _asset("AAVE", 4.0),
                _asset("DOGE", 10.0),
            ]
        },
    )
    monkeypatch.setattr(
        cov,
        "get_open_margin_orders",
        lambda: [
            _order("LINKUSDT", "SELL", 10.0),
            _order("BTCFDUSD", "SELL", 0.01),
            _order("ETHUSDT", "SELL", 1.0),
            _order("ETHUSDT", "BUY", 0.5),
        ],
    )
    prices = {
        "LINKUSDT": 20.0,
        "BTCUSDT": 80000.0,
        "ETHUSDT": 2000.0,
        "AAVEUSDT": 100.0,
        "DOGEUSDT": 0.1,
    }
    monkeypatch.setattr(cov.bw, "get_symbol_price", lambda symbol: prices[symbol])

    rows = cov.load_coverage(min_value_usdt=12.0)

    by_asset = {row.asset: row for row in rows}
    assert set(by_asset) == {"LINK", "BTC", "ETH", "AAVE"}
    assert by_asset["LINK"].status == "covered"
    assert by_asset["LINK"].sell_coverage_ratio == 1.0
    assert by_asset["BTC"].status == "covered"
    assert "BTCFDUSD" in by_asset["BTC"].pair_candidates
    assert by_asset["ETH"].status == "partial"
    assert by_asset["ETH"].open_buy_qty == 0.5
    assert by_asset["AAVE"].status == "missing"


def test_load_coverage_uses_remaining_order_quantity(monkeypatch) -> None:
    monkeypatch.setattr(
        cov,
        "get_margin_account",
        lambda: {"userAssets": [_asset("LINK", 10.0)]},
    )
    monkeypatch.setattr(
        cov,
        "get_open_margin_orders",
        lambda: [_order("LINKUSDT", "SELL", 20.0, executed=10.2)],
    )
    monkeypatch.setattr(cov.bw, "get_symbol_price", lambda _symbol: 20.0)

    [row] = cov.load_coverage(min_value_usdt=12.0)

    assert row.status == "covered"
    assert row.open_sell_qty == pytest.approx(9.8)
    assert row.sell_coverage_ratio == pytest.approx(0.98)
