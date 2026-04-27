from __future__ import annotations

import scripts.binance_margin_exit_coverage as cov
import pytest


def _asset(asset: str, net: float) -> dict[str, str]:
    return {"asset": asset, "netAsset": str(net)}


def _asset_with_free(asset: str, net: float, free: float) -> dict[str, str]:
    return {"asset": asset, "netAsset": str(net), "free": str(free)}


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
    assert by_asset["ETH"].coverage_gap_qty == pytest.approx(1.0)
    assert by_asset["ETH"].free_qty == pytest.approx(2.0)
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


def test_build_repair_plan_places_only_uncovered_free_quantity(monkeypatch) -> None:
    monkeypatch.setattr(
        cov,
        "get_margin_account",
        lambda: {
            "userAssets": [
                _asset_with_free("ETH", 2.0, 0.75),
                _asset_with_free("AAVE", 4.0, 4.0),
                _asset_with_free("LINK", 10.0, 10.0),
            ]
        },
    )
    monkeypatch.setattr(
        cov,
        "get_open_margin_orders",
        lambda: [
            _order("ETHUSDT", "SELL", 1.0),
            _order("LINKUSDT", "SELL", 10.0),
        ],
    )
    prices = {"ETHUSDT": 2000.0, "AAVEUSDT": 100.0, "LINKUSDT": 20.0}
    monkeypatch.setattr(cov.bw, "get_symbol_price", lambda symbol: prices[symbol])

    rows = cov.load_coverage(min_value_usdt=12.0)
    plans = cov.build_repair_plan(rows, target_markup_pct=0.20, min_order_value_usdt=12.0)

    by_asset = {plan.asset: plan for plan in plans}
    assert set(by_asset) == {"ETH", "AAVE"}
    assert by_asset["ETH"].quantity == pytest.approx(0.75)
    assert by_asset["ETH"].price == pytest.approx(2400.0)
    assert by_asset["ETH"].status == "planned"
    assert by_asset["ETH"].reason == "partial_gap"
    assert by_asset["AAVE"].quantity == pytest.approx(4.0)
    assert by_asset["AAVE"].price == pytest.approx(120.0)
    assert by_asset["AAVE"].reason == "missing_exit"


def test_place_repair_orders_quantizes_and_submits(monkeypatch) -> None:
    class Rules:
        tick_size = 0.1
        step_size = 0.01
        min_qty = 0.01
        min_notional = 5.0

    submitted: list[tuple[str, float, float, str]] = []
    monkeypatch.setattr(cov, "resolve_symbol_rules", lambda _pair: Rules())

    def fake_create_margin_limit_sell(pair, qty, price, *, side_effect_type):
        submitted.append((pair, qty, price, side_effect_type))
        return {"symbol": pair, "orderId": 123, "origQty": str(qty), "price": str(price)}

    monkeypatch.setattr(cov, "create_margin_limit_sell", fake_create_margin_limit_sell)
    plans = [
        cov.RepairPlan(
            asset="AAVE",
            pair="AAVEUSDT",
            quantity=4.009,
            price=120.04,
            notional_usdt=481.0,
            status="planned",
            reason="missing_exit",
        )
    ]

    placed = cov.place_repair_orders(plans, side_effect_type="NO_SIDE_EFFECT")

    assert submitted == [("AAVEUSDT", 4.0, 120.1, "NO_SIDE_EFFECT")]
    assert placed[0].status == "placed"
    assert placed[0].order == {"symbol": "AAVEUSDT", "orderId": 123, "origQty": "4.0", "price": "120.1"}
