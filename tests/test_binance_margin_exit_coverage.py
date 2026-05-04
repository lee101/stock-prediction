from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.binance_margin_exit_coverage as coverage


def test_load_coverage_detects_short_liability_buy_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        coverage,
        "get_margin_account",
        lambda: {
            "userAssets": [
                {"asset": "DOGE", "netAsset": "-100", "free": "0", "borrowed": "100", "interest": "0"},
                {"asset": "AAVE", "netAsset": "5", "free": "5", "borrowed": "0", "interest": "0"},
            ]
        },
    )
    monkeypatch.setattr(
        coverage,
        "get_open_margin_orders",
        lambda: [
            {"symbol": "DOGEUSDT", "side": "BUY", "origQty": "60", "executedQty": "0"},
            {"symbol": "AAVEUSDT", "side": "SELL", "origQty": "5", "executedQty": "0"},
        ],
    )
    monkeypatch.setattr(
        coverage,
        "_asset_price_and_pair",
        lambda asset: (0.10 if asset == "DOGE" else 100.0, f"{asset}USDT"),
    )

    rows = coverage.load_coverage(min_value_usdt=1.0)
    by_asset = {row.asset: row for row in rows}

    assert by_asset["DOGE"].direction == "short"
    assert by_asset["DOGE"].coverage_side == "BUY"
    assert by_asset["DOGE"].position_qty == pytest.approx(100.0)
    assert by_asset["DOGE"].open_buy_qty == pytest.approx(60.0)
    assert by_asset["DOGE"].coverage_gap_qty == pytest.approx(40.0)
    assert by_asset["DOGE"].status == "partial"

    assert by_asset["AAVE"].direction == "long"
    assert by_asset["AAVE"].coverage_side == "SELL"
    assert by_asset["AAVE"].status == "covered"


def test_build_repair_plan_uses_buy_below_market_for_short_gap() -> None:
    row = coverage.CoverageRow(
        asset="DOGE",
        pair="DOGEUSDT",
        pair_candidates=("DOGEUSDT",),
        direction="short",
        coverage_side="BUY",
        net_qty=-100.0,
        free_qty=0.0,
        borrowed_qty=100.0,
        interest_qty=0.0,
        position_qty=100.0,
        market_price=0.10,
        est_value_usdt=10.0,
        open_sell_qty=0.0,
        open_buy_qty=60.0,
        coverage_gap_qty=40.0,
        sell_coverage_ratio=0.60,
        status="partial",
    )

    plan = coverage.build_repair_plan([row], target_markup_pct=0.20, min_order_value_usdt=1.0)

    assert len(plan) == 1
    assert plan[0].side == "BUY"
    assert plan[0].quantity == pytest.approx(40.0)
    assert plan[0].price == pytest.approx(0.08)
    assert plan[0].status == "planned"


def test_place_repair_orders_uses_auto_repay_for_short_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(
        coverage,
        "resolve_symbol_rules",
        lambda _pair: SimpleNamespace(tick_size=0.01, step_size=0.1, min_qty=0.1, min_notional=1.0),
    )
    monkeypatch.setattr(coverage, "quantize_price", lambda price, *, tick_size, side: price)
    monkeypatch.setattr(coverage, "quantize_qty", lambda qty, *, step_size: qty)

    def fake_buy(pair, qty, price, *, side_effect_type):
        captured.update(
            {
                "pair": pair,
                "qty": qty,
                "price": price,
                "side_effect_type": side_effect_type,
            }
        )
        return {"orderId": 42}

    monkeypatch.setattr(coverage, "create_margin_limit_buy", fake_buy)

    placed = coverage.place_repair_orders(
        [
            coverage.RepairPlan(
                asset="DOGE",
                pair="DOGEUSDT",
                side="BUY",
                quantity=40.0,
                price=0.08,
                notional_usdt=3.2,
                status="planned",
                reason="partial_gap",
            )
        ],
        side_effect_type="NO_SIDE_EFFECT",
        short_side_effect_type="AUTO_REPAY",
    )

    assert placed[0].status == "placed"
    assert captured["side_effect_type"] == "AUTO_REPAY"
