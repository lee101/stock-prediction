from __future__ import annotations

import pytest

from src.binan.margin_smoke import (
    MarginAssetSnapshot,
    build_excess_flatten_qty,
    build_liability_cleanup_qty,
    build_market_qty_from_notional,
)


def test_margin_asset_snapshot_computes_liability_fields() -> None:
    snapshot = MarginAssetSnapshot(
        free=445.3389,
        borrowed=450.47538678,
        interest=0.05545739,
        net_asset=-5.19194417,
    )
    assert snapshot.liability == pytest.approx(450.53084417)
    assert snapshot.deficit == pytest.approx(5.19194417)
    assert snapshot.repayable == pytest.approx(445.3389)


def test_build_liability_cleanup_qty_rounds_up_to_step() -> None:
    snapshot = MarginAssetSnapshot(
        free=445.3389,
        borrowed=450.47538678,
        interest=0.05545739,
    )
    qty = build_liability_cleanup_qty(
        snapshot=snapshot,
        market_price=0.21,
        step_size=1.0,
        min_qty=1.0,
        min_notional=1.0,
    )
    assert qty == pytest.approx(6.0)


def test_build_liability_cleanup_qty_returns_zero_when_no_deficit() -> None:
    snapshot = MarginAssetSnapshot(
        free=10.0,
        borrowed=5.0,
        interest=0.1,
    )
    qty = build_liability_cleanup_qty(
        snapshot=snapshot,
        market_price=0.21,
        step_size=1.0,
        min_qty=1.0,
        min_notional=1.0,
    )
    assert qty == 0.0


def test_build_liability_cleanup_qty_respects_min_notional() -> None:
    snapshot = MarginAssetSnapshot(
        free=0.0,
        borrowed=0.25,
        interest=0.0,
    )
    qty = build_liability_cleanup_qty(
        snapshot=snapshot,
        market_price=10.0,
        step_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
    )
    assert qty == pytest.approx(0.5)


def test_build_market_qty_from_notional_rounds_up_to_cover_target() -> None:
    qty = build_market_qty_from_notional(
        target_notional=5.10,
        market_price=0.21,
        step_size=1.0,
        min_qty=1.0,
        min_notional=1.0,
    )
    assert qty == pytest.approx(25.0)


def test_build_market_qty_from_notional_respects_min_qty_and_min_notional() -> None:
    qty = build_market_qty_from_notional(
        target_notional=0.2,
        market_price=100.0,
        step_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
    )
    assert qty == pytest.approx(0.05)


def test_build_excess_flatten_qty_only_returns_sellable_excess() -> None:
    qty = build_excess_flatten_qty(
        snapshot=MarginAssetSnapshot(
            free=6.8,
            borrowed=0.0,
            interest=0.0,
            net_asset=6.8,
        ),
        market_price=0.09011,
        step_size=1.0,
        min_qty=1.0,
        min_notional=1.0,
    )
    assert qty == 0.0

    qty2 = build_excess_flatten_qty(
        snapshot=MarginAssetSnapshot(
            free=13.2,
            borrowed=0.0,
            interest=0.0,
            net_asset=13.2,
        ),
        market_price=0.09011,
        step_size=1.0,
        min_qty=1.0,
        min_notional=1.0,
    )
    assert qty2 == pytest.approx(13.0)
