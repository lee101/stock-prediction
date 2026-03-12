from __future__ import annotations

from binanceneural.execution import SymbolRules
from src.binan.binance_conversion import (
    StableQuoteConversionPlan,
    _normalize_sell_conversion_plan,
)


def test_normalize_sell_conversion_plan_quantizes_quantity(monkeypatch) -> None:
    def fake_rules(symbol: str) -> SymbolRules:
        assert symbol == "FDUSDUSDT"
        return SymbolRules(min_qty=0.1, step_size=0.1)

    monkeypatch.setattr("src.binan.binance_conversion.resolve_symbol_rules", fake_rules)

    normalized = _normalize_sell_conversion_plan(
        StableQuoteConversionPlan(
            symbol="FDUSDUSDT",
            side="SELL",
            from_asset="FDUSD",
            to_asset="USDT",
            amount=67.79731399,
            quantity=67.79731399,
        )
    )

    assert normalized.quantity == 67.7


def test_normalize_sell_conversion_plan_rejects_below_min_qty(monkeypatch) -> None:
    def fake_rules(symbol: str) -> SymbolRules:
        assert symbol == "FDUSDUSDT"
        return SymbolRules(min_qty=1.0, step_size=0.1)

    monkeypatch.setattr("src.binan.binance_conversion.resolve_symbol_rules", fake_rules)

    try:
        _normalize_sell_conversion_plan(
            StableQuoteConversionPlan(
                symbol="FDUSDUSDT",
                side="SELL",
                from_asset="FDUSD",
                to_asset="USDT",
                amount=0.95,
                quantity=0.95,
            )
        )
    except ValueError as exc:
        assert "below minQty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for quantity below minQty")
