from __future__ import annotations

from types import SimpleNamespace

from src.binan import stable_quote_router as module


def test_preferred_spot_execution_symbol_uses_fdusd_for_zero_fee_majors(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_TLD", raising=False)
    monkeypatch.delenv("BINANCE_DEFAULT_QUOTE", raising=False)

    assert module.preferred_spot_execution_symbol("BTCUSD") == "BTCFDUSD"
    assert module.preferred_spot_execution_symbol("ETHUSD") == "ETHFDUSD"
    assert module.preferred_spot_execution_symbol("SOLUSD") == "SOLUSDT"
    assert module.preferred_spot_execution_symbol("DOGEUSD") == "DOGEUSDT"


def test_get_spendable_quote_balance_includes_convertible_stablecoin(monkeypatch) -> None:
    balances = [
        {"asset": "FDUSD", "free": "12.5", "locked": "0"},
        {"asset": "USDT", "free": "7.25", "locked": "0"},
    ]
    monkeypatch.setattr(module.binance_wrapper, "get_account_balances", lambda: balances)

    assert module.get_spendable_quote_balance("BTCFDUSD") == 19.75
    assert module.get_spendable_quote_balance("DOGEUSDT") == 19.75


def test_ensure_stable_quote_balance_converts_shortfall(monkeypatch) -> None:
    balances = [
        {"asset": "FDUSD", "free": "20.0", "locked": "0"},
        {"asset": "USDT", "free": "100.0", "locked": "0"},
    ]
    monkeypatch.setattr(module.binance_wrapper, "get_account_balances", lambda: balances)

    calls: list[tuple[str, str, float]] = []
    monkeypatch.setattr(
        module,
        "build_stable_quote_conversion_plan",
        lambda *, from_asset, to_asset, amount, available_pairs: (
            calls.append((from_asset, to_asset, amount))
            or SimpleNamespace(symbol="FDUSDUSDT", side="BUY", quote_order_qty=amount)
        ),
    )

    executed: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        module,
        "execute_stable_quote_conversion",
        lambda plan, *, dry_run=False: executed.append((plan, dry_run)) or {"status": "ok"},
    )

    ok = module.ensure_stable_quote_balance("BTCFDUSD", 65.0, dry_run=False)

    assert ok is True
    assert calls == [("USDT", "FDUSD", 45.0)]
    assert len(executed) == 1
