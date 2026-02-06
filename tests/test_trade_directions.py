from __future__ import annotations

from src.trade_directions import resolve_trade_directions


def test_crypto_is_always_long_only() -> None:
    dirs = resolve_trade_directions("BTCUSD", allow_short=True)
    assert dirs.can_long is True
    assert dirs.can_short is False


def test_allow_short_false_disables_shorting_even_for_stocks() -> None:
    dirs = resolve_trade_directions("EBAY", allow_short=False)
    assert dirs.can_long is True
    assert dirs.can_short is False


def test_default_groups_apply_when_allow_short_true() -> None:
    short_only = resolve_trade_directions("EBAY", allow_short=True)
    assert short_only.can_long is False
    assert short_only.can_short is True

    long_only = resolve_trade_directions("NVDA", allow_short=True)
    assert long_only.can_long is True
    assert long_only.can_short is False


def test_conflicting_overrides_disable_trading() -> None:
    dirs = resolve_trade_directions(
        "XYZ",
        allow_short=True,
        long_only_symbols=["XYZ"],
        short_only_symbols=["XYZ"],
        use_default_groups=False,
    )
    assert dirs.can_long is False
    assert dirs.can_short is False

