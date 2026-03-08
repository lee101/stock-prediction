from __future__ import annotations

from src.trade_directions import (
    DEFAULT_ALPACA_LIVE8_STOCKS,
    is_long_only_symbol,
    is_short_only_symbol,
    resolve_trade_directions,
    trade_direction_name,
)


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

    dbx_short_only = resolve_trade_directions("DBX", allow_short=True)
    assert dbx_short_only.can_long is False
    assert dbx_short_only.can_short is True

    yelp_short_only = resolve_trade_directions("YELP", allow_short=True)
    assert yelp_short_only.can_long is False
    assert yelp_short_only.can_short is True

    long_only = resolve_trade_directions("NVDA", allow_short=True)
    assert long_only.can_long is True
    assert long_only.can_short is False

    pltr_long_only = resolve_trade_directions("PLTR", allow_short=True)
    assert pltr_long_only.can_long is True
    assert pltr_long_only.can_short is False

    tsla_long_only = resolve_trade_directions("TSLA", allow_short=True)
    assert tsla_long_only.can_long is True
    assert tsla_long_only.can_short is False


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


def test_live8_universe_keeps_tsla_and_excludes_yelp() -> None:
    assert DEFAULT_ALPACA_LIVE8_STOCKS == (
        "NVDA",
        "PLTR",
        "GOOG",
        "TSLA",
        "DBX",
        "TRIP",
        "MTCH",
        "NYT",
    )
    assert "YELP" not in DEFAULT_ALPACA_LIVE8_STOCKS


def test_direction_helpers_match_resolver() -> None:
    assert is_long_only_symbol("TSLA") is True
    assert is_short_only_symbol("TSLA") is False
    assert trade_direction_name("TSLA") == "long"

    assert is_short_only_symbol("YELP") is True
    assert is_long_only_symbol("YELP") is False
    assert trade_direction_name("YELP") == "short"

    assert trade_direction_name("XYZ") == "both"
