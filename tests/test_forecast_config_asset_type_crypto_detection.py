from __future__ import annotations

from src.preaug.forecast_config import ForecastTag


def test_forecast_tag_classifies_u_suffix_stock_as_stock() -> None:
    tag = ForecastTag.from_symbols_and_targets(["MU"], targets=("close",))
    assert tag.asset_type == "stock"


def test_forecast_tag_classifies_mixed_stock_and_btc_u_as_mixed() -> None:
    tag = ForecastTag.from_symbols_and_targets(["AAPL", "BTCU"], targets=("close",))
    assert tag.asset_type == "mixed"


def test_forecast_tag_classifies_u_quote_crypto_only_list_as_crypto() -> None:
    tag = ForecastTag.from_symbols_and_targets(["BTCU", "ETHU"], targets=("close",))
    assert tag.asset_type == "crypto"
