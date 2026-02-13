from __future__ import annotations

import sys
import types

import pytest

from src import fees
from stockagent.constants import CRYPTO_TRADING_FEE, TRADING_FEE


def _patch_asset_metadata(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_asset_record,
    get_trading_fee,
) -> None:
    fake_module = types.ModuleType("hftraining.asset_metadata")
    fake_module.get_asset_record = get_asset_record
    fake_module.get_trading_fee = get_trading_fee
    monkeypatch.setitem(sys.modules, "hftraining.asset_metadata", fake_module)


def test_get_fee_for_symbol_resolves_metadata_alias_for_compact_crypto_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_map = {"LINK-USD": {"asset_class": "crypto"}}

    def _get_asset_record(symbol: str):
        return record_map.get(symbol.upper(), {})

    def _get_trading_fee(symbol: str):
        if symbol.upper() == "LINK-USD":
            return 0.0015
        return 0.0005

    _patch_asset_metadata(
        monkeypatch,
        get_asset_record=_get_asset_record,
        get_trading_fee=_get_trading_fee,
    )
    assert fees.get_fee_for_symbol("LINKUSD") == pytest.approx(0.0015)


def test_get_fee_for_symbol_falls_back_when_metadata_record_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _get_asset_record(symbol: str):
        return {}

    def _get_trading_fee(symbol: str):
        return 0.0001

    _patch_asset_metadata(
        monkeypatch,
        get_asset_record=_get_asset_record,
        get_trading_fee=_get_trading_fee,
    )
    assert fees.get_fee_for_symbol("DOGEUSD") == pytest.approx(CRYPTO_TRADING_FEE)
    assert fees.get_fee_for_symbol("AAPL") == pytest.approx(TRADING_FEE)

