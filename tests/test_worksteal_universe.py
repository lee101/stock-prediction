"""Tests for binance_worksteal.universe module."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from binance_worksteal.universe import (
    SymbolInfo,
    load_universe,
    get_symbols,
    get_fee,
    validate_universe,
)


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "test_universe.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


@pytest.fixture
def minimal_universe(tmp_path):
    data = {
        "symbols": [
            {"symbol": "BTCUSD", "usdt_pair": "BTCUSDT", "fee_tier": "fdusd",
             "margin_eligible": True, "has_lora": False, "min_notional": 10.0},
            {"symbol": "DOGEUSD", "usdt_pair": "DOGEUSDT", "fee_tier": "usdt",
             "margin_eligible": True, "has_lora": True, "min_notional": 10.0},
        ]
    }
    return _write_yaml(tmp_path, data)


def test_load_universe_basic(minimal_universe):
    uni = load_universe(minimal_universe)
    assert len(uni) == 2
    assert isinstance(uni[0], SymbolInfo)
    assert uni[0].symbol == "BTCUSD"
    assert uni[0].fee_tier == "fdusd"
    assert uni[1].symbol == "DOGEUSD"
    assert uni[1].has_lora is True


def test_get_symbols(minimal_universe):
    uni = load_universe(minimal_universe)
    syms = get_symbols(uni)
    assert syms == ["BTCUSD", "DOGEUSD"]


def test_get_fee_fdusd(minimal_universe):
    uni = load_universe(minimal_universe)
    assert get_fee("BTCUSD", uni) == 0.0


def test_get_fee_usdt(minimal_universe):
    uni = load_universe(minimal_universe)
    assert get_fee("DOGEUSD", uni) == 0.001


def test_get_fee_unknown(minimal_universe):
    uni = load_universe(minimal_universe)
    assert get_fee("UNKNOWN", uni) == 0.001


def test_duplicate_symbol_raises(tmp_path):
    data = {
        "symbols": [
            {"symbol": "BTCUSD"},
            {"symbol": "BTCUSD"},
        ]
    }
    p = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="Duplicate"):
        load_universe(p)


def test_missing_symbol_field_raises(tmp_path):
    data = {"symbols": [{"fee_tier": "usdt"}]}
    p = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="missing 'symbol'"):
        load_universe(p)


def test_invalid_fee_tier_raises(tmp_path):
    data = {"symbols": [{"symbol": "BTCUSD", "fee_tier": "invalid"}]}
    p = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="Invalid fee_tier"):
        load_universe(p)


def test_missing_symbols_key_raises(tmp_path):
    data = {"coins": [{"symbol": "BTCUSD"}]}
    p = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="'symbols' key"):
        load_universe(p)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_universe("/nonexistent/path.yaml")


def test_defaults_applied(tmp_path):
    data = {"symbols": [{"symbol": "ETHUSD"}]}
    p = _write_yaml(tmp_path, data)
    uni = load_universe(p)
    assert uni[0].usdt_pair == "ETHUSDT"
    assert uni[0].fee_tier == "usdt"
    assert uni[0].margin_eligible is True
    assert uni[0].has_lora is False
    assert uni[0].min_notional == 10.0


def test_validate_empty():
    warns = validate_universe([])
    assert any("empty" in w.lower() for w in warns)


def test_validate_missing_data(tmp_path):
    uni = [SymbolInfo("FAKEUSD", "FAKEUSDT", "usdt", True, False, 10.0)]
    warns = validate_universe(uni, data_dir=str(tmp_path))
    assert any("FAKEUSD" in w for w in warns)


def test_load_v2_yaml():
    p = Path(__file__).resolve().parents[1] / "binance_worksteal" / "universe_v2.yaml"
    if not p.exists():
        pytest.skip("universe_v2.yaml not found")
    uni = load_universe(p)
    syms = get_symbols(uni)
    assert len(syms) >= 50
    # FDUSD symbols present
    for s in ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"]:
        assert s in syms
        assert get_fee(s, uni) == 0.0
    # LoRA symbols
    lora_expected = {"XRPUSD", "DOTUSD", "AVAXUSD", "LINKUSD", "DOGEUSD", "AAVEUSD", "LTCUSD", "SOLUSD"}
    lora_actual = {s.symbol for s in uni if s.has_lora}
    assert lora_expected == lora_actual
    # No duplicates
    assert len(syms) == len(set(syms))


def test_case_insensitive_symbol(tmp_path):
    data = {"symbols": [{"symbol": "btcusd"}]}
    p = _write_yaml(tmp_path, data)
    uni = load_universe(p)
    assert uni[0].symbol == "BTCUSD"


def test_case_insensitive_fee_tier(tmp_path):
    data = {"symbols": [{"symbol": "BTCUSD", "fee_tier": "FDUSD"}]}
    p = _write_yaml(tmp_path, data)
    uni = load_universe(p)
    assert uni[0].fee_tier == "fdusd"
    assert get_fee("BTCUSD", uni) == 0.0
