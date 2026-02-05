from __future__ import annotations

from pathlib import Path

import pytest

from newnanoalpacahourlyexp.trade_alpaca_hourly import (
    _allocation_usd,
    _ensure_valid_levels,
    _parse_checkpoint_map,
    _parse_horizon_map,
    _parse_symbols,
)


def test_parse_symbols_default():
    assert _parse_symbols(None) == ["SOLUSD", "LINKUSD", "UNIUSD"]


def test_parse_symbols_strip():
    parsed = _parse_symbols(" solusd, linkusd ,, UNIUSD ")
    assert parsed == ["SOLUSD", "LINKUSD", "UNIUSD"]


def test_parse_checkpoint_map(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_text("x")
    mapping = _parse_checkpoint_map(f"SOLUSD={model_path}")
    assert mapping["SOLUSD"] == model_path.resolve()


def test_parse_checkpoint_map_invalid():
    with pytest.raises(ValueError):
        _parse_checkpoint_map("SOLUSD")


def test_parse_horizon_map():
    mapping = _parse_horizon_map("SOLUSD=1,24;UNIUSD=1")
    assert mapping["SOLUSD"] == (1, 24)
    assert mapping["UNIUSD"] == (1,)


def test_ensure_valid_levels_rejects_nonpositive():
    assert _ensure_valid_levels(-1.0, 2.0, min_gap_pct=0.01) is None


def test_ensure_valid_levels_enforces_gap():
    buy, sell = _ensure_valid_levels(100.0, 99.0, min_gap_pct=0.01)
    assert sell > buy


def test_allocation_usd_prefers_fixed():
    class Account:
        buying_power = 1000.0
        equity = 500.0

    assert _allocation_usd(Account(), allocation_usd=123.0, allocation_pct=0.5) == 123.0


def test_allocation_usd_pct_uses_buying_power():
    class Account:
        buying_power = 1000.0
        equity = 500.0

    assert _allocation_usd(Account(), allocation_usd=None, allocation_pct=0.1) == 100.0


def test_allocation_usd_pct_falls_back_to_equity():
    class Account:
        buying_power = 0.0
        equity = 400.0

    assert _allocation_usd(Account(), allocation_usd=None, allocation_pct=0.25) == 100.0
