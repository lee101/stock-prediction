"""Tests for tiered dip fallback in work-stealing strategy."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.strategy import WorkStealConfig, build_entry_candidates
from binance_worksteal.trade_live import _build_tiered_candidates


def _make_bars(symbol: str, prices: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [1e6] * len(prices),
        "symbol": symbol,
    })


def _flat_then_dip(symbol: str, base: float, dip_pct: float, n_flat: int = 25) -> pd.DataFrame:
    """Generate bars: n_flat days at base, then a dip day."""
    prices = [base] * n_flat + [base * (1 - dip_pct)]
    return _make_bars(symbol, prices)


BASE_CONFIG = WorkStealConfig(
    dip_pct=0.20,
    proximity_pct=0.02,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    max_positions=5,
    max_hold_days=14,
    lookback_days=20,
    sma_filter_period=0,
    trailing_stop_pct=0.0,
    maker_fee=0.001,
    fdusd_fee=0.0,
    initial_cash=10000.0,
)


class TestTieredDipDeepDips:
    """When dips are deep enough for tier 0, all entries come from tier 0."""

    def test_all_fill_at_tier_0(self):
        bars_a = _flat_then_dip("SYMA", 100.0, 0.22)
        bars_b = _flat_then_dip("SYMB", 50.0, 0.21)
        bars_c = _flat_then_dip("SYMC", 200.0, 0.25)
        all_bars = {"SYMA": bars_a, "SYMB": bars_b, "SYMC": bars_c}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
        )
        assert len(candidates) == 3
        for sym in ["SYMA", "SYMB", "SYMC"]:
            assert tier_map[sym] == 0, f"{sym} should be tier 0"

    def test_no_tier_1_2_when_tier_0_fills_all(self):
        syms = [f"SYM{i}" for i in range(5)]
        all_bars = {sym: _flat_then_dip(sym, 100.0 + i * 10, 0.22) for i, sym in enumerate(syms)}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
        )
        assert len(candidates) == 5
        assert all(t == 0 for t in tier_map.values())


class TestTieredDipShallowDips:
    """When only shallow dips exist, entries come from tier 1 or 2."""

    def test_shallow_dips_fill_at_later_tiers(self):
        # 13% dip with high=base*1.02 means actual dip from ref_high ~14.7%
        # tier 1 (15%) catches it with 2% proximity
        bars_a = _flat_then_dip("SYMA", 100.0, 0.13)
        # 11% dip -> actual ~12.7% from ref_high, caught by tier 2 (12%)
        bars_b = _flat_then_dip("SYMB", 50.0, 0.11)
        all_bars = {"SYMA": bars_a, "SYMB": bars_b}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
        )
        assert len(candidates) >= 1
        # SYMA: 13% dip caught by tier 1 (15% dip threshold)
        assert tier_map.get("SYMA") == 1
        # SYMB: 11% dip caught by tier 2 (12% dip threshold)
        assert tier_map.get("SYMB") == 2

    def test_mixed_dips_fill_from_multiple_tiers(self):
        # DEEP: 22% dip -> tier 0 (20%)
        bars_deep = _flat_then_dip("DEEP", 100.0, 0.22)
        # MID: 16% dip -> price below 15% target, tier 1
        bars_mid = _flat_then_dip("MID", 100.0, 0.16)
        # SHALLOW: 11% dip -> caught by tier 2 (12% dip threshold)
        bars_shallow = _flat_then_dip("SHALLOW", 100.0, 0.11)
        all_bars = {"DEEP": bars_deep, "MID": bars_mid, "SHALLOW": bars_shallow}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
        )
        assert len(candidates) == 3
        assert tier_map["DEEP"] == 0
        assert tier_map["MID"] == 1
        assert tier_map["SHALLOW"] == 2

    def test_no_candidates_when_no_dips(self):
        bars = _make_bars("FLAT", [100.0] * 26)
        all_bars = {"FLAT": bars}
        current_bars = {"FLAT": bars.iloc[-1]}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
        )
        assert len(candidates) == 0
        assert len(tier_map) == 0


class TestTieredDipMaxPositions:
    """max_positions is respected across all tiers."""

    def test_max_positions_respected(self):
        syms = [f"S{i}" for i in range(10)]
        all_bars = {sym: _flat_then_dip(sym, 100.0 + i * 5, 0.13) for i, sym in enumerate(syms)}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=3,
            pending_entries={},
        )
        assert len(candidates) <= 3

    def test_existing_positions_reduce_slots(self):
        bars_a = _flat_then_dip("SYMA", 100.0, 0.22)
        bars_b = _flat_then_dip("SYMB", 50.0, 0.22)
        bars_c = _flat_then_dip("SYMC", 200.0, 0.22)
        all_bars = {"SYMA": bars_a, "SYMB": bars_b, "SYMC": bars_c}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={"SYMA": {"entry_price": 80.0}},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=2,
            pending_entries={},
        )
        assert len(candidates) <= 1
        assert "SYMA" not in tier_map

    def test_pending_entries_reduce_slots(self):
        bars = {f"S{i}": _flat_then_dip(f"S{i}", 100.0, 0.22) for i in range(5)}
        current_bars = {sym: df.iloc[-1] for sym, df in bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=3,
            pending_entries={"S0": {"buy_price": 80.0}, "S1": {"buy_price": 80.0}},
        )
        assert len(candidates) <= 1


class TestBackwardsCompat:
    """Single dip_pct (no fallback) still works as before."""

    def test_single_dip_no_fallback(self):
        bars = _flat_then_dip("SYM", 100.0, 0.22)
        all_bars = {"SYM": bars}
        current_bars = {"SYM": bars.iloc[-1]}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates = build_entry_candidates(
            date=date,
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            config=BASE_CONFIG,
            base_symbol=None,
        )
        assert len(candidates) == 1
        assert candidates[0][0] == "SYM"

    def test_single_tier_list_same_as_no_fallback(self):
        bars = _flat_then_dip("SYM", 100.0, 0.22)
        all_bars = {"SYM": bars}
        current_bars = {"SYM": bars.iloc[-1]}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
        )
        assert len(candidates) == 1
        assert tier_map["SYM"] == 0


class TestTieredDipDiagnostics:
    """Diagnostics are collected from tier 0."""

    def test_diagnostics_populated(self):
        bars_a = _flat_then_dip("SYMA", 100.0, 0.22)
        bars_b = _flat_then_dip("SYMB", 50.0, 0.05)
        all_bars = {"SYMA": bars_a, "SYMB": bars_b}
        current_bars = {sym: df.iloc[-1] for sym, df in all_bars.items()}
        date = pd.Timestamp("2025-01-26", tz="UTC")

        diagnostics = []
        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=[0.20, 0.15, 0.12],
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={},
            date=date,
            entry_config=BASE_CONFIG,
            max_positions=5,
            pending_entries={},
            diagnostics=diagnostics,
        )
        assert len(diagnostics) > 0
        syms_in_diag = {d.symbol for d in diagnostics}
        assert "SYMA" in syms_in_diag or "SYMB" in syms_in_diag


class TestArgParser:
    """CLI arg --dip-pct-fallback parses correctly."""

    def test_fallback_arg_parsed(self):
        from binance_worksteal.trade_live import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--dip-pct-fallback", "0.20", "0.15", "0.12"])
        assert args.dip_pct_fallback == [0.20, 0.15, 0.12]

    def test_fallback_arg_default_none(self):
        from binance_worksteal.trade_live import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.dip_pct_fallback is None

    def test_single_fallback_value(self):
        from binance_worksteal.trade_live import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--dip-pct-fallback", "0.18"])
        assert args.dip_pct_fallback == [0.18]
