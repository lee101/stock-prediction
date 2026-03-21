"""Tests for worksteal diagnostic logging and adaptive dip features."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from dataclasses import replace

from binance_worksteal.strategy import (
    WorkStealConfig,
    SymbolDiagnostic,
    build_entry_candidates,
    compute_market_breadth_skip,
    passes_sma_filter,
    compute_ref_price,
    compute_buy_target,
    _risk_off_triggered,
)


def _make_bars(n=30, base_price=100.0, trend=0.0):
    dates = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    prices = base_price + np.arange(n) * trend
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.uniform(1e6, 5e6, n),
    })


def _make_current_bar(price):
    return pd.Series({
        "timestamp": pd.Timestamp("2026-02-01", tz="UTC"),
        "open": price * 0.99,
        "high": price * 1.01,
        "low": price * 0.98,
        "close": price,
        "volume": 2e6,
    })


class TestSymbolDiagnostic:
    def test_dataclass_defaults(self):
        d = SymbolDiagnostic(symbol="BTCUSD")
        assert d.symbol == "BTCUSD"
        assert d.close == 0.0
        assert d.filter_reason == ""
        assert d.is_candidate is False
        assert d.sma_pass is True

    def test_dataclass_with_values(self):
        d = SymbolDiagnostic(
            symbol="ETHUSD", close=2000.0, ref_high=2200.0,
            buy_target=1760.0, dist_pct=0.109, filter_reason="proximity"
        )
        assert d.close == 2000.0
        assert d.dist_pct == 0.109


class TestBuildEntryCandidatesWithDiagnostics:
    def test_diagnostics_None_default(self):
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=5)
        bars = _make_bars(10, base_price=100)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        candidates = build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=None,
        )
        assert isinstance(candidates, list)

    def test_diagnostics_populated(self):
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=5)
        bars = _make_bars(10, base_price=100)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=diags,
        )
        assert len(diags) == 1
        assert diags[0].symbol == "SYM1"
        assert diags[0].close > 0

    def test_already_held_diagnostic(self):
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=5)
        bars = _make_bars(10, base_price=100)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={"SYM1": {}},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=diags,
        )
        assert len(diags) == 1
        assert diags[0].filter_reason == "already_held"

    def test_cooldown_diagnostic(self):
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=5, reentry_cooldown_days=2)
        bars = _make_bars(10, base_price=100)
        now = pd.Timestamp("2026-02-01", tz="UTC")
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={"SYM1": now - pd.Timedelta(days=1)},
            date=now, config=config, diagnostics=diags,
        )
        assert any(d.filter_reason == "cooldown" for d in diags)

    def test_insufficient_history_diagnostic(self):
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=50)
        bars = _make_bars(10, base_price=100)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=diags,
        )
        assert any(d.filter_reason == "insufficient_history" for d in diags)

    def test_sma_filter_diagnostic(self):
        config = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.10, lookback_days=5,
            sma_filter_period=5, sma_check_method="current",
        )
        prices = [110, 108, 106, 104, 102, 100, 98, 96, 94, 92]
        dates = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
        bars = pd.DataFrame({
            "timestamp": dates, "open": prices, "high": [p+2 for p in prices],
            "low": [p-2 for p in prices], "close": prices, "volume": [1e6]*10,
        })
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=dates[-1], config=config, diagnostics=diags,
        )
        sma_diags = [d for d in diags if "sma_filter" in d.filter_reason]
        assert len(sma_diags) == 1
        assert not sma_diags[0].sma_pass

    def test_proximity_diagnostic(self):
        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        bars = _make_bars(10, base_price=100, trend=0.0)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=diags,
        )
        prox_diags = [d for d in diags if "proximity" in d.filter_reason]
        assert len(prox_diags) == 1
        assert prox_diags[0].dist_pct > config.proximity_pct
        assert prox_diags[0].ref_high > 0
        assert prox_diags[0].buy_target > 0

    def test_candidate_diagnostic(self):
        config = WorkStealConfig(dip_pct=0.05, proximity_pct=0.10, lookback_days=5)
        prices = [100, 102, 104, 106, 108, 110, 100, 98, 96, 95]
        dates = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
        bars = pd.DataFrame({
            "timestamp": dates, "open": prices, "high": [p+2 for p in prices],
            "low": [p-2 for p in prices], "close": prices, "volume": [1e6]*10,
        })
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        candidates = build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=dates[-1], config=config, diagnostics=diags,
        )
        cand_diags = [d for d in diags if d.is_candidate]
        assert len(cand_diags) >= 1
        assert cand_diags[0].filter_reason == ""

    def test_multiple_symbols_all_diagnosed(self):
        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        bars1 = _make_bars(10, base_price=100)
        bars2 = _make_bars(10, base_price=200)
        bars3 = _make_bars(10, base_price=50)
        current = {"A": bars1.iloc[-1], "B": bars2.iloc[-1], "C": bars3.iloc[-1]}
        history = {"A": bars1, "B": bars2, "C": bars3}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=diags,
        )
        diagnosed_syms = {d.symbol for d in diags}
        assert diagnosed_syms == {"A", "B", "C"}

    def test_momentum_diagnostic(self):
        config = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.10, lookback_days=5,
            momentum_period=5, momentum_min=-0.05,
        )
        prices = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55]
        dates = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
        bars = pd.DataFrame({
            "timestamp": dates, "open": prices, "high": [p+2 for p in prices],
            "low": [p-2 for p in prices], "close": prices, "volume": [1e6]*10,
        })
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        diags = []
        build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=dates[-1], config=config, diagnostics=diags,
        )
        mom_diags = [d for d in diags if "momentum" in d.filter_reason]
        assert len(mom_diags) == 1


class TestAdaptiveDip:
    def test_adaptive_dip_reduces_after_threshold(self):
        from binance_worksteal.trade_live import run_daily_cycle, load_state, save_state, STATE_FILE
        import tempfile
        import os

        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = {"positions": {}, "pending_entries": {}, "last_exit": {},
                     "recent_trades": [], "peak_equity": 0, "zero_candidate_cycles": 5}
            state_file.write_text(json.dumps(state))

            with patch("binance_worksteal.trade_live.STATE_FILE", state_file):
                with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={}):
                    run_daily_cycle(
                        None, ["BTCUSD"], config, dry_run=True,
                        min_dip_pct=0.10, adaptive_dip_cycles=3,
                    )
            result = json.loads(state_file.read_text())
            assert result.get("zero_candidate_cycles", 0) >= 5

    def test_zero_candidate_counter_increments(self):
        from binance_worksteal.trade_live import STATE_FILE
        import tempfile

        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        bars = _make_bars(10, base_price=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = {"positions": {}, "pending_entries": {}, "last_exit": {},
                     "recent_trades": [], "peak_equity": 0, "zero_candidate_cycles": 0}
            state_file.write_text(json.dumps(state))

            with patch("binance_worksteal.trade_live.STATE_FILE", state_file):
                with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars}):
                    with patch("binance_worksteal.trade_live.get_account_equity", return_value=10000):
                        from binance_worksteal.trade_live import run_daily_cycle
                        run_daily_cycle(
                            None, ["SYM1"], config, dry_run=True,
                            min_dip_pct=0.10, adaptive_dip_cycles=3,
                        )
            result = json.loads(state_file.read_text())
            assert result["zero_candidate_cycles"] == 1

    def test_zero_candidate_counter_resets_on_staged(self):
        from binance_worksteal.trade_live import STATE_FILE
        import tempfile

        # Use market_breadth_filter=0 and risk-off disabled to avoid blocking
        config = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.50, lookback_days=5,
            initial_cash=10000.0, max_position_pct=0.25,
            market_breadth_filter=0.0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
            risk_off_market_breadth_filter=0.0,
        )
        # Price drops from 110 to 95 -- a dip candidate with dip_pct=0.05
        prices = [100, 102, 104, 106, 108, 110, 100, 98, 96, 95]
        dates = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
        bars = pd.DataFrame({
            "timestamp": dates, "open": prices, "high": [p+2 for p in prices],
            "low": [p-2 for p in prices], "close": prices, "volume": [1e6]*10,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = {"positions": {}, "pending_entries": {}, "last_exit": {},
                     "recent_trades": [], "peak_equity": 0, "zero_candidate_cycles": 5}
            state_file.write_text(json.dumps(state))

            with patch("binance_worksteal.trade_live.STATE_FILE", state_file), \
                 patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars}), \
                 patch("binance_worksteal.trade_live.log_trade"), \
                 patch("binance_worksteal.trade_live.log_event"):
                from binance_worksteal.trade_live import run_daily_cycle
                run_daily_cycle(
                    None, ["SYM1"], config, dry_run=True,
                    min_dip_pct=0.05, adaptive_dip_cycles=3,
                )
            result = json.loads(state_file.read_text())
            assert result["zero_candidate_cycles"] == 0


    def test_zero_candidate_counter_skips_on_insufficient_data(self):
        """Fix 1: counter should NOT increment when data is insufficient (API outage)."""
        from binance_worksteal.trade_live import STATE_FILE
        import tempfile

        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = {"positions": {}, "pending_entries": {}, "last_exit": {},
                     "recent_trades": [], "peak_equity": 0, "zero_candidate_cycles": 2}
            state_file.write_text(json.dumps(state))

            # Return only 1 symbol's data out of 5 symbols (< 50% threshold)
            bars = _make_bars(10, base_price=100)
            with patch("binance_worksteal.trade_live.STATE_FILE", state_file):
                with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars}):
                    with patch("binance_worksteal.trade_live.get_account_equity", return_value=10000):
                        from binance_worksteal.trade_live import run_daily_cycle
                        run_daily_cycle(
                            None, ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"],
                            config, dry_run=True,
                            min_dip_pct=0.10, adaptive_dip_cycles=3,
                        )
            result = json.loads(state_file.read_text())
            assert result["zero_candidate_cycles"] == 2  # unchanged

    def test_zero_candidate_counter_resets_on_candidates_without_staging(self):
        """Fix 2: counter resets when candidates exist even if none were staged."""
        from binance_worksteal.trade_live import STATE_FILE
        import tempfile

        # max_positions=2 with 1 position + 1 pending = 0 slots, but scan still runs
        config = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.50, lookback_days=5,
            initial_cash=10000.0, max_position_pct=0.25, max_positions=2,
            market_breadth_filter=0.0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
            risk_off_market_breadth_filter=0.0,
        )
        prices = [100, 102, 104, 106, 108, 110, 100, 98, 96, 95]
        dates = pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")
        bars = pd.DataFrame({
            "timestamp": dates, "open": prices, "high": [p+2 for p in prices],
            "low": [p-2 for p in prices], "close": prices, "volume": [1e6]*10,
        })

        other_bars = _make_bars(10, base_price=50)
        existing_pos = {"OTHERSYMUSD": {"entry_price": 50, "target_sell": 60,
                                         "stop_price": 45, "trailing_stop_price": 45,
                                         "quantity": 1, "side": "long",
                                         "entry_time": "2026-01-01T00:00:00+00:00"}}
        pending = {"PENDING1USD": {"symbol": "PENDING1USD", "order_id": "123",
                                   "price": 100, "quantity": 1,
                                   "placed_at": "2026-01-01T00:00:00+00:00"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = {"positions": existing_pos, "pending_entries": pending,
                     "last_exit": {}, "recent_trades": [], "peak_equity": 0,
                     "zero_candidate_cycles": 5}
            state_file.write_text(json.dumps(state))

            with patch("binance_worksteal.trade_live.STATE_FILE", state_file), \
                 patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars, "OTHERSYMUSD": other_bars}), \
                 patch("binance_worksteal.trade_live.log_trade"), \
                 patch("binance_worksteal.trade_live.log_event"):
                from binance_worksteal.trade_live import run_daily_cycle
                run_daily_cycle(
                    None, ["SYM1"], config, dry_run=True,
                    min_dip_pct=0.05, adaptive_dip_cycles=3,
                )
            result = json.loads(state_file.read_text())
            assert result["zero_candidate_cycles"] == 0


class TestDiagnoseFunction:
    def test_diagnose_with_data(self, capsys):
        from binance_worksteal.trade_live import run_diagnose, STATE_FILE
        import tempfile

        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5,
                                 sma_filter_period=5, sma_check_method="current")
        bars = _make_bars(30, base_price=100)
        all_bars = {"SYM1": bars, "SYM2": _make_bars(30, base_price=200)}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = {"positions": {}, "pending_entries": {}, "last_exit": {},
                     "recent_trades": [], "peak_equity": 0}
            state_file.write_text(json.dumps(state))

            with patch("binance_worksteal.trade_live.STATE_FILE", state_file):
                with patch("binance_worksteal.trade_live._fetch_all_bars", return_value=all_bars):
                    run_diagnose(None, ["SYM1", "SYM2"], config)

        output = capsys.readouterr().out
        assert "WORKSTEAL DIAGNOSTIC" in output
        assert "Market breadth:" in output
        assert "SMA filter:" in output
        assert "FILTER SUMMARY" in output

    def test_diagnose_no_data(self, capsys):
        from binance_worksteal.trade_live import run_diagnose
        config = WorkStealConfig()
        with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={}):
            run_diagnose(None, ["BTCUSD"], config)

    def test_diagnose_initializes_client(self):
        """Fix 3: --diagnose should attempt to create a real Binance client."""
        from binance_worksteal.trade_live import main
        mock_client = object()
        with patch("binance_worksteal.trade_live.BinanceClient", return_value=mock_client) as mock_cls, \
             patch("binance_worksteal.trade_live.run_diagnose") as mock_diag, \
             patch("sys.argv", ["trade_live.py", "--diagnose"]), \
             patch.dict("sys.modules", {"env_real": type(sys)("env_real")}):
            sys.modules["env_real"].BINANCE_API_KEY = "test_key"
            sys.modules["env_real"].BINANCE_SECRET = "test_secret"
            result = main()
            assert result == 0
            mock_cls.assert_called_once_with("test_key", "test_secret")
            assert mock_diag.call_args[0][0] is mock_client


class TestBackwardCompatibility:
    def test_build_entry_candidates_without_diagnostics(self):
        """Ensure old callers that don't pass diagnostics still work."""
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=5)
        bars = _make_bars(10, base_price=100)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        candidates = build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config,
        )
        assert isinstance(candidates, list)

    def test_build_entry_candidates_explicit_None_diagnostics(self):
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, lookback_days=5)
        bars = _make_bars(10, base_price=100)
        current = {"SYM1": bars.iloc[-1]}
        history = {"SYM1": bars}
        candidates = build_entry_candidates(
            current_bars=current, history=history, positions={},
            last_exit={}, date=pd.Timestamp("2026-02-01", tz="UTC"),
            config=config, diagnostics=None,
        )
        assert isinstance(candidates, list)
