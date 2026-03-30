"""Tests for worksteal diagnostic logging and adaptive dip features."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch

from binance_worksteal.strategy import WorkStealConfig, SymbolDiagnostic, build_entry_candidates


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
        build_entry_candidates(
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
        from binance_worksteal.trade_live import run_daily_cycle
        import tempfile

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
        from binance_worksteal.trade_live import run_diagnose
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
                    rc = run_diagnose(None, ["SYM1", "SYM2"], config)

        output = capsys.readouterr().out
        assert rc == 0
        assert "WORKSTEAL DIAGNOSTIC" in output
        assert "Market breadth:" in output
        assert "SMA filter:" in output
        assert "FILTER SUMMARY" in output

    def test_diagnose_no_data(self, capsys):
        from binance_worksteal.trade_live import run_diagnose
        config = WorkStealConfig()
        with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={}):
            rc = run_diagnose(None, ["BTCUSD"], config)
        output = capsys.readouterr().out
        assert rc == 1
        assert "Symbols with data:  0/1" in output
        assert "Data coverage:     0/1 loaded (0.0%)" in output
        assert "Missing symbols:   BTCUSD" in output
        assert "Action summary:    no actionable signal; no market data loaded" in output
        assert "Warnings:" in output
        assert "  - missing data for 1 symbol: BTCUSD" in output
        assert "ERROR: No bar data fetched for any symbol" in output

    def test_collect_diagnose_result_passes_symbol_metrics_to_strategy_calls(self):
        from types import SimpleNamespace

        from binance_worksteal.trade_live import _collect_diagnose_result

        config = WorkStealConfig()
        bars = _make_bars(30, base_price=100)
        all_bars = {"SYM1": bars}
        symbol_metrics = {"SYM1": object()}
        calls: dict[str, object] = {}

        def fake_resolve_entry_regime(*, symbol_metrics=None, **kwargs):
            calls["resolve"] = symbol_metrics
            return SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False)

        def fake_count_sma_pass_fail(*args, symbol_metrics=None, **kwargs):
            calls["sma"] = symbol_metrics
            return (1, 0)

        def fake_build_entry_candidates(*args, symbol_metrics=None, diagnostics=None, **kwargs):
            calls["build"] = symbol_metrics
            if diagnostics is not None:
                diagnostics.append(SymbolDiagnostic(symbol="SYM1", is_candidate=True))
            return []

        with patch("binance_worksteal.trade_live._fetch_all_bars", return_value=all_bars),              patch("binance_worksteal.trade_live._build_symbol_metric_cache", return_value=symbol_metrics) as mock_cache,              patch("binance_worksteal.trade_live.resolve_entry_regime", side_effect=fake_resolve_entry_regime),              patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 1)),              patch("binance_worksteal.trade_live._count_sma_pass_fail", side_effect=fake_count_sma_pass_fail),              patch("binance_worksteal.trade_live.load_state", return_value={"positions": {}, "pending_entries": {}, "last_exit": {}}),              patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            result = _collect_diagnose_result(None, ["SYM1"], config)

        assert result is not None
        mock_cache.assert_called_once()
        assert calls == {"resolve": symbol_metrics, "sma": symbol_metrics, "build": symbol_metrics}

    def test_collect_diagnose_result_skips_invalid_candidate_and_proximity_rows(self):
        from types import SimpleNamespace

        from binance_worksteal.trade_live import _collect_diagnose_result

        config = WorkStealConfig()
        bars = _make_bars(30, base_price=100)
        bad_bar = bars.iloc[-1].copy()
        bad_bar["close"] = np.nan

        def fake_build_entry_candidates(*args, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.extend(
                    [
                        SymbolDiagnostic(
                            symbol="BAD",
                            close=float("nan"),
                            ref_high=120.0,
                            buy_target=95.0,
                            dist_pct=0.01,
                            is_candidate=True,
                        ),
                        SymbolDiagnostic(
                            symbol="BLOCKED_BAD",
                            close=float("nan"),
                            ref_high=120.0,
                            buy_target=95.0,
                            dist_pct=0.02,
                            filter_reason="sma filter blocked",
                        ),
                        SymbolDiagnostic(
                            symbol="BLOCKED_GOOD",
                            close=100.0,
                            ref_high=120.0,
                            buy_target=95.0,
                            dist_pct=0.03,
                            filter_reason="breadth filter blocked",
                        ),
                    ]
                )
            return [
                ("BAD", "long", 0.4, 95.0, bad_bar),
                ("GOOD", "long", 0.8, 96.0, bars.iloc[-1]),
            ]

        with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"GOOD": bars}),              patch(
                 "binance_worksteal.trade_live.resolve_entry_regime",
                 return_value=SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False),
             ),              patch("binance_worksteal.trade_live._build_symbol_metric_cache", return_value={}),              patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 1)),              patch("binance_worksteal.trade_live._count_sma_pass_fail", return_value=(1, 0)),              patch("binance_worksteal.trade_live.load_state", return_value={"positions": {}, "pending_entries": {}, "last_exit": {}}),              patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            result = _collect_diagnose_result(None, ["GOOD"], config)

        assert result is not None
        assert result["candidate_count"] == 1
        assert result["candidate_symbols"] == ["GOOD"]
        assert result["top_candidate"] == {
            "symbol": "GOOD",
            "direction": "long",
            "score": 0.8,
            "fill_price": 96.0,
            "close": float(bars.iloc[-1]["close"]),
        }
        assert result["nearest_blocked"] == {
            "symbol": "BLOCKED_GOOD",
            "close": 100.0,
            "ref_high": 120.0,
            "buy_target": 95.0,
            "dist_pct": 0.03,
            "dip_needed_pct": 3.0,
            "filter_reason": "breadth filter blocked",
            "is_candidate": False,
        }
        assert result["watchlist"] == [result["nearest_blocked"]]
        assert result["closest_to_entry"] == [result["nearest_blocked"]]
        assert result["min_entry_distance_pct"] == pytest.approx(0.03)
        assert result["action_summary"] == {
            "status": "candidate_available",
            "symbol": "GOOD",
            "reason": None,
            "fill_price": 96.0,
            "target_price": 96.0,
            "current_price": float(bars.iloc[-1]["close"]),
            "distance_pct": 0.0,
            "summary": "stage GOOD near $96.0000",
        }

    def test_collect_diagnose_result_returns_empty_valid_rows_when_all_candidate_and_proximity_rows_are_invalid(self):
        from types import SimpleNamespace

        from binance_worksteal.trade_live import _collect_diagnose_result

        config = WorkStealConfig()
        bars = _make_bars(30, base_price=100)
        bad_bar = bars.iloc[-1].copy()
        bad_bar["close"] = np.nan

        def fake_build_entry_candidates(*args, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.extend(
                    [
                        SymbolDiagnostic(
                            symbol="BAD_CANDIDATE",
                            close=float("nan"),
                            ref_high=120.0,
                            buy_target=95.0,
                            dist_pct=float("nan"),
                            is_candidate=True,
                        ),
                        SymbolDiagnostic(
                            symbol="BAD_BLOCKED",
                            close=float("nan"),
                            ref_high=120.0,
                            buy_target=95.0,
                            dist_pct=float("nan"),
                            filter_reason="sma filter blocked",
                        ),
                    ]
                )
            return [("BAD_CANDIDATE", "long", 0.4, 95.0, bad_bar)]

        with patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"GOOD": bars}),              patch(
                 "binance_worksteal.trade_live.resolve_entry_regime",
                 return_value=SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False),
             ),              patch("binance_worksteal.trade_live._build_symbol_metric_cache", return_value={}),              patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 1)),              patch("binance_worksteal.trade_live._count_sma_pass_fail", return_value=(1, 0)),              patch("binance_worksteal.trade_live.load_state", return_value={"positions": {}, "pending_entries": {}, "last_exit": {}}),              patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            result = _collect_diagnose_result(None, ["GOOD"], config)

        assert result is not None
        assert result["candidate_count"] == 0
        assert result["candidate_symbols"] == []
        assert result["top_candidate"] is None
        assert result["nearest_blocked"] is None
        assert result["watchlist"] == []
        assert result["closest_to_entry"] == []
        assert result["min_entry_distance_pct"] is None
        assert result["action_summary"] == {
            "status": "no_actionable_signal",
            "symbol": None,
            "reason": None,
            "fill_price": None,
            "target_price": None,
            "current_price": None,
            "distance_pct": None,
            "summary": "no actionable signal from current data",
        }

    def test_build_diagnose_action_summary_covers_blocked_wait_paths(self):
        from binance_worksteal.trade_live import _build_diagnose_action_summary

        wait_breadth = _build_diagnose_action_summary(
            top_candidate=None,
            nearest_blocked={
                "symbol": "SYM1",
                "close": 100.0,
                "buy_target": 95.0,
                "filter_reason": "breadth filter blocked",
                "dist_pct": 0.04,
            },
            min_dist=0.04,
            breadth_skip=True,
            risk_off=False,
        )
        wait_entry = _build_diagnose_action_summary(
            top_candidate=None,
            nearest_blocked={
                "symbol": "SYM2",
                "close": 200.0,
                "buy_target": 180.0,
                "filter_reason": "sma filter blocked",
                "dist_pct": 0.03,
            },
            min_dist=0.03,
            breadth_skip=False,
            risk_off=False,
        )
        distance_only = _build_diagnose_action_summary(
            top_candidate=None,
            nearest_blocked=None,
            min_dist=0.02,
            breadth_skip=False,
            risk_off=False,
        )

        assert wait_breadth == {
            "status": "wait_breadth",
            "symbol": "SYM1",
            "reason": "breadth filter blocked",
            "fill_price": None,
            "target_price": 95.0,
            "current_price": 100.0,
            "distance_pct": 0.04,
            "summary": "wait for breadth recovery; SYM1 near $95.0000 blocked by breadth filter blocked",
        }
        assert wait_entry == {
            "status": "wait_for_entry",
            "symbol": "SYM2",
            "reason": "sma filter blocked",
            "fill_price": None,
            "target_price": 180.0,
            "current_price": 200.0,
            "distance_pct": 0.03,
            "summary": "watch SYM2 near $180.0000; needs 3.0% more dip (sma filter blocked)",
        }
        assert distance_only == {
            "status": "wait_for_entry",
            "symbol": None,
            "reason": None,
            "fill_price": None,
            "target_price": None,
            "current_price": None,
            "distance_pct": 0.02,
            "summary": "wait for setup; nearest valid entry is 2.0% away",
        }

    def test_print_diagnose_coverage_context_formats_symbol_previews_and_command(self, capsys):
        from binance_worksteal.trade_live import _print_diagnose_coverage_context

        loaded_symbols = [f"SYM{i:02d}" for i in range(12)]
        missing_symbols = [f"MISS{i:02d}" for i in range(11)]

        _print_diagnose_coverage_context(
            loaded_symbol_count=12,
            total_symbols=23,
            data_coverage_ratio=12 / 23,
            loaded_symbols=loaded_symbols,
            missing_symbols=missing_symbols,
            command_preview="python -m binance_worksteal.trade_live --diagnose",
        )

        output = capsys.readouterr().out
        assert "Command:" in output
        assert "  python -m binance_worksteal.trade_live --diagnose" in output
        assert "Symbols with data:  12/23" in output
        assert "Data coverage:     12/23 loaded (52.2%)" in output
        assert "Loaded symbols:    SYM00, SYM01, SYM02, SYM03, SYM04, SYM05, SYM06, SYM07, SYM08, SYM09 (+2 more)" in output
        assert "Missing symbols:   MISS00, MISS01, MISS02, MISS03, MISS04, MISS05, MISS06, MISS07, MISS08, MISS09 (+1 more)" in output

    def test_summarize_validated_proximity_rows_limits_and_selects_nearest_blocked(self):
        from binance_worksteal.trade_live import _summarize_validated_proximity_rows

        rows = [
            {"symbol": "R7", "dist_pct": 0.07, "is_candidate": False, "filter_reason": "late"},
            {"symbol": "R3", "dist_pct": 0.03, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R11", "dist_pct": 0.11, "is_candidate": False, "filter_reason": "later"},
            {"symbol": "R1", "dist_pct": 0.01, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R6", "dist_pct": 0.06, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R2", "dist_pct": 0.02, "is_candidate": False, "filter_reason": "blocked early"},
            {"symbol": "R10", "dist_pct": 0.10, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R4", "dist_pct": 0.04, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R8", "dist_pct": 0.08, "is_candidate": False, "filter_reason": "late blocked"},
            {"symbol": "R5", "dist_pct": 0.05, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R9", "dist_pct": 0.09, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R12", "dist_pct": 0.12, "is_candidate": False, "filter_reason": "latest"},
        ]

        closest_rows, nearest_blocked, min_dist = _summarize_validated_proximity_rows(rows, limit=10)

        assert [row["symbol"] for row in closest_rows] == [
            "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"
        ]
        assert nearest_blocked == rows[5]
        assert min_dist == pytest.approx(0.01)

    def test_diagnose_watchlist_rows_limits_blocked_symbols(self):
        from binance_worksteal.trade_live import _diagnose_watchlist_rows

        rows = [
            {"symbol": "R7", "dist_pct": 0.07, "is_candidate": False, "filter_reason": "late"},
            {"symbol": "R3", "dist_pct": 0.03, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R11", "dist_pct": 0.11, "is_candidate": False, "filter_reason": "later"},
            {"symbol": "R1", "dist_pct": 0.01, "is_candidate": True, "filter_reason": ""},
            {"symbol": "R2", "dist_pct": 0.02, "is_candidate": False, "filter_reason": "blocked early"},
            {"symbol": "R8", "dist_pct": 0.08, "is_candidate": False, "filter_reason": "late blocked"},
        ]

        watchlist = _diagnose_watchlist_rows(rows, limit=2)

        assert [row["symbol"] for row in watchlist] == ["R2", "R7"]

    def test_main_diagnose_summary_dash_prints_json_to_stdout(self, tmp_path, capsys):
        from types import SimpleNamespace

        from binance_worksteal.reporting import render_command_preview
        from binance_worksteal.trade_live import main

        bars1 = _make_bars(30, base_price=100)
        bars2 = _make_bars(30, base_price=200)
        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({
                "positions": {},
                "pending_entries": {},
                "last_exit": {},
                "recent_trades": [],
                "peak_equity": 0,
            }),
            encoding="utf-8",
        )

        candidate_bar = bars1.iloc[-1]

        def fake_build_entry_candidates(*args, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.extend(
                    [
                        SymbolDiagnostic(
                            symbol="SYM1",
                            close=100.0,
                            ref_high=120.0,
                            buy_target=95.0,
                            dist_pct=0.01,
                            is_candidate=True,
                        ),
                        SymbolDiagnostic(
                            symbol="SYM2",
                            close=200.0,
                            ref_high=220.0,
                            buy_target=180.0,
                            dist_pct=0.05,
                            filter_reason="sma filter blocked",
                            sma_pass=False,
                            sma_value=210.0,
                        ),
                    ]
                )
            return [("SYM1", "long", 0.5, 95.0, candidate_bar)]

        argv = ["--symbols", "SYM1", "SYM2", "--diagnose", "--summary-json", "-"]
        expected_command = render_command_preview("binance_worksteal.trade_live", argv)

        with patch("binance_worksteal.trade_live.STATE_FILE", state_file),              patch("binance_worksteal.trade_live.BinanceClient", None),              patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars1, "SYM2": bars2}),              patch(
                 "binance_worksteal.trade_live.resolve_entry_regime",
                 return_value=SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False),
             ),              patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 2)),              patch("binance_worksteal.trade_live._count_sma_pass_fail", return_value=(1, 1)),              patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert rc == 0
        assert payload["tool"] == "trade_live"
        assert payload["diagnose"] is True
        assert payload["inspection_mode"] == "diagnose"
        assert payload["symbol_source"] == "command line --symbols"
        assert payload["requested_symbol_count"] == 2
        assert payload["symbol_count"] == 2
        assert payload["symbols"] == ["SYM1", "SYM2"]
        assert payload["loaded_symbol_count"] == 2
        assert payload["missing_symbol_count"] == 0
        assert payload["universe_complete"] is True
        assert payload["data_coverage_ratio"] == pytest.approx(1.0)
        assert payload["warnings"] == []
        assert payload["candidate_count"] == 1
        assert payload["candidate_symbols"] == ["SYM1"]
        assert payload["top_candidate"] == {
            "symbol": "SYM1",
            "direction": "long",
            "score": 0.5,
            "fill_price": 95.0,
            "close": float(candidate_bar["close"]),
        }
        assert payload["nearest_blocked"] == {
            "symbol": "SYM2",
            "close": 200.0,
            "ref_high": 220.0,
            "buy_target": 180.0,
            "dist_pct": 0.05,
            "dip_needed_pct": 5.0,
            "filter_reason": "sma filter blocked",
            "is_candidate": False,
        }
        assert payload["watchlist"] == [payload["nearest_blocked"]]
        assert payload["action_summary"] == {
            "status": "candidate_available",
            "symbol": "SYM1",
            "reason": None,
            "fill_price": 95.0,
            "target_price": 95.0,
            "current_price": float(candidate_bar["close"]),
            "distance_pct": 0.0,
            "summary": "stage SYM1 near $95.0000",
        }
        assert payload["candidates"] == [
            {
                "symbol": "SYM1",
                "direction": "long",
                "score": 0.5,
                "fill_price": 95.0,
                "close": float(candidate_bar["close"]),
            }
        ]
        assert payload["filter_summary"] == [{"reason": "sma filter blocked", "count": 1}]
        assert payload["command"] == expected_command
        assert payload["status"] == "success"
        assert payload["exit_code"] == 0
        assert payload["invocation"]["argv"][-2:] == ["--summary-json", "-"]
        assert "WORKSTEAL DIAGNOSTIC" in captured.err
        assert "Data coverage:     2/2 loaded (100.0%)" in captured.err
        assert "Loaded symbols:    SYM1, SYM2" in captured.err
        assert (
            f"Top candidate:     SYM1 score=0.5000 current=${float(candidate_bar['close']):.4f} fill=$95.0000"
            in captured.err
        )
        assert (
            "Nearest blocked:   SYM2 current=$200.0000 target=$180.0000 dist=0.0500 reason=sma filter blocked"
            in captured.err
        )
        assert "Action summary:    stage SYM1 near $95.0000" in captured.err
        assert "Watchlist:         SYM2 near $180.0000 (5.0%, sma filter blocked)" in captured.err
        assert "Command:" in captured.err
        assert f"  {expected_command}" in captured.err

    def test_main_diagnose_summary_dash_surfaces_partial_data_coverage(self, tmp_path, capsys):
        from types import SimpleNamespace

        from binance_worksteal.trade_live import main

        bars1 = _make_bars(30, base_price=100)
        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({
                "positions": {},
                "pending_entries": {},
                "last_exit": {},
                "recent_trades": [],
                "peak_equity": 0,
            }),
            encoding="utf-8",
        )
        candidate_bar = bars1.iloc[-1]

        def fake_build_entry_candidates(*args, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.extend([
                    SymbolDiagnostic(
                        symbol="SYM1",
                        close=100.0,
                        ref_high=120.0,
                        buy_target=95.0,
                        dist_pct=0.01,
                        is_candidate=True,
                    )
                ])
            return [("SYM1", "long", 0.5, 95.0, candidate_bar)]

        argv = ["--symbols", "SYM1", "SYM2", "--diagnose", "--summary-json", "-"]

        with patch("binance_worksteal.trade_live.STATE_FILE", state_file), \
             patch("binance_worksteal.trade_live.BinanceClient", None), \
             patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars1}), \
             patch(
                 "binance_worksteal.trade_live.resolve_entry_regime",
                 return_value=SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False),
             ), \
             patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 1)), \
             patch("binance_worksteal.trade_live._count_sma_pass_fail", return_value=(1, 0)), \
             patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert rc == 0
        assert payload["loaded_symbol_count"] == 1
        assert payload["missing_symbol_count"] == 1
        assert payload["missing_symbols"] == ["SYM2"]
        assert payload["universe_complete"] is False
        assert payload["data_coverage_ratio"] == pytest.approx(0.5)
        assert payload["warnings"] == ["missing data for 1 symbol: SYM2"]
        assert "Data coverage:     1/2 loaded (50.0%)" in captured.err
        assert "Loaded symbols:    SYM1" in captured.err
        assert "Missing symbols:   SYM2" in captured.err
        assert "Warnings:" in captured.err
        assert "  - missing data for 1 symbol: SYM2" in captured.err

    def test_main_diagnose_summary_dash_surfaces_omitted_symbols_and_pair_routing(self, tmp_path, capsys):
        from types import SimpleNamespace

        from binance_worksteal.reporting import render_command_preview
        from binance_worksteal.trade_live import main

        bars1 = _make_bars(30, base_price=100)
        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({
                "positions": {},
                "pending_entries": {},
                "last_exit": {},
                "recent_trades": [],
                "peak_equity": 0,
            }),
            encoding="utf-8",
        )

        candidate_bar = bars1.iloc[-1]

        def fake_build_entry_candidates(*args, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.append(
                    SymbolDiagnostic(
                        symbol="BTCUSD",
                        close=100.0,
                        ref_high=120.0,
                        buy_target=95.0,
                        dist_pct=0.01,
                        is_candidate=True,
                    )
                )
            return [("BTCUSD", "long", 0.5, 95.0, candidate_bar)]

        argv = ["--symbols", "BTCUSD", "ETHUSD", "--max-symbols", "1", "--diagnose", "--summary-json", "-"]
        expected_command = render_command_preview("binance_worksteal.trade_live", argv)

        with patch("binance_worksteal.trade_live.STATE_FILE", state_file), \
             patch("binance_worksteal.trade_live.BinanceClient", None), \
             patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"BTCUSD": bars1}), \
             patch(
                 "binance_worksteal.trade_live.resolve_entry_regime",
                 return_value=SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False),
             ), \
             patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 2)), \
             patch("binance_worksteal.trade_live._count_sma_pass_fail", return_value=(1, 0)), \
             patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert rc == 0
        assert payload["inspection_mode"] == "diagnose"
        assert payload["requested_symbol_count"] == 2
        assert payload["symbol_count"] == 1
        assert payload["omitted_symbol_count"] == 1
        assert payload["omitted_symbols"] == ["ETHUSD"]
        assert payload["symbols"] == ["BTCUSD"]
        assert payload["pair_routing"] == [
            {"symbol": "BTCUSD", "data_pair": "BTCUSDT", "order_pair": "BTCFDUSD"},
        ]
        assert payload["command"] == expected_command
        assert "Excluded by --max-symbols:" in captured.err
        assert "  ETHUSD" in captured.err
        assert "Pair routing:" in captured.err
        assert "  BTCUSD: data=BTCUSDT order=BTCFDUSD" in captured.err
        assert "Routing summary:" in captured.err
        assert "  required data quotes: USDT" in captured.err
        assert "  required order quotes: FDUSD" in captured.err

    def test_main_diagnose_summary_dash_prints_structured_error_on_no_data(self, capsys):
        from binance_worksteal.reporting import render_command_preview
        from binance_worksteal.trade_live import main

        argv = ["--symbols", "BTCUSD", "--diagnose", "--summary-json", "-"]
        expected_command = render_command_preview("binance_worksteal.trade_live", argv)

        with patch("binance_worksteal.trade_live.BinanceClient", None),              patch("binance_worksteal.trade_live._fetch_all_bars", return_value={}):
            rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert rc == 1
        assert payload["tool"] == "trade_live"
        assert payload["diagnose"] is True
        assert payload["inspection_mode"] == "diagnose"
        assert payload["error"] == "ERROR: No bar data fetched for any symbol"
        assert payload["error_type"] == "RuntimeError"
        assert payload["symbol_source"] == "command line --symbols"
        assert payload["symbols"] == ["BTCUSD"]
        assert payload["loaded_symbol_count"] == 0
        assert payload["loaded_symbols"] == []
        assert payload["missing_symbol_count"] == 1
        assert payload["missing_symbols"] == ["BTCUSD"]
        assert payload["universe_complete"] is False
        assert payload["data_coverage_ratio"] == pytest.approx(0.0)
        assert payload["candidate_count"] == 0
        assert payload["candidate_symbols"] == []
        assert payload["top_candidate"] is None
        assert payload["nearest_blocked"] is None
        assert payload["watchlist"] == []
        assert payload["action_summary"] == {
            "status": "no_data",
            "symbol": None,
            "reason": "no_market_data",
            "fill_price": None,
            "target_price": None,
            "current_price": None,
            "distance_pct": None,
            "summary": "no actionable signal; no market data loaded",
        }
        assert payload["candidates"] == []
        assert payload["closest_to_entry"] == []
        assert payload["filter_summary"] == []
        assert payload["diagnostics"] == []
        assert payload["min_entry_distance_pct"] is None
        assert payload["warnings"] == ["missing data for 1 symbol: BTCUSD"]
        assert payload["command"] == expected_command
        assert payload["status"] == "error"
        assert payload["exit_code"] == 1
        assert captured.err.count("Command:") == 1
        assert captured.err.count(expected_command) == 1
        assert "Symbols with data:  0/1" in captured.err
        assert "Data coverage:     0/1 loaded (0.0%)" in captured.err
        assert "Missing symbols:   BTCUSD" in captured.err
        assert "Action summary:    no actionable signal; no market data loaded" in captured.err
        assert "Warnings:" in captured.err
        assert "  - missing data for 1 symbol: BTCUSD" in captured.err
        assert "ERROR: No bar data fetched for any symbol" in captured.err

    def test_main_diagnose_summary_file_writes_artifacts(self, tmp_path, capsys):
        from types import SimpleNamespace

        from binance_worksteal.reporting import render_command_preview
        from binance_worksteal.trade_live import main

        bars1 = _make_bars(30, base_price=100)
        bars2 = _make_bars(30, base_price=200)
        config = WorkStealConfig(dip_pct=0.20, proximity_pct=0.02, lookback_days=5)
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({
                "positions": {},
                "pending_entries": {},
                "last_exit": {},
                "recent_trades": [],
                "peak_equity": 0,
            }),
            encoding="utf-8",
        )
        summary_path = tmp_path / "diagnose_summary.json"
        candidate_bar = bars1.iloc[-1]

        def fake_build_entry_candidates(*args, diagnostics=None, **kwargs):
            if diagnostics is not None:
                diagnostics.extend([
                    SymbolDiagnostic(
                        symbol="SYM1",
                        close=100.0,
                        ref_high=120.0,
                        buy_target=95.0,
                        dist_pct=0.01,
                        is_candidate=True,
                    )
                ])
            return [("SYM1", "long", 0.5, 95.0, candidate_bar)]

        argv = ["--symbols", "SYM1", "SYM2", "--diagnose", "--summary-json", str(summary_path)]
        expected_command = render_command_preview("binance_worksteal.trade_live", argv)

        with patch("binance_worksteal.trade_live.STATE_FILE", state_file),              patch("binance_worksteal.trade_live.BinanceClient", None),              patch("binance_worksteal.trade_live._fetch_all_bars", return_value={"SYM1": bars1, "SYM2": bars2}),              patch(
                 "binance_worksteal.trade_live.resolve_entry_regime",
                 return_value=SimpleNamespace(config=config, market_breadth_skip=False, risk_off=False),
             ),              patch("binance_worksteal.trade_live._entry_regime_breadth_snapshot", return_value=(0.5, 1, 2)),              patch("binance_worksteal.trade_live._count_sma_pass_fail", return_value=(1, 1)),              patch("binance_worksteal.trade_live.build_entry_candidates", side_effect=fake_build_entry_candidates):
            rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert rc == 0
        assert payload["tool"] == "trade_live"
        assert payload["diagnose"] is True
        assert payload["inspection_mode"] == "diagnose"
        assert payload["command"] == expected_command
        assert payload["candidate_symbols"] == ["SYM1"]
        assert payload["top_candidate"] == {
            "symbol": "SYM1",
            "direction": "long",
            "score": 0.5,
            "fill_price": 95.0,
            "close": float(candidate_bar["close"]),
        }
        assert payload["nearest_blocked"] is None
        assert payload["action_summary"] == {
            "status": "candidate_available",
            "symbol": "SYM1",
            "reason": None,
            "fill_price": 95.0,
            "target_price": 95.0,
            "current_price": float(candidate_bar["close"]),
            "distance_pct": 0.0,
            "summary": "stage SYM1 near $95.0000",
        }
        assert payload["summary_json_file"] == str(summary_path)
        assert "WORKSTEAL DIAGNOSTIC" in captured.out
        assert (
            f"Top candidate:     SYM1 score=0.5000 current=${float(candidate_bar['close']):.4f} fill=$95.0000"
            in captured.out
        )
        assert "Nearest blocked:   none" in captured.out
        assert "Action summary:    stage SYM1 near $95.0000" in captured.out
        assert "Command:" in captured.out
        assert f"  {expected_command}" in captured.out
        assert "Generated artifacts:" in captured.out
        assert f"  summary_json: {summary_path}" in captured.out
        assert "Reproduce:" in captured.out
        assert payload["invocation"]["command"] in captured.out

    def test_main_diagnose_summary_file_writes_structured_error_on_no_data(self, tmp_path, capsys):
        from binance_worksteal.reporting import render_command_preview
        from binance_worksteal.trade_live import main

        summary_path = tmp_path / "diagnose_error_summary.json"
        argv = ["--symbols", "BTCUSD", "--diagnose", "--summary-json", str(summary_path)]
        expected_command = render_command_preview("binance_worksteal.trade_live", argv)

        with patch("binance_worksteal.trade_live.BinanceClient", None),              patch("binance_worksteal.trade_live._fetch_all_bars", return_value={}):
            rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert rc == 1
        assert payload["tool"] == "trade_live"
        assert payload["diagnose"] is True
        assert payload["inspection_mode"] == "diagnose"
        assert payload["error"] == "ERROR: No bar data fetched for any symbol"
        assert payload["error_type"] == "RuntimeError"
        assert payload["symbol_source"] == "command line --symbols"
        assert payload["symbols"] == ["BTCUSD"]
        assert payload["loaded_symbol_count"] == 0
        assert payload["loaded_symbols"] == []
        assert payload["missing_symbol_count"] == 1
        assert payload["missing_symbols"] == ["BTCUSD"]
        assert payload["universe_complete"] is False
        assert payload["data_coverage_ratio"] == pytest.approx(0.0)
        assert payload["candidate_count"] == 0
        assert payload["candidate_symbols"] == []
        assert payload["top_candidate"] is None
        assert payload["nearest_blocked"] is None
        assert payload["watchlist"] == []
        assert payload["action_summary"] == {
            "status": "no_data",
            "symbol": None,
            "reason": "no_market_data",
            "fill_price": None,
            "target_price": None,
            "current_price": None,
            "distance_pct": None,
            "summary": "no actionable signal; no market data loaded",
        }
        assert payload["candidates"] == []
        assert payload["closest_to_entry"] == []
        assert payload["filter_summary"] == []
        assert payload["diagnostics"] == []
        assert payload["min_entry_distance_pct"] is None
        assert payload["warnings"] == ["missing data for 1 symbol: BTCUSD"]
        assert payload["command"] == expected_command
        assert payload["summary_json_file"] == str(summary_path)
        assert payload["status"] == "error"
        assert payload["exit_code"] == 1
        assert captured.out.count("Command:") == 1
        assert "Symbols with data:  0/1" in captured.out
        assert "Data coverage:     0/1 loaded (0.0%)" in captured.out
        assert "Missing symbols:   BTCUSD" in captured.out
        assert "Action summary:    no actionable signal; no market data loaded" in captured.out
        assert "Warnings:" in captured.out
        assert "  - missing data for 1 symbol: BTCUSD" in captured.out
        assert "ERROR: No bar data fetched for any symbol" in captured.out
        assert "Diagnostic artifacts:" in captured.out
        assert f"  summary_json: {summary_path}" in captured.out
        assert "Reproduce:" in captured.out
        assert payload["invocation"]["command"] in captured.out

    def test_main_diagnose_invalid_universe_summary_dash_keeps_inspection_mode(self, tmp_path, capsys):
        from binance_worksteal.reporting import render_command_preview
        from binance_worksteal.trade_live import main

        missing_path = tmp_path / "missing.yaml"
        argv = ["--universe-file", str(missing_path), "--diagnose", "--summary-json", "-"]
        expected_command = render_command_preview("binance_worksteal.trade_live", argv)

        rc = main(argv)

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert rc == 1
        assert payload["tool"] == "trade_live"
        assert payload["diagnose"] is True
        assert payload["inspection_mode"] == "diagnose"
        assert payload["error"] == f"ERROR: Universe file not found: {missing_path}"
        assert payload["error_type"] == "FileNotFoundError"
        assert payload["command"] == expected_command
        assert payload["status"] == "error"
        assert payload["exit_code"] == 1
        assert captured.err.strip() == f"ERROR: Universe file not found: {missing_path}"

    def test_diagnose_initializes_client(self):
        """Fix 3: --diagnose should attempt to create a real Binance client."""
        from binance_worksteal.trade_live import main
        mock_client = object()
        with patch("binance_worksteal.trade_live.BinanceClient", return_value=mock_client) as mock_cls, \
             patch("binance_worksteal.trade_live.run_diagnose", return_value=0) as mock_diag, \
             patch("sys.argv", ["trade_live.py", "--diagnose"]), \
             patch.dict("sys.modules", {"env_real": type(sys)("env_real")}):
            sys.modules["env_real"].BINANCE_API_KEY = "test_key"
            sys.modules["env_real"].BINANCE_SECRET = "test_secret"
            result = main()
            assert result == 0
            mock_cls.assert_called_once_with("test_key", "test_secret")
            assert mock_diag.call_args[0][0] is mock_client

    def test_main_returns_diagnose_failure_exit_code(self):
        from binance_worksteal.trade_live import main
        with patch("binance_worksteal.trade_live.run_diagnose", return_value=1) as mock_diag, \
             patch("sys.argv", ["trade_live.py", "--diagnose"]):
            result = main()
            assert result == 1
            mock_diag.assert_called_once()


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
