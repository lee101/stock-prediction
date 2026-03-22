"""Tests for edge cases in binance_worksteal pipeline."""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from binance_worksteal.strategy import (
    WorkStealConfig,
    compute_atr,
    compute_buy_target,
    compute_metrics,
    compute_ref_low,
    compute_ref_price,
    compute_rsi,
    compute_sma,
    compute_volume_ratio,
    passes_sma_filter,
    run_worksteal_backtest,
    _risk_off_triggered,
    build_entry_candidates,
)


def make_bars(prices, start="2026-01-01", symbol="BTCUSD"):
    dates = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    rows = []
    for d, p in zip(dates, prices):
        noise = p * 0.02
        rows.append({
            "timestamp": d, "open": p - noise * 0.5,
            "high": p + noise, "low": p - noise,
            "close": p, "volume": 1000.0, "symbol": symbol,
        })
    return pd.DataFrame(rows)


def make_empty_bars():
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])


# ---- strategy.py edge cases ----

class TestEmptyBarsHandling:
    def test_compute_ref_price_empty(self):
        assert compute_ref_price(make_empty_bars(), "high", 20) == 0.0

    def test_compute_ref_low_empty(self):
        assert compute_ref_low(make_empty_bars(), 20) == 0.0

    def test_compute_sma_empty(self):
        assert compute_sma(make_empty_bars(), 20) == 0.0

    def test_compute_atr_empty(self):
        assert compute_atr(make_empty_bars(), 14) == 0.0

    def test_compute_rsi_empty(self):
        assert compute_rsi(make_empty_bars(), 14) == 50.0

    def test_compute_volume_ratio_empty(self):
        assert compute_volume_ratio(make_empty_bars(), 20) == 1.0


class TestSingleBarHandling:
    def test_compute_ref_price_single(self):
        bars = make_bars([100.0])
        ref = compute_ref_price(bars, "high", 20)
        assert ref == pytest.approx(100.0, rel=0.05)

    def test_compute_ref_low_single(self):
        bars = make_bars([100.0])
        ref = compute_ref_low(bars, 20)
        assert ref == pytest.approx(100.0, rel=0.05)

    def test_compute_atr_single(self):
        bars = make_bars([100.0])
        atr = compute_atr(bars, 14)
        assert atr > 0


class TestComputeBuyTargetEdgeCases:
    def test_zero_ref_high(self):
        bars = make_bars([100.0] * 5)
        result = compute_buy_target(bars, 0.0, WorkStealConfig())
        assert result == 0.0

    def test_zero_ref_high_adaptive(self):
        bars = make_bars([100.0] * 20)
        config = WorkStealConfig(adaptive_dip=True)
        result = compute_buy_target(bars, 0.0, config)
        assert result == 0.0

    def test_adaptive_dip_normal(self):
        bars = make_bars([100.0] * 20)
        config = WorkStealConfig(adaptive_dip=True, dip_pct=0.20)
        result = compute_buy_target(bars, 100.0, config)
        assert 0 < result < 100.0


class TestPassesSmaFilterEdgeCases:
    def test_empty_bars(self):
        config = WorkStealConfig(sma_filter_period=20)
        assert passes_sma_filter(make_empty_bars(), config, 100.0) is True

    def test_single_bar_pre_dip_falls_back_to_close_check(self):
        bars = make_bars([100.0])
        config = WorkStealConfig(sma_filter_period=20, sma_check_method="pre_dip")
        assert passes_sma_filter(bars, config, 100.0) is True
        assert passes_sma_filter(bars, config, 50.0) is False

    def test_two_bars_pre_dip(self):
        bars = make_bars([100.0, 90.0])
        config = WorkStealConfig(sma_filter_period=20, sma_check_method="pre_dip")
        result = passes_sma_filter(bars, config, 50.0)
        assert result is True  # bar[0] close=100 >= SMA of ~95

    def test_disabled_filter(self):
        assert passes_sma_filter(make_bars([50.0] * 5), WorkStealConfig(sma_filter_period=0), 10.0) is True


class TestComputeMetricsEdgeCases:
    def test_empty_equity_df(self):
        assert compute_metrics(pd.DataFrame(), WorkStealConfig()) == {}

    def test_single_row(self):
        eq_df = pd.DataFrame({"equity": [10000.0]})
        assert compute_metrics(eq_df, WorkStealConfig()) == {}

    def test_zero_initial_equity(self):
        eq_df = pd.DataFrame({"equity": [0.0, 100.0, 200.0]})
        assert compute_metrics(eq_df, WorkStealConfig()) == {}

    def test_nan_in_equity(self):
        eq_df = pd.DataFrame({"equity": [10000.0, float("nan"), 10100.0]})
        assert compute_metrics(eq_df, WorkStealConfig()) == {}

    def test_flat_equity(self):
        eq_df = pd.DataFrame({"equity": [10000.0, 10000.0, 10000.0, 10000.0]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] == pytest.approx(0.0, abs=1e-10)
        assert np.isfinite(m["sortino"])
        assert np.isfinite(m["sharpe"])

    def test_with_trades(self):
        from binance_worksteal.strategy import TradeLog
        eq_df = pd.DataFrame({"equity": [10000.0, 10100.0, 10200.0]})
        trades = [
            TradeLog(timestamp=pd.Timestamp.now(), symbol="BTC", side="sell",
                     price=100, quantity=1, notional=100, fee=0.1, pnl=50),
            TradeLog(timestamp=pd.Timestamp.now(), symbol="BTC", side="sell",
                     price=100, quantity=1, notional=100, fee=0.1, pnl=-20),
        ]
        m = compute_metrics(eq_df, WorkStealConfig(), trades)
        assert m["win_rate"] == 50.0


class TestRiskOffEdgeCases:
    def test_all_symbols_insufficient_history(self):
        bars = make_bars([100.0, 95.0])
        current_bars = {"BTCUSD": bars.iloc[-1]}
        history = {"BTCUSD": bars}
        config = WorkStealConfig(
            risk_off_trigger_sma_period=30,
            risk_off_trigger_momentum_period=30,
            risk_off_momentum_threshold=-0.05,
        )
        result = _risk_off_triggered(current_bars, history, config)
        assert result is False

    def test_empty_current_bars(self):
        config = WorkStealConfig(
            risk_off_trigger_sma_period=5,
            risk_off_trigger_momentum_period=5,
        )
        result = _risk_off_triggered({}, {}, config)
        assert result is False

    def test_momentum_with_zero_past_close(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
        rows = []
        for i, d in enumerate(dates):
            close = 0.0 if i < 5 else 100.0
            rows.append({"timestamp": d, "open": close, "high": close + 1,
                         "low": max(close - 1, 0), "close": close, "volume": 1000.0, "symbol": "SYM"})
        bars = pd.DataFrame(rows)
        current_bars = {"SYM": bars.iloc[-1]}
        history = {"SYM": bars}
        config = WorkStealConfig(
            risk_off_trigger_momentum_period=5,
            risk_off_momentum_threshold=-0.05,
        )
        result = _risk_off_triggered(current_bars, history, config)
        assert isinstance(result, bool)


class TestBuildEntryCandidatesEdgeCases:
    def test_empty_history(self):
        candidates = build_entry_candidates(
            current_bars={}, history={}, positions={},
            last_exit={}, date=pd.Timestamp("2026-01-01", tz="UTC"),
            config=WorkStealConfig(lookback_days=5),
        )
        assert candidates == []

    def test_all_symbols_already_positioned(self):
        bars = make_bars([100.0] * 30)
        current_bars = {"BTCUSD": bars.iloc[-1]}
        history = {"BTCUSD": bars}
        positions = {"BTCUSD": object()}
        candidates = build_entry_candidates(
            current_bars=current_bars, history=history,
            positions=positions, last_exit={},
            date=pd.Timestamp("2026-01-30", tz="UTC"),
            config=WorkStealConfig(lookback_days=20),
        )
        assert candidates == []

    def test_insufficient_lookback(self):
        bars = make_bars([100.0] * 5)
        current_bars = {"BTCUSD": bars.iloc[-1]}
        history = {"BTCUSD": bars}
        candidates = build_entry_candidates(
            current_bars=current_bars, history=history,
            positions={}, last_exit={},
            date=pd.Timestamp("2026-01-05", tz="UTC"),
            config=WorkStealConfig(lookback_days=20),
        )
        assert candidates == []


class TestBacktestEdgeCases:
    def test_no_data_returns_empty(self):
        eq, trades, metrics = run_worksteal_backtest({}, WorkStealConfig())
        assert eq.empty
        assert trades == []
        assert metrics == {}

    def test_single_symbol_single_bar(self):
        bars = {"BTCUSD": make_bars([100.0])}
        eq, trades, metrics = run_worksteal_backtest(bars, WorkStealConfig())
        assert trades == []


# ---- trade_live.py edge cases ----

class TestSaveStateAtomic:
    def test_atomic_write(self, tmp_path):
        from binance_worksteal import trade_live
        original_state_file = trade_live.STATE_FILE
        try:
            trade_live.STATE_FILE = tmp_path / "live_state.json"
            trade_live.save_state({"positions": {}, "test": True})
            assert trade_live.STATE_FILE.exists()
            data = json.loads(trade_live.STATE_FILE.read_text())
            assert data["test"] is True
            assert not (tmp_path / "live_state.tmp").exists()
        finally:
            trade_live.STATE_FILE = original_state_file

    def test_overwrite_existing(self, tmp_path):
        from binance_worksteal import trade_live
        original_state_file = trade_live.STATE_FILE
        try:
            trade_live.STATE_FILE = tmp_path / "live_state.json"
            trade_live.save_state({"version": 1})
            trade_live.save_state({"version": 2})
            data = json.loads(trade_live.STATE_FILE.read_text())
            assert data["version"] == 2
        finally:
            trade_live.STATE_FILE = original_state_file


class TestPlanLegacyRebalanceExitsMissingBars:
    def test_position_without_bars_skipped(self):
        from binance_worksteal.trade_live import (
            plan_legacy_rebalance_exits, build_runtime_config, build_arg_parser,
        )
        history = {
            "BTCUSD": make_bars([100.0] * 30, symbol="BTCUSD"),
        }
        current_bars = {"BTCUSD": history["BTCUSD"].iloc[-1]}
        positions = {
            "BTCUSD": {
                "entry_price": 100.0,
                "entry_date": "2026-03-01T00:00:00+00:00",
                "quantity": 1.0,
                "peak_price": 100.0,
                "target_sell": 103.0,
                "stop_price": 92.0,
                "source": "legacy",
            },
            "MISSINGUSDUSD": {
                "entry_price": 50.0,
                "entry_date": "2026-03-01T00:00:00+00:00",
                "quantity": 1.0,
                "peak_price": 50.0,
                "target_sell": 53.0,
                "stop_price": 45.0,
                "source": "legacy",
            },
        }
        config = build_runtime_config(build_arg_parser().parse_args([
            "--dip-pct", "0.10", "--lookback-days", "20",
        ]))
        exits, _ = plan_legacy_rebalance_exits(
            now=datetime(2026, 3, 18, tzinfo=timezone.utc),
            positions=positions,
            current_bars=current_bars,
            history=history,
            last_exit={},
            config=config,
        )
        exit_syms = [sym for sym, *_ in exits]
        assert "MISSINGUSDUSD" not in exit_syms


class TestNormalizeLivePositionsEdgeCases:
    def test_none_input(self):
        from binance_worksteal.trade_live import normalize_live_positions, DEFAULT_CONFIG
        result = normalize_live_positions(None, DEFAULT_CONFIG)
        assert result == {}

    def test_non_dict_values_skipped(self):
        from binance_worksteal.trade_live import normalize_live_positions, DEFAULT_CONFIG
        result = normalize_live_positions({"BTC": "invalid"}, DEFAULT_CONFIG)
        assert result == {}

    def test_zero_quantity_skipped(self):
        from binance_worksteal.trade_live import normalize_live_positions, DEFAULT_CONFIG
        result = normalize_live_positions({
            "BTCUSD": {"entry_price": 100.0, "quantity": 0.0}
        }, DEFAULT_CONFIG)
        assert result == {}

    def test_zero_entry_price_skipped(self):
        from binance_worksteal.trade_live import normalize_live_positions, DEFAULT_CONFIG
        result = normalize_live_positions({
            "BTCUSD": {"entry_price": 0.0, "quantity": 1.0}
        }, DEFAULT_CONFIG)
        assert result == {}


# ---- model.py edge cases ----

class TestModelEdgeCases:
    def test_odd_dim_positional_encoding(self):
        from binance_worksteal.model import PositionalEncoding
        pe = PositionalEncoding(dim=31, dropout=0.0, max_len=50)
        x = torch.randn(2, 10, 31)
        out = pe(x)
        assert out.shape == (2, 10, 31)
        assert not torch.isnan(out).any()

    def test_nan_input_propagation(self):
        from binance_worksteal.model import DailyWorkStealPolicy
        model = DailyWorkStealPolicy(n_features=11, n_symbols=3, hidden_dim=32,
                                     num_layers=1, num_heads=2, seq_len=10)
        x = torch.randn(2, 10, 3, 11)
        out_clean = model(x)
        assert not torch.isnan(out_clean["buy_offset"]).any()

    def test_single_symbol(self):
        from binance_worksteal.model import PerSymbolWorkStealPolicy
        model = PerSymbolWorkStealPolicy(
            n_features=11, n_symbols=1, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=10,
        )
        x = torch.randn(2, 10, 1, 11)
        out = model(x)
        assert out["buy_offset"].shape == (2, 1)
        assert (out["intensity"] >= 0).all() and (out["intensity"] <= 1.0).all()


# ---- data.py edge cases ----

class TestDataEdgeCases:
    def test_compute_features_no_volume_column(self):
        from binance_worksteal.data import compute_features
        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.rand(30) * 100 + 50,
            "high": np.random.rand(30) * 100 + 55,
            "low": np.random.rand(30) * 100 + 45,
            "close": np.random.rand(30) * 100 + 50,
        })
        feats = compute_features(df)
        assert len(feats) == 30
        assert not feats.isna().any().any()
        assert (feats["volume_norm"] == 0.0).all()

    def test_compute_features_zero_close(self):
        from binance_worksteal.data import compute_features
        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        close = np.ones(30) * 100.0
        close[15] = 0.0  # one zero close
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close,
            "high": close + 1,
            "low": np.maximum(close - 1, 0),
            "close": close,
            "volume": np.ones(30) * 1000.0,
        })
        feats = compute_features(df)
        assert not feats.isna().any().any()
        assert (feats.values >= -5.0).all() and (feats.values <= 5.0).all()

    def test_feature_clipping(self):
        from binance_worksteal.data import compute_features
        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.ones(30) * 100.0,
            "high": np.ones(30) * 200.0,
            "low": np.ones(30) * 1.0,
            "close": np.ones(30) * 100.0,
            "volume": np.ones(30) * 1e15,
        })
        feats = compute_features(df)
        assert (feats.values >= -5.0).all() and (feats.values <= 5.0).all()


# ---- sweep.py edge case ----

class TestSweepNoStrictFill:
    def test_workstealconfig_no_strict_fill_field(self):
        try:
            config = WorkStealConfig(initial_cash=5000.0, dip_pct=0.10)
            assert config.initial_cash == 5000.0
        except TypeError:
            pytest.fail("WorkStealConfig should not have strict_fill")

        import inspect
        fields = {f.name for f in __import__("dataclasses").fields(WorkStealConfig)}
        assert "strict_fill" not in fields


# ---- gemini_overlay.py edge cases ----

class TestGeminiOverlayEdgeCases:
    def test_call_gemini_daily_no_genai(self, monkeypatch):
        from binance_worksteal import gemini_overlay
        monkeypatch.setattr(gemini_overlay, "genai", None)
        result = gemini_overlay.call_gemini_daily("test prompt")
        assert result is None

    def test_call_gemini_daily_no_api_key(self, monkeypatch):
        from binance_worksteal import gemini_overlay
        if gemini_overlay.genai is None:
            pytest.skip("genai not installed")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = gemini_overlay.call_gemini_daily("test prompt", api_key="")
        assert result is None

    def test_build_daily_prompt_minimal(self):
        from binance_worksteal.gemini_overlay import build_daily_prompt
        bars = make_bars([100.0] * 25)
        prompt = build_daily_prompt(
            symbol="BTCUSD", bars=bars, current_price=100.0,
            rule_signal={}, fee_bps=10,
        )
        assert "BTCUSD" in prompt
        assert "100.00" in prompt

    def test_build_daily_prompt_with_all_optional(self):
        from binance_worksteal.gemini_overlay import build_daily_prompt
        bars = make_bars([100.0 + i for i in range(60)])
        prompt = build_daily_prompt(
            symbol="ETHUSD", bars=bars, current_price=159.0,
            rule_signal={"buy_target": 150.0, "dip_score": 0.05, "ref_price": 160.0, "sma_ok": True},
            position_info={"entry_price": 145.0, "quantity": 0.5, "held_days": 3,
                           "peak_price": 155.0, "target_sell": 170.0, "stop_price": 130.0},
            recent_trades=[{"timestamp": "2026-01-01", "side": "buy", "symbol": "ETHUSD",
                            "price": 145.0, "pnl": 0, "reason": "dip_buy"}],
            forecast_24h={"predicted_close_p50": 160.0, "predicted_high_p50": 165.0,
                          "predicted_low_p50": 155.0,
                          "predicted_close_p90": 170.0, "predicted_close_p10": 150.0},
            universe_summary="BTC: +2%, ETH: -1%",
            fee_bps=0,
            entry_proximity_bps=3000.0,
        )
        assert "ETHUSD" in prompt
        assert "FDUSD" in prompt
        assert "CHRONOS2" in prompt
        assert "POSITION" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
