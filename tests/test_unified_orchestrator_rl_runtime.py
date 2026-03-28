from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from llm_hourly_trader.gemini_wrapper import TradePlan
from unified_orchestrator import orchestrator
from unified_orchestrator.rl_gemini_bridge import RLSignal


def _make_history_frame(symbol: str, *, periods: int = 72, base_price: float = 100.0, freq: str = "h") -> pd.DataFrame:
    index = pd.date_range("2026-03-10 00:00:00+00:00", periods=periods, freq=freq, tz="UTC")
    closes = [base_price + i * 0.5 for i in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": index,
            "symbol": symbol,
            "open": [price - 0.2 for price in closes],
            "high": [price + 0.4 for price in closes],
            "low": [price - 0.6 for price in closes],
            "close": closes,
            "volume": [1_000 + i for i in range(periods)],
        }
    )


def _make_forecast_frame(frame: pd.DataFrame, *, offset: float) -> pd.DataFrame:
    index = pd.to_datetime(frame["timestamp"], utc=True)
    close = frame["close"].astype(float).to_numpy()
    return pd.DataFrame(
        {
            "predicted_close_p50": close + offset,
            "predicted_close_p10": close + offset - 1.0,
            "predicted_close_p90": close + offset + 1.0,
            "predicted_high_p50": close + offset + 0.5,
            "predicted_low_p50": close + offset - 0.5,
        },
        index=index,
    )


class _FakeBridge:
    def __init__(self, *, checkpoint_path: str, obs_size: int):
        self.checkpoint_path = Path(checkpoint_path)
        self._spec = SimpleNamespace(obs_size=obs_size, alloc_bins=1, level_bins=1)
        self.calls: list[dict[str, object]] = []

    def get_checkpoint_spec(self):
        return self._spec

    def get_rl_signals(self, obs, num_symbols, symbol_names, top_k):
        self.calls.append(
            {
                "obs_shape": tuple(obs.shape),
                "num_symbols": num_symbols,
                "symbol_names": list(symbol_names),
                "top_k": top_k,
            }
        )
        return [
            RLSignal(
                symbol_idx=idx,
                symbol_name=symbol,
                direction="long",
                confidence=0.6 + idx * 0.01,
                logit_gap=1.5,
                allocation_pct=0.5,
            )
            for idx, symbol in enumerate(symbol_names)
        ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_crypto_rl_signal_map_uses_full_trained_universe(monkeypatch):
    trained_symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    history_frames = {
        symbol: _make_history_frame(symbol, base_price=100.0 + idx * 10.0)
        for idx, symbol in enumerate(trained_symbols)
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt",
        obs_size=90,
    )

    signal_map = orchestrator._build_crypto_rl_signal_map(history_frames, bridge)

    assert set(signal_map) == set(trained_symbols)
    assert bridge.calls == [
        {
            "obs_shape": (90,),
            "num_symbols": 5,
            "symbol_names": trained_symbols,
            "top_k": 5,
        }
    ]


def test_maybe_use_rl_only_fallback_plan_uses_rl_signal_when_llm_exhausted():
    plan = TradePlan(direction="hold", buy_price=0.0, sell_price=0.0, confidence=0.0, reasoning="API exhausted")
    signal = RLSignal(
        symbol_idx=0,
        symbol_name="ETHUSD",
        direction="long",
        confidence=0.75,
        logit_gap=1.2,
        allocation_pct=0.5,
    )

    out = orchestrator._maybe_use_rl_only_fallback_plan("ETHUSD", plan, signal, 2000.0)

    assert out.direction == "long"
    assert out.buy_price == pytest.approx(1996.0)
    assert out.sell_price == pytest.approx(2020.0)
    assert out.confidence == pytest.approx(0.6)
    assert "llm_unavailable" in out.reasoning


def test_maybe_use_rl_only_fallback_plan_keeps_non_exhausted_hold():
    plan = TradePlan(direction="hold", buy_price=0.0, sell_price=0.0, confidence=0.0, reasoning="model says hold")
    signal = RLSignal(
        symbol_idx=0,
        symbol_name="ETHUSD",
        direction="long",
        confidence=0.75,
        logit_gap=1.2,
        allocation_pct=0.5,
    )

    out = orchestrator._maybe_use_rl_only_fallback_plan("ETHUSD", plan, signal, 2000.0)

    assert out == plan


def test_load_recent_stock_edges_from_meta_log_prefers_latest_fresh_positive_edges(tmp_path: Path):
    event_log = tmp_path / "stock_event_log.jsonl"
    _write_jsonl(
        event_log,
        [
            {
                "event_type": "meta_signal_ready",
                "symbol": "AAPL",
                "edge": 0.01,
                "logged_at": "2026-03-28T07:00:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "AAPL",
                "edge": 0.03,
                "logged_at": "2026-03-28T08:50:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "NVDA",
                "edge": 0.025,
                "logged_at": "2026-03-28T08:40:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "TSLA",
                "edge": -0.02,
                "logged_at": "2026-03-28T08:55:00Z",
            },
            {
                "event_type": "meta_signal_skipped",
                "symbol": "MSFT",
                "edge": 0.04,
                "logged_at": "2026-03-28T08:56:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "AAPL",
                "edge": 0.05,
                "logged_at": "2026-03-27T10:00:00Z",
            },
        ],
    )

    out = orchestrator._load_recent_stock_edges_from_meta_log(
        ["AAPL", "NVDA", "TSLA"],
        as_of=datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc),
        event_log=event_log,
    )

    assert out == {"AAPL": 0.03, "NVDA": 0.025}


def test_load_recent_stock_edges_from_meta_log_prefers_latest_real_cycle_over_isolated_signal_rows(tmp_path: Path):
    event_log = tmp_path / "stock_event_log.jsonl"
    _write_jsonl(
        event_log,
        [
            {
                "event_type": "meta_cycle_start",
                "pid": 153050,
                "logged_at": "2026-03-28T09:15:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "pid": 153050,
                "symbol": "AAPL",
                "edge": 0.04,
                "logged_at": "2026-03-28T09:15:10Z",
            },
            {
                "event_type": "meta_signal_ready",
                "pid": 999999,
                "symbol": "NVDA",
                "edge": 0.50,
                "logged_at": "2026-03-28T09:20:00Z",
            },
        ],
    )

    out = orchestrator._load_recent_stock_edges_from_meta_log(
        ["AAPL", "NVDA"],
        as_of=datetime(2026, 3, 28, 9, 30, tzinfo=timezone.utc),
        event_log=event_log,
    )

    assert out == {"AAPL": 0.04}


def test_run_cycle_pre_market_passes_recent_stock_edges_to_backout(monkeypatch):
    snapshot = SimpleNamespace(
        regime="PRE_MARKET",
        total_stock_value=10_000.0,
        alpaca_positions={},
        binance_positions={},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(orchestrator, "build_snapshot", lambda now=None: snapshot)
    monkeypatch.setattr(orchestrator, "save_snapshot", lambda snapshot: None)
    monkeypatch.setattr(orchestrator, "read_pending_fills", lambda since_minutes=65: [])
    monkeypatch.setattr(orchestrator, "get_crypto_signals", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        orchestrator,
        "_load_recent_stock_edges_from_meta_log",
        lambda stock_symbols, as_of=None: {"AAPL": 0.04, "NVDA": 0.02},
    )

    def _fake_select_backout_candidates(snapshot_arg, best_stock_edges, min_edge_ratio=2.0, only_profitable=True):
        captured["snapshot"] = snapshot_arg
        captured["best_stock_edges"] = best_stock_edges
        return []

    monkeypatch.setattr(orchestrator, "select_backout_candidates", _fake_select_backout_candidates)

    results = orchestrator.run_cycle(
        crypto_symbols=["BTCUSD"],
        stock_symbols=["AAPL", "NVDA"],
        dry_run=True,
    )

    assert captured["snapshot"] is snapshot
    assert captured["best_stock_edges"] == {"AAPL": 0.04, "NVDA": 0.02}
    assert results["best_stock_edges"] == {"AAPL": 0.04, "NVDA": 0.02}


def test_build_stock_rl_signal_map_uses_forecast_features(monkeypatch):
    trained_symbols = ["AAPL", "NVDA"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    history_frames = {
        symbol: _make_history_frame(symbol, base_price=150.0 + idx * 25.0)
        for idx, symbol in enumerate(trained_symbols)
    }
    forecasts_1h = {
        symbol: _make_forecast_frame(frame, offset=1.0)
        for symbol, frame in history_frames.items()
    }
    forecasts_24h = {
        symbol: _make_forecast_frame(frame, offset=3.0)
        for symbol, frame in history_frames.items()
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt",
        obs_size=39,
    )

    signal_map = orchestrator._build_stock_rl_signal_map(
        history_frames,
        bridge,
        forecasts_1h,
        forecasts_24h,
    )

    assert set(signal_map) == set(trained_symbols)
    assert bridge.calls == [
        {
            "obs_shape": (39,),
            "num_symbols": 2,
            "symbol_names": trained_symbols,
            "top_k": 2,
        }
    ]


# ---------------------------------------------------------------------------
# _is_daily_checkpoint tests
# ---------------------------------------------------------------------------

def _write_mktd_header(path: Path, symbols: list[str]) -> None:
    """Write a minimal MKTD header for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD", 2, len(symbols), 1, 16, 5, b"\x00" * 40,
    )
    with path.open("wb") as handle:
        handle.write(header)
        for sym in symbols:
            handle.write(sym.encode("ascii").ljust(16, b"\x00"))


def test_is_daily_checkpoint_detects_daily_data_hint(monkeypatch, tmp_path):
    """When the data-hint .bin filename contains 'daily', checkpoint is daily."""
    data_path = tmp_path / "crypto5_daily_train.bin"
    _write_mktd_header(data_path, ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"])

    checkpoint = tmp_path / "checkpoints" / "tp0.15_s314" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    monkeypatch.setitem(orchestrator._CHECKPOINT_DATA_HINTS, "tp0.15_s314", data_path)
    assert orchestrator._is_daily_checkpoint(checkpoint) is True


def test_is_daily_checkpoint_detects_daily_in_path():
    """When checkpoint path contains 'mass_daily', checkpoint is daily."""
    checkpoint = Path("pufferlib_market/checkpoints/mass_daily/tp0.15_s314/best.pt")
    assert orchestrator._is_daily_checkpoint(checkpoint) is True


def test_is_daily_checkpoint_returns_false_for_hourly():
    """Hourly checkpoints (no 'daily' in path or data hint) return False."""
    checkpoint = Path("pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt")
    assert orchestrator._is_daily_checkpoint(checkpoint) is False


def test_is_daily_checkpoint_returns_false_for_stock_hourly():
    """Stock hourly checkpoints return False."""
    checkpoint = Path("pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt")
    assert orchestrator._is_daily_checkpoint(checkpoint) is False


# ---------------------------------------------------------------------------
# _build_crypto_rl_signal_map with daily checkpoint
# ---------------------------------------------------------------------------

def test_build_crypto_rl_signal_map_daily_checkpoint(monkeypatch):
    """Daily checkpoint uses compute_daily_features and requires 60 bars."""
    trained_symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    # Create daily frames with 90 bars (> 60 minimum)
    history_frames = {
        symbol: _make_history_frame(symbol, periods=90, base_price=100.0 + idx * 10.0, freq="D")
        for idx, symbol in enumerate(trained_symbols)
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/mass_daily/tp0.15_s314/best.pt",
        obs_size=90,
    )

    signal_map = orchestrator._build_crypto_rl_signal_map(history_frames, bridge)

    assert set(signal_map) == set(trained_symbols)
    assert bridge.calls[0]["num_symbols"] == 5


def test_build_crypto_rl_signal_map_daily_skips_insufficient_bars(monkeypatch):
    """Daily checkpoint skips symbols with < 60 bars."""
    trained_symbols = ["BTCUSD", "ETHUSD"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    history_frames = {
        "BTCUSD": _make_history_frame("BTCUSD", periods=90, base_price=80000.0, freq="D"),
        "ETHUSD": _make_history_frame("ETHUSD", periods=30, base_price=2000.0, freq="D"),  # too few
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/mass_daily/tp0.10_s42/best.pt",
        obs_size=39,
    )

    signal_map = orchestrator._build_crypto_rl_signal_map(history_frames, bridge)

    # BTCUSD should be in signal map, ETHUSD skipped
    assert "BTCUSD" in signal_map
    # ETHUSD was skipped (< 60 bars) but bridge still gets called with all trained_symbols
    assert bridge.calls[0]["num_symbols"] == 2


# ---------------------------------------------------------------------------
# compute_daily_features smoke test
# ---------------------------------------------------------------------------

def test_compute_daily_features_returns_16_floats():
    """compute_daily_features returns a 16-element float32 array."""
    from pufferlib_market.inference_daily import compute_daily_features

    df = pd.DataFrame({
        "open": np.random.uniform(90, 110, 100),
        "high": np.random.uniform(100, 120, 100),
        "low": np.random.uniform(80, 100, 100),
        "close": np.random.uniform(90, 110, 100),
        "volume": np.random.uniform(1e5, 1e7, 100),
    })
    features = compute_daily_features(df)
    assert features.shape == (16,)
    assert features.dtype == np.float32
    # At least some features should be non-zero
    assert (features != 0).sum() > 0
