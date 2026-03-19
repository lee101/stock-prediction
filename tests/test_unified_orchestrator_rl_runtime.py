from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
