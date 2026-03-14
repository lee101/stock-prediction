from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from unified_orchestrator import orchestrator
from unified_orchestrator.rl_gemini_bridge import RLSignal


def _make_history_frame(symbol: str, *, periods: int = 72, base_price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2026-03-10 00:00:00+00:00", periods=periods, freq="h", tz="UTC")
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
