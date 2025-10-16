#!/usr/bin/env python3
"""Tests for hfinference HFTradingEngine using synthetic data and mocks.

These tests bypass real checkpoints and network calls to validate
signal generation, trade execution, and backtest integration.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure repository root is on import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip if torch is not installed, since the engine and dummy model use it
pytest.importorskip("torch", reason="hfinference engine tests require torch installed")

import hfinference.hf_trading_engine as hfe


class _DummyModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        # x: [B, seq_len, features]
        B, L, F = x.shape
        horizon = self.cfg.get("prediction_horizon", 5)
        features = self.cfg.get("input_features", F)
        # Predict slight increase on close (index 3) and strong buy prob
        price_preds = np.zeros((B, horizon, features), dtype=np.float32)
        price_preds[..., 3] = 0.2  # normalized positive delta
        action_logits = np.array([[5.0, 0.1, -5.0]], dtype=np.float32)  # buy/hold/sell
        import torch
        return {
            "price_predictions": torch.from_numpy(price_preds),
            "action_logits": torch.from_numpy(action_logits).repeat(B, 1),
            "action_probs": torch.softmax(torch.from_numpy(action_logits).repeat(B, 1), dim=-1),
        }


def _make_ohlcv(days=100, start=100.0, drift=0.2, seed=7):
    rng = np.random.RandomState(seed)
    close = start + np.cumsum(rng.randn(days) * 0.5 + drift)
    open_ = close + rng.randn(days) * 0.2
    high = np.maximum(open_, close) + np.abs(rng.randn(days)) * 0.5
    low = np.minimum(open_, close) - np.abs(rng.randn(days)) * 0.5
    vol = rng.randint(1_000_000, 5_000_000, size=days)
    idx = pd.date_range(end=datetime.now(), periods=days, freq="D")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # Patch load_model to bypass checkpoint reading and return dummy model
    def _fake_load_model(self, checkpoint_path):
        model_cfg = {
            "hidden_size": 64,
            "num_heads": 4,
            "num_layers": 2,
            "intermediate_size": 128,
            "dropout": 0.0,
            "input_features": 21,
            "sequence_length": 60,
            "prediction_horizon": 5,
        }
        return _DummyModel(model_cfg)

    monkeypatch.setattr(hfe.HFTradingEngine, "load_model", _fake_load_model)
    yield


def test_generate_signal_buy_action(monkeypatch):
    # Instantiate engine with fake checkpoint (won't be used by patched load_model)
    engine = hfe.HFTradingEngine(checkpoint_path="hftraining/checkpoints/fake.pt", config_path=None, device="cpu")

    # Synthetic data with enough length
    df = _make_ohlcv(days=80)

    signal = engine.generate_signal("TEST", df)
    assert signal is not None
    assert signal.action in {"buy", "hold", "sell"}
    # Our dummy logits bias should choose buy with high confidence
    assert signal.action == "buy"
    assert signal.confidence > 0.7
    # Position size should be positive with positive expected_return
    assert signal.position_size > 0


def test_run_backtest_with_mocked_yfinance(monkeypatch):
    # Allow all trades by bypassing risk manager for this integration test
    monkeypatch.setattr(hfe.RiskManager, "check_risk_limits", lambda *a, **k: True)
    engine = hfe.HFTradingEngine(checkpoint_path="hftraining/checkpoints/fake.pt", config_path=None, device="cpu")

    # Patch yfinance.download used inside hf_trading_engine to return synthetic data
    def _fake_download(symbol, start=None, end=None, progress=False):
        return _make_ohlcv(days=100)

    monkeypatch.setattr(hfe.yf, "download", _fake_download)

    results = engine.run_backtest(symbols=["AAPL"], start_date="2022-01-01", end_date="2022-03-01")

    assert isinstance(results, dict)
    assert "metrics" in results
    assert "equity_curve" in results and len(results["equity_curve"]) > 0
    # With buy-biased dummy, we should have executed some trades
    executed = [t for t in results.get("trades", []) if t.get("status") == "executed"]
    assert len(executed) > 0
