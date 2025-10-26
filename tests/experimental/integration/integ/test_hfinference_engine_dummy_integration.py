#!/usr/bin/env python3
"""Integration test for HFTradingEngine using a minimal DummyModel.

This test exercises the code paths where:
- price_predictions are 1D per batch item: shape [B, horizon]
- only action_logits are returned (no action_probs)
- yfinance is patched to provide synthetic OHLCV data

It validates end-to-end signal generation and backtest execution without
depending on real checkpoints or network calls.
"""

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on import path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip if torch is not installed
pytest.importorskip("torch", reason="hfinference engine tests require torch installed")
import torch
import hfinference.hf_trading_engine as hfe


class _DummyModel1D:
    def __init__(self, cfg):
        self.cfg = cfg

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        # x: [B, seq_len, features]
        B = x.shape[0]
        horizon = int(self.cfg.get("prediction_horizon", 5))
        # Positive normalized close to encourage buys when denormalized
        price_preds = torch.full((B, horizon), 0.15, dtype=torch.float32)
        # Strong buy logits (buy/hold/sell)
        action_logits = torch.tensor([[4.0, 0.0, -4.0]], dtype=torch.float32).repeat(B, 1)
        return {
            "price_predictions": price_preds,
            # Intentionally omit action_probs to test logits-only branch
            "action_logits": action_logits,
        }


def _make_synthetic_ohlcv(days=120, start=100.0, drift=0.2, seed=11):
    rng = np.random.RandomState(seed)
    close = start + np.cumsum(rng.randn(days) * 0.5 + drift)
    open_ = close + rng.randn(days) * 0.2
    high = np.maximum(open_, close) + np.abs(rng.randn(days)) * 0.5
    low = np.minimum(open_, close) - np.abs(rng.randn(days)) * 0.5
    vol = rng.randint(1_000_000, 5_000_000, size=days)
    idx = pd.date_range(end=datetime.now(), periods=days, freq="D")
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol
    }, index=idx)


@pytest.fixture(autouse=True)
def patch_engine_deps(monkeypatch):
    # Patch load_model to bypass real checkpoints
    def _fake_load_model(self, checkpoint_path):
        model_cfg = {
            "input_features": 21,
            "sequence_length": 60,
            "prediction_horizon": 5,
        }
        return _DummyModel1D(model_cfg)

    monkeypatch.setattr(hfe.HFTradingEngine, "load_model", _fake_load_model)

    # Patch yfinance.download to synthetic data
    monkeypatch.setattr(hfe.yf, "download", lambda *a, **k: _make_synthetic_ohlcv())

    # Relax risk manager to always allow trades in this integration test
    monkeypatch.setattr(hfe.RiskManager, "check_risk_limits", lambda *a, **k: True)
    yield


def test_generate_signal_logits_only_1d_preds():
    engine = hfe.HFTradingEngine(checkpoint_path="hftraining/checkpoints/fake.pt", device="cpu")
    df = _make_synthetic_ohlcv(days=80)

    sig = engine.generate_signal("DUMMY", df)
    assert sig is not None
    assert sig.action in {"buy", "hold", "sell"}
    # With strong buy logits and positive normalized close, expect buy
    assert sig.action == "buy"
    assert sig.confidence > 0.6
    assert sig.expected_return >= 0
    assert sig.position_size >= 0


def test_run_backtest_end_to_end_with_dummy():
    engine = hfe.HFTradingEngine(checkpoint_path="hftraining/checkpoints/fake.pt", device="cpu")
    results = engine.run_backtest(symbols=["AAPL"], start_date="2022-01-01", end_date="2022-04-01")

    assert isinstance(results, dict)
    assert "metrics" in results
    assert "equity_curve" in results and len(results["equity_curve"]) > 0
    # Should execute some trades given relaxed risk and buy bias
    executed = [t for t in results.get("trades", []) if t.get("status") == "executed"]
    assert len(executed) > 0

