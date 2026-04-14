"""Tests for src/meta_selector.py (daily stock meta-selector)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.meta_selector import MetaSelector, MetaSignal, _mask_shorts


class TestMaskShorts:
    def test_masks_short_actions(self):
        logits = torch.zeros(1, 25)  # 0 + 12 longs + 12 shorts
        masked = _mask_shorts(logits, 12)
        assert masked[0, 0].item() == 0.0
        for i in range(1, 13):
            assert masked[0, i].item() == 0.0
        for i in range(13, 25):
            assert masked[0, i].item() < -1e30

    def test_preserves_long_logits(self):
        logits = torch.randn(1, 7)  # 0 + 3 longs + 3 shorts
        orig = logits.clone()
        masked = _mask_shorts(logits, 3)
        assert torch.allclose(masked[0, :4], orig[0, :4])


def _make_checkpoint(path: Path, n_sym: int = 12, hidden: int = 64):
    obs_size = n_sym * 16 + 5 + n_sym
    num_actions = 1 + 2 * n_sym
    state = {
        "encoder.0.weight": torch.randn(hidden, obs_size),
        "encoder.0.bias": torch.zeros(hidden),
        "encoder.2.weight": torch.randn(hidden, hidden),
        "encoder.2.bias": torch.zeros(hidden),
        "encoder.4.weight": torch.randn(hidden, hidden),
        "encoder.4.bias": torch.zeros(hidden),
        "actor.0.weight": torch.randn(hidden // 2, hidden),
        "actor.0.bias": torch.zeros(hidden // 2),
        "actor.2.weight": torch.randn(num_actions, hidden // 2),
        "actor.2.bias": torch.zeros(num_actions),
        "critic.0.weight": torch.randn(hidden // 2, hidden),
        "critic.0.bias": torch.zeros(hidden // 2),
        "critic.2.weight": torch.randn(1, hidden // 2),
        "critic.2.bias": torch.zeros(1),
        "encoder_norm.weight": torch.ones(hidden),
        "encoder_norm.bias": torch.zeros(hidden),
    }
    torch.save({"model": state, "arch": "mlp", "use_encoder_norm": True}, path)


class TestMetaSelector:
    @pytest.fixture
    def checkpoints(self, tmp_path):
        paths = []
        for name in ["model_a", "model_b", "model_c"]:
            cp = tmp_path / f"{name}.pt"
            _make_checkpoint(cp)
            paths.append(cp)
        return paths

    def test_init(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=2, lookback=5)
        assert len(sel.policies) == 3
        assert len(sel.names) == 3
        assert sel.top_k == 2

    def test_signal_returns_correct_k(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=2, lookback=5)
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 + i for i in range(12)}
        sig = sel.get_meta_signal(features, prices)
        assert isinstance(sig, MetaSignal)
        assert len(sig.selected_models) == 2
        assert len(sig.selected_symbols) == 2
        assert len(sig.confidences) == 2

    def test_momentum_prefers_winning_model(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=1, lookback=3)
        sel.model_equity["model_a"] = [10000, 10100, 10200, 10500, 10800]
        sel.model_equity["model_b"] = [10000, 9900, 9800, 9700, 9600]
        sel.model_equity["model_c"] = [10000, 10050, 10080, 10100, 10120]
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 for i in range(12)}
        sig = sel.get_meta_signal(features, prices)
        assert "model_a" in sig.selected_models

    def test_model_returns_populated(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=2, lookback=5)
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 for i in range(12)}
        sig = sel.get_meta_signal(features, prices)
        assert len(sig.model_returns) == 3
        for v in sig.model_returns.values():
            assert isinstance(v, float)

    def test_multiple_calls_build_equity(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=1, lookback=3)
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 for i in range(12)}
        for _ in range(10):
            sel.get_meta_signal(features, prices)
        for name in sel.names:
            assert len(sel.model_equity[name]) >= 1

    def test_state_persistence(self, checkpoints, tmp_path):
        state_file = tmp_path / "meta_state.json"
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=1, lookback=3, state_path=state_file)
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 for i in range(12)}
        for _ in range(5):
            sel.get_meta_signal(features, prices)
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["day_count"] == 5
        assert len(data["equity"]) == 3

        # Reload and verify state restored
        sel2 = MetaSelector(checkpoints, symbols, top_k=1, lookback=3, state_path=state_file)
        assert sel2._day_count == 5
        for name in sel2.names:
            assert len(sel2.model_equity[name]) == len(sel.model_equity[name])

    def test_drawdown_filter_skips_model_in_drawdown(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=1, lookback=3, max_drawdown_filter=0.05)
        # model_a: in 10% drawdown (peak 11000, current 9900)
        sel.model_equity["model_a"] = [10000, 10500, 11000, 10500, 10000, 9900]
        # model_b: small drawdown, still within 5%
        sel.model_equity["model_b"] = [10000, 10100, 10200, 10300, 10250, 10200]
        # model_c: losing badly
        sel.model_equity["model_c"] = [10000, 9500, 9000, 8500, 8000, 7500]
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 for i in range(12)}
        sig = sel.get_meta_signal(features, prices)
        # model_b should be selected (not in drawdown, model_a is in >5% DD)
        assert "model_b" in sig.selected_models

    def test_drawdown_filter_fallback_when_all_filtered(self, checkpoints):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=1, lookback=3, max_drawdown_filter=0.01)
        # All models in significant drawdown
        sel.model_equity["model_a"] = [10000, 10500, 10000, 9500, 9000]
        sel.model_equity["model_b"] = [10000, 10300, 9800, 9500, 9200]
        sel.model_equity["model_c"] = [10000, 10200, 9700, 9300, 8900]
        features = np.random.randn(12, 16).astype(np.float32)
        prices = {f"SYM{i}": 100.0 for i in range(12)}
        sig = sel.get_meta_signal(features, prices)
        # Should fallback to unfiltered selection (select best momentum)
        assert len(sig.selected_models) == 1

    def test_warmup_skips_if_already_run(self, checkpoints, tmp_path):
        symbols = [f"SYM{i}" for i in range(12)]
        sel = MetaSelector(checkpoints, symbols, top_k=1, lookback=3)
        sel._day_count = 10
        n_bars = 50
        frames = {}
        for sym in symbols:
            frames[sym] = _make_dummy_frame(n_bars)
        sel.warmup_from_frames(frames)
        assert sel._day_count == 10  # unchanged


def _make_dummy_frame(n_days: int) -> "pd.DataFrame":
    import pandas as pd
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.randint(1000, 10000, n_days).astype(float),
    })
