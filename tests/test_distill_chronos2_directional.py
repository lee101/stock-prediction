"""Tests for scripts/distill_chronos2_directional.py.

The full distillation training is a multi-minute job; these tests cover the
pure-python helpers, the model forward, and a tiny end-to-end run on synthetic
panels so CI can guard against regressions in the data pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_compute_simple_features_zero_pad_and_shape():
    from scripts.distill_chronos2_directional import compute_simple_features

    rng = np.random.default_rng(0)
    closes = np.cumsum(rng.standard_normal(120) * 0.5) + 100.0
    highs = closes + 0.5
    lows = closes - 0.5
    vols = np.full_like(closes, 1_000_000.0)
    feats = compute_simple_features(closes, highs, lows, vols, lookback=60)
    assert feats.shape == (120, 8)
    assert feats.dtype == np.float32
    # Pre-lookback rows must be zero (we drop them downstream)
    assert np.all(feats[:60] == 0.0)
    # Post-lookback rows must be finite
    assert np.all(np.isfinite(feats[60:]))


def test_distillee_mlp_forward_shapes():
    from scripts.distill_chronos2_directional import DirectionalDistilleeMLP

    model = DirectionalDistilleeMLP(input_dim=8, hidden=16)
    x = torch.randn(32, 8)
    reg, dir_logit = model(x)
    assert reg.shape == (32,)
    assert dir_logit.shape == (32,)


def _make_synthetic_panel(symbol: str, seed: int):
    """Build a SymbolPanel where the teacher target is a noisy linear function
    of feature[0] — so the student should learn a non-trivial fit.
    """
    from scripts.distill_chronos2_directional import SymbolPanel

    rng = np.random.default_rng(seed)
    T = 200
    F = 8
    feats = rng.standard_normal((T, F)).astype(np.float32) * 0.05
    target = feats[:, 0] * 0.5 + rng.standard_normal(T).astype(np.float32) * 0.02
    realized = feats[:, 0] * 0.3 + rng.standard_normal(T).astype(np.float32) * 0.05
    dates = np.array([np.datetime64("2024-01-01", "D") + np.timedelta64(i, "D") for i in range(T)])
    return SymbolPanel(symbol=symbol, dates=dates, features=feats,
                       target_move=target.astype(np.float32),
                       realized_move=realized.astype(np.float32))


def test_train_student_smoke_drops_loss_and_beats_chance():
    from scripts.distill_chronos2_directional import train_student

    panels = [_make_synthetic_panel(f"SYN{i}", seed=i) for i in range(3)]
    train_end = "2024-06-01"
    result = train_student(
        panels,
        train_end=train_end,
        epochs=20,
        batch_size=64,
        lr=1e-2,
        direction_weight=1.0,
        device="cpu",
        seed=42,
    )
    assert result["status"] == "ok"
    losses = result["loss_per_epoch"]
    assert len(losses) == 20
    # Last 25% of epochs must average lower than first 25% — distillation works.
    quarter = len(losses) // 4
    early = float(np.mean(losses[:quarter]))
    late = float(np.mean(losses[-quarter:]))
    assert late < early, f"loss did not drop: early={early}, late={late}"
    # Distillee should track the teacher's direction better than coin flip.
    assert result["teacher_distillation_accuracy"] > 0.55, (
        f"distillee fails to mimic teacher direction: "
        f"{result['teacher_distillation_accuracy']}"
    )
    # Pearson IC vs teacher should be positive — student is learning the signal.
    assert result["regression_pearson_ic"] > 0.30, (
        f"regression IC too low: {result['regression_pearson_ic']}"
    )


def test_train_student_handles_empty_split():
    """If everything falls into one side of the cutoff, return status=skip without crashing."""
    from scripts.distill_chronos2_directional import train_student

    panels = [_make_synthetic_panel("SYNX", seed=0)]
    # Panel covers 2024-01-01 to 2024-07-18 (~200 days). Cutoff in 2030 → all train, no val.
    result = train_student(panels, train_end="2030-01-01", epochs=2, device="cpu")
    assert result["status"] == "skip"
