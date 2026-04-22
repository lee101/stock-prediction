"""Unit tests for the Muon optimizer + MuonMLPStockModel.

Ensures:
  - Newton-Schulz produces a near-orthogonal update
  - Muon.step() only works on 2D params
  - MuonMLP fit → save → load → predict_scores is numerically identical
    (same seed → same scores)
  - split_params_muon routes input-proj + head weights to AdamW group
  - registry dispatches mlp_muon family correctly
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _make_toy_frame(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feats = [f"f{i}" for i in range(8)]
    X = rng.standard_normal((n, 8)).astype(np.float32)
    # learnable: target is sign of weighted sum + noise
    w = rng.standard_normal(8).astype(np.float32)
    logits = X @ w + 0.3 * rng.standard_normal(n).astype(np.float32)
    y = (logits > 0).astype(np.int32)
    df = pd.DataFrame(X, columns=feats)
    df["target_oc_up"] = y
    return df, feats


def test_newton_schulz_near_orthogonal():
    from xgbnew.muon import zeropower_via_newtonschulz5

    G = torch.randn(64, 48)
    Y = zeropower_via_newtonschulz5(G.clone(), steps=10).float()
    # Singular values of Y should cluster around 1 (allowing Muon's 0.5-1.5 spread)
    U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
    assert S.min() > 0.3
    assert S.max() < 1.7


def test_muon_rejects_1d_params():
    from xgbnew.muon import Muon

    p = torch.zeros(8, requires_grad=True)
    p.grad = torch.ones(8)
    opt = Muon([p], lr=0.01)
    with pytest.raises(AssertionError):
        opt.step()


def test_muon_step_reduces_loss_on_linreg():
    """Sanity: Muon on a linear layer actually minimizes MSE."""
    from xgbnew.muon import Muon

    torch.manual_seed(0)
    W_true = torch.randn(16, 8)
    X = torch.randn(1024, 16)
    Y = X @ W_true + 0.01 * torch.randn(1024, 8)

    lin = torch.nn.Linear(16, 8, bias=False)
    opt = Muon([lin.weight], lr=0.05, momentum=0.9)
    loss0 = ((lin(X) - Y) ** 2).mean().item()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = ((lin(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    loss1 = ((lin(X) - Y) ** 2).mean().item()
    assert loss1 < loss0 * 0.5


def test_mlp_muon_fit_save_load_roundtrip(tmp_path):
    from xgbnew.model_mlp_muon import MuonMLPStockModel
    from xgbnew.model_registry import load_any_model

    df, feats = _make_toy_frame(n=2000, seed=7)
    model = MuonMLPStockModel(
        device="cpu", hidden_dim=32, n_blocks=2, epochs=3,
        batch_size=512, random_state=42,
    )
    model.fit(df, feature_cols=feats, verbose=False)

    scores_before = model.predict_scores(df).values
    assert scores_before.min() >= 0.0 and scores_before.max() <= 1.0

    out = tmp_path / "muon_mlp_seed42.pkl"
    model.save(out)

    loaded = load_any_model(out)
    assert loaded.family == "mlp_muon"
    scores_after = loaded.predict_scores(df).values
    assert np.allclose(scores_before, scores_after, atol=1e-6)


def test_split_params_routes_head_and_inproj_to_adam():
    from xgbnew.model_mlp_muon import _build_net, _split_params_muon

    net = _build_net(in_dim=8, hidden_dim=32, n_blocks=2, dropout=0.0)
    muon, adam = _split_params_muon(net)
    adam_ids = {id(p) for p in adam}
    muon_ids = {id(p) for p in muon}
    assert id(net.in_proj.weight) in adam_ids
    assert id(net.head.weight) in adam_ids
    # every block's hidden Linear weight should be Muon
    for blk in net.blocks:
        assert id(blk.fc1.weight) in muon_ids
        assert id(blk.fc2.weight) in muon_ids


def test_determinism_same_seed_same_scores():
    from xgbnew.model_mlp_muon import MuonMLPStockModel

    df, feats = _make_toy_frame(n=2000, seed=11)
    kw = dict(device="cpu", hidden_dim=32, n_blocks=2, epochs=3,
              batch_size=512, random_state=13)

    a = MuonMLPStockModel(**kw).fit(df, feature_cols=feats, verbose=False)
    b = MuonMLPStockModel(**kw).fit(df, feature_cols=feats, verbose=False)
    sa = a.predict_scores(df).values
    sb = b.predict_scores(df).values
    assert np.allclose(sa, sb, atol=1e-5)
