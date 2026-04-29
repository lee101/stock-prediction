"""Smoke tests for the cross-attention transformer (no GPU required)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.cross_attn_transformer_v1 import (
    CrossAttnConfig,
    CrossAttnTransformerV1,
    apply_rope,
    precompute_rope,
)


def test_rope_roundtrip_shape() -> None:
    cos, sin = precompute_rope(seq_len=32, head_dim=16)
    assert cos.shape == (32, 16)
    assert sin.shape == (32, 16)


def test_apply_rope_preserves_shape() -> None:
    B, T, H, hd = 2, 16, 4, 8
    x = torch.randn(B, T, H, hd)
    cos, sin = precompute_rope(seq_len=T, head_dim=hd)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # Magnitude preserved per-token (rotation is orthogonal up to fp error)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_model_forward_shape() -> None:
    cfg = CrossAttnConfig(
        n_features=5, seq_len=8, d_model=32,
        n_temporal_layers=1, n_cross_layers=1, n_heads=4,
        grad_checkpoint=False,
    )
    model = CrossAttnTransformerV1(cfg)
    x = torch.randn(7, 8, 5)
    out = model(x)
    assert out.shape == (7,)
    assert torch.isfinite(out).all().item()


def test_model_with_valid_mask() -> None:
    cfg = CrossAttnConfig(
        n_features=5, seq_len=8, d_model=32,
        n_temporal_layers=1, n_cross_layers=1, n_heads=4,
        grad_checkpoint=False,
    )
    model = CrossAttnTransformerV1(cfg)
    x = torch.randn(10, 8, 5)
    mask = torch.tensor([True] * 6 + [False] * 4)
    out = model(x, valid_mask=mask)
    assert out.shape == (10,)
    # Outputs at masked positions are still finite (head still applies, just
    # the cross-sectional attention ignores them as keys).
    assert torch.isfinite(out).all().item()


def test_model_train_step() -> None:
    cfg = CrossAttnConfig(
        n_features=5, seq_len=8, d_model=32,
        n_temporal_layers=1, n_cross_layers=1, n_heads=4,
        grad_checkpoint=False,
    )
    model = CrossAttnTransformerV1(cfg)
    x = torch.randn(7, 8, 5, requires_grad=False)
    y = torch.randint(0, 2, (7,)).float()
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    grad = next(model.parameters()).grad
    assert grad is not None and torch.isfinite(grad).all().item()


def test_param_count_in_target_range() -> None:
    cfg = CrossAttnConfig(
        n_features=15, seq_len=256, d_model=512,
        n_temporal_layers=6, n_cross_layers=3, n_heads=8,
        grad_checkpoint=False,
    )
    model = CrossAttnTransformerV1(cfg)
    n = model.num_params()
    assert 20_000_000 <= n <= 35_000_000, f"unexpected param count {n}"
