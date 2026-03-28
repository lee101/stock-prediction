from __future__ import annotations

import math

import torch

from pufferlib_market.train import (
    _backoff_optimizer_lr,
    _classify_update_stability,
    _max_abs_grad,
)


def test_max_abs_grad_ignores_missing_grads() -> None:
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float32))
    p1.grad = torch.tensor([0.25, -0.75], dtype=torch.float32)
    p2.grad = None

    assert _max_abs_grad([p1, p2]) == 0.75


def test_backoff_optimizer_lr_respects_floor() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-4)

    old_lr, new_lr = _backoff_optimizer_lr(optimizer, factor=0.5, min_lr=7.5e-5)

    assert old_lr == 1e-4
    assert new_lr == 7.5e-5
    assert optimizer.param_groups[0]["lr"] == 7.5e-5


def test_classify_update_stability_flags_nonfinite_loss() -> None:
    reason = _classify_update_stability(
        loss_value=float("nan"),
        grad_norm_value=1.0,
        max_abs_grad_value=0.1,
        grad_norm_skip_threshold=100.0,
    )

    assert reason == "nonfinite_loss=nan"


def test_classify_update_stability_flags_nonfinite_grad_norm() -> None:
    reason = _classify_update_stability(
        loss_value=1.0,
        grad_norm_value=float("inf"),
        max_abs_grad_value=0.1,
        grad_norm_skip_threshold=100.0,
    )

    assert reason == "nonfinite_grad_norm=inf"


def test_classify_update_stability_flags_large_grad_norm() -> None:
    reason = _classify_update_stability(
        loss_value=1.0,
        grad_norm_value=250.0,
        max_abs_grad_value=0.1,
        grad_norm_skip_threshold=100.0,
    )

    assert reason is not None
    assert "grad_norm=250.0000 exceeds grad_norm_skip_threshold=100.0000" in reason


def test_classify_update_stability_accepts_finite_update() -> None:
    reason = _classify_update_stability(
        loss_value=1.0,
        grad_norm_value=2.5,
        max_abs_grad_value=0.1,
        grad_norm_skip_threshold=100.0,
    )

    assert reason is None
