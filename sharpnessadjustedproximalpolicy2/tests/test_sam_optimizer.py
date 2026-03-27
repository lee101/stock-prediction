"""Tests for directional sharpness-aware optimizer variants."""

import pytest
import torch
import torch.nn as nn

from sharpnessadjustedproximalpolicy2.sam_optimizer import (
    DirectionalSharpnessProximalOptimizer,
    FullSAMOptimizer,
    LookSAMOptimizer,
    SharpnessState,
)


class ScalarModel(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self):
        return self.weight


def _quadratic_loss(model: ScalarModel) -> torch.Tensor:
    return 0.5 * model.weight.pow(2).sum()


def _simple_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))


def _mse_loss(model, x, y):
    return nn.functional.mse_loss(model(x), y)


def test_sharpness_state_lr_scale_alias():
    state = SharpnessState(step_scale=0.8)
    assert state.lr_scale == pytest.approx(0.8)
    state.lr_scale = 1.1
    assert state.step_scale == pytest.approx(1.1)


def test_directional_proximal_shrinks_when_candidate_is_sharper(monkeypatch):
    model = ScalarModel(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sam = DirectionalSharpnessProximalOptimizer(
        opt,
        rho=0.05,
        probe_every=1,
        warmup_probes=0,
        min_scale=0.3,
        max_scale=1.2,
    )

    curvature_values = iter([1.0, 4.0])
    monkeypatch.setattr(sam, "_directional_curvature", lambda *args, **kwargs: next(curvature_values))
    monkeypatch.setattr(sam, "_evaluate_loss", lambda loss_fn: 0.75)

    loss = _quadratic_loss(model)
    sam.zero_grad()
    loss.backward()
    sam.step(loss_fn=lambda: _quadratic_loss(model), base_loss=loss.item())

    assert sam.state.source_raw == pytest.approx(1.0)
    assert sam.state.raw == pytest.approx(4.0)
    assert sam.state.loss_delta == pytest.approx(0.25)
    assert sam.state.step_scale == pytest.approx(0.3)
    assert model.weight.item() == pytest.approx(0.7, abs=1e-6)


def test_directional_proximal_expands_when_candidate_is_flatter(monkeypatch):
    model = ScalarModel(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sam = DirectionalSharpnessProximalOptimizer(
        opt,
        rho=0.05,
        probe_every=1,
        warmup_probes=0,
        min_scale=0.3,
        max_scale=1.2,
        flat_bonus=0.2,
    )

    curvature_values = iter([4.0, 1.0])
    monkeypatch.setattr(sam, "_directional_curvature", lambda *args, **kwargs: next(curvature_values))
    monkeypatch.setattr(sam, "_evaluate_loss", lambda loss_fn: 0.0)

    loss = _quadratic_loss(model)
    sam.zero_grad()
    loss.backward()
    sam.step(loss_fn=lambda: _quadratic_loss(model), base_loss=loss.item())

    assert sam.state.step_scale == pytest.approx(1.2)
    assert model.weight.item() == pytest.approx(-0.2, abs=1e-6)


def test_non_probe_step_reuses_previous_step_scale():
    model = ScalarModel(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sam = DirectionalSharpnessProximalOptimizer(opt, rho=0.05, probe_every=2, warmup_probes=0)
    sam.state.step_scale = 0.5

    loss = _quadratic_loss(model)
    sam.zero_grad()
    loss.backward()
    sam.step(loss_fn=lambda: _quadratic_loss(model), base_loss=loss.item())

    assert sam.state.step == 1
    assert sam.state.step_scale == pytest.approx(0.5)
    assert model.weight.item() == pytest.approx(0.5, abs=1e-6)


def test_nan_base_loss_keeps_existing_scale():
    model = ScalarModel(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sam = DirectionalSharpnessProximalOptimizer(opt, rho=0.05, probe_every=1, warmup_probes=0)
    sam.state.step_scale = 0.75

    loss = _quadratic_loss(model)
    sam.zero_grad()
    loss.backward()
    sam.step(loss_fn=lambda: _quadratic_loss(model), base_loss=float("nan"))

    assert sam.state.raw == 0.0
    assert sam.state.step_scale == pytest.approx(0.75)
    assert model.weight.item() == pytest.approx(0.25, abs=1e-6)


def test_update_base_lrs_tracks_external_schedule():
    model = ScalarModel(1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sam = DirectionalSharpnessProximalOptimizer(opt, rho=0.05)

    for group in opt.param_groups:
        group["lr"] = 5e-4
    sam.update_base_lrs()
    assert sam._base_lrs[0] == pytest.approx(5e-4)


def test_full_sam_sharpness_uses_loss_difference_not_absolute_loss():
    model = ScalarModel(0.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sam = FullSAMOptimizer(opt, rho=0.05, adaptive=False)

    def loss_fn():
        return (model.weight - 1.0).pow(2).sum() + 100.0

    sam.zero_grad()
    base_loss = loss_fn()
    base_loss.backward()
    perturbed_val, sharpness = sam.step_with_sam(loss_fn, base_loss=base_loss.item())

    expected = abs(perturbed_val - base_loss.item()) / 0.05
    assert sharpness == pytest.approx(expected, rel=1e-5)
    assert sharpness < 100.0
    assert sam.state.step == 1


def test_looksam_basic_step_count():
    model = _simple_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sam = LookSAMOptimizer(opt, rho=0.05, sam_every=3)

    x = torch.randn(8, 10)
    y = torch.randn(8, 1)

    for _ in range(6):
        sam.zero_grad()
        loss = _mse_loss(model, x, y)
        loss.backward()
        sam.step(loss_fn=lambda: _mse_loss(model, x, y))

    assert sam.state.step == 6
