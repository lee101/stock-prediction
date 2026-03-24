"""Tests for SAM optimizer variants."""

import math

import pytest
import torch
import torch.nn as nn

from sharpnessadjustedproximalpolicy.sam_optimizer import (
    FullSAMOptimizer,
    LookSAMOptimizer,
    SharpnessAdjustedOptimizer,
    SharpnessState,
)


def _simple_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))


def _loss_fn(model, x, y):
    return nn.functional.mse_loss(model(x), y)


class TestSharpnessAdjustedOptimizer:
    def test_basic_step(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(opt, rho=0.05, probe_every=5)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        for i in range(10):
            sam.zero_grad()
            loss = _loss_fn(model, x, y)
            loss.backward()
            sam.step()

        assert sam.state.step == 10

    def test_sharpness_probe(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(opt, rho=0.05, probe_every=2)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        sam.zero_grad()
        loss = _loss_fn(model, x, y)
        loss.backward()
        sam.step()

        # probe on step 2
        sam.zero_grad()
        loss = _loss_fn(model, x, y)
        loss.backward()
        sam.step()
        assert sam.should_probe()  # step=2, probe_every=2

        params_before = [p.data.clone() for p in model.parameters()]
        def probe_fn():
            return _loss_fn(model, x, y)
        sharpness = sam.probe_sharpness(probe_fn, loss.item())
        params_after = [p.data.clone() for p in model.parameters()]

        # params restored after probe
        for pb, pa in zip(params_before, params_after):
            assert torch.allclose(pb, pa), "params changed after probe"

        assert sharpness >= 0
        assert sam.state.ema > 0

    def test_lr_scaling(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(
            opt, rho=0.05, probe_every=1, target_sharpness=1.0,
            min_scale=0.1, max_scale=5.0,
        )

        x = torch.randn(16, 10)
        y = torch.randn(16, 1)

        for _ in range(5):
            sam.zero_grad()
            loss = _loss_fn(model, x, y)
            loss.backward()
            sam.step()
            def probe_fn():
                return _loss_fn(model, x, y)
            sam.probe_sharpness(probe_fn, loss.item())

        scale = sam.state.lr_scale
        assert 0.1 <= scale <= 5.0
        actual_lr = sam.param_groups[0]["lr"]
        expected_lr = 1e-3 * scale
        assert abs(actual_lr - expected_lr) < 1e-8

    def test_log_scale_mode(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(opt, scale_mode="log", target_sharpness=1.0)
        # test internal scale computation
        assert sam._compute_scale(1.0) == pytest.approx(1.0, abs=0.01)
        assert sam._compute_scale(math.e) == pytest.approx(2.0, abs=0.01)

    def test_update_base_lrs(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(opt, rho=0.05)

        # simulate external lr schedule changing base lr
        for g in opt.param_groups:
            g["lr"] = 5e-4
        sam.update_base_lrs()
        assert sam._base_lrs[0] == pytest.approx(5e-4)


    def test_nan_handling(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(opt, rho=0.05, probe_every=1, warmup_probes=0)

        # probe with NaN base loss should be a no-op
        def probe_fn():
            return torch.tensor(1.0)
        result = sam.probe_sharpness(probe_fn, float("nan"))
        assert sam.state.lr_scale == 1.0  # unchanged

        # probe returning NaN should be a no-op
        def nan_probe():
            return torch.tensor(float("nan"))
        result = sam.probe_sharpness(nan_probe, 1.0)
        assert sam.state.lr_scale == 1.0

    def test_warmup_autocalibrate(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SharpnessAdjustedOptimizer(opt, rho=0.05, probe_every=1, warmup_probes=3)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        for i in range(5):
            sam.zero_grad()
            loss = _loss_fn(model, x, y)
            loss.backward()
            sam.step()
            def probe_fn():
                return _loss_fn(model, x, y)
            sam.probe_sharpness(probe_fn, loss.item())

        # after 3 warmup probes, target_sharpness should be calibrated
        assert sam.target_sharpness > 0
        # after warmup, lr_scale should be active
        assert sam._probe_count >= 3


class TestFullSAMOptimizer:
    def test_basic(self):
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = FullSAMOptimizer(opt, rho=0.05)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        params_init = [p.data.clone() for p in model.parameters()]

        sam.zero_grad()
        loss = _loss_fn(model, x, y)
        loss.backward()

        def loss_fn():
            return _loss_fn(model, x, y)

        pl, sh = sam.step_with_sam(loss_fn)
        assert isinstance(pl, float)
        assert isinstance(sh, float)
        assert sam.state.step == 1

        # params should have changed
        changed = False
        for pi, p in zip(params_init, model.parameters()):
            if not torch.allclose(pi, p.data):
                changed = True
        assert changed

    def test_effective_rho_adapts(self):
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = FullSAMOptimizer(opt, rho=0.05, target_sharpness=1.0, rho_min=0.01, rho_max=0.2)

        # simulate high sharpness
        sam.state.ema = 3.0
        sam.state.step = 1
        eff = sam._effective_rho()
        assert eff > 0.05  # should be larger for high sharpness
        assert eff <= 0.2

        # simulate low sharpness
        sam.state.ema = 0.1
        eff = sam._effective_rho()
        assert eff < 0.05
        assert eff >= 0.01


class TestLookSAMOptimizer:
    def test_basic(self):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = LookSAMOptimizer(opt, rho=0.05, sam_every=3)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        for i in range(6):
            sam.zero_grad()
            loss = _loss_fn(model, x, y)
            loss.backward()
            def loss_fn():
                return _loss_fn(model, x, y)
            sam.step(loss_fn=loss_fn)

        assert sam.state.step == 6

    def test_projected_steps(self):
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = LookSAMOptimizer(opt, rho=0.05, sam_every=5, alpha=0.5)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        # first step is full SAM (step 0 % 5 == 0)
        sam.zero_grad()
        loss = _loss_fn(model, x, y)
        loss.backward()
        sam.step(loss_fn=lambda: _loss_fn(model, x, y))

        # steps 1-4 are projected
        for _ in range(4):
            sam.zero_grad()
            loss = _loss_fn(model, x, y)
            loss.backward()
            sam.step()

        assert sam.state.step == 5


class TestSharpnessState:
    def test_defaults(self):
        s = SharpnessState()
        assert s.raw == 0.0
        assert s.ema == 0.0
        assert s.lr_scale == 1.0
        assert s.step == 0
