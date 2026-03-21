"""
Tests for pufferlib_market/muon.py — Muon optimizer and Newton-Schulz helper.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from pufferlib_market.muon import Muon, make_muon_optimizer, zeropower_via_newtonschulz5


# ── zeropower_via_newtonschulz5 ──────────────────────────────────────


def test_ns_output_shape_square():
    G = torch.randn(32, 32)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape


def test_ns_output_shape_tall():
    G = torch.randn(64, 32)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape


def test_ns_output_shape_wide():
    G = torch.randn(32, 64)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape


def test_ns_preserves_dtype_float32():
    G = torch.randn(16, 16, dtype=torch.float32)
    X = zeropower_via_newtonschulz5(G)
    assert X.dtype == torch.float32


def test_ns_preserves_dtype_float64():
    G = torch.randn(16, 16, dtype=torch.float64)
    X = zeropower_via_newtonschulz5(G)
    assert X.dtype == torch.float64


def test_ns_flattens_singular_spectrum_square():
    """
    NS orthogonalisation equalises singular values (flat spectrum).
    It does NOT produce a unitary matrix (X @ X.T != I in general).
    The relative std of singular values should decrease significantly.
    """
    torch.manual_seed(0)
    G = torch.randn(64, 64)
    X = zeropower_via_newtonschulz5(G, steps=5)
    S_in = torch.linalg.svdvals(G.float())
    S_out = torch.linalg.svdvals(X.float())
    sv_std_in = (S_in.std() / S_in.mean()).item()
    sv_std_out = (S_out.std() / S_out.mean()).item()
    # NS should reduce singular-value spread by at least 2x
    assert sv_std_out < sv_std_in / 2, (
        f"NS did not flatten spectrum: relative_std {sv_std_in:.4f} -> {sv_std_out:.4f}"
    )


def test_ns_flattens_singular_spectrum_tall():
    """For a tall matrix (rows>cols), NS should still equalise singular values."""
    torch.manual_seed(1)
    G = torch.randn(64, 32)
    X = zeropower_via_newtonschulz5(G, steps=5)
    S_in = torch.linalg.svdvals(G.float())
    S_out = torch.linalg.svdvals(X.float())
    sv_std_in = (S_in.std() / S_in.mean()).item()
    sv_std_out = (S_out.std() / S_out.mean()).item()
    assert sv_std_out < sv_std_in / 2, (
        f"NS did not flatten spectrum: relative_std {sv_std_in:.4f} -> {sv_std_out:.4f}"
    )


def test_ns_no_nan_inf():
    torch.manual_seed(2)
    G = torch.randn(48, 48)
    X = zeropower_via_newtonschulz5(G)
    assert torch.isfinite(X).all()


def test_ns_requires_2d():
    with pytest.raises(AssertionError):
        zeropower_via_newtonschulz5(torch.randn(16))


# ── Muon instantiation ────────────────────────────────────────────────


def test_muon_instantiate_basic():
    model = nn.Linear(32, 16)
    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    opt = Muon(matrix_params, lr=0.02)
    assert opt is not None


def test_muon_instantiate_with_adamw_fallback():
    model = nn.Linear(32, 16)
    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    opt = Muon(matrix_params, lr=0.02, adamw_params=scalar_params, adamw_lr=3e-4)
    assert opt._adamw is not None


def test_muon_no_adamw_when_no_scalar_params():
    model = nn.Linear(32, 16, bias=False)
    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    opt = Muon(matrix_params, lr=0.02)
    assert opt._adamw is None


# ── make_muon_optimizer ───────────────────────────────────────────────


def test_make_muon_optimizer_splits_params():
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.LayerNorm(32),
        nn.Linear(32, 8),
    )
    opt = make_muon_optimizer(model, muon_lr=0.02, adamw_lr=3e-4)
    assert isinstance(opt, Muon)
    # 1D params (biases + LayerNorm weight/bias) go to adamw
    assert opt._adamw is not None


def test_make_muon_optimizer_bias_free():
    model = nn.Linear(16, 8, bias=False)
    opt = make_muon_optimizer(model)
    # No 1D params → no adamw
    assert opt._adamw is None


# ── Training step ─────────────────────────────────────────────────────


def _simple_step(opt, model, x):
    """Run one forward/backward/step cycle, return loss value."""
    opt.zero_grad()
    loss = model(x).mean()
    loss.backward()
    opt.step()
    return loss.item()


def test_muon_step_no_crash():
    torch.manual_seed(42)
    model = nn.Linear(64, 32)
    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    opt = Muon(matrix_params, lr=0.02, adamw_params=scalar_params, adamw_lr=3e-4)
    x = torch.randn(8, 64)
    _simple_step(opt, model, x)


def test_muon_parameters_change_after_step():
    torch.manual_seed(0)
    model = nn.Linear(32, 16)
    w_before = model.weight.data.clone()
    b_before = model.bias.data.clone()

    opt = make_muon_optimizer(model)
    x = torch.randn(4, 32)
    _simple_step(opt, model, x)

    assert not torch.equal(model.weight.data, w_before), "Weight unchanged after step"
    assert not torch.equal(model.bias.data, b_before), "Bias unchanged after step"


def test_muon_step_no_nan():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = make_muon_optimizer(model)
    x = torch.randn(16, 32)
    _simple_step(opt, model, x)
    for p in model.parameters():
        assert torch.isfinite(p.data).all(), f"NaN/Inf in param after step"


def test_muon_multiple_steps():
    torch.manual_seed(7)
    model = nn.Linear(16, 8)
    opt = make_muon_optimizer(model, muon_lr=0.02, adamw_lr=3e-4)
    x = torch.randn(4, 16)
    for _ in range(10):
        _simple_step(opt, model, x)
    for p in model.parameters():
        assert torch.isfinite(p.data).all()


def test_muon_gradients_flow():
    """Ensure .grad is populated on all parameters."""
    torch.manual_seed(3)
    model = nn.Linear(16, 8)
    opt = make_muon_optimizer(model)
    x = torch.randn(4, 16)
    opt.zero_grad()
    loss = model(x).mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None, "Gradient not populated"


# ── Parameter separation (2D vs 1D) ──────────────────────────────────


def test_param_separation_correctness():
    model = nn.Sequential(
        nn.Linear(8, 16),  # weight 2D, bias 1D
        nn.LayerNorm(16),  # weight 1D, bias 1D
        nn.Linear(16, 4),  # weight 2D, bias 1D
    )
    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]

    # 2 Linear weights = 2D
    assert len(matrix_params) == 2
    # 2 Linear biases + 2 LayerNorm params = 4 scalar
    assert len(scalar_params) == 4


def test_make_muon_applies_adamw_to_scalars():
    """Scalar params should receive AdamW updates (not Muon SGD)."""
    torch.manual_seed(5)
    model = nn.Linear(16, 8)  # weight 2D, bias 1D
    opt = make_muon_optimizer(model, muon_lr=0.02, adamw_lr=1e-2)

    b_before = model.bias.data.clone()
    x = torch.randn(4, 16)
    _simple_step(opt, model, x)

    # Bias should have changed because AdamW ran
    assert not torch.equal(model.bias.data, b_before)


# ── CLI argument acceptance ───────────────────────────────────────────


def test_optimizer_arg_accepted():
    """Verify argparse accepts --optimizer muon without error."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.train",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert "--optimizer" in result.stdout, "--optimizer not found in --help output"
    assert "muon" in result.stdout, "'muon' not listed as optimizer choice"


# ── E2E smoke-test using make_muon_optimizer ─────────────────────────


def test_e2e_muon_convergence_toy():
    """
    Toy regression: Muon should drive loss down over 50 steps on a simple
    linear regression task.
    """
    torch.manual_seed(99)
    model = nn.Linear(8, 1)
    opt = make_muon_optimizer(model, muon_lr=0.02, adamw_lr=1e-2)

    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 1)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss did not decrease over 50 steps"
