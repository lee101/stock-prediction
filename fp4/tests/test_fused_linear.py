"""Tests for FusedLinearGELU."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fp4.fused_linear import FusedLinearGELU


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def test_forward_matches_unfused_fp32():
    _seed(0)
    fused = FusedLinearGELU(16, 8).double()
    x = torch.randn(4, 16, dtype=torch.float64, requires_grad=False)
    y_fused = fused(x)
    y_ref = F.gelu(F.linear(x, fused.weight, fused.bias))
    assert torch.allclose(y_fused, y_ref, atol=1e-12, rtol=1e-12)


def test_backward_matches_unfused_fp32():
    _seed(1)
    fused = FusedLinearGELU(12, 7)
    # Reference module sharing the same params.
    w = fused.weight.detach().clone().requires_grad_(True)
    b = fused.bias.detach().clone().requires_grad_(True)

    x1 = torch.randn(5, 12, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = fused(x1)
    y2 = F.gelu(F.linear(x2, w, b))

    g = torch.randn_like(y1)
    y1.backward(g)
    y2.backward(g)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused.weight.grad, w.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused.bias.grad, b.grad, atol=1e-5, rtol=1e-5)


def test_gradcheck_small():
    _seed(2)
    fused = FusedLinearGELU(4, 3).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    # gradcheck against the autograd Function.
    assert torch.autograd.gradcheck(
        lambda inp: fused(inp), (x,), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_no_bias():
    _seed(3)
    fused = FusedLinearGELU(8, 5, bias=False)
    x = torch.randn(3, 8)
    y = fused(x)
    ref = F.gelu(F.linear(x, fused.weight, None))
    assert torch.allclose(y, ref, atol=1e-6)


def test_policy_fuse_gelu_flag():
    """TwoLayerPolicy with fuse_gelu=True should run and match unfused outputs."""
    from fp4.policy_two_layer import TwoLayerPolicy

    _seed(4)
    obs_dim = 10
    p_unfused = TwoLayerPolicy(obs_dim=obs_dim, precision="fp32", seed=7, fuse_gelu=False)
    _seed(4)
    p_fused = TwoLayerPolicy(obs_dim=obs_dim, precision="fp32", seed=7, fuse_gelu=True)

    # Copy params so the two policies are numerically identical.
    p_fused.load_state_dict(p_unfused.state_dict(), strict=False)
    # Also copy encoder weights/bias from nn.Linear -> FusedLinearGELU.
    for name in ("l1", "l2", "l3"):
        src = getattr(p_unfused.encoder, name)
        dst = getattr(p_fused.encoder, name)
        with torch.no_grad():
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

    x = torch.randn(2, obs_dim)
    out_u = p_unfused(x)
    out_f = p_fused(x)
    for k in ("layer_a", "layer_b", "value"):
        assert torch.allclose(out_u[k], out_f[k], atol=1e-5, rtol=1e-5), k
