"""Correctness tests for the fused CUDA pair-step kernel.

Compares forward outputs and backward gradients against the pure-PyTorch
reference in ``wrapper.fused_pair_step_reference``.
"""

from __future__ import annotations

import math
import pytest
import torch

from pair_sim_cuda.wrapper import (
    PairStepConfig,
    build_extension,
    fused_pair_step,
    fused_pair_step_reference,
    daily_eod_interest,
)

CUDA_AVAILABLE = torch.cuda.is_available()


def _make_inputs(B: int, P: int, seed: int = 0, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    target_pos = torch.empty(B, P, device=device).uniform_(-1.0, 1.0, generator=g)
    offset_bps = torch.empty(B, P, device=device).uniform_(0.0, 3.0, generator=g)
    prev_pos = torch.empty(B, P, device=device).uniform_(-1.0, 1.0, generator=g)
    pair_ret = torch.empty(B, P, device=device).uniform_(-0.02, 0.02, generator=g)
    reach_side = torch.empty(B, P, device=device).uniform_(0.0, 20.0, generator=g)
    half_spread = torch.empty(B, P, device=device).uniform_(0.5, 5.0, generator=g)
    sm = torch.empty(B, device=device).uniform_(0.0, 1.0, generator=g)
    # Binarise session_mask so the tests exercise both on- and off-session:
    session_mask = (sm > 0.3).float()
    return (target_pos, offset_bps, prev_pos, pair_ret, reach_side,
            half_spread, session_mask)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_forward_matches_reference():
    build_extension(verbose=False)
    B, P = 8, 16
    cfg = PairStepConfig(commission_bps=1.0, fill_temp_bps=2.0, fee_bp=5.0)
    tp, ob, pp, pr, rs, hs, sm = _make_inputs(B, P, seed=1)
    for t in (tp, ob, pp):
        t.requires_grad_(False)

    np_f, pn_f, tv_f = fused_pair_step(tp, ob, pp, pr, rs, hs, sm, cfg)
    np_r, pn_r, tv_r = fused_pair_step_reference(tp, ob, pp, pr, rs, hs, sm, cfg)

    torch.testing.assert_close(np_f, np_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pn_f, pn_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tv_f, tv_r, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_backward_matches_reference(seed: int):
    """Same random loss applied to both impls; grads must match."""
    build_extension(verbose=False)
    B, P = 6, 12
    cfg = PairStepConfig(commission_bps=1.0, fill_temp_bps=2.0, fee_bp=5.0)
    tp, ob, pp, pr, rs, hs, sm = _make_inputs(B, P, seed=seed)

    g = torch.Generator(device="cuda").manual_seed(100 + seed)
    w_np = torch.empty(B, P, device="cuda").uniform_(-1, 1, generator=g)
    w_pn = torch.empty(B, P, device="cuda").uniform_(-1, 1, generator=g)
    w_tv = torch.empty(B, P, device="cuda").uniform_(-1, 1, generator=g)

    def run(impl):
        tp_, ob_, pp_ = tp.clone(), ob.clone(), pp.clone()
        for x in (tp_, ob_, pp_):
            x.requires_grad_(True)
        np_, pn_, tv_ = impl(tp_, ob_, pp_, pr, rs, hs, sm, cfg)
        loss = (np_ * w_np + pn_ * w_pn + tv_ * w_tv).sum()
        loss.backward()
        return tp_.grad.detach(), ob_.grad.detach(), pp_.grad.detach()

    tg_f, og_f, pg_f = run(fused_pair_step)
    tg_r, og_r, pg_r = run(fused_pair_step_reference)

    torch.testing.assert_close(tg_f, tg_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(og_f, og_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(pg_f, pg_r, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_gradcheck_small():
    """torch.autograd.gradcheck at fp64 precision on a small shape."""
    build_extension(verbose=False)
    B, P = 2, 3
    cfg = PairStepConfig(commission_bps=1.0, fill_temp_bps=2.0, fee_bp=5.0)

    tp, ob, pp, pr, rs, hs, sm = _make_inputs(B, P, seed=42)
    tp = tp.double().requires_grad_(True)
    ob = ob.double().requires_grad_(True)
    pp = pp.double().requires_grad_(True)

    def pyref(tp_, ob_, pp_):
        return fused_pair_step_reference(
            tp_, ob_, pp_,
            pr.double(), rs.double(), hs.double(), sm.double(), cfg,
        )

    # gradcheck ref in fp64. We only probe the pure-PyTorch path; the CUDA
    # path is tested via the equality check above.
    assert torch.autograd.gradcheck(
        pyref, (tp, ob, pp), eps=1e-6, atol=1e-4, rtol=1e-3,
        check_undefined_grad=False, nondet_tol=0.0,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_eod_interest_two_leverage():
    """Verify EOD interest matches a hand-calculated value."""
    # One batch, two pairs. next_pos sums to 2.0 (leverage = 2x), borrowed = 1.0.
    next_pos = torch.tensor([[1.0, -1.0]], device="cuda")
    charge = daily_eod_interest(next_pos, annual_rate=0.0625, one_x_free=True)
    expected = 1.0 * (0.0625 / 252.0)
    torch.testing.assert_close(charge.item(), expected, rtol=0, atol=1e-8)

    # Leverage under 1.0 → no charge.
    np2 = torch.tensor([[0.3, -0.4]], device="cuda")
    c2 = daily_eod_interest(np2)
    assert c2.item() == 0.0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_session_mask_zero_disables_trade():
    """When session_mask=0 the next_pos must equal prev_pos and turnover=0."""
    build_extension(verbose=False)
    B, P = 3, 5
    cfg = PairStepConfig()
    tp, ob, pp, pr, rs, hs, _ = _make_inputs(B, P, seed=7)
    sm_zero = torch.zeros(B, device="cuda")
    np_f, pn_f, tv_f = fused_pair_step(tp, ob, pp, pr, rs, hs, sm_zero, cfg)
    torch.testing.assert_close(np_f, pp, rtol=0, atol=0)
    torch.testing.assert_close(tv_f, torch.zeros_like(tv_f), rtol=0, atol=0)
    # pair_pnl = prev_pos * pair_ret - 0
    torch.testing.assert_close(pn_f, pp * pr, rtol=1e-6, atol=1e-7)
