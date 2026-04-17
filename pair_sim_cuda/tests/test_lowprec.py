"""bf16 and fp16 correctness + sanity for the fused pair step.

Low-precision outputs compare to the pure-PyTorch reference at the same
dtype (tolerances loosened to match dtype precision). Relying on absolute
tolerances rather than rtol for small values to stay realistic for bf16.
"""

from __future__ import annotations

import pytest
import torch

from pair_sim_cuda.wrapper import (
    PairStepConfig,
    build_extension,
    fused_pair_step,
    fused_pair_step_reference,
)

CUDA = torch.cuda.is_available()


def _make(B: int, P: int, dtype: torch.dtype, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    f = lambda lo, hi: torch.empty(B, P, device="cuda").uniform_(lo, hi, generator=g).to(dtype)
    target_pos = f(-1.0, 1.0)
    offset_bps = f(0.0, 3.0)
    prev_pos = f(-1.0, 1.0)
    pair_ret = f(-0.02, 0.02)
    reach = f(0.0, 20.0)
    hs = f(0.5, 5.0)
    sm = torch.ones(B, device="cuda", dtype=dtype)
    return target_pos, offset_bps, prev_pos, pair_ret, reach, hs, sm


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("dtype,atol", [
    (torch.bfloat16, 0.03),   # bf16: 8-bit mantissa
    (torch.float16, 0.01),    # fp16: 10-bit mantissa
])
def test_forward_lowprec_vs_reference(dtype, atol):
    build_extension(verbose=False)
    cfg = PairStepConfig()
    B, P = 16, 32
    tp, ob, pp, pr, rs, hs, sm = _make(B, P, dtype, seed=11)
    np_f, pn_f, tv_f = fused_pair_step(tp, ob, pp, pr, rs, hs, sm, cfg)
    np_r, pn_r, tv_r = fused_pair_step_reference(tp, ob, pp, pr, rs, hs, sm, cfg)
    assert np_f.dtype == dtype
    torch.testing.assert_close(np_f.float(), np_r.float(), rtol=0, atol=atol)
    torch.testing.assert_close(pn_f.float(), pn_r.float(), rtol=0, atol=atol)
    torch.testing.assert_close(tv_f.float(), tv_r.float(), rtol=0, atol=atol)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_backward_lowprec_runs_and_matches_fp32_shape(dtype):
    build_extension(verbose=False)
    cfg = PairStepConfig()
    B, P = 8, 16
    tp, ob, pp, pr, rs, hs, sm = _make(B, P, dtype, seed=22)
    tp.requires_grad_(True); ob.requires_grad_(True); pp.requires_grad_(True)

    np_f, pn_f, tv_f = fused_pair_step(tp, ob, pp, pr, rs, hs, sm, cfg)
    loss = pn_f.sum().float()
    loss.backward()

    assert tp.grad is not None and tp.grad.dtype == dtype
    assert ob.grad is not None and ob.grad.dtype == dtype
    assert pp.grad is not None and pp.grad.dtype == dtype

    # Compare to fp32 gradients (same inputs cast up).
    tp32 = tp.detach().float().clone().requires_grad_(True)
    ob32 = ob.detach().float().clone().requires_grad_(True)
    pp32 = pp.detach().float().clone().requires_grad_(True)
    rs32 = rs.float(); hs32 = hs.float(); pr32 = pr.float(); sm32 = sm.float()
    np32, pn32, tv32 = fused_pair_step(tp32, ob32, pp32, pr32, rs32, hs32, sm32, cfg)
    pn32.sum().backward()

    atol = 0.05 if dtype == torch.bfloat16 else 0.02
    torch.testing.assert_close(tp.grad.float(), tp32.grad, rtol=0, atol=atol)
    torch.testing.assert_close(ob.grad.float(), ob32.grad, rtol=0, atol=atol)
    torch.testing.assert_close(pp.grad.float(), pp32.grad, rtol=0, atol=atol)
