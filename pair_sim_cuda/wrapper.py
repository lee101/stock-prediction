"""Fused differentiable daily pair-step.

Model (per pair, per sample):
    trade       = target_pos - prev_pos
    threshold   = fee_bp + half_spread_bps + offset_bps
    signal_bps  = reach_side_bps - threshold
    fill        = session_mask * sigmoid(signal_bps / fill_temp_bps)
    exec        = fill * trade
    next_pos    = prev_pos + exec
    turnover    = |exec|
    cost_frac   = turnover * (commission_bps + half_spread_bps + fee_bp
                              + 0.5 * offset_bps) * 1e-4
    pair_pnl    = next_pos * pair_ret - cost_frac

End-of-day interest (applied in Python, still differentiable):
    leverage_used = sum(|next_pos|)
    borrowed      = max(0, leverage_used - 1.0)
    interest_frac = borrowed * (annual_rate / 252)

All inputs in bps are *bps*, not fractions (i.e. 5bp is 5.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_EXT = None


def build_extension(verbose: bool = False):
    """Compile the CUDA extension in-place via torch.utils.cpp_extension.load.

    Cached across calls. Uses sm_120 (Blackwell/RTX 5090) if detected.
    """
    global _EXT
    if _EXT is not None:
        return _EXT

    import os
    # Torch sometimes picks /usr/local/cuda-12.0 even when only 12.9 is
    # installed. Override with the real install if CUDA_HOME is unset or
    # points to a missing dir.
    cuda_home = os.environ.get("CUDA_HOME", "")
    if not cuda_home or not Path(cuda_home).exists():
        for candidate in ("/usr/local/cuda-12.9", "/usr/local/cuda-12", "/usr/local/cuda"):
            if Path(candidate, "bin", "nvcc").exists():
                os.environ["CUDA_HOME"] = candidate
                break

    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    src = here / "src" / "fused_pair_step.cu"
    assert src.exists(), f"missing source: {src}"

    # Pick a sensible sm for the host device. 5090 is sm_120.
    arch = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
    cuda_flags = ["-O3", "--use_fast_math"]
    if arch is not None:
        cuda_flags += [f"-arch={arch}"]

    build_dir = here / ".build_cache"
    build_dir.mkdir(parents=True, exist_ok=True)

    _EXT = load(
        name="pair_sim_cuda_ext",
        sources=[str(src)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3"],
        verbose=verbose,
        build_directory=str(build_dir),
    )
    return _EXT


@dataclass
class PairStepConfig:
    commission_bps: float = 1.0
    fill_temp_bps: float = 2.0
    fee_bp: float = 5.0  # 5bp exec buffer around the bar


class _FusedPairStep(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target_pos: torch.Tensor,
        offset_bps: torch.Tensor,
        prev_pos: torch.Tensor,
        pair_ret: torch.Tensor,
        reach_side_bps: torch.Tensor,
        half_spread_bps: torch.Tensor,
        session_mask: torch.Tensor,
        commission_bps: float,
        fill_temp_bps: float,
        fee_bp: float,
    ):
        ext = build_extension()
        next_pos, pair_pnl, turnover, fill, sign_exec = ext.pair_step_fwd(
            target_pos.contiguous(),
            offset_bps.contiguous(),
            prev_pos.contiguous(),
            pair_ret.contiguous(),
            reach_side_bps.contiguous(),
            half_spread_bps.contiguous(),
            session_mask.contiguous(),
            float(commission_bps),
            float(fill_temp_bps),
            float(fee_bp),
        )
        ctx.save_for_backward(
            target_pos, offset_bps, prev_pos, pair_ret,
            half_spread_bps, session_mask, fill, sign_exec,
        )
        ctx.commission_bps = float(commission_bps)
        ctx.fill_temp_bps = float(fill_temp_bps)
        ctx.fee_bp = float(fee_bp)
        return next_pos, pair_pnl, turnover

    @staticmethod
    def backward(ctx, g_next_pos, g_pair_pnl, g_turnover):
        ext = build_extension()
        (target_pos, offset_bps, prev_pos, pair_ret,
         half_spread_bps, session_mask, fill, sign_exec) = ctx.saved_tensors
        d_target_pos, d_offset_bps, d_prev_pos = ext.pair_step_bwd(
            g_next_pos.contiguous(),
            g_pair_pnl.contiguous(),
            g_turnover.contiguous(),
            target_pos, offset_bps, prev_pos, pair_ret,
            half_spread_bps, session_mask, fill, sign_exec,
            ctx.commission_bps, ctx.fill_temp_bps, ctx.fee_bp,
        )
        # Order of forward inputs: (target_pos, offset_bps, prev_pos,
        # pair_ret, reach_side_bps, half_spread_bps, session_mask,
        # commission_bps, fill_temp_bps, fee_bp).
        return (
            d_target_pos, d_offset_bps, d_prev_pos,
            None, None, None, None,
            None, None, None,
        )


def fused_pair_step(
    target_pos: torch.Tensor,
    offset_bps: torch.Tensor,
    prev_pos: torch.Tensor,
    pair_ret: torch.Tensor,
    reach_side_bps: torch.Tensor,
    half_spread_bps: torch.Tensor,
    session_mask: torch.Tensor,
    cfg: Optional[PairStepConfig] = None,
):
    """Fused CUDA pair step. Returns (next_pos, pair_pnl, turnover)."""
    cfg = cfg or PairStepConfig()
    return _FusedPairStep.apply(
        target_pos, offset_bps, prev_pos,
        pair_ret, reach_side_bps, half_spread_bps, session_mask,
        cfg.commission_bps, cfg.fill_temp_bps, cfg.fee_bp,
    )


def fused_pair_step_reference(
    target_pos: torch.Tensor,
    offset_bps: torch.Tensor,
    prev_pos: torch.Tensor,
    pair_ret: torch.Tensor,
    reach_side_bps: torch.Tensor,
    half_spread_bps: torch.Tensor,
    session_mask: torch.Tensor,
    cfg: Optional[PairStepConfig] = None,
):
    """Pure-PyTorch reference; used for correctness tests and as a
    fallback when the CUDA extension fails to build."""
    cfg = cfg or PairStepConfig()
    trade = target_pos - prev_pos
    threshold = cfg.fee_bp + half_spread_bps + offset_bps
    signal_bps = reach_side_bps - threshold
    sm = session_mask.view(-1, 1)
    fill = sm * torch.sigmoid(signal_bps / cfg.fill_temp_bps)
    exec_ = fill * trade
    next_pos = prev_pos + exec_
    turnover = exec_.abs()
    c_unit = (cfg.commission_bps + half_spread_bps + cfg.fee_bp
              + 0.5 * offset_bps) * 1e-4
    cost = turnover * c_unit
    pair_pnl = next_pos * pair_ret - cost
    return next_pos, pair_pnl, turnover


def daily_eod_interest(
    next_pos: torch.Tensor,
    annual_rate: float = 0.0625,
    one_x_free: bool = True,
) -> torch.Tensor:
    """Differentiable EOD interest charge.

    leverage_used_b = sum_p |next_pos[b, p]|
    borrowed_b     = max(0, leverage_used_b - 1.0)   (if one_x_free)
                   = leverage_used_b                 (otherwise)
    interest_frac  = borrowed_b * (annual_rate / 252)
    """
    per_day = annual_rate / 252.0
    leverage_used = next_pos.abs().sum(dim=-1)
    borrowed = torch.clamp(leverage_used - 1.0, min=0.0) if one_x_free else leverage_used
    return borrowed * per_day
