"""Shared test helpers for cutechronos preprocessing tests."""

import torch


def make_context(B: int, L: int, nan_frac: float = 0.1, seed: int = 42) -> torch.Tensor:
    """Create a random context tensor with some NaN values sprinkled in."""
    gen = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, L, generator=gen)
    mask = torch.rand(B, L, generator=gen) < nan_frac
    ctx[mask] = float("nan")
    return ctx


def compare_outputs(
    result: tuple,
    ref: tuple,
    atol: float = 1e-5,
    label: str = "",
):
    """Compare preprocessing output against reference, asserting shape and value match."""
    patched, attn, loc, scale = result
    ref_patched, ref_attn, ref_loc, ref_scale = ref

    prefix = f"[{label}] " if label else ""

    # Shapes
    assert patched.shape == ref_patched.shape, (
        f"{prefix}patched shape mismatch: {patched.shape} vs {ref_patched.shape}"
    )
    assert attn.shape == ref_attn.shape, (
        f"{prefix}attn_mask shape mismatch: {attn.shape} vs {ref_attn.shape}"
    )
    assert loc.shape == ref_loc.shape, (
        f"{prefix}loc shape mismatch: {loc.shape} vs {ref_loc.shape}"
    )
    assert scale.shape == ref_scale.shape, (
        f"{prefix}scale shape mismatch: {scale.shape} vs {ref_scale.shape}"
    )

    # Values
    max_err_patched = (patched.float() - ref_patched.float()).abs().max().item()
    max_err_attn = (attn.float() - ref_attn.float()).abs().max().item()
    max_err_loc = (loc.float() - ref_loc.float()).abs().max().item()
    max_err_scale = (scale.float() - ref_scale.float()).abs().max().item()

    assert max_err_loc < atol, f"{prefix}loc max error {max_err_loc} >= {atol}"
    assert max_err_scale < atol, f"{prefix}scale max error {max_err_scale} >= {atol}"
    assert max_err_attn < atol, f"{prefix}attn_mask max error {max_err_attn} >= {atol}"
    assert max_err_patched < atol, (
        f"{prefix}patched max error {max_err_patched} >= {atol}"
    )
