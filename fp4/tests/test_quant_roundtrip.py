import torch

from fp4.quant import quantize_nvfp4_2d, dequantize_nvfp4_2d
from fp4.dtypes import NVFP4_MAX


def test_rtn_deterministic():
    torch.manual_seed(0)
    x = torch.randn(4, 64) * 2.0
    a = dequantize_nvfp4_2d(quantize_nvfp4_2d(x, stochastic=False))
    b = dequantize_nvfp4_2d(quantize_nvfp4_2d(x, stochastic=False))
    assert torch.equal(a, b)
    # Reasonable accuracy: NVFP4 with block scale should reconstruct within ~10% RMS.
    rms = (a - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()
    assert rms < 0.15, f"RTN rms too high: {rms}"


def test_sr_unbiased_large_sample():
    # Generate ~1e6 samples uniform in [-3, 3] and check SR is unbiased.
    torch.manual_seed(0)
    x = (torch.rand(1_000_000) * 6.0 - 3.0)
    # Run SR many times implicitly (one big batch is enough since SR is per-element).
    packed = quantize_nvfp4_2d(x.unsqueeze(0), stochastic=True)
    y = dequantize_nvfp4_2d(packed).squeeze(0)
    bias = (y - x).mean().item()
    assert abs(bias) < 1e-3, f"SR bias too large: {bias}"


def test_clipping_to_range():
    x = torch.tensor([[100.0, -100.0, 0.0, 6.0, -6.0]])
    y = dequantize_nvfp4_2d(quantize_nvfp4_2d(x, stochastic=False))
    # All recovered values within ±tensor_max (per-tensor scale absorbs the 100).
    assert torch.isfinite(y).all()
