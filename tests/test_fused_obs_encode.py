"""Tests for fused_obs_norm_linear_relu Triton kernel.

Verifies correctness of the fused observation normalization + first linear +
ReLU kernel against a FP32-accumulated reference, and checks dtype/shape
handling for integration with TradingPolicy.

Run with:
    PYTHONPATH=. pytest tests/test_fused_obs_encode.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp32_reference(obs, mean, std, weight, bias, eps=1e-5):
    """FP32-accumulated reference: normalize -> linear -> relu -> cast to weight.dtype."""
    obs_norm = (obs.float() - mean.float()) / (std.float() + eps)
    h = F.linear(obs_norm, weight.float(), bias.float())
    return F.relu(h).to(weight.dtype)


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"Fused obs-encode test skipped under shared-GPU resource pressure: {exc}")


def _allocate_on_device_or_skip(factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _to_device_or_skip(obj, *args, **kwargs):
    try:
        return obj.to(*args, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("B,OBS,H", [
    (32,  209, 1024),   # stocks12 exact dims
    (64,  209, 1024),   # B=64 stocks12
    (256, 209, 1024),   # B=256 stocks12
    (16,  32,  64),     # tiny dims
    (1,   16,  32),     # batch size 1 edge case
    (128, 512, 512),    # larger OBS
])
def test_fused_obs_encode_correctness(device, B, OBS, H):
    """Fused kernel output must match FP32-accumulated reference within BF16 tolerance.

    BF16 output quantization causes up to ~0.5 absolute error at large values
    (O(100) output magnitude at OBS=209, H=1024). Tolerance of 0.5 is appropriate
    for correct FP32-accumulated kernels producing BF16 output.
    """
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu, HAS_TRITON

    torch.manual_seed(42 + B + OBS + H)
    obs = _allocate_on_device_or_skip(torch.randn, B, OBS, device=device, dtype=torch.float32)
    mean = _allocate_on_device_or_skip(torch.randn, OBS, device=device, dtype=torch.float32)
    std = torch.abs(_allocate_on_device_or_skip(torch.randn, OBS, device=device)) + 0.1
    weight = _allocate_on_device_or_skip(torch.randn, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.randn, H, device=device, dtype=torch.float32)

    ref = _fp32_reference(obs, mean, std, weight, bias)

    if HAS_TRITON:
        out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
        assert out.shape == (B, H), f"Shape mismatch: {out.shape}"
        assert out.dtype == torch.bfloat16, f"Expected BF16, got {out.dtype}"
        torch.testing.assert_close(
            out.float(), ref.float(),
            atol=0.5, rtol=0.1,
            msg=f"Fused obs-encode mismatch at B={B}, OBS={OBS}, H={H}",
        )
    else:
        # Fallback should match reference closely (same computation path)
        out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
        assert out.shape == (B, H)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_obs_encode_relu_gating(device):
    """Verify ReLU is applied: all-negative pre-activations should produce zeros."""
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu

    B, OBS, H = 16, 32, 64
    obs = _allocate_on_device_or_skip(torch.ones, B, OBS, device=device, dtype=torch.float32)
    mean = _allocate_on_device_or_skip(torch.zeros, OBS, device=device)
    std = _allocate_on_device_or_skip(torch.ones, OBS, device=device)
    # Negative weights -> linear output is negative -> ReLU -> zero
    weight = -_allocate_on_device_or_skip(torch.ones, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.zeros, H, device=device)

    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    assert out.abs().max().item() < 1e-3, f"ReLU not applied: max abs = {out.abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_obs_encode_normalization(device):
    """Verify normalization (obs - mean) / std is applied correctly.

    Uses zero weight/bias to isolate the normalization from the linear layer.
    After zero linear, output should be all zeros (ReLU of 0 = 0).
    A non-zero mean should shift the obs, changing which outputs are non-zero
    when we use non-zero weights.
    """
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu

    B, OBS, H = 8, 16, 32
    torch.manual_seed(1)
    obs = _allocate_on_device_or_skip(torch.randn, B, OBS, device=device)
    mean = _allocate_on_device_or_skip(torch.randn, OBS, device=device)
    std = torch.abs(_allocate_on_device_or_skip(torch.randn, OBS, device=device)) + 0.5
    weight = _allocate_on_device_or_skip(torch.randn, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.zeros, H, device=device)

    # Manually compute reference
    obs_norm = (obs - mean) / (std + 1e-5)
    ref = F.relu(F.linear(obs_norm, weight.float())).to(torch.bfloat16)

    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    # Use generous tolerance since BF16 quantization + different accumulation order
    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_obs_encode_fallback_matches_triton(device):
    """Fallback and Triton paths must agree within BF16 output tolerance."""
    from pufferlib_market.kernels.fused_obs_encode import (
        fused_obs_norm_linear_relu,
        _fused_obs_norm_linear_relu_fallback,
        HAS_TRITON,
    )

    if not HAS_TRITON:
        pytest.skip("Triton not available -- cannot compare with fallback")

    B, OBS, H = 32, 64, 128
    torch.manual_seed(99)
    obs = _allocate_on_device_or_skip(torch.randn, B, OBS, device=device)
    mean = _allocate_on_device_or_skip(torch.randn, OBS, device=device)
    std = torch.abs(_allocate_on_device_or_skip(torch.randn, OBS, device=device)) + 0.1
    weight = _allocate_on_device_or_skip(torch.randn, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.randn, H, device=device)

    out_triton   = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    out_fallback = _fused_obs_norm_linear_relu_fallback(obs, mean, std, weight, bias, eps=1e-5)

    # Both are BF16; should agree very closely since both accumulate in FP32
    torch.testing.assert_close(out_triton.float(), out_fallback.float(), atol=0.5, rtol=0.1)


# ---------------------------------------------------------------------------
# Dtype and shape tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_obs_encode_output_dtype(device):
    """Kernel must produce BF16 output when Triton is active."""
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu, HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton not available")

    B, OBS, H = 32, 209, 1024
    obs = _allocate_on_device_or_skip(torch.randn, B, OBS, device=device)
    mean = _allocate_on_device_or_skip(torch.zeros, OBS, device=device)
    std = _allocate_on_device_or_skip(torch.ones, OBS, device=device)
    weight = _allocate_on_device_or_skip(torch.randn, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.zeros, H, device=device)

    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    assert out.dtype == torch.bfloat16, f"Expected BF16, got {out.dtype}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_obs_encode_3d_input(device):
    """Accepts 3-D input (..., OBS) and restores original batch shape."""
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu

    B1, B2, OBS, H = 4, 8, 32, 64
    obs = _allocate_on_device_or_skip(torch.randn, B1, B2, OBS, device=device)
    mean = _allocate_on_device_or_skip(torch.zeros, OBS, device=device)
    std = _allocate_on_device_or_skip(torch.ones, OBS, device=device)
    weight = _allocate_on_device_or_skip(torch.randn, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.zeros, H, device=device)

    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    assert out.shape == (B1, B2, H), f"Unexpected shape: {out.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_obs_encode_no_nan(device):
    """Output must not contain NaN or Inf for normal inputs."""
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu

    B, OBS, H = 64, 209, 1024
    torch.manual_seed(7)
    obs = _allocate_on_device_or_skip(torch.randn, B, OBS, device=device)
    mean = _allocate_on_device_or_skip(torch.randn, OBS, device=device)
    std = torch.abs(_allocate_on_device_or_skip(torch.randn, OBS, device=device)) + 1e-3
    weight = _allocate_on_device_or_skip(torch.randn, H, OBS, device=device, dtype=torch.bfloat16)
    bias = _allocate_on_device_or_skip(torch.randn, H, device=device)

    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    assert not out.isnan().any(), "NaN in output"
    assert not out.isinf().any(), "Inf in output"


# ---------------------------------------------------------------------------
# Integration test: TradingPolicy with set_obs_norm_stats
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trading_policy_fused_obs_encode(device):
    """TradingPolicy._encode uses fused obs-encode when obs_mean/obs_std are set."""
    from pufferlib_market.train import TradingPolicy

    obs_size, hidden, num_actions = 209, 1024, 50
    policy = _to_device_or_skip(TradingPolicy(obs_size, num_actions, hidden=hidden), device)
    policy.eval()

    # Before setting obs_norm stats: standard path
    obs = _allocate_on_device_or_skip(torch.randn, 32, obs_size, device=device)
    with torch.no_grad():
        h_standard = policy._encode(obs)
    assert h_standard.shape == (32, hidden)

    # Set obs norm stats
    mean_np = np.zeros(obs_size, dtype=np.float32)
    std_np  = np.ones(obs_size, dtype=np.float32)
    policy.set_obs_norm_stats(mean_np, std_np)

    assert policy.obs_mean is not None
    assert policy.obs_std is not None
    assert policy.obs_mean.device.type == "cuda"

    # After setting: fused path (when HAS_TRITON)
    with torch.no_grad():
        h_fused = policy._encode(obs)
    assert h_fused.shape == (32, hidden)
    assert not h_fused.isnan().any()

    # _encode always returns a tensor with the actor/critic-compatible dtype.
    # The fused obs-encode kernel produces BF16 internally, but the policy casts
    # the encoder output to match the actor's weight dtype (typically FP32).
    assert h_fused.dtype == policy.actor[0].weight.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trading_policy_fused_obs_encode_forward(device):
    """TradingPolicy.forward works end-to-end with fused obs-encode active."""
    from pufferlib_market.train import TradingPolicy

    obs_size, hidden, num_actions = 209, 512, 50
    policy = _to_device_or_skip(TradingPolicy(obs_size, num_actions, hidden=hidden), device)
    policy.eval()

    mean_np = np.random.randn(obs_size).astype(np.float32)
    std_np  = np.abs(np.random.randn(obs_size).astype(np.float32)) + 0.1
    policy.set_obs_norm_stats(mean_np, std_np)

    obs = _allocate_on_device_or_skip(torch.randn, 64, obs_size, device=device)
    with torch.no_grad():
        logits, value = policy(obs)

    assert logits.shape == (64, num_actions)
    assert value.shape == (64,)
    assert not logits.isnan().any()
    assert not value.isnan().any()
