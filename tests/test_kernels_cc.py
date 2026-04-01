"""CC-aware tests for fused MLP Triton kernel.

Tests correctness of the fused_mlp_relu kernel against a sequential
nn.Linear+ReLU+nn.Linear reference, and verifies dtype handling required
for the RL policy (BF16 output must be castable back to policy weight dtype).

Run with:
    pytest tests/test_kernels_cc.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_weights(in_dim, hidden, out_dim, device, dtype=torch.bfloat16):
    """Return (W1, b1, W2, b2) random weights on device."""
    W1 = _allocate_on_device_or_skip(torch.randn, hidden, in_dim, device=device, dtype=dtype)
    b1 = _allocate_on_device_or_skip(torch.zeros, hidden, device=device, dtype=dtype)
    W2 = _allocate_on_device_or_skip(torch.randn, out_dim, hidden, device=device, dtype=dtype)
    b2 = _allocate_on_device_or_skip(torch.zeros, out_dim, device=device, dtype=dtype)
    return W1, b1, W2, b2


def _reference(x, W1, b1, W2, b2):
    """Sequential reference: relu(x @ W1.T + b1) @ W2.T + b2 in F32."""
    x32 = x.float()
    h = F.linear(x32, W1.float(), b1.float())
    h = F.relu(h)
    return F.linear(h, W2.float(), b2.float())


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"Kernel CC test skipped under shared-GPU resource pressure: {exc}")


def _allocate_on_device_or_skip(factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _to_device_or_skip(tensor, *args, **kwargs):
    try:
        return tensor.to(*args, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def cc(device):
    major, minor = torch.cuda.get_device_capability(device)
    return major, minor


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hidden_size", [256, 512, 1024])
@pytest.mark.parametrize("batch_size", [64, 128, 256])
def test_fused_mlp_correctness(device, cc, hidden_size, batch_size):
    """Fused MLP output must match F32 sequential reference within BF16 tolerance."""
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu, HAS_TRITON

    in_dim = 221   # typical RL obs dim
    out_dim = hidden_size

    torch.manual_seed(42)
    x = _allocate_on_device_or_skip(torch.randn, batch_size, in_dim, device=device, dtype=torch.bfloat16)
    W1, b1, W2, b2 = _build_weights(in_dim, hidden_size, out_dim, device)

    # Reference in FP32
    ref = _reference(x, W1, b1, W2, b2)

    if HAS_TRITON:
        out = fused_mlp_relu(x, W1, b1, W2, b2)
        assert out.shape == (batch_size, out_dim), f"Shape mismatch: {out.shape}"
        # BF16 has ~1% relative error; use generous atol since we're checking
        # accumulated matmul results across large hidden dims.
        torch.testing.assert_close(
            out.float(), ref.float(),
            atol=1e-1, rtol=1e-1,
            msg=f"Fused MLP mismatch at CC {cc[0]}.{cc[1]}, hidden={hidden_size}, batch={batch_size}",
        )
    else:
        # Fallback path: verify PyTorch fallback is correct
        out = fused_mlp_relu(x, W1, b1, W2, b2)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_mlp_correctness_small_batch(device, cc):
    """Edge case: batch size 1."""
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu

    torch.manual_seed(7)
    in_dim, hidden, out_dim = 64, 128, 64
    x = _allocate_on_device_or_skip(torch.randn, 1, in_dim, device=device, dtype=torch.bfloat16)
    W1, b1, W2, b2 = _build_weights(in_dim, hidden, out_dim, device)

    ref = _reference(x, W1, b1, W2, b2)
    out = fused_mlp_relu(x, W1, b1, W2, b2)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_mlp_correctness_relu_gating(device):
    """Verify ReLU is actually applied: negative pre-activations should be zeroed."""
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu

    # Craft a W1 that produces all-negative hidden activations for a known x.
    in_dim, hidden, out_dim = 32, 64, 32
    x = _allocate_on_device_or_skip(torch.ones, 4, in_dim, device=device, dtype=torch.bfloat16)
    # W1 all negative -> hidden = x @ W1.T + 0 = all negative -> ReLU -> all zero
    W1 = -_allocate_on_device_or_skip(torch.ones, hidden, in_dim, device=device, dtype=torch.bfloat16)
    b1 = _allocate_on_device_or_skip(torch.zeros, hidden, device=device, dtype=torch.bfloat16)
    W2 = _allocate_on_device_or_skip(torch.randn, out_dim, hidden, device=device, dtype=torch.bfloat16)
    b2 = _allocate_on_device_or_skip(torch.zeros, out_dim, device=device, dtype=torch.bfloat16)

    out = fused_mlp_relu(x, W1, b1, W2, b2)
    # After ReLU kills all hidden activations, output should equal b2 = 0
    assert out.abs().max().item() < 1e-3, f"ReLU not applied correctly; max abs = {out.abs().max().item()}"


# ---------------------------------------------------------------------------
# Dtype handling tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_mlp_dtype_bf16_output(device):
    """Fused kernel must produce BF16 output when Triton is active."""
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu, HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("Triton not available")

    in_dim, hidden, out_dim = 221, 1024, 1024
    x = _allocate_on_device_or_skip(torch.randn, 128, in_dim, device=device, dtype=torch.bfloat16)
    W1, b1, W2, b2 = _build_weights(in_dim, hidden, out_dim, device)

    out = fused_mlp_relu(x, W1, b1, W2, b2)
    assert out.dtype == torch.bfloat16, f"Expected BF16 output, got {out.dtype}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_mlp_dtype_castable_to_policy_weights(device):
    """Output BF16 must be castable to the same dtype as policy encoder weights.

    This exercises the BF16 dtype bug documented in MEMORY.md: fused_mlp output
    must be cast to encoder.weight.dtype before being passed to the next layer.
    """
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu

    in_dim, hidden, out_dim = 221, 1024, 1024
    x = _allocate_on_device_or_skip(torch.randn, 128, in_dim, device=device, dtype=torch.bfloat16)
    W1, b1, W2, b2 = _build_weights(in_dim, hidden, out_dim, device)

    out = fused_mlp_relu(x, W1, b1, W2, b2)

    # Simulate the policy: next layer is BF16
    next_layer = _to_device_or_skip(nn.Linear(out_dim, 512, dtype=torch.bfloat16), device)
    out_cast = _to_device_or_skip(out, next_layer.weight.dtype)
    result = next_layer(out_cast)
    assert result.shape == (128, 512)
    assert not result.isnan().any(), "NaN in next layer output after dtype cast"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_mlp_dtype_float32_input(device):
    """Fused kernel accepts FP32 input (auto-casts to BF16 internally)."""
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu

    in_dim, hidden, out_dim = 64, 256, 64
    x = _allocate_on_device_or_skip(torch.randn, 32, in_dim, device=device, dtype=torch.float32)
    W1 = _allocate_on_device_or_skip(torch.randn, hidden, in_dim, device=device, dtype=torch.bfloat16)
    b1 = _allocate_on_device_or_skip(torch.zeros, hidden, device=device, dtype=torch.bfloat16)
    W2 = _allocate_on_device_or_skip(torch.randn, out_dim, hidden, device=device, dtype=torch.bfloat16)
    b2 = _allocate_on_device_or_skip(torch.zeros, out_dim, device=device, dtype=torch.bfloat16)

    # Should not raise
    out = fused_mlp_relu(x, W1, b1, W2, b2)
    assert out.shape == (32, out_dim)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Shape handling tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_mlp_3d_input(device):
    """Fused MLP accepts 3-D input (..., in_dim) and restores original shape."""
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu

    in_dim, hidden, out_dim = 64, 128, 64
    x = _allocate_on_device_or_skip(torch.randn, 4, 8, in_dim, device=device, dtype=torch.bfloat16)
    W1, b1, W2, b2 = _build_weights(in_dim, hidden, out_dim, device)

    out = fused_mlp_relu(x, W1, b1, W2, b2)
    assert out.shape == (4, 8, out_dim), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# CC detection test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autotune_configs_nonempty(device, cc):
    """_get_autotune_configs() must return at least one config for any GPU."""
    from pufferlib_market.kernels.fused_mlp import _get_autotune_configs
    configs = _get_autotune_configs()
    assert len(configs) >= 1, "No autotune configs returned"
    major, minor = cc
    # Verify each config has the required keys
    required_keys = {"BLOCK_M", "BLOCK_D", "BLOCK_H", "BLOCK_K"}
    for cfg in configs:
        for key in required_keys:
            assert key in cfg.kwargs, f"Config missing key {key}: {cfg.kwargs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autotune_configs_cc_routing(device, cc):
    """Check that configs returned match the expected CC tier."""
    from pufferlib_market.kernels.fused_mlp import _get_autotune_configs

    major, _minor = cc
    configs = _get_autotune_configs()

    if major >= 9:
        # Expect at least one large-tile config (BLOCK_D >= 128)
        large = [c for c in configs if c.kwargs.get("BLOCK_D", 0) >= 128]
        assert large, f"CC {major}: expected large-tile configs, got {[c.kwargs for c in configs]}"
    elif major == 8:
        # Expect at least one mid-large tile (BLOCK_D >= 128)
        mid = [c for c in configs if c.kwargs.get("BLOCK_D", 0) >= 128]
        assert mid, f"CC {major}: expected mid-large-tile configs, got {[c.kwargs for c in configs]}"
    else:
        # Conservative configs: BLOCK_D <= 128
        oversized = [c for c in configs if c.kwargs.get("BLOCK_D", 0) > 128]
        assert not oversized, f"CC {major}: conservative configs expected, got oversized {[c.kwargs for c in oversized]}"
