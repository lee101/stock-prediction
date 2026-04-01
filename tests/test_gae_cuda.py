"""Tests for GPU-parallel GAE kernel vs CPU reference."""

import pytest
import torch

from pufferlib_market.gae_cuda import compute_gae_gpu, _compute_gae_cpu, HAS_TRITON


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"GAE CUDA test skipped under shared-GPU resource pressure: {exc}")


def _cuda_or_skip(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return tensor.cuda()
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _compute_gae_gpu_or_skip(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    try:
        return compute_gae_gpu(
            _cuda_or_skip(rewards),
            _cuda_or_skip(values),
            _cuda_or_skip(dones),
            _cuda_or_skip(next_value),
            gamma,
            gae_lambda,
        )
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _ref_gae_sequential(rewards, values, dones, next_value, gamma, gae_lambda):
    """Scalar sequential reference -- guaranteed correct."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    for env in range(N):
        last_gae = 0.0
        for t in reversed(range(T)):
            nv = next_value[env].item() if t == T - 1 else values[t + 1, env].item()
            nd = 1.0 - dones[t, env].item()
            r = rewards[t, env].item()
            v = values[t, env].item()
            delta = r + gamma * nv * nd - v
            last_gae = delta + gamma * gae_lambda * nd * last_gae
            advantages[t, env] = last_gae
    returns = advantages + values
    return advantages, returns


@pytest.mark.parametrize("T,N", [(256, 128), (256, 64), (128, 32), (16, 4), (1, 1)])
def test_cpu_matches_reference(T, N):
    torch.manual_seed(42)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = (torch.rand(T, N) > 0.95).float()
    next_value = torch.randn(N)
    gamma, lam = 0.99, 0.95

    adv_ref, ret_ref = _ref_gae_sequential(rewards, values, dones, next_value, gamma, lam)
    adv_cpu, ret_cpu = _compute_gae_cpu(rewards, values, dones, next_value, gamma, lam)

    torch.testing.assert_close(adv_cpu, adv_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ret_cpu, ret_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("T,N", [(256, 128), (256, 64), (128, 32), (16, 4), (1, 1), (512, 256)])
def test_gpu_matches_reference(T, N):
    torch.manual_seed(42)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = (torch.rand(T, N) > 0.95).float()
    next_value = torch.randn(N)
    gamma, lam = 0.99, 0.95

    adv_ref, ret_ref = _ref_gae_sequential(rewards, values, dones, next_value, gamma, lam)

    adv_gpu, ret_gpu = _compute_gae_gpu_or_skip(rewards, values, dones, next_value, gamma, lam)

    torch.testing.assert_close(adv_gpu.cpu(), adv_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ret_gpu.cpu(), ret_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_all_terminals():
    T, N = 64, 16
    torch.manual_seed(7)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.ones(T, N)
    next_value = torch.randn(N)

    adv_ref, ret_ref = _ref_gae_sequential(rewards, values, dones, next_value, 0.99, 0.95)
    adv_gpu, ret_gpu = _compute_gae_gpu_or_skip(rewards, values, dones, next_value, 0.99, 0.95)
    torch.testing.assert_close(adv_gpu.cpu(), adv_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ret_gpu.cpu(), ret_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_no_terminals():
    T, N = 256, 128
    torch.manual_seed(13)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    next_value = torch.randn(N)

    adv_ref, ret_ref = _ref_gae_sequential(rewards, values, dones, next_value, 0.99, 0.95)
    adv_gpu, ret_gpu = _compute_gae_gpu_or_skip(rewards, values, dones, next_value, 0.99, 0.95)
    torch.testing.assert_close(adv_gpu.cpu(), adv_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ret_gpu.cpu(), ret_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_single_step():
    rewards = torch.tensor([[0.5]])
    values = torch.tensor([[0.3]])
    dones = torch.tensor([[0.0]])
    next_value = torch.tensor([0.7])

    expected_delta = 0.5 + 0.99 * 0.7 * 1.0 - 0.3
    expected_adv = expected_delta

    adv, ret = _compute_gae_gpu_or_skip(rewards, values, dones, next_value, 0.99, 0.95)
    assert abs(adv.item() - expected_adv) < 1e-5
    assert abs(ret.item() - (expected_adv + 0.3)) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_single_step_terminal():
    rewards = torch.tensor([[0.5]])
    values = torch.tensor([[0.3]])
    dones = torch.tensor([[1.0]])
    next_value = torch.tensor([0.7])

    # done=1 at step 0 -> not_done=0 -> next_value zeroed
    expected_adv = 0.5 + 0.0 - 0.3  # 0.2

    adv, ret = _compute_gae_gpu_or_skip(rewards, values, dones, next_value, 0.99, 0.95)
    assert abs(adv.item() - expected_adv) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fallback_on_cpu_tensor():
    """compute_gae_gpu with CPU tensors should use CPU fallback."""
    T, N = 32, 8
    torch.manual_seed(99)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = (torch.rand(T, N) > 0.9).float()
    next_value = torch.randn(N)

    adv_ref, ret_ref = _ref_gae_sequential(rewards, values, dones, next_value, 0.99, 0.95)
    adv, ret = compute_gae_gpu(rewards, values, dones, next_value, 0.99, 0.95)

    torch.testing.assert_close(adv, adv_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ret, ret_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_different_gamma_lambda():
    T, N = 64, 32
    torch.manual_seed(55)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = (torch.rand(T, N) > 0.9).float()
    next_value = torch.randn(N)

    for gamma, lam in [(0.95, 0.9), (0.999, 0.99), (0.5, 0.5), (1.0, 1.0)]:
        adv_ref, ret_ref = _ref_gae_sequential(rewards, values, dones, next_value, gamma, lam)
        adv_gpu, ret_gpu = _compute_gae_gpu_or_skip(rewards, values, dones, next_value, gamma, lam)
        torch.testing.assert_close(adv_gpu.cpu(), adv_ref, rtol=1e-4, atol=1e-4,
                                   msg=f"Failed for gamma={gamma}, lambda={lam}")
        torch.testing.assert_close(ret_gpu.cpu(), ret_ref, rtol=1e-4, atol=1e-4)
