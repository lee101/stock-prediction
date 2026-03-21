"""
Tests for new trading policy architectures in pufferlib_market/train.py:
  - relu_sq activation
  - TransformerTradingPolicy
  - GRUTradingPolicy
  - DepthRecurrenceTradingPolicy

Each architecture is tested for:
  - Correct output shapes
  - Gradient flow (loss.backward() succeeds)
  - Multiple obs_size values (crypto6=107, crypto12=209, mixed23=396)
  - DepthRecurrenceTradingPolicy has fewer params than equivalent deep MLP
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pufferlib_market.train import (
    TradingPolicy,
    relu_sq,
    TransformerTradingPolicy,
    GRUTradingPolicy,
    DepthRecurrenceTradingPolicy,
)


# Canonical obs_sizes used in production
# crypto6:   6*16 + 5 + 6  = 107
# crypto12: 12*16 + 5 + 12 = 209
# mixed23:  23*16 + 5 + 23 = 396
OBS_SIZES = [107, 209, 396]
BATCH_SIZE = 4
HIDDEN = 256


def _num_actions_for_obs(obs_size: int) -> int:
    """Reconstruct num_actions from obs_size (matches production formula with 1 action bin each)."""
    # obs_size = S*17 + 5  =>  S = (obs_size - 5) // 17
    num_symbols = (obs_size - 5) // 17
    # With action_allocation_bins=1, action_level_bins=1:
    #   per_symbol_actions = 1
    #   num_actions = 1 + 2 * S * 1
    return 1 + 2 * num_symbols


# ─── relu_sq ──────────────────────────────────────────────────────────

def test_relu_sq_positive():
    x = torch.tensor([1.0, 2.0, 3.0])
    out = relu_sq(x)
    expected = torch.tensor([1.0, 4.0, 9.0])
    assert torch.allclose(out, expected)


def test_relu_sq_negative_zeroed():
    x = torch.tensor([-1.0, -0.5, 0.0])
    out = relu_sq(x)
    assert torch.all(out == 0.0)


def test_relu_sq_mixed():
    x = torch.tensor([-2.0, 0.0, 2.0])
    out = relu_sq(x)
    assert out[0].item() == pytest.approx(0.0)
    assert out[1].item() == pytest.approx(0.0)
    assert out[2].item() == pytest.approx(4.0)


def test_relu_sq_gradient():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    out = relu_sq(x).sum()
    out.backward()
    # d/dx relu(x)^2 = 2*relu(x) at x>0
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([2.0, 4.0]))


# ─── TransformerTradingPolicy ─────────────────────────────────────────

@pytest.mark.parametrize("obs_size", OBS_SIZES)
def test_transformer_output_shapes(obs_size):
    num_actions = _num_actions_for_obs(obs_size)
    policy = TransformerTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,), f"actions shape mismatch: {actions.shape}"
    assert log_probs.shape == (BATCH_SIZE,), f"log_probs shape mismatch: {log_probs.shape}"
    assert entropy.shape == (BATCH_SIZE,), f"entropy shape mismatch: {entropy.shape}"
    assert values.shape == (BATCH_SIZE,), f"values shape mismatch: {values.shape}"


@pytest.mark.parametrize("obs_size", OBS_SIZES)
def test_transformer_gradient_flow(obs_size):
    num_actions = _num_actions_for_obs(obs_size)
    policy = TransformerTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    loss = -log_probs.mean() + values.mean() - 0.01 * entropy.mean()
    loss.backward()
    # All parameters should have gradients
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_transformer_get_value_shape():
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = TransformerTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    values = policy.get_value(obs)
    assert values.shape == (BATCH_SIZE,)


@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu"])
def test_transformer_activations(activation):
    obs_size = 107
    num_actions = _num_actions_for_obs(obs_size)
    policy = TransformerTradingPolicy(obs_size, num_actions, hidden=HIDDEN, activation=activation)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,)
    assert not torch.isnan(values).any(), f"NaN in values with {activation}"


def test_transformer_disable_shorts():
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = TransformerTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, _, _, _ = policy.get_action_and_value(obs, disable_shorts=True)
    # With disable_shorts, actions should be in [0, 1+num_symbols) range
    num_symbols = (obs_size - 5) // 17
    assert torch.all(actions < 1 + num_symbols), "disable_shorts: short actions should be masked"


# ─── GRUTradingPolicy ─────────────────────────────────────────────────

@pytest.mark.parametrize("obs_size", OBS_SIZES)
def test_gru_output_shapes(obs_size):
    num_actions = _num_actions_for_obs(obs_size)
    policy = GRUTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,)
    assert log_probs.shape == (BATCH_SIZE,)
    assert entropy.shape == (BATCH_SIZE,)
    assert values.shape == (BATCH_SIZE,)


@pytest.mark.parametrize("obs_size", OBS_SIZES)
def test_gru_gradient_flow(obs_size):
    num_actions = _num_actions_for_obs(obs_size)
    policy = GRUTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    loss = -log_probs.mean() + values.mean() - 0.01 * entropy.mean()
    loss.backward()
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_gru_get_value_shape():
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = GRUTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    values = policy.get_value(obs)
    assert values.shape == (BATCH_SIZE,)


@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu"])
def test_gru_activations(activation):
    obs_size = 107
    num_actions = _num_actions_for_obs(obs_size)
    policy = GRUTradingPolicy(obs_size, num_actions, hidden=HIDDEN, activation=activation)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,)
    assert not torch.isnan(values).any()


def test_gru_disable_shorts():
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = GRUTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, _, _, _ = policy.get_action_and_value(obs, disable_shorts=True)
    num_symbols = (obs_size - 5) // 17
    assert torch.all(actions < 1 + num_symbols)


# ─── DepthRecurrenceTradingPolicy ─────────────────────────────────────

@pytest.mark.parametrize("obs_size", OBS_SIZES)
def test_depth_recurrence_output_shapes(obs_size):
    num_actions = _num_actions_for_obs(obs_size)
    policy = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,)
    assert log_probs.shape == (BATCH_SIZE,)
    assert entropy.shape == (BATCH_SIZE,)
    assert values.shape == (BATCH_SIZE,)


@pytest.mark.parametrize("obs_size", OBS_SIZES)
def test_depth_recurrence_gradient_flow(obs_size):
    num_actions = _num_actions_for_obs(obs_size)
    policy = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    loss = -log_probs.mean() + values.mean() - 0.01 * entropy.mean()
    loss.backward()
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_depth_recurrence_get_value_shape():
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    values = policy.get_value(obs)
    assert values.shape == (BATCH_SIZE,)


@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu"])
def test_depth_recurrence_activations(activation):
    obs_size = 107
    num_actions = _num_actions_for_obs(obs_size)
    policy = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN, activation=activation)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,)
    assert not torch.isnan(values).any()


def test_depth_recurrence_fewer_params_than_deep_mlp():
    """
    DepthRecurrence with N=3 blocks × K=2 passes = 6 effective layers
    should have fewer parameters than a DepthRecurrence with N=6 blocks × K=1
    (equivalent depth but every block is unique — no reuse).

    Both have the same overall architecture (same input_proj, actor, critic, out_norm),
    so only the block count differs, making this a clean parameter-efficiency comparison.
    """
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)

    # N=3 blocks, K=2 passes (6 effective layers, 3-block cost)
    dr_3x2 = DepthRecurrenceTradingPolicy(
        obs_size, num_actions, hidden=HIDDEN,
        num_blocks=3, num_recurrences=2,
    )
    # N=6 blocks, K=1 pass (6 effective layers, 6-block cost — no reuse)
    dr_6x1 = DepthRecurrenceTradingPolicy(
        obs_size, num_actions, hidden=HIDDEN,
        num_blocks=6, num_recurrences=1,
    )

    params_3x2 = sum(p.numel() for p in dr_3x2.parameters())
    params_6x1 = sum(p.numel() for p in dr_6x1.parameters())

    assert params_3x2 < params_6x1, (
        f"DepthRecurrence N=3×K=2 ({params_3x2:,}) should have fewer params "
        f"than N=6×K=1 ({params_6x1:,})"
    )


def test_depth_recurrence_num_recurrences():
    """More recurrences increases effective depth without increasing param count."""
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)

    policy_k1 = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN,
                                              num_blocks=3, num_recurrences=1)
    policy_k2 = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN,
                                              num_blocks=3, num_recurrences=2)
    policy_k4 = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN,
                                              num_blocks=3, num_recurrences=4)

    params_k1 = sum(p.numel() for p in policy_k1.parameters())
    params_k2 = sum(p.numel() for p in policy_k2.parameters())
    params_k4 = sum(p.numel() for p in policy_k4.parameters())

    # Parameter count should be identical across recurrences (blocks are reused)
    assert params_k1 == params_k2 == params_k4, (
        f"Param counts should be equal: k1={params_k1}, k2={params_k2}, k4={params_k4}"
    )


def test_depth_recurrence_disable_shorts():
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=HIDDEN)
    obs = torch.randn(BATCH_SIZE, obs_size)
    actions, _, _, _ = policy.get_action_and_value(obs, disable_shorts=True)
    num_symbols = (obs_size - 5) // 17
    assert torch.all(actions < 1 + num_symbols)


# ─── Interface compatibility (all policies share same API) ────────────

@pytest.mark.parametrize("PolicyClass,kwargs", [
    (TransformerTradingPolicy, {}),
    (GRUTradingPolicy, {}),
    (DepthRecurrenceTradingPolicy, {}),
])
def test_policy_interface_compatibility(PolicyClass, kwargs):
    """All new policies must satisfy the same interface as TradingPolicy."""
    obs_size = 209
    num_actions = _num_actions_for_obs(obs_size)
    policy = PolicyClass(obs_size, num_actions, hidden=HIDDEN, **kwargs)
    obs = torch.randn(BATCH_SIZE, obs_size)

    # Test get_action_and_value
    actions, log_probs, entropy, values = policy.get_action_and_value(obs)
    assert actions.shape == (BATCH_SIZE,)
    assert log_probs.shape == (BATCH_SIZE,)
    assert entropy.shape == (BATCH_SIZE,)
    assert values.shape == (BATCH_SIZE,)

    # Test with explicit action
    fixed_action = torch.zeros(BATCH_SIZE, dtype=torch.long)
    _, lp2, ent2, val2 = policy.get_action_and_value(obs, action=fixed_action)
    assert lp2.shape == (BATCH_SIZE,)

    # Test get_value
    val3 = policy.get_value(obs)
    assert val3.shape == (BATCH_SIZE,)

    # Test obs_size and num_actions attributes
    assert policy.obs_size == obs_size
    assert policy.num_actions == num_actions
