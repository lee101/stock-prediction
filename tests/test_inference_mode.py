"""
Tests verifying correct behavior of torch.inference_mode() with TradingPolicy.

Covers:
  1. test_policy_forward_inference_mode — output shapes and no requires_grad in
     outputs under inference_mode; numerical match with no_grad.
  2. test_inference_mode_no_grad_output_match — inference_mode and no_grad
     produce numerically identical policy outputs.
  3. test_inference_mode_tensor_properties — tensors created inside
     inference_mode have no gradient history.
  4. test_cuda_graph_ppo_smoke — CUDA graph PPO harness initialises and runs
     several steps without error (CUDA required).

All CUDA-dependent tests are skipped automatically when no GPU is available.
CPU tests run in any environment.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Reuse the CUDA graph harness and helpers from the dedicated PPO graph test.
sys.path.insert(0, str(Path(__file__).parent))
from test_ppo_cuda_graph import _CUDAGraphPPOHarness, _make_fake_minibatch

OBS_SIZE = 107    # crypto6: 6*16 + 5 + 6
NUM_ACTIONS = 13  # 1 + 2*6
BATCH = 8
HIDDEN = 64       # small for fast tests


def _make_policy(device: torch.device | str = "cpu") -> nn.Module:
    from pufferlib_market.train import TradingPolicy
    return TradingPolicy(OBS_SIZE, NUM_ACTIONS, hidden=HIDDEN).to(device)


def test_policy_forward_inference_mode():
    """Forward pass inside inference_mode produces correct shapes and no grad."""
    torch.manual_seed(0)
    policy = _make_policy("cpu")
    policy.eval()
    obs = torch.randn(BATCH, OBS_SIZE)

    with torch.inference_mode():
        logits_im, value_im = policy(obs)

    assert logits_im.shape == (BATCH, NUM_ACTIONS), (
        f"logits shape: expected ({BATCH}, {NUM_ACTIONS}), got {logits_im.shape}"
    )
    assert value_im.shape == (BATCH,), (
        f"value shape: expected ({BATCH},), got {value_im.shape}"
    )
    assert not logits_im.requires_grad, "logits must not require grad under inference_mode"
    assert not value_im.requires_grad, "value must not require grad under inference_mode"

    with torch.no_grad():
        logits_ng, value_ng = policy(obs)

    assert torch.equal(logits_im, logits_ng), "logits differ between inference_mode and no_grad"
    assert torch.equal(value_im, value_ng), "values differ between inference_mode and no_grad"


def test_inference_mode_no_grad_output_match():
    """inference_mode and no_grad must produce bit-identical results on CPU FP32."""
    torch.manual_seed(42)
    policy = _make_policy("cpu")
    policy.eval()

    obs = torch.randn(BATCH, OBS_SIZE)

    with torch.inference_mode():
        logits_im, value_im = policy(obs)

    with torch.no_grad():
        logits_ng, value_ng = policy(obs)

    assert torch.equal(logits_im, logits_ng), (
        "logits are not bit-identical between inference_mode and no_grad"
    )
    assert torch.equal(value_im, value_ng), (
        "values are not bit-identical between inference_mode and no_grad"
    )


def test_inference_mode_tensor_properties():
    """Tensors created or returned inside inference_mode have no grad history."""
    torch.manual_seed(7)
    policy = _make_policy("cpu")
    policy.eval()

    obs = torch.randn(BATCH, OBS_SIZE)

    with torch.inference_mode():
        logits, value = policy(obs)
        intermediate = torch.randn(4, 4)
        scaled = intermediate * 2.0

    assert not logits.requires_grad
    assert not value.requires_grad
    assert logits.grad_fn is None
    assert value.grad_fn is None
    assert not intermediate.requires_grad
    assert not scaled.requires_grad
    assert scaled.grad_fn is None

    # A leaf created outside the context keeps its grad requirement after exit.
    leaf = torch.randn(3, requires_grad=True)
    assert leaf.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_ppo_smoke():
    """CUDA graph PPO harness runs several steps without error or non-finite loss."""
    import torch.optim as optim

    device = torch.device("cuda")
    torch.manual_seed(99)

    MB = 128
    policy = _make_policy(device)
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4)

    harness = _CUDAGraphPPOHarness(policy, optimizer, MB, device)
    obs, act, logprob, adv, ret, val = _make_fake_minibatch(MB, device)

    losses = []
    for _ in range(5):
        loss_val, _, _, _ = harness.step(obs, act, logprob, adv, ret, val)
        assert math.isfinite(loss_val), f"CUDA graph PPO loss is not finite: {loss_val}"
        losses.append(loss_val)

    assert len(losses) == 5
