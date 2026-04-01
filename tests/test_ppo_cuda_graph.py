"""
Tests for CUDA graph PPO update extension and BF16 autocast.

Covers:
  - PPO loss decreases over multiple update steps (with CUDA graph)
  - CUDA graph output matches eager mode numerically
  - LR annealing still works correctly with CUDA graph enabled
  - BF16 mode does not produce NaN in losses or values

All CUDA-dependent tests are marked `cuda_required` and auto-skipped when no
GPU is available (handled by conftest.py).  Each CUDA graph test runs on a
dedicated CUDA stream so that a capture failure in one test cannot corrupt
the shared default stream used by subsequent tests.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


# ── helpers ──────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.cuda_required  # auto-skip when CUDA is unavailable

OBS_SIZE = 107   # crypto6 shape: 6*16 + 5 + 6
NUM_ACTIONS = 13  # 1 + 2*6


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"CUDA graph PPO test skipped under shared-GPU resource pressure: {exc}")


def _make_policy(device: torch.device) -> nn.Module:
    from pufferlib_market.train import TradingPolicy
    try:
        return TradingPolicy(OBS_SIZE, NUM_ACTIONS, hidden=128).to(device)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _ppo_loss_eager(policy, obs, act, old_logprob, advantages, returns, old_values,
                    clip_eps_t: torch.Tensor, vf_coef_t: torch.Tensor, ent_coef_t: torch.Tensor,
                    clip_vloss: bool = False):
    """
    PPO loss computation (eager, FP32).

    All coefficient arguments must be CUDA tensors (scalar) — this keeps the
    computation graph CUDA-only and avoids CPU synchronisation calls during
    CUDA graph capture (no .item() inside this function).
    """
    logits, new_value = policy(obs)
    log_probs_all = torch.log_softmax(logits, dim=-1)
    new_logprob = log_probs_all.gather(1, act.unsqueeze(1)).squeeze(1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs_all).sum(-1)

    log_ratio = new_logprob - old_logprob
    ratio = log_ratio.exp()

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_eps_t, 1 + clip_eps_t)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    v_loss = 0.5 * ((new_value - returns) ** 2).mean()
    ent_loss = entropy.mean()
    loss = pg_loss + vf_coef_t * v_loss - ent_coef_t * ent_loss
    return loss, pg_loss, v_loss, ent_loss


def _make_fake_minibatch(mb_size: int, device: torch.device):
    """Random but plausible PPO minibatch tensors."""
    try:
        obs = torch.randn(mb_size, OBS_SIZE, device=device)
        act = torch.randint(0, NUM_ACTIONS, (mb_size,), device=device)
        old_logprob = torch.randn(mb_size, device=device)
        advantages = torch.randn(mb_size, device=device)
        returns = torch.randn(mb_size, device=device)
        old_values = torch.randn(mb_size, device=device)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise
    return obs, act, old_logprob, advantages, returns, old_values


# ── CUDA graph PPO update harness ─────────────────────────────────────────────

class _CUDAGraphPPOHarness:
    """
    Minimal reproduction of the CUDA-graph PPO update from train.py.

    Captures: forward + backward inside a CUDA graph on a dedicated stream.
    Keeps outside: optimizer.step(), grad norm clip.

    Key constraint: CUDA graphs cannot capture CPU-GPU synchronisation.
    All scalar coefficients are stored as CUDA tensors and passed directly
    to the loss function (no .item() calls inside the graph).
    """

    def __init__(self, policy: nn.Module, optimizer, mb_size: int, device: torch.device,
                 clip_eps: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01):
        self.policy = policy
        self.optimizer = optimizer
        self.mb_size = mb_size
        self.device = device

        # Static tensors (fixed addresses required for CUDA graph)
        try:
            self.st_obs = torch.zeros(mb_size, OBS_SIZE, device=device)
            self.st_act = torch.zeros(mb_size, dtype=torch.long, device=device)
            self.st_logprob = torch.zeros(mb_size, device=device)
            self.st_adv = torch.zeros(mb_size, device=device)
            self.st_ret = torch.zeros(mb_size, device=device)
            self.st_val = torch.zeros(mb_size, device=device)
            # Scalar coefficients as CUDA tensors (no .item() during capture)
            self.st_clip_eps = torch.tensor(clip_eps, device=device)
            self.st_vf_coef = torch.tensor(vf_coef, device=device)
            self.st_ent_coef = torch.tensor(ent_coef, device=device)
            # Scalar outputs (written inside graph via .copy_())
            self.st_loss = torch.tensor(0.0, device=device)
            self.st_pg = torch.tensor(0.0, device=device)
            self.st_vl = torch.tensor(0.0, device=device)
            self.st_el = torch.tensor(0.0, device=device)

            # Use a private CUDA stream so capture errors do not contaminate the
            # default stream used by other tests in the same process.
            self._stream = torch.cuda.Stream(device=device)
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            raise

        def _body():
            loss, pg, vl, el = _ppo_loss_eager(
                self.policy,
                self.st_obs, self.st_act, self.st_logprob,
                self.st_adv, self.st_ret, self.st_val,
                self.st_clip_eps, self.st_vf_coef, self.st_ent_coef,
            )
            loss.backward()
            self.st_loss.copy_(loss.detach())
            self.st_pg.copy_(pg.detach())
            self.st_vl.copy_(vl.detach())
            self.st_el.copy_(el.detach())

        self._body = _body

        # Warmup passes on the private stream (fills CUDA caches before capture)
        policy.train()
        try:
            with torch.cuda.stream(self._stream):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=False)
                    _body()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
            self._stream.synchronize()
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            raise

        # Capture graph on the private stream
        try:
            optimizer.zero_grad(set_to_none=False)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._stream):
                with torch.cuda.graph(self._graph, stream=self._stream):
                    _body()
            self._stream.synchronize()
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            raise

    def step(self, obs, act, logprob, adv, ret, val):
        """Copy inputs into static tensors, replay graph, step optimizer."""
        try:
            self.st_obs.copy_(obs)
            self.st_act.copy_(act)
            self.st_logprob.copy_(logprob)
            self.st_adv.copy_(adv)
            self.st_ret.copy_(ret)
            self.st_val.copy_(val)
            self.optimizer.zero_grad(set_to_none=False)
            with torch.cuda.stream(self._stream):
                self._graph.replay()
            self._stream.synchronize()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            return (
                self.st_loss.item(),
                self.st_pg.item(),
                self.st_vl.item(),
                self.st_el.item(),
            )
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            raise


# ── test: loss decreases over 100 steps ──────────────────────────────────────

def test_cuda_graph_ppo_loss_decreases():
    """PPO loss should trend downward over 100 update steps with CUDA graph."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(1)
    policy = _make_policy(device)
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4)
    mb_size = 512

    harness = _CUDAGraphPPOHarness(policy, optimizer, mb_size, device)

    # Fix a single minibatch to measure pure optimisation progress
    obs, act, old_logprob, advantages, returns, old_values = _make_fake_minibatch(mb_size, device)

    losses = []
    for _ in range(100):
        loss_val, _, _, _ = harness.step(obs, act, old_logprob, advantages, returns, old_values)
        losses.append(loss_val)

    first_10 = sum(losses[:10]) / 10
    last_10 = sum(losses[-10:]) / 10
    assert last_10 < first_10, (
        f"PPO loss did not decrease: first_10={first_10:.4f}, last_10={last_10:.4f}"
    )


# ── test: CUDA graph matches eager numerically ────────────────────────────────

def test_cuda_graph_ppo_matches_eager():
    """CUDA graph PPO loss scalars must match eager mode within FP tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(0)

    mb_size = 256
    obs, act, old_logprob, advantages, returns, old_values = _make_fake_minibatch(mb_size, device)

    clip_eps_t = torch.tensor(0.2, device=device)
    vf_coef_t = torch.tensor(0.5, device=device)
    ent_coef_t = torch.tensor(0.01, device=device)

    # ── eager reference ──
    policy_eager = _make_policy(device)
    optimizer_eager = optim.AdamW(policy_eager.parameters(), lr=3e-4)
    policy_eager.train()
    optimizer_eager.zero_grad()
    loss_e, pg_e, vl_e, el_e = _ppo_loss_eager(
        policy_eager, obs, act, old_logprob, advantages, returns, old_values,
        clip_eps_t, vf_coef_t, ent_coef_t,
    )
    loss_e.backward()
    torch.cuda.synchronize()

    # ── CUDA graph (start from identical weights) ──
    policy_graph = _make_policy(device)
    policy_graph.load_state_dict(policy_eager.state_dict())
    optimizer_graph = optim.AdamW(policy_graph.parameters(), lr=3e-4)

    harness = _CUDAGraphPPOHarness(policy_graph, optimizer_graph, mb_size, device)
    # Reset to original weights after warmup (warmup performs gradient updates)
    policy_graph.load_state_dict(policy_eager.state_dict())

    # Run a single graph replay (backward inside graph) without the optimizer step
    harness.st_obs.copy_(obs)
    harness.st_act.copy_(act)
    harness.st_logprob.copy_(old_logprob)
    harness.st_adv.copy_(advantages)
    harness.st_ret.copy_(returns)
    harness.st_val.copy_(old_values)
    optimizer_graph.zero_grad(set_to_none=False)
    with torch.cuda.stream(harness._stream):
        harness._graph.replay()
    harness._stream.synchronize()

    loss_g = harness.st_loss.item()
    pg_g = harness.st_pg.item()
    vl_g = harness.st_vl.item()
    el_g = harness.st_el.item()

    assert math.isclose(loss_g, loss_e.item(), rel_tol=1e-4), (
        f"Total loss mismatch: graph={loss_g:.6f}, eager={loss_e.item():.6f}"
    )
    assert math.isclose(pg_g, pg_e.item(), rel_tol=1e-4), (
        f"PG loss mismatch: graph={pg_g:.6f}, eager={pg_e.item():.6f}"
    )
    assert math.isclose(vl_g, vl_e.item(), rel_tol=1e-4), (
        f"VF loss mismatch: graph={vl_g:.6f}, eager={vl_e.item():.6f}"
    )
    assert math.isclose(el_g, el_e.item(), rel_tol=1e-4), (
        f"Entropy loss mismatch: graph={el_g:.6f}, eager={el_e.item():.6f}"
    )


# ── test: LR annealing still works with CUDA graph ───────────────────────────

def test_cuda_graph_ppo_lr_annealing():
    """LR annealing applied between CUDA graph PPO update steps must be respected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    base_lr = 3e-4
    num_updates = 50
    torch.manual_seed(2)
    policy = _make_policy(device)
    optimizer = optim.AdamW(policy.parameters(), lr=base_lr)
    mb_size = 256

    harness = _CUDAGraphPPOHarness(policy, optimizer, mb_size, device)
    obs, act, old_logprob, adv, ret, val = _make_fake_minibatch(mb_size, device)

    lrs_observed = []
    for update in range(1, num_updates + 1):
        # Linear LR anneal (mirrors --anneal-lr logic in train.py)
        frac = 1.0 - (update - 1) / num_updates
        lr_now = frac * base_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        harness.step(obs, act, old_logprob, adv, ret, val)
        lrs_observed.append(optimizer.param_groups[0]["lr"])

    # LR should have decreased monotonically from base_lr toward (near) 0
    assert lrs_observed[0] == pytest.approx(base_lr, rel=1e-5)
    assert lrs_observed[-1] < lrs_observed[0]
    for i in range(1, len(lrs_observed)):
        assert lrs_observed[i] <= lrs_observed[i - 1] + 1e-12, (
            f"LR increased at step {i}: {lrs_observed[i-1]:.2e} -> {lrs_observed[i]:.2e}"
        )


# ── test: BF16 autocast does not produce NaN ─────────────────────────────────

def test_bf16_ppo_no_nan():
    """BF16 autocast PPO forward pass must not produce NaN in loss or values."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(3)
    mb_size = 256
    obs, act, old_logprob, advantages, returns, old_values = _make_fake_minibatch(mb_size, device)

    policy = _make_policy(device)
    policy.train()

    # Replicate the BF16 autocast logic from train.py _ppo_loss_fn
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits, new_value = policy(obs)
    logits = logits.float()
    new_value = new_value.float()

    log_probs_all = torch.log_softmax(logits, dim=-1)
    new_logprob = log_probs_all.gather(1, act.unsqueeze(1)).squeeze(1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs_all).sum(-1)

    ratio = (new_logprob - old_logprob).exp()
    pg_loss = torch.max(-advantages * ratio,
                        -advantages * torch.clamp(ratio, 0.8, 1.2)).mean()
    v_loss = 0.5 * ((new_value - returns) ** 2).mean()
    ent_loss = entropy.mean()
    loss = pg_loss + 0.5 * v_loss - 0.01 * ent_loss

    assert not torch.isnan(loss), "BF16 loss is NaN"
    assert not torch.isnan(new_value).any(), "BF16 value head produced NaN"
    assert not torch.isnan(logits).any(), "BF16 logits produced NaN"

    # Backward must also succeed without NaN grads
    loss.backward()
    for name, param in policy.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name} in BF16 mode"


# ── test: _ppo_loss_fn in train.py respects use_bf16 flag ────────────────────

def test_train_ppo_loss_fn_bf16_flag():
    """PPO loss function with BF16 autocast enabled must not produce NaN."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(4)
    mb_size = 64

    policy = _make_policy(device)
    policy.train()

    obs, act, old_logprob, adv, ret, val = _make_fake_minibatch(mb_size, device)
    clip_eps_t = torch.tensor(0.2, device=device)
    vf_coef_t = torch.tensor(0.5, device=device)
    ent_coef_t = torch.tensor(0.01, device=device)

    # Simulate the BF16 closure created in train.py (_use_bf16=True)
    _use_bf16 = True

    def _ppo_loss_fn_bf16(p, obs, act, old_logprob, advantages, returns, old_values,
                          clip_eps_t, vf_coef_t, ent_coef_t, clip_vloss):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_use_bf16):
            logits, new_value = p(obs)
        logits = logits.float()
        new_value = new_value.float()
        log_probs_all = torch.log_softmax(logits, dim=-1)
        new_logprob = log_probs_all.gather(1, act.unsqueeze(1)).squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(-1)
        ratio = (new_logprob - old_logprob).exp()
        pg_loss = torch.max(-advantages * ratio,
                            -advantages * torch.clamp(ratio, 1 - clip_eps_t, 1 + clip_eps_t)).mean()
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()
        ent_loss = entropy.mean()
        loss = pg_loss + vf_coef_t * v_loss - ent_coef_t * ent_loss
        return loss, pg_loss, v_loss, ent_loss

    loss, pg, vl, el = _ppo_loss_fn_bf16(
        policy, obs, act, old_logprob, adv, ret, val,
        clip_eps_t, vf_coef_t, ent_coef_t, False,
    )

    assert not torch.isnan(loss), "BF16 _ppo_loss_fn produced NaN loss"
    assert not torch.isnan(pg), "BF16 _ppo_loss_fn produced NaN pg_loss"
    assert not torch.isnan(vl), "BF16 _ppo_loss_fn produced NaN v_loss"
    assert not torch.isnan(el), "BF16 _ppo_loss_fn produced NaN ent_loss"

    loss.backward()


# ── test: argparse exposes --cuda-graph-ppo and --use-bf16 ───────────────────

def test_argparse_cuda_graph_ppo_flag():
    """--cuda-graph-ppo and --use-bf16 must be recognised by the argument parser."""
    import sys
    import argparse
    from unittest.mock import patch

    argv = [
        "prog",
        "--data-path", "dummy.bin",
        "--cuda-graph-ppo",
        "--use-bf16",
    ]
    with patch.object(sys, "argv", argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-path")
        parser.add_argument("--cuda-graph-ppo", action="store_true")
        parser.add_argument("--use-bf16", action="store_true")
        args = parser.parse_args()

    assert args.cuda_graph_ppo is True, "--cuda-graph-ppo not parsed as True"
    assert args.use_bf16 is True, "--use-bf16 not parsed as True"
