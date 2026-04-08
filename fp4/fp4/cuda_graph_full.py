"""Full rollout+update CUDA graph capture.

Phase 3 (`fp4/fp4/cuda_graph.py`) only captures the PPO *update* step. This
module captures a single graph that contains the **entire** PPO inner loop:

    env_step -> policy_fwd -> rollout buffer write -> GAE -> policy/value
    loss -> backward -> optimizer.step

Pattern (mirrors Phase 3's `capture_step`):

1. Caller provides a `step_fn()` that, on each call, performs *one full PPO
   iteration* using only **persistent** tensors that live in closure / module
   state. No Python-level allocations during the body, no host syncs, no
   dynamic shapes.
2. We warm up `step_fn` 3x on a private CUDA stream so the allocator caches
   are populated and any lazy init runs.
3. Then we capture one more invocation under `torch.cuda.graph(...)`.
4. `replay()` runs the captured graph. Outputs are read directly from the
   persistent tensors the caller exposes via the returned handle.

This is intentionally a *thin* wrapper -- the real complexity (allocation
hygiene, static rollout buffer, persistent optimizer state) is the caller's
job, exactly like the Phase 3 update-only graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


@dataclass
class CapturedFullStep:
    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    outputs: Dict[str, torch.Tensor]

    def replay(self) -> Dict[str, torch.Tensor]:
        self.graph.replay()
        return self.outputs


def capture_full_step(
    step_fn: Callable[[], Dict[str, torch.Tensor]],
    *,
    warmup: int = 3,
    stream: Optional[torch.cuda.Stream] = None,
    device: Optional[torch.device] = None,
) -> CapturedFullStep:
    """Capture a CUDA graph for one full PPO iteration.

    `step_fn` must:
        - take no arguments,
        - read all per-iteration *inputs* from persistent tensors in its
          closure (caller controls these and may `.copy_()` new data in
          between replays),
        - return a `dict[str, Tensor]` whose values are persistent tensors
          (the same Python/CUDA addresses on every call). Output addresses
          are locked in during warmup.

    Raises if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("capture_full_step requires CUDA")

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    capture_stream = stream if stream is not None else torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream())

    locked_outputs: Dict[str, torch.Tensor] = {}
    with torch.cuda.stream(capture_stream):
        for i in range(max(1, int(warmup))):
            out = step_fn()
            if not locked_outputs:
                for k, v in out.items():
                    locked_outputs[k] = v
            else:
                for k, v in out.items():
                    if locked_outputs[k].data_ptr() != v.data_ptr():
                        # Caller didn't reuse the same tensor address; copy
                        # so the locked-in handle still observes the value.
                        locked_outputs[k].copy_(v)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            out = step_fn()
            for k, v in out.items():
                if locked_outputs[k].data_ptr() != v.data_ptr():
                    locked_outputs[k].copy_(v)
    capture_stream.synchronize()
    torch.cuda.current_stream().wait_stream(capture_stream)

    return CapturedFullStep(graph=graph, stream=capture_stream, outputs=locked_outputs)


# ---------------------------------------------------------------------------
# Reference builder: assemble a `step_fn` from a tiny PPO-style training step
# over a *pure-tensor* synthetic env. This is what `trainer.py`'s
# `full_graph_capture` opt-in path uses, and what the bench/test exercise.
# ---------------------------------------------------------------------------

def build_synthetic_full_step(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    num_envs: int,
    rollout_len: int,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
):
    """Build (`step_fn`, `state`) for a fully-static synthetic PPO iteration.

    `state` is a dict of persistent tensors:
        - `obs`            [N, obs_dim]      current obs (caller may overwrite)
        - `target`         [N, act_dim]      synthetic reward direction
        - `obs_buf`        [T, N, obs_dim]
        - `act_buf`        [T, N, act_dim]
        - `logp_buf`       [T, N]
        - `val_buf`        [T, N]
        - `rew_buf`        [T, N]
        - `done_buf`       [T, N]            (zeros, fixed-length episodes)
        - `adv_buf`        [T, N]
        - `ret_buf`        [T, N]
        - `loss`           []                last loss scalar (output)
        - `mean_reward`    []                output

    `step_fn()` performs one full rollout + one PPO update minibatch and
    returns `{"loss": loss, "mean_reward": mean_reward}`.
    """
    T, N = int(rollout_len), int(num_envs)
    state: Dict[str, torch.Tensor] = {
        "obs":      torch.randn(N, obs_dim, device=device),
        "target":   torch.randn(N, act_dim, device=device),
        "obs_buf":  torch.zeros(T, N, obs_dim, device=device),
        "act_buf":  torch.zeros(T, N, act_dim, device=device),
        "logp_buf": torch.zeros(T, N, device=device),
        "val_buf":  torch.zeros(T, N, device=device),
        "rew_buf":  torch.zeros(T, N, device=device),
        "done_buf": torch.zeros(T, N, device=device),
        "adv_buf":  torch.zeros(T, N, device=device),
        "ret_buf":  torch.zeros(T, N, device=device),
        "loss":     torch.zeros((), device=device),
        "mean_reward": torch.zeros((), device=device),
    }
    # Normalize target so reward stays bounded.
    state["target"] = state["target"] / (state["target"].norm(dim=-1, keepdim=True) + 1e-6)

    def _policy_act(obs):
        mean, std, value = policy(obs)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        var = std * std
        logp = (-0.5 * (((action - mean) ** 2) / (var + 1e-8)
                        + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))).sum(dim=-1)
        return action, logp, value

    def step_fn() -> Dict[str, torch.Tensor]:
        obs = state["obs"]
        target = state["target"]
        # Rollout: T steps. All ops write into pre-allocated slots.
        with torch.no_grad():
            for t in range(T):
                action, logp, value = _policy_act(obs)
                state["obs_buf"][t].copy_(obs)
                state["act_buf"][t].copy_(action)
                state["logp_buf"][t].copy_(logp)
                state["val_buf"][t].copy_(value)
                # Synthetic reward: dot(action, target) - 0.05 * |a|^2.
                reward = (action * target).sum(dim=-1) - 0.05 * (action * action).sum(dim=-1)
                state["rew_buf"][t].copy_(reward)
                # Next obs: small drift toward target so the policy can learn.
                # `target` lives in action space (N, act_dim); only nudge the
                # first act_dim columns of `obs` (N, obs_dim) so the shapes
                # always agree even when obs_dim != act_dim.
                obs = obs.clone()
                obs[:, :act_dim] = obs[:, :act_dim] + 0.01 * (target - obs[:, :act_dim])
            state["obs"].copy_(obs)
            _, _, last_value = policy(obs)

        # GAE — fully on device, fixed-length, no python-side dynamic shapes.
        adv = state["adv_buf"]
        ret = state["ret_buf"]
        last_gae = torch.zeros(N, device=device)
        next_value = last_value
        for t in range(T - 1, -1, -1):
            not_done = 1.0 - state["done_buf"][t]
            delta = state["rew_buf"][t] + gamma * next_value * not_done - state["val_buf"][t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            adv[t].copy_(last_gae)
            next_value = state["val_buf"][t]
        ret.copy_(adv + state["val_buf"])

        # Single PPO update over the full rollout.
        b_obs = state["obs_buf"].reshape(T * N, obs_dim)
        b_act = state["act_buf"].reshape(T * N, act_dim)
        b_logp = state["logp_buf"].reshape(T * N)
        b_adv = adv.reshape(T * N)
        b_ret = ret.reshape(T * N)

        optimizer.zero_grad(set_to_none=False)
        mean, std, value = policy(b_obs)
        var = std * std
        new_logp = (-0.5 * (((b_act - mean) ** 2) / (var + 1e-8)
                            + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))).sum(dim=-1)
        entropy = (0.5 * math.log(2 * math.pi * math.e) + torch.log(std + 1e-8)).sum(dim=-1).mean()
        ratio = torch.exp(new_logp - b_logp)
        adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((value - b_ret) ** 2).mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        loss.backward()
        optimizer.step()

        state["loss"].copy_(loss.detach())
        state["mean_reward"].copy_(state["rew_buf"].mean())
        return {"loss": state["loss"], "mean_reward": state["mean_reward"]}

    return step_fn, state


# ---------------------------------------------------------------------------
# Real-env builder: same shape as build_synthetic_full_step, but the rollout
# uses a real ``EnvHandle`` (P5-1 env_adapter) backed by gpu_trading_env
# (P4-1 CUDA kernel). The kernel is graph-safe by design — every step writes
# into persistent SoA tensors with no host syncs. We add the only piece the
# adapter doesn't already give us: persistent obs/reward/done staging buffers
# so the captured graph reads/writes the same CUDA addresses every replay.
# ---------------------------------------------------------------------------


def build_real_full_step(
    env_handle,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    rollout_len: int,
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    example_inputs: Optional[Dict[str, torch.Tensor]] = None,
):
    """Build (`step_fn`, `state`) for a fully-static PPO iteration whose
    rollout invokes the **real** env (gpu_trading_env via env_adapter).

    `env_handle` must expose ``reset() -> obs`` and ``step(action) -> (obs,
    reward, done, cost_or_None)`` with persistent CUDA-resident outputs.
    The kernel writes its SoA state in-place; the small per-call allocation
    (the obs view returned by ``EnvHandle._obs()``) is absorbed by copying
    into a persistent staging buffer before the captured replay reads it.

    `example_inputs` is accepted for API symmetry with future extensions but
    is currently unused — shapes are derived from ``env_handle`` directly.
    """
    del example_inputs  # reserved for future use
    T = int(rollout_len)
    N = int(env_handle.num_envs)
    obs_dim = int(env_handle.obs_dim)
    act_dim = int(env_handle.action_dim)

    # Prime the env so reset() has populated state, then take its first obs
    # into a persistent buffer.
    first_obs = env_handle.reset().to(device).to(torch.float32)
    if first_obs.shape != (N, obs_dim):
        raise RuntimeError(
            f"env_handle.reset() returned {tuple(first_obs.shape)}, expected ({N},{obs_dim})"
        )

    state: Dict[str, torch.Tensor] = {
        "obs":      first_obs.contiguous().clone(),
        "obs_buf":  torch.zeros(T, N, obs_dim, device=device),
        "act_buf":  torch.zeros(T, N, act_dim, device=device),
        "logp_buf": torch.zeros(T, N, device=device),
        "val_buf":  torch.zeros(T, N, device=device),
        "rew_buf":  torch.zeros(T, N, device=device),
        "done_buf": torch.zeros(T, N, device=device),
        "adv_buf":  torch.zeros(T, N, device=device),
        "ret_buf":  torch.zeros(T, N, device=device),
        "loss":     torch.zeros((), device=device),
        "mean_reward": torch.zeros((), device=device),
    }

    def _policy_act(obs):
        mean, std, value = policy(obs)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        var = std * std
        logp = (-0.5 * (((action - mean) ** 2) / (var + 1e-8)
                        + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))).sum(dim=-1)
        return action, logp, value

    def step_fn() -> Dict[str, torch.Tensor]:
        obs = state["obs"]
        with torch.no_grad():
            for t in range(T):
                action, logp, value = _policy_act(obs)
                state["obs_buf"][t].copy_(obs)
                state["act_buf"][t].copy_(action)
                state["logp_buf"][t].copy_(logp)
                state["val_buf"][t].copy_(value)
                next_obs, reward, done, _cost = env_handle.step(action)
                state["rew_buf"][t].copy_(reward.to(torch.float32))
                state["done_buf"][t].copy_(done.to(torch.float32))
                # Persistent staging: copy the (possibly freshly-allocated)
                # obs view into the locked buffer the next iteration reads.
                state["obs"].copy_(next_obs.to(torch.float32))
                obs = state["obs"]
            _, _, last_value = policy(obs)

        adv = state["adv_buf"]
        ret = state["ret_buf"]
        last_gae = torch.zeros(N, device=device)
        next_value = last_value
        for t in range(T - 1, -1, -1):
            not_done = 1.0 - state["done_buf"][t]
            delta = state["rew_buf"][t] + gamma * next_value * not_done - state["val_buf"][t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            adv[t].copy_(last_gae)
            next_value = state["val_buf"][t]
        ret.copy_(adv + state["val_buf"])

        b_obs = state["obs_buf"].reshape(T * N, obs_dim)
        b_act = state["act_buf"].reshape(T * N, act_dim)
        b_logp = state["logp_buf"].reshape(T * N)
        b_adv = adv.reshape(T * N)
        b_ret = ret.reshape(T * N)

        optimizer.zero_grad(set_to_none=False)
        mean, std, value = policy(b_obs)
        var = std * std
        new_logp = (-0.5 * (((b_act - mean) ** 2) / (var + 1e-8)
                            + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))).sum(dim=-1)
        entropy = (0.5 * math.log(2 * math.pi * math.e) + torch.log(std + 1e-8)).sum(dim=-1).mean()
        ratio = torch.exp(new_logp - b_logp)
        adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((value - b_ret) ** 2).mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        loss.backward()
        optimizer.step()

        state["loss"].copy_(loss.detach())
        state["mean_reward"].copy_(state["rew_buf"].mean())
        return {"loss": state["loss"], "mean_reward": state["mean_reward"]}

    return step_fn, state


__all__ = [
    "CapturedFullStep",
    "capture_full_step",
    "build_synthetic_full_step",
    "build_real_full_step",
]
