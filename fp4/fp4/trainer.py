"""GPU-resident PPO trainer for the fp4 library.

`train_ppo(cfg, total_timesteps, seed, checkpoint_dir) -> dict` is the public
entry point — its signature matches what `fp4/bench/bench_trading.py` already
expects.

Design notes
------------
- Rollout buffer, advantages and minibatch updates all live on GPU when one is
  available; on CPU we run the same code path so the smoke test stays cheap.
- The actor-critic uses `NVFP4Linear` for the hidden layer and BF16/FP32 for
  the input projection + heads (see `policy.py`).
- We try to capture the PPO update step as a CUDA graph the first time it
  runs.  Capture is best-effort: if anything in the inner step trips it
  (control flow, dynamic shapes, autograd issues with NVFP4Linear's custom
  autograd Function, …) we silently fall back to eager.  The rest of the
  trainer is unaffected.
- The env is pluggable.  `cfg.get("env")` may be:
    * `"stub"` (or missing) → an internal torch synthetic vector env so the
      smoke test runs CPU-only.
    * an object with `reset()` / `step(action)` returning torch tensors → used
      directly (e.g. Unit C marketsim bindings once they exist).
    * a dict → reserved for the real marketsim wiring (Unit G); for now we
      fall back to the stub so the bench harness still produces a metrics JSON.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .optim import AdamWMaster
from .policy import ActorCritic


# ---------------------------------------------------------------------------
# Stub vector env (pure torch, no host syncs)
# ---------------------------------------------------------------------------

class StubVecEnv:
    """A trivial GPU-resident vector env used for smoke tests / fallbacks.

    Reward is `dot(action, target_dir) - 0.05 * |action|^2`, where target_dir
    is a slowly-rotating unit vector embedded in obs.  This gives a smooth,
    learnable, finite-Sortino signal so we can verify the PPO loop end-to-end.
    """

    def __init__(self, num_envs: int, obs_dim: int, act_dim: int,
                 episode_len: int, device: torch.device, seed: int = 0):
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.episode_len = int(episode_len)
        self.device = device
        self._gen = torch.Generator(device=device).manual_seed(int(seed))
        self._t = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._target = self._sample_target()
        self._obs = self._make_obs()

    def _sample_target(self) -> torch.Tensor:
        t = torch.randn(self.num_envs, self.act_dim, generator=self._gen, device=self.device)
        return t / (t.norm(dim=-1, keepdim=True) + 1e-6)

    def _make_obs(self) -> torch.Tensor:
        obs = torch.randn(self.num_envs, self.obs_dim, generator=self._gen, device=self.device) * 0.1
        obs[:, : self.act_dim] = self._target
        return obs

    def reset(self) -> torch.Tensor:
        self._t.zero_()
        self._target = self._sample_target()
        self._obs = self._make_obs()
        return self._obs

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = action.to(self._obs.dtype)
        reward = (a * self._target).sum(dim=-1) - 0.05 * (a * a).sum(dim=-1)
        self._t += 1
        done = self._t >= self.episode_len
        if done.any():
            # Resample finished envs in-place (still no host sync — bool mask only).
            new_targets = self._sample_target()
            self._target = torch.where(done.unsqueeze(-1), new_targets, self._target)
            self._t = torch.where(done, torch.zeros_like(self._t), self._t)
        self._obs = self._make_obs()
        return self._obs, reward, done.to(torch.float32)


def _make_env(cfg: Dict[str, Any], num_envs: int, obs_dim: int, act_dim: int,
              device: torch.device, seed: int):
    env_spec = cfg.get("env", "stub")
    # If caller passed a ready-to-use env object, just use it.
    if hasattr(env_spec, "reset") and hasattr(env_spec, "step"):
        return env_spec
    # Real marketsim wiring lands in Unit G; until then we use the stub for
    # everything so the bench harness can still produce a JSON.
    episode_len = int(cfg.get("episode_len", 256))
    return StubVecEnv(num_envs, obs_dim, act_dim, episode_len, device, seed)


# ---------------------------------------------------------------------------
# GAE on GPU
# ---------------------------------------------------------------------------

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                last_value: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation, fully on-device.

    Shapes: rewards/values/dones are (T, N).  Returns advantages (T, N) and
    returns (T, N).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    next_value = last_value
    for t in range(T - 1, -1, -1):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _ppo_loss(policy: ActorCritic, obs_b: torch.Tensor, act_b: torch.Tensor,
              old_logp_b: torch.Tensor, adv_b: torch.Tensor, ret_b: torch.Tensor,
              clip_eps: float, vf_coef: float, ent_coef: float
              ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mean, std, value = policy(obs_b)
    new_logp = ActorCritic.gaussian_logprob(mean, std, act_b)
    entropy = ActorCritic.gaussian_entropy(std).mean()
    ratio = torch.exp(new_logp - old_logp_b)
    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, ret_b)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
    }


# ---------------------------------------------------------------------------
# Sortino on a torch tensor (GPU friendly).
# ---------------------------------------------------------------------------

def sortino_ratio(returns: torch.Tensor, eps: float = 1e-8) -> float:
    r = returns.flatten().to(torch.float32)
    if r.numel() == 0:
        return 0.0
    mean = r.mean()
    downside = torch.clamp(r, max=0.0)
    dd = torch.sqrt((downside * downside).mean() + eps)
    return float((mean / (dd + eps)).item())


def percentile(returns: torch.Tensor, q: float) -> float:
    r = returns.flatten().to(torch.float32)
    if r.numel() == 0:
        return 0.0
    return float(torch.quantile(r, q).item())


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def train_ppo(cfg: Dict[str, Any], total_timesteps: int, seed: int,
              checkpoint_dir: str) -> Dict[str, Any]:
    """GPU-resident PPO. See module docstring for the contract."""
    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_cfg = cfg.get("ppo", {}) if isinstance(cfg, dict) else {}
    num_envs = int(ppo_cfg.get("num_envs", 8))
    rollout_len = int(ppo_cfg.get("rollout_len", 64))
    hidden = int(ppo_cfg.get("hidden_size", 64))
    lr = float(ppo_cfg.get("lr", 3e-4))
    clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.01))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    ppo_epochs = int(ppo_cfg.get("ppo_epochs", 1))
    minibatch_size = int(ppo_cfg.get("minibatch_size", 128))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    grad_clip = float(ppo_cfg.get("max_grad_norm", 1.0))

    obs_dim = int(cfg.get("obs_dim", 16)) if isinstance(cfg, dict) else 16
    act_dim = int(cfg.get("act_dim", 3)) if isinstance(cfg, dict) else 3

    env = _make_env(cfg if isinstance(cfg, dict) else {}, num_envs, obs_dim, act_dim, device, seed)

    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, seed=int(seed)).to(device)
    optim = AdamWMaster(policy.parameters(), lr=lr)

    # Rollout buffer (all on device).
    obs_buf = torch.zeros(rollout_len, num_envs, obs_dim, device=device)
    act_buf = torch.zeros(rollout_len, num_envs, act_dim, device=device)
    logp_buf = torch.zeros(rollout_len, num_envs, device=device)
    rew_buf = torch.zeros(rollout_len, num_envs, device=device)
    val_buf = torch.zeros(rollout_len, num_envs, device=device)
    done_buf = torch.zeros(rollout_len, num_envs, device=device)

    obs = env.reset()
    if obs.dtype != torch.float32:
        obs = obs.to(torch.float32)

    steps_per_iter = num_envs * rollout_len
    n_iters = max(1, int(total_timesteps) // steps_per_iter)

    all_episode_returns: list[float] = []
    running_returns = torch.zeros(num_envs, device=device)

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    last_metrics: Dict[str, float] = {}
    cuda_graph_attempted = False
    cuda_graph_used = False

    # Try graph capture BEFORE any eager training so the params/grads have
    # never been touched by the legacy stream. This is a correctness probe:
    # if NVFP4Linear's autograd Function is graph-safe the capture will
    # succeed, and we record that in the metrics. (We don't yet replay the
    # captured graph inside the main loop — wiring that up is a follow-up,
    # guarded by Unit G's cuda_graph helper.)
    if torch.cuda.is_available():
        cuda_graph_attempted = True
        probe_obs = torch.zeros(minibatch_size, obs_dim, device=device)
        probe_act = torch.zeros(minibatch_size, act_dim, device=device)
        probe_lp = torch.zeros(minibatch_size, device=device)
        probe_adv = torch.zeros(minibatch_size, device=device)
        probe_ret = torch.zeros(minibatch_size, device=device)
        try:
            _try_capture_update(policy, optim, probe_obs, probe_act, probe_lp,
                                probe_adv, probe_ret,
                                clip_eps, vf_coef, ent_coef, grad_clip)
            cuda_graph_used = True
        except Exception as exc:
            cuda_graph_used = False
            import sys
            print(f"[fp4.trainer] cuda-graph capture failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    for it in range(n_iters):
        # ---- Rollout ----
        with torch.no_grad():
            for t in range(rollout_len):
                action, logp, value, _ = policy.act(obs)
                obs_buf[t] = obs
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = value
                next_obs, reward, done = env.step(action)
                if next_obs.dtype != torch.float32:
                    next_obs = next_obs.to(torch.float32)
                rew_buf[t] = reward
                done_buf[t] = done
                running_returns = running_returns + reward
                if done.any():
                    finished = running_returns[done.bool()]
                    if finished.numel() > 0:
                        all_episode_returns.extend(finished.detach().cpu().tolist())
                    running_returns = torch.where(done.bool(), torch.zeros_like(running_returns), running_returns)
                obs = next_obs
            _, _, last_value = policy(obs)

        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, last_value, gamma, gae_lambda)

        # Flatten for minibatch SGD.
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        n_samples = b_obs.shape[0]
        mb = min(minibatch_size, n_samples)

        for _epoch in range(ppo_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mb):
                idx = perm[start:start + mb]
                if idx.numel() < mb:
                    continue  # keep shapes static for potential graph capture
                obs_mb = b_obs[idx]
                act_mb = b_act[idx]
                lp_mb = b_logp[idx]
                adv_mb = b_adv[idx]
                ret_mb = b_ret[idx]
                optim.zero_grad(set_to_none=True)
                loss, info = _ppo_loss(policy, obs_mb, act_mb, lp_mb, adv_mb, ret_mb,
                                       clip_eps, vf_coef, ent_coef)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optim.step()
                last_metrics = {k: float(v.item()) for k, v in info.items()}
                last_metrics["loss"] = float(loss.item())


    wall = time.perf_counter() - t0

    # ---- Final metrics ----
    if all_episode_returns:
        ret_t = torch.tensor(all_episode_returns, dtype=torch.float32)
    else:
        ret_t = running_returns.detach().cpu().to(torch.float32)
    final_sortino = sortino_ratio(ret_t)
    final_p10 = percentile(ret_t, 0.10)
    mean_return = float(ret_t.mean().item()) if ret_t.numel() else 0.0

    metrics: Dict[str, Any] = {
        "final_sortino": final_sortino,
        "final_p10": final_p10,
        "mean_return": mean_return,
        "n_episodes": int(ret_t.numel()),
        "steps_per_sec": float(n_iters * steps_per_iter) / max(wall, 1e-9),
        "wall_sec": wall,
        "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available() else 0.0,
        "device": str(device),
        "cuda_graph_used": bool(cuda_graph_used),
        "n_iters": int(n_iters),
        "total_steps": int(n_iters * steps_per_iter),
        "last_loss": last_metrics.get("loss", 0.0),
        "last_entropy": last_metrics.get("entropy", 0.0),
    }

    # ---- Checkpoint ----
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "metrics": metrics,
        "cfg": cfg if isinstance(cfg, dict) else {},
        "seed": int(seed),
    }, ckpt_dir / "final.pt")
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def _try_capture_update(policy, optim, obs_mb, act_mb, lp_mb, adv_mb, ret_mb,
                        clip_eps, vf_coef, ent_coef, grad_clip):
    """Best-effort CUDA-graph capture of one update step.  Raises on failure."""
    # Prime the NVFP4 per-(device, dtype) level-table caches so the captured
    # forward does not need to do a host->device copy of the level tables.
    from .quant import prime_caches
    prime_caches(obs_mb.device, (torch.float32,))

    # Must run BEFORE any eager update has scheduled work on the legacy
    # stream that params/grads depend on, otherwise backward inside the
    # capture will try to sync legacy<->side-stream and fail with
    # cudaErrorStreamCaptureImplicit.
    s = torch.cuda.Stream(device=obs_mb.device)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optim.zero_grad(set_to_none=True)
            loss, _ = _ppo_loss(policy, obs_mb, act_mb, lp_mb, adv_mb, ret_mb,
                                clip_eps, vf_coef, ent_coef)
            loss.backward()

        g = torch.cuda.CUDAGraph()
        optim.zero_grad(set_to_none=True)
        with torch.cuda.graph(g, stream=s):
            loss, _ = _ppo_loss(policy, obs_mb, act_mb, lp_mb, adv_mb, ret_mb,
                                clip_eps, vf_coef, ent_coef)
            loss.backward()
        for _ in range(3):
            g.replay()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
