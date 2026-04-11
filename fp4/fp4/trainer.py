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


class _Env3Tuple:
    """Adapt an ``env_adapter.EnvHandle`` (4-tuple step) to the legacy
    ``(obs, reward, done)`` 3-tuple contract used inside the trainers."""

    def __init__(self, handle):
        self._h = handle
        self.num_envs = int(handle.num_envs)
        self.obs_dim = int(handle.obs_dim)
        self.act_dim = int(handle.action_dim)
        self.backend_name = handle.backend_name

    def reset(self):
        return self._h.reset()

    def step(self, action):
        obs, rew, done, _cost = self._h.step(action)
        return obs, rew, done


def _make_env(cfg: Dict[str, Any], num_envs: int, obs_dim: int, act_dim: int,
              device: torch.device, seed: int):
    """Legacy entry point retained for callers that still import it from
    ``fp4.trainer`` directly. New code should go through ``env_adapter.make_env``
    + ``_Env3Tuple`` (used inline in ``train_ppo``/``train_sac``/``train_qr_ppo``).
    This shim preserves backwards compatibility: it routes a dict env spec
    through ``env_adapter`` and falls back to ``StubVecEnv`` only when the
    adapter is not importable.
    """
    env_spec = cfg.get("env", "stub")
    if hasattr(env_spec, "reset") and hasattr(env_spec, "step"):
        return env_spec
    if isinstance(env_spec, dict):
        try:
            from .env_adapter import make_env as _adapter_make_env
            return _Env3Tuple(_adapter_make_env(
                cfg, num_envs=num_envs, obs_dim=obs_dim,
                act_dim=act_dim, device=device, seed=seed,
            ))
        except Exception as exc:
            import sys as _sys
            print(
                f"[fp4.trainer] env_adapter.make_env unavailable "
                f"({type(exc).__name__}: {exc}); falling back to StubVecEnv",
                file=_sys.stderr,
            )
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

def _save_checkpoint(ckpt_dir: Path, name: str, policy, cfg, seed, metrics):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "metrics": metrics,
        "cfg": cfg if isinstance(cfg, dict) else {},
        "seed": int(seed),
    }, ckpt_dir / name)


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

    from .env_adapter import make_env as _adapter_make_env
    _raw_env = _adapter_make_env(cfg if isinstance(cfg, dict) else {},
                                 num_envs=num_envs, obs_dim=obs_dim,
                                 act_dim=act_dim, device=device, seed=seed)
    env_backend_name = _raw_env.backend_name
    env = _Env3Tuple(_raw_env)
    # Re-read shapes from the constructed env so the policy + rollout buffers
    # match the real environment, not whatever default lived in cfg. This is
    # what unblocks evaluation against the marketsim — the policy is born
    # the right size from the start.
    obs_dim = int(getattr(env, "obs_dim", obs_dim))
    act_dim = int(getattr(env, "act_dim", act_dim))
    num_envs = int(getattr(env, "num_envs", num_envs))

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
    _last_info_t: Dict[str, torch.Tensor] | None = None
    _last_loss_t: torch.Tensor | None = None
    # Sync-free rollout accounting: stage finished-episode returns into a
    # GPU buffer each step, then host-sync once per iter (instead of per-step
    # via .cpu().tolist()).
    finished_buf = torch.zeros(rollout_len, num_envs, device=device)

    # ---- Phase 6 Unit P6-2: 27%/month Lagrangian + leverage curriculum ----
    # Both opt-in via cfg['lagrangian'] / cfg['leverage_curriculum']. When
    # present, the reward buffer is shaped after each rollout: a drawdown
    # penalty (per-step dd growth * lambda_dd) is subtracted and a target
    # residual bonus (lambda_pnl * positive_reward) is added. The Lagrangian
    # multipliers are updated once per iter from the per-iter equity curve.
    lagrangian = None
    lag_cfg = cfg.get("lagrangian") if isinstance(cfg, dict) else None
    if isinstance(lag_cfg, dict):
        try:
            from .lagrangian import Lagrangian
            targets = lag_cfg.get("targets", {}) or {}
            # We represent the "monthly PnL" constraint with a cost = -actual,
            # target = -target_monthly (so cost - target = target - actual,
            # positive when we're below target). max_dd cost = avg_worst_dd.
            target_d = {
                "monthly_pnl": -float(targets.get("monthly_pnl", 0.27)),
                "max_dd": float(targets.get("max_dd", 0.08)),
            }
            lagrangian = Lagrangian(
                constraint_names=list(target_d.keys()),
                init_lambda=0.0,
                lr_lambda=float(lag_cfg.get("lr_lambda", 1e-2)),
                slow_update_every=int(lag_cfg.get("slow_update_every", 1)),
                target_d=target_d,
            )
        except Exception as _exc:
            import sys as _sys
            print(f"[fp4.trainer] lagrangian disabled: {_exc}", file=_sys.stderr)
            lagrangian = None

    lev_curriculum = None
    lev_cfg = cfg.get("leverage_curriculum") if isinstance(cfg, dict) else None
    if isinstance(lev_cfg, dict):
        try:
            from .curriculum import LeverageCurriculum
            lev_curriculum = LeverageCurriculum(
                start=float(lev_cfg.get("start", 1.0)),
                target=float(lev_cfg.get("target", 2.0)),
                ramp_steps=int(lev_cfg.get("ramp_steps", 500_000)),
                cap=float(lev_cfg.get("cap", 5.0)),
                step_size=float(lev_cfg.get("step_size", 0.05)),
            )
        except Exception as _exc:
            import sys as _sys
            print(f"[fp4.trainer] curriculum disabled: {_exc}", file=_sys.stderr)
            lev_curriculum = None

    # Per-iter telemetry we'll record in the metrics JSON.
    last_monthly_return: float = 0.0
    last_avg_max_dd: float = 0.0
    last_lam_pnl: float = 0.0
    last_lam_dd: float = 0.0
    last_leverage_cap: float = float(lev_cfg.get("start", 1.0)) if isinstance(lev_cfg, dict) else 0.0
    cuda_graph_attempted = False
    cuda_graph_used = False

    # ----- P5-3 opt-in: full rollout+update CUDA graph capture -----
    # Captures one full PPO iter (rollout on the real env + update) into a
    # single CUDA graph, replays it `n_iters` times. After each replay we
    # read the persistent finished_buf + done_buf to harvest real episode
    # returns, so sortino / n_episodes / entropy are all real, not stubbed.
    if (isinstance(cfg, dict) and cfg.get("full_graph_capture")
            and torch.cuda.is_available()):
        from .cuda_graph_full import build_real_full_step, capture_full_step
        step_fn, fg_state = build_real_full_step(
            _raw_env, policy, optim,
            rollout_len=rollout_len, device=device,
            gamma=gamma, gae_lambda=gae_lambda,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
        )
        captured = capture_full_step(step_fn)
        best_sortino = -float("inf")
        last_entropy_f = 0.0
        last_loss_f = 0.0
        last_n_done_f = 0.0
        for it in range(n_iters):
            captured.replay()
            # Per-iter host sync to harvest finished episode returns. This
            # is cheap (one sync per iter, not per step).
            done_mask_flat = (fg_state["done_buf"] > 0.5).reshape(-1)
            if bool(done_mask_flat.any().item()):
                all_episode_returns.extend(
                    fg_state["finished_buf"].reshape(-1)[done_mask_flat].cpu().tolist()
                )
            last_entropy_f = float(fg_state["entropy"].item())
            last_loss_f = float(fg_state["loss"].item())
            last_n_done_f = float(fg_state["n_done"].item())
            # Track best-so-far so we can save a best.pt checkpoint that
            # `scripts/eval_100d.py` can consume.
            if all_episode_returns:
                running_ret_t = torch.tensor(all_episode_returns[-max(256, num_envs):],
                                             dtype=torch.float32)
                cur_sortino = sortino_ratio(running_ret_t)
                if cur_sortino > best_sortino:
                    best_sortino = cur_sortino
                    _save_checkpoint(Path(checkpoint_dir), "best.pt",
                                     policy, cfg, seed, {"sortino": cur_sortino,
                                                         "iter": it})
            if lev_curriculum is not None:
                r_flat = fg_state["rew_buf"].detach().reshape(-1)
                downside = r_flat.clamp_max(0.0)
                dd_std = torch.sqrt((downside * downside).mean() + 1e-8)
                proxy_sortino = float((r_flat.mean() / (dd_std + 1e-8)).item())
                last_leverage_cap = lev_curriculum.current_cap(
                    step=int((it + 1) * steps_per_iter), last_sortino=proxy_sortino
                )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if all_episode_returns:
            ret_t = torch.tensor(all_episode_returns, dtype=torch.float32)
        else:
            ret_t = fg_state["running_returns"].detach().cpu().to(torch.float32)
        final_sortino = sortino_ratio(ret_t)
        final_p10 = percentile(ret_t, 0.10)
        mean_return = float(ret_t.mean().item()) if ret_t.numel() else 0.0
        metrics = {
            "final_sortino": final_sortino,
            "final_p10": final_p10,
            "mean_return": mean_return,
            "n_episodes": int(ret_t.numel()),
            "steps_per_sec": float(n_iters * steps_per_iter) / max(wall, 1e-9),
            "wall_sec": wall,
            "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "device": str(device),
            "cuda_graph_used": True,
            "full_graph_used": True,
            "env_backend": env_backend_name,
            "n_iters": int(n_iters),
            "total_steps": int(n_iters * steps_per_iter),
            "last_loss": last_loss_f,
            "last_entropy": last_entropy_f,
            "last_leverage_cap": float(last_leverage_cap),
            "n_done_last_iter": last_n_done_f,
        }
        # Always write final.pt too. If best.pt never got written (e.g. no
        # episodes closed), promote final.pt as the only ckpt the evaluator
        # can consume.
        _save_checkpoint(Path(checkpoint_dir), "final.pt", policy, cfg, seed, metrics)
        if not (Path(checkpoint_dir) / "best.pt").exists():
            _save_checkpoint(Path(checkpoint_dir), "best.pt", policy, cfg, seed, metrics)
        (Path(checkpoint_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics
    # ----- end P5-3 branch -----

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
                # Sync-free episode-return capture: stage `running_returns`
                # into finished_buf[t] at finished envs only; reset
                # `running_returns` via masked multiply (no host branch).
                done_f = done.to(running_returns.dtype)
                finished_buf[t] = running_returns * done_f
                running_returns = running_returns * (1.0 - done_f)
                obs = next_obs
            _, _, last_value = policy(obs)
            # Single host sync per rollout iter to harvest finished returns.
            done_mask_flat = (done_buf > 0.5).reshape(-1)
            if bool(done_mask_flat.any().item()):
                all_episode_returns.extend(
                    finished_buf.reshape(-1)[done_mask_flat].cpu().tolist()
                )

        # ---- Lagrangian reward shaping (P6-2 opt-in) ----
        # Equity curve from rewards: eq[n, t] = 1 + cumsum_t rew[n, t].
        # Shape: rew_buf is [T, N]; transpose to [N, T].
        if lagrangian is not None:
            rew_NT = rew_buf.detach().transpose(0, 1)  # [N, T]
            eq = 1.0 + torch.cumsum(rew_NT, dim=1)  # [N, T]
            peak = torch.cummax(eq, dim=1).values
            dd_path = (peak - eq).clamp_min(0.0)  # [N, T] non-negative
            try:
                from .objective_27pct import monthly_return_value
                actual_monthly_t = monthly_return_value(eq)
                actual_monthly = float(actual_monthly_t.item())
            except Exception:
                actual_monthly = 0.0
            avg_max_dd = float(dd_path.amax(dim=1).mean().item())
            lam_state = lagrangian.step({
                "monthly_pnl": -actual_monthly,   # cost = -return; constraint satisfied when actual >= target
                "max_dd": avg_max_dd,
            })
            lam_pnl = float(lam_state.get("monthly_pnl", 0.0))
            lam_dd = float(lam_state.get("max_dd", 0.0))
            last_monthly_return = actual_monthly
            last_avg_max_dd = avg_max_dd
            last_lam_pnl = lam_pnl
            last_lam_dd = lam_dd
            # Drawdown-growth per step (on-device): dd_growth[n, t] = max(0, dd[t] - dd[t-1]).
            dd_prev = torch.cat([dd_path[:, :1], dd_path[:, :-1]], dim=1)
            dd_growth = (dd_path - dd_prev).clamp_min(0.0)  # [N, T]
            dd_growth_TN = dd_growth.transpose(0, 1)        # [T, N]
            # Shape rewards: penalise drawdown growth, amplify upside when behind target.
            shaped = rew_buf - lam_dd * dd_growth_TN
            if lam_pnl > 0.0:
                shaped = shaped + lam_pnl * rew_buf.clamp_min(0.0)
            rew_buf = shaped

        # ---- Leverage curriculum telemetry ----
        if lev_curriculum is not None:
            # Use a running proxy Sortino from the iter: mean/std of rew_buf.
            r_flat = rew_buf.detach().reshape(-1)
            downside = r_flat.clamp_max(0.0)
            dd_std = torch.sqrt((downside * downside).mean() + 1e-8)
            proxy_sortino = float((r_flat.mean() / (dd_std + 1e-8)).item())
            last_leverage_cap = lev_curriculum.current_cap(
                step=int(it * steps_per_iter), last_sortino=proxy_sortino
            )

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
                # Defer host syncs: stash detached tensors and convert to
                # python floats at the very end (saves N_minibatch syncs/iter).
                _last_info_t = info
                _last_loss_t = loss.detach()


    wall = time.perf_counter() - t0

    # Convert deferred metrics to python floats once, after the training loop.
    if _last_info_t is not None and _last_loss_t is not None:
        last_metrics = {k: float(v.item()) for k, v in _last_info_t.items()}
        last_metrics["loss"] = float(_last_loss_t.item())

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
        "env_backend": env_backend_name,
        "n_iters": int(n_iters),
        "total_steps": int(n_iters * steps_per_iter),
        "last_loss": last_metrics.get("loss", 0.0),
        "last_entropy": last_metrics.get("entropy", 0.0),
        "last_monthly_return": float(last_monthly_return),
        "last_avg_max_dd": float(last_avg_max_dd),
        "last_lambda_monthly_pnl": float(last_lam_pnl),
        "last_lambda_max_dd": float(last_lam_dd),
        "last_leverage_cap": float(last_leverage_cap),
        "lagrangian_enabled": bool(lagrangian is not None),
        "leverage_curriculum_enabled": bool(lev_curriculum is not None),
    }

    # ---- Checkpoint ----
    ckpt_dir = Path(checkpoint_dir)
    _save_checkpoint(ckpt_dir, "final.pt", policy, cfg, seed, metrics)
    # Promote final.pt as best.pt if no best-so-far exists (eval_100d gates
    # on the presence of a best.pt file).
    if not (ckpt_dir / "best.pt").exists():
        _save_checkpoint(ckpt_dir, "best.pt", policy, cfg, seed, metrics)
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
