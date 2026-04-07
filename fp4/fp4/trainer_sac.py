"""SAC trainer variant for the fp4 library.

``train_sac(cfg, total_timesteps, seed, checkpoint_dir) -> dict`` mirrors the
signature of ``fp4.trainer.train_ppo`` so the bench harness can dispatch on it.

- Twin Q networks + target networks (soft Polyak updates).
- Automatic entropy temperature tuning toward ``target_entropy = -act_dim``.
- GPU-resident replay buffer (see ``fp4.replay.GPUReplayBuffer``).
- Uses the Layer A/B two-timescale policy from Unit P4-2 when importable,
  otherwise falls back to a tiny Gaussian MLP so this unit isn't blocked.
- Falls back to the stub vector env from ``fp4.trainer`` when the real market
  bindings aren't importable (same contract as the PPO trainer).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .optim import AdamWMaster
from .replay import GPUReplayBuffer
from .trainer import StubVecEnv, sortino_ratio, percentile


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class _SquashedGaussianPolicy(nn.Module):
    """Squashed-Gaussian MLP policy used when ``policy_two_layer`` is missing.

    Outputs actions in ``(-1, 1)`` via tanh-squashing and exposes the
    log-prob correction the SAC objective needs.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        action = torch.tanh(pre_tanh)
        # log N(pre_tanh; mu, std^2) — tanh squashing correction.
        logp = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2
                       + 2.0 * log_std + math.log(2 * math.pi))
        logp = logp.sum(dim=-1)
        logp = logp - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, logp


def _build_actor(obs_dim: int, act_dim: int, hidden: int) -> nn.Module:
    # Prefer P4-2's two-timescale policy if it's importable and exposes a
    # compatible `.sample(obs) -> (action, logp)` surface; otherwise fall back
    # to a plain squashed-Gaussian MLP so this unit isn't blocked on P4-2.
    try:
        from .policy_two_layer import TwoLayerPolicy  # type: ignore
        cand = TwoLayerPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        if hasattr(cand, "sample"):
            return cand
    except Exception:
        pass
    return _SquashedGaussianPolicy(obs_dim, act_dim, hidden)


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Env construction (mirrors trainer.py fallback pattern)
# ---------------------------------------------------------------------------

def _make_env(cfg: Dict[str, Any], num_envs: int, obs_dim: int, act_dim: int,
              device: torch.device, seed: int):
    env_spec = cfg.get("env", "stub") if isinstance(cfg, dict) else "stub"
    if hasattr(env_spec, "reset") and hasattr(env_spec, "step"):
        return env_spec
    episode_len = int(cfg.get("episode_len", 256)) if isinstance(cfg, dict) else 256
    return StubVecEnv(num_envs, obs_dim, act_dim, episode_len, device, seed)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def train_sac(cfg: Dict[str, Any], total_timesteps: int, seed: int,
              checkpoint_dir: str) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sac_cfg = {}
    if isinstance(cfg, dict):
        sac_cfg = cfg.get("sac", cfg.get("ppo", {})) or {}

    num_envs = int(sac_cfg.get("num_envs", 8))
    hidden = int(sac_cfg.get("hidden_size", 128))
    lr = float(sac_cfg.get("lr", 3e-4))
    gamma = float(sac_cfg.get("gamma", 0.99))
    tau = float(sac_cfg.get("tau", 0.005))
    batch_size = int(sac_cfg.get("batch_size", 256))
    capacity = int(sac_cfg.get("replay_capacity", 100_000))
    warmup_steps = int(sac_cfg.get("warmup_steps", 512))
    updates_per_step = int(sac_cfg.get("updates_per_step", 1))
    target_entropy_scale = float(sac_cfg.get("target_entropy_scale", 1.0))

    obs_dim = int(cfg.get("obs_dim", 16)) if isinstance(cfg, dict) else 16
    act_dim = int(cfg.get("act_dim", 3)) if isinstance(cfg, dict) else 3

    env = _make_env(cfg if isinstance(cfg, dict) else {}, num_envs, obs_dim, act_dim, device, seed)

    actor = _build_actor(obs_dim, act_dim, hidden).to(device)
    q1 = _QNet(obs_dim, act_dim, hidden).to(device)
    q2 = _QNet(obs_dim, act_dim, hidden).to(device)
    q1_tgt = _QNet(obs_dim, act_dim, hidden).to(device)
    q2_tgt = _QNet(obs_dim, act_dim, hidden).to(device)
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())
    for p in q1_tgt.parameters():
        p.requires_grad_(False)
    for p in q2_tgt.parameters():
        p.requires_grad_(False)

    actor_opt = AdamWMaster(actor.parameters(), lr=lr)
    q_params = list(q1.parameters()) + list(q2.parameters())
    q_opt = AdamWMaster(q_params, lr=lr)

    # Auto-tuned entropy temperature (learned via log_alpha).
    log_alpha = torch.zeros((), device=device, requires_grad=True)
    alpha_opt = AdamWMaster([log_alpha], lr=lr)
    target_entropy = -float(act_dim) * target_entropy_scale

    replay = GPUReplayBuffer(
        capacity=capacity, obs_dim=obs_dim, act_dim=act_dim, device=device,
    )

    obs = env.reset()
    if obs.dtype != torch.float32:
        obs = obs.to(torch.float32)

    all_episode_returns: list[float] = []
    running_returns = torch.zeros(num_envs, device=device)

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    last_metrics: Dict[str, float] = {}
    total_env_steps = 0
    max_env_steps = int(total_timesteps)

    while total_env_steps < max_env_steps:
        # ---- Env interaction ----
        with torch.no_grad():
            if total_env_steps < warmup_steps:
                action = torch.empty(num_envs, act_dim, device=device).uniform_(-1.0, 1.0)
            else:
                action, _ = actor.sample(obs)
            next_obs, reward, done = env.step(action)
            if next_obs.dtype != torch.float32:
                next_obs = next_obs.to(torch.float32)
            replay.add(obs, action, reward, next_obs, done)
            running_returns = running_returns + reward
            if done.any():
                finished = running_returns[done.bool()]
                if finished.numel() > 0:
                    all_episode_returns.extend(finished.detach().cpu().tolist())
                running_returns = torch.where(done.bool(), torch.zeros_like(running_returns), running_returns)
            obs = next_obs
            total_env_steps += num_envs

        # ---- Learn ----
        if len(replay) >= max(batch_size, warmup_steps):
            for _ in range(updates_per_step):
                batch = replay.sample(batch_size)
                b_obs = batch["obs"]
                b_act = batch["act"]
                b_rew = batch["rew"]
                b_next = batch["next_obs"]
                b_done = batch["done"]

                alpha = log_alpha.exp().detach()

                # --- Critic target ---
                with torch.no_grad():
                    next_a, next_logp = actor.sample(b_next)
                    q1_t = q1_tgt(b_next, next_a)
                    q2_t = q2_tgt(b_next, next_a)
                    q_t = torch.min(q1_t, q2_t) - alpha * next_logp
                    target_q = b_rew + gamma * (1.0 - b_done) * q_t

                q1_pred = q1(b_obs, b_act)
                q2_pred = q2(b_obs, b_act)
                q1_loss = F.mse_loss(q1_pred, target_q)
                q2_loss = F.mse_loss(q2_pred, target_q)
                q_loss = q1_loss + q2_loss

                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_params, 1.0)
                q_opt.step()

                # --- Actor ---
                new_a, new_logp = actor.sample(b_obs)
                q1_pi = q1(b_obs, new_a)
                q2_pi = q2(b_obs, new_a)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * new_logp - q_pi).mean()

                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()

                # --- Temperature ---
                alpha_loss = -(log_alpha * (new_logp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()

                # --- Target soft update ---
                with torch.no_grad():
                    for p, tp in zip(q1.parameters(), q1_tgt.parameters()):
                        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
                    for p, tp in zip(q2.parameters(), q2_tgt.parameters()):
                        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

                last_metrics = {
                    "q_loss": float(q_loss.detach().item()),
                    "actor_loss": float(actor_loss.detach().item()),
                    "alpha": float(alpha.item()),
                    "alpha_loss": float(alpha_loss.detach().item()),
                }

    wall = time.perf_counter() - t0

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
        "steps_per_sec": float(total_env_steps) / max(wall, 1e-9),
        "wall_sec": wall,
        "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available() else 0.0,
        "device": str(device),
        "total_steps": int(total_env_steps),
        "replay_size": int(len(replay)),
        "last_q_loss": last_metrics.get("q_loss", 0.0),
        "last_actor_loss": last_metrics.get("actor_loss", 0.0),
        "last_alpha": last_metrics.get("alpha", 0.0),
        "algorithm": "sac",
    }

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "q1_tgt": q1_tgt.state_dict(),
        "q2_tgt": q2_tgt.state_dict(),
        "log_alpha": log_alpha.detach(),
        "metrics": metrics,
        "cfg": cfg if isinstance(cfg, dict) else {},
        "seed": int(seed),
    }, ckpt_dir / "final.pt")
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
