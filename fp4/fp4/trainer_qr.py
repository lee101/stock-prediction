"""Distributional PPO (QR-PPO) trainer.

Identical to ``fp4.trainer.train_ppo`` except the scalar value head is
replaced by a ``QuantileValueHead`` and the value loss becomes the
quantile Huber loss against sample returns.

The public entry ``train_qr_ppo(cfg, total_timesteps, seed, checkpoint_dir)``
matches the signature contract of ``train_ppo`` / ``train_sac`` so the
benchmark dispatch in ``fp4/bench/bench_trading.py`` can call any of them
uniformly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn

from .distributional import QuantileValueHead, cvar_from_quantiles, quantile_huber_loss
from .optim import AdamWMaster
from .policy import ActorCritic
from .trainer import _make_env, compute_gae, percentile, sortino_ratio


class QuantileActorCritic(nn.Module):
    """Actor-critic with a distributional (quantile) value head.

    Reuses ``ActorCritic``'s feature extractor, log-std and action logic so
    the rollout code path stays bit-identical to ``train_ppo``.  Only the
    value branch changes: instead of a scalar, we produce K quantiles and
    expose ``.mean_value`` for bootstrap/GAE (and optional CVaR).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 num_quantiles: int = 51, seed: int = 0):
        super().__init__()
        self.backbone = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, seed=seed)
        # Replace the scalar value head with a quantile head.  The backbone's
        # v_head is kept unused to preserve NVFP4Linear init determinism.
        self.num_quantiles = int(num_quantiles)
        self.q_head = QuantileValueHead(in_dim=hidden, num_quantiles=num_quantiles)

    @property
    def log_std(self) -> torch.Tensor:
        return self.backbone.log_std

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(obs)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean, std, scalar_value, quantiles)."""
        z = self.backbone.features(obs)
        mean = self.backbone.pi_head(z)
        quantiles = self.q_head(z)  # [B, K]
        value = self.q_head.mean_value(quantiles)
        std = torch.exp(self.backbone.log_std).expand_as(mean)
        return mean, std, value, quantiles

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value, _q = self.forward(obs)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        logp = ActorCritic.gaussian_logprob(mean, std, action)
        return action, logp, value, mean


def _qr_ppo_loss(policy: QuantileActorCritic, obs_b, act_b, old_logp_b, adv_b, ret_b,
                 clip_eps: float, vf_coef: float, ent_coef: float, kappa: float
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mean, std, _value, quantiles = policy(obs_b)
    new_logp = ActorCritic.gaussian_logprob(mean, std, act_b)
    entropy = ActorCritic.gaussian_entropy(std).mean()
    ratio = torch.exp(new_logp - old_logp_b)
    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = quantile_huber_loss(quantiles, ret_b, kappa=kappa)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
    }


def train_qr_ppo(cfg: Dict[str, Any], total_timesteps: int, seed: int,
                 checkpoint_dir: str) -> Dict[str, Any]:
    """GPU-resident distributional PPO. Mirrors ``train_ppo`` exactly
    apart from the quantile value head + quantile Huber loss."""
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
    num_quantiles = int(ppo_cfg.get("num_quantiles", 51))
    kappa = float(ppo_cfg.get("huber_kappa", 1.0))
    cvar_alpha = float(ppo_cfg.get("cvar_alpha", 0.05))

    obs_dim = int(cfg.get("obs_dim", 16)) if isinstance(cfg, dict) else 16
    act_dim = int(cfg.get("act_dim", 3)) if isinstance(cfg, dict) else 3

    from .env_adapter import make_env as _adapter_make_env
    from .trainer import _Env3Tuple
    _raw_env = _adapter_make_env(cfg if isinstance(cfg, dict) else {},
                                 num_envs=num_envs, obs_dim=obs_dim,
                                 act_dim=act_dim, device=device, seed=seed)
    env_backend_name = _raw_env.backend_name
    env = _Env3Tuple(_raw_env)
    obs_dim = int(getattr(env, "obs_dim", obs_dim))
    act_dim = int(getattr(env, "act_dim", act_dim))
    num_envs = int(getattr(env, "num_envs", num_envs))

    policy = QuantileActorCritic(
        obs_dim=obs_dim, act_dim=act_dim, hidden=hidden,
        num_quantiles=num_quantiles, seed=int(seed),
    ).to(device)
    optim = AdamWMaster(policy.parameters(), lr=lr)

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
    last_cvar: float = 0.0

    for _it in range(n_iters):
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
            _, _, last_value, last_q = policy(obs)
            last_cvar = float(cvar_from_quantiles(last_q, alpha=cvar_alpha).mean().item())

        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, last_value, gamma, gae_lambda)

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
                    continue
                obs_mb = b_obs[idx]
                act_mb = b_act[idx]
                lp_mb = b_logp[idx]
                adv_mb = b_adv[idx]
                ret_mb = b_ret[idx]
                optim.zero_grad(set_to_none=True)
                loss, info = _qr_ppo_loss(policy, obs_mb, act_mb, lp_mb, adv_mb, ret_mb,
                                          clip_eps, vf_coef, ent_coef, kappa)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optim.step()
                last_metrics = {k: float(v.item()) for k, v in info.items()}
                last_metrics["loss"] = float(loss.item())

    wall = time.perf_counter() - t0

    if all_episode_returns:
        ret_t = torch.tensor(all_episode_returns, dtype=torch.float32)
    else:
        ret_t = running_returns.detach().cpu().to(torch.float32)
    final_sortino = sortino_ratio(ret_t)
    final_p10 = percentile(ret_t, 0.10)
    mean_return = float(ret_t.mean().item()) if ret_t.numel() else 0.0

    metrics: Dict[str, Any] = {
        "trainer": "qr_ppo",
        "final_sortino": final_sortino,
        "final_p10": final_p10,
        "mean_return": mean_return,
        "n_episodes": int(ret_t.numel()),
        "steps_per_sec": float(n_iters * steps_per_iter) / max(wall, 1e-9),
        "wall_sec": wall,
        "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available() else 0.0,
        "device": str(device),
        "num_quantiles": num_quantiles,
        "cvar_alpha": cvar_alpha,
        "last_cvar": last_cvar,
        "n_iters": int(n_iters),
        "total_steps": int(n_iters * steps_per_iter),
        "last_loss": last_metrics.get("loss", 0.0),
        "last_value_loss": last_metrics.get("value_loss", 0.0),
        "last_entropy": last_metrics.get("entropy", 0.0),
        "env_backend": env_backend_name,
    }

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "metrics": metrics,
        "cfg": cfg if isinstance(cfg, dict) else {},
        "seed": int(seed),
        "num_quantiles": num_quantiles,
    }, ckpt_dir / "final.pt")
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
