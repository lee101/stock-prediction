"""PPO trainer for PortfolioBracketEnv (continuous [B, S, 4] action).

Native vectorised: the env IS a B-parallel batch in a single CUDA kernel,
so we skip PufferLib's vec-env wrapping entirely and write a direct PPO
loop that calls env.step() once per rollout step. The actor/critic forward
runs in BF16 autocast on the 5090; PPO update does standard fp32 math on
the loss.

Action layout per (env, sym):
    a[..., 0] = lim_buy_offset_bps    in ~[-30, 30]
    a[..., 1] = lim_sell_offset_bps   in ~[-30, 30]
    a[..., 2] = buy_qty_pct           >= 0  (>1 OK; leverage cap clips)
    a[..., 3] = sell_qty_pct          >= 0

Distribution: independent Gaussian per (sym, dim) with shared learned
log_std. Sampled means are passed straight to the env (the env clips
qty_pct to [0, +inf) inside the kernel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Actor + Critic
# ---------------------------------------------------------------------------

class PortfolioBracketActor(nn.Module):
    """Shared MLP encoder + Gaussian actor head + scalar critic.

    obs: [B, F_obs] float32
    output:
        mu      [B, S, 4]
        log_std [S, 4]    (state-independent, learned)
        value   [B]
    """

    def __init__(
        self,
        obs_dim: int,
        num_symbols: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        bps_scale: float = 30.0,
        qty_scale: float = 0.3,
        log_std_init: float = -1.5,
    ):
        super().__init__()
        self.num_symbols = num_symbols
        self.bps_scale = bps_scale
        self.qty_scale = qty_scale

        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        self.encoder = nn.Sequential(*layers)

        self.actor_head = nn.Linear(hidden_dim, num_symbols * 4)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.log_std = nn.Parameter(torch.full((num_symbols, 4), log_std_init))

        # Init last actor layer small so initial actions are near zero.
        nn.init.zeros_(self.actor_head.bias)
        nn.init.normal_(self.actor_head.weight, std=0.01)

    def _scales(self, device, dtype):
        # Per-dim scale: [bps, bps, qty, qty]
        return torch.tensor(
            [self.bps_scale, self.bps_scale, self.qty_scale, self.qty_scale],
            device=device, dtype=dtype,
        )

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        raw = self.actor_head(h).reshape(-1, self.num_symbols, 4)
        scale = self._scales(raw.device, raw.dtype)
        # Bound the mean smoothly so the policy can't sample huge bps offsets.
        mu = torch.tanh(raw) * scale
        value = self.critic_head(h).squeeze(-1)
        return mu, value

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 0.5

    def _log_std_clamped(self) -> torch.Tensor:
        return self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def dist_log_prob(self, mu: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # action: [B, S, 4]; sum log_prob across (S, 4).
        log_std = self._log_std_clamped()
        std = log_std.exp().to(mu.dtype)                 # [S, 4]
        log_prob = -0.5 * ((action - mu) / std).pow(2) \
                   - log_std.to(mu.dtype) \
                   - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=mu.device, dtype=mu.dtype))
        return log_prob.flatten(1).sum(dim=-1)           # [B]

    def entropy(self) -> torch.Tensor:
        # Diff entropy of Gaussian summed across (S, 4): independent of state.
        log_std = self._log_std_clamped()
        return (0.5 + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=log_std.device))
                + log_std).sum()


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 4
    minibatches: int = 4
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.05        # early-stop epoch loop if approx_kl exceeds this
    bf16: bool = True
    reward_scale: float = 1.0
    reward_clip: float = 5.0       # hard clip on per-step reward to keep PPO stable
    log_every: int = 1


@dataclass
class RolloutBuffer:
    obs:        torch.Tensor   # [T, B, F_obs]
    actions:    torch.Tensor   # [T, B, S, 4]
    log_probs:  torch.Tensor   # [T, B]
    rewards:    torch.Tensor   # [T, B]
    values:     torch.Tensor   # [T, B]
    dones:      torch.Tensor   # [T, B]


def collect_rollout(
    env,
    policy: PortfolioBracketActor,
    cfg: PPOConfig,
) -> tuple[RolloutBuffer, torch.Tensor]:
    T = cfg.rollout_steps
    B = env.B
    S = env.S

    obs0 = env.obs()
    F_obs = obs0.shape[-1]
    device = obs0.device

    obs_buf      = torch.empty(T, B, F_obs, device=device, dtype=torch.float32)
    action_buf   = torch.empty(T, B, S, 4, device=device, dtype=torch.float32)
    logp_buf     = torch.empty(T, B, device=device, dtype=torch.float32)
    reward_buf   = torch.empty(T, B, device=device, dtype=torch.float32)
    value_buf    = torch.empty(T, B, device=device, dtype=torch.float32)
    done_buf     = torch.empty(T, B, device=device, dtype=torch.float32)

    cur_obs = obs0
    autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    cur_obs = _sanitize(cur_obs)

    for t in range(T):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=cfg.bf16):
                mu, value = policy(cur_obs)
            std = policy._log_std_clamped().exp().to(mu.dtype)
            noise = torch.randn_like(mu)
            action = (mu + std * noise).to(torch.float32)
            log_prob = policy.dist_log_prob(mu.float(), action)

        reward, done, _info = env.step(action)

        obs_buf[t]    = cur_obs
        action_buf[t] = action
        logp_buf[t]   = log_prob
        # Reward is per-step relative return; can spike if env equity nearly
        # collapsed. Sanitize then clip so PPO targets stay bounded.
        scaled_r = torch.nan_to_num(reward * cfg.reward_scale,
                                    nan=0.0, posinf=cfg.reward_clip,
                                    neginf=-cfg.reward_clip)
        scaled_r = scaled_r.clamp(-cfg.reward_clip, cfg.reward_clip)
        reward_buf[t] = scaled_r
        # Force-reset envs whose equity has gone non-positive — the env's
        # auto-reset only fires on episode_len/T-end, not on bankruptcy.
        bankrupt = env.state["equity"] <= 1e-3
        if bankrupt.any():
            env.reset(bankrupt)
        value_buf[t]  = value.float()
        done_buf[t]   = done.to(torch.float32)

        cur_obs = _sanitize(env.obs())

    # Bootstrap value for the final step.
    with torch.no_grad():
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=cfg.bf16):
            _mu_last, last_value = policy(cur_obs)
        last_value = torch.nan_to_num(last_value.float(),
                                      nan=0.0, posinf=cfg.reward_clip,
                                      neginf=-cfg.reward_clip)

    buf = RolloutBuffer(obs_buf, action_buf, logp_buf, reward_buf, value_buf, done_buf)
    return buf, last_value


def compute_gae(buf: RolloutBuffer, last_value: torch.Tensor, cfg: PPOConfig):
    T, B = buf.rewards.shape
    advantages = torch.zeros_like(buf.rewards)
    last_adv = torch.zeros(B, device=buf.rewards.device, dtype=torch.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = buf.values[t + 1]
        not_done = 1.0 - buf.dones[t]
        delta = buf.rewards[t] + cfg.gamma * next_value * not_done - buf.values[t]
        last_adv = delta + cfg.gamma * cfg.gae_lambda * not_done * last_adv
        advantages[t] = last_adv
    returns = advantages + buf.values
    return advantages, returns


def ppo_update(
    buf: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    policy: PortfolioBracketActor,
    optim: torch.optim.Optimizer,
    cfg: PPOConfig,
) -> dict:
    T, B = buf.rewards.shape
    F_obs = buf.obs.shape[-1]
    S = buf.actions.shape[2]

    flat_obs   = buf.obs.reshape(T * B, F_obs)
    flat_act   = buf.actions.reshape(T * B, S, 4)
    flat_logp  = buf.log_probs.reshape(T * B)
    flat_adv   = advantages.reshape(T * B)
    flat_ret   = returns.reshape(T * B)

    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    N = T * B
    mb_size = max(1, N // cfg.minibatches)
    autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    pg_losses, vf_losses, ents, kls = [], [], [], []
    early_stopped = False
    for epoch in range(cfg.epochs):
        if early_stopped:
            break
        perm = torch.randperm(N, device=flat_obs.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            mb_obs  = flat_obs[idx]
            mb_act  = flat_act[idx]
            mb_logp = flat_logp[idx]
            mb_adv  = flat_adv[idx]
            mb_ret  = flat_ret[idx]

            with torch.autocast("cuda", dtype=autocast_dtype, enabled=cfg.bf16):
                mu, value = policy(mb_obs)
            mu = mu.float()
            value = value.float()
            new_logp = policy.dist_log_prob(mu, mb_act)
            ratio = (new_logp - mb_logp).exp()

            unclipped = ratio * mb_adv
            clipped   = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            pg_loss = -torch.min(unclipped, clipped).mean()
            vf_loss = 0.5 * (value - mb_ret).pow(2).mean()
            ent     = policy.entropy()
            loss    = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optim.step()

            with torch.no_grad():
                approx_kl = (mb_logp - new_logp).mean()
            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ents.append(ent.item())
            kls.append(approx_kl.item())
            if cfg.target_kl > 0 and approx_kl.item() > cfg.target_kl:
                early_stopped = True
                break

    return {
        "pg_loss":   sum(pg_losses) / len(pg_losses),
        "vf_loss":   sum(vf_losses) / len(vf_losses),
        "entropy":   sum(ents) / len(ents),
        "approx_kl": sum(kls) / len(kls),
    }


def train(
    env,
    policy: Optional[PortfolioBracketActor] = None,
    cfg: Optional[PPOConfig] = None,
    iters: int = 100,
) -> dict:
    cfg = cfg or PPOConfig()
    if policy is None:
        obs_dim = env.obs().shape[-1]
        policy = PortfolioBracketActor(
            obs_dim=obs_dim, num_symbols=env.S,
        ).to("cuda")
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    history = {
        "iter": [], "mean_reward": [], "pg_loss": [], "vf_loss": [],
        "entropy": [], "approx_kl": [],
    }
    for it in range(iters):
        buf, last_value = collect_rollout(env, policy, cfg)
        adv, ret = compute_gae(buf, last_value, cfg)
        stats = ppo_update(buf, adv, ret, policy, optim, cfg)
        mean_r = buf.rewards.mean().item()
        history["iter"].append(it)
        history["mean_reward"].append(mean_r)
        history["pg_loss"].append(stats["pg_loss"])
        history["vf_loss"].append(stats["vf_loss"])
        history["entropy"].append(stats["entropy"])
        history["approx_kl"].append(stats["approx_kl"])
        if (it + 1) % cfg.log_every == 0:
            print(
                f"[ppo it={it+1}/{iters}] mean_r={mean_r:+.5f} "
                f"pg={stats['pg_loss']:+.4f} vf={stats['vf_loss']:.4f} "
                f"ent={stats['entropy']:+.3f} kl={stats['approx_kl']:+.4f}",
                flush=True,
            )
    return {"policy": policy, "history": history}


__all__ = ["PortfolioBracketActor", "PPOConfig", "RolloutBuffer",
           "collect_rollout", "compute_gae", "ppo_update", "train"]
