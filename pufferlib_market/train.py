"""
PPO training for the C trading environment using PufferLib.

Designed to stay under 8GB GPU memory:
  - Small MLP policy (~500K params)
  - 64 parallel envs
  - 256-step rollouts
  - Batch size 2048

Improvements from autoresearch agent (2026-03):
  - RunningObsNorm: online observation normalization
  - Cosine LR with warmup (5% floor)
  - Entropy annealing (start→end over training)
  - Clip eps annealing (start→end over training)
  - Value function clipping (PPO-style)
  - AdamW with weight decay
  - TF32 for faster matmuls
  - non_blocking GPU transfers
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# PufferLib imports
import pufferlib
import pufferlib.vector

# Local
from pufferlib_market.environment import TradingEnvConfig, TradingEnv
from pufferlib_market.metrics import annualize_total_return


# ─── Running Observation Normalizer ──────────────────────────────────


class RunningObsNorm:
    """Online observation normalization using Welford's algorithm."""

    def __init__(self, shape: int, clip: float = 10.0, eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # avoid div-by-zero
        self.clip = clip
        self.eps = eps

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of observations."""
        batch_mean = batch.mean(axis=0).astype(np.float64)
        batch_var = batch.var(axis=0).astype(np.float64)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics."""
        std = np.sqrt(self.var + self.eps).astype(np.float32)
        normed = (obs - self.mean.astype(np.float32)) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


# ─── Schedule Helpers ────────────────────────────────────────────────


def cosine_lr_with_warmup(update: int, num_updates: int, base_lr: float,
                          warmup_frac: float = 0.02, min_ratio: float = 0.05) -> float:
    """Cosine annealing LR with linear warmup and minimum floor."""
    warmup_updates = int(num_updates * warmup_frac)
    if update <= warmup_updates:
        return base_lr * update / max(warmup_updates, 1)
    progress = (update - warmup_updates) / max(num_updates - warmup_updates, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * max(cosine_decay, min_ratio)


def linear_anneal(update: int, num_updates: int, start: float, end: float) -> float:
    """Linearly anneal a value from start to end over training."""
    frac = min(update / max(num_updates, 1), 1.0)
    return start + (end - start) * frac


# ─── Policy Network ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ResumeState:
    update: int = 0
    global_step: int = 0
    best_return: float = -float("inf")


def _checkpoint_payload(
    policy: nn.Module,
    optimizer: optim.Optimizer,
    *,
    update: int,
    global_step: int,
    best_return: float,
    disable_shorts: bool,
    action_meta: dict[str, int | float],
) -> dict[str, object]:
    return {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "global_step": int(global_step),
        "best_return": float(best_return),
        "disable_shorts": bool(disable_shorts),
        **action_meta,
    }


def _optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _validate_resume_payload(
    payload: dict[str, object],
    *,
    disable_shorts: bool,
    action_meta: dict[str, int | float],
) -> None:
    del disable_shorts

    for key, expected in action_meta.items():
        if key not in payload:
            continue
        actual = payload[key]
        if isinstance(expected, float):
            if not np.isclose(float(actual), float(expected)):
                raise ValueError(
                    f"Resume checkpoint {key}={actual} does not match current run {key}={expected}"
                )
            continue
        if int(actual) != int(expected):
            raise ValueError(f"Resume checkpoint {key}={actual} does not match current run {key}={expected}")


def _load_resume_checkpoint(
    checkpoint_path: str | Path,
    *,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    disable_shorts: bool,
    action_meta: dict[str, int | float],
) -> ResumeState:
    path = Path(checkpoint_path)
    payload = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported resume checkpoint format at {path} (expected dict with 'model')")

    _validate_resume_payload(payload, disable_shorts=disable_shorts, action_meta=action_meta)

    policy.load_state_dict(payload["model"])
    optimizer_state = payload.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _optimizer_state_to_device(optimizer, device)

    return ResumeState(
        update=max(int(payload.get("update", 0)), 0),
        global_step=max(int(payload.get("global_step", 0)), 0),
        best_return=float(payload.get("best_return", -float("inf"))),
    )


def _mask_short_logits(logits: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Mask short-action branch in discrete action logits."""
    num_symbols = (int(num_actions) - 1) // 2
    if num_symbols <= 0:
        return logits
    masked = logits.clone()
    masked[:, 1 + num_symbols :] = torch.finfo(masked.dtype).min
    return masked

class TradingPolicy(nn.Module):
    """
    Small MLP policy + value head for trading.
    ~500K params to stay well under 8GB GPU.
    """

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Shared feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for policy output
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None, *, disable_shorts: bool = False):
        logits, value = self.forward(x)
        if disable_shorts:
            logits = _mask_short_logits(logits, self.num_actions)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x):
        h = self.encoder(x)
        return self.critic(h).squeeze(-1)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Linear → Add."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class ResidualTradingPolicy(nn.Module):
    """Residual MLP with LayerNorm for more stable training of larger models."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, num_blocks: int = 3):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        self.input_proj = nn.Linear(obs_size, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.out_norm(h)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None, *, disable_shorts: bool = False):
        logits, value = self.forward(x)
        if disable_shorts:
            logits = _mask_short_logits(logits, self.num_actions)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.out_norm(h)
        return self.critic(h).squeeze(-1)


# ─── PPO Training Loop ────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  TF32 enabled")

    # ── Load shared data ──
    data_path = str(Path(args.data_path).resolve())
    import pufferlib_market.binding as binding
    binding.shared(data_path=data_path)
    print(f"Loaded market data from {data_path}")

    # ── Read binary header to get num_symbols ──
    import struct
    with open(data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])
    print(f"  {num_symbols} symbols, {num_timesteps} timesteps")

    # ── Env config ──
    config = TradingEnvConfig(
        data_path=data_path,
        max_steps=args.max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.periods_per_year,
        num_symbols=num_symbols,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        action_allocation_bins=args.action_allocation_bins,
        action_level_bins=args.action_level_bins,
        action_max_offset_bps=args.action_max_offset_bps,
        cash_penalty=args.cash_penalty,
        drawdown_penalty=args.drawdown_penalty,
        downside_penalty=args.downside_penalty,
        smooth_downside_penalty=args.smooth_downside_penalty,
        smooth_downside_temperature=args.smooth_downside_temperature,
        trade_penalty=args.trade_penalty,
        fill_slippage_bps=args.fill_slippage_bps,
        fill_probability=args.fill_probability,
        max_hold_hours=args.max_hold_hours,
    )

    obs_size = num_symbols * 16 + 5 + num_symbols
    per_symbol_actions = config.action_allocation_bins * config.action_level_bins
    num_actions = 1 + 2 * num_symbols * per_symbol_actions
    print(f"  obs_size={obs_size}, num_actions={num_actions}")
    print(
        "  action_grid: alloc_bins={} level_bins={} max_offset_bps={:.1f}".format(
            int(config.action_allocation_bins),
            int(config.action_level_bins),
            float(config.action_max_offset_bps),
        )
    )

    # ── Create vectorised envs ──
    # PufferLib vec_init handles batched numpy buffers
    num_envs = args.num_envs
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        num_envs, args.seed,
        max_steps=config.max_steps,
        fee_rate=config.fee_rate,
        max_leverage=config.max_leverage,
        short_borrow_apr=config.short_borrow_apr,
        periods_per_year=config.periods_per_year,
        reward_scale=config.reward_scale,
        reward_clip=config.reward_clip,
        action_allocation_bins=config.action_allocation_bins,
        action_level_bins=config.action_level_bins,
        action_max_offset_bps=config.action_max_offset_bps,
        cash_penalty=config.cash_penalty,
        drawdown_penalty=config.drawdown_penalty,
        downside_penalty=config.downside_penalty,
        smooth_downside_penalty=config.smooth_downside_penalty,
        smooth_downside_temperature=config.smooth_downside_temperature,
        trade_penalty=config.trade_penalty,
        fill_slippage_bps=config.fill_slippage_bps,
        fill_probability=config.fill_probability,
        max_hold_hours=config.max_hold_hours,
    )
    binding.vec_reset(vec_handle, args.seed)
    print(f"  Created {num_envs} parallel envs")

    # ── Observation normalizer ──
    obs_norm = RunningObsNorm(obs_size) if args.obs_norm else None
    if obs_norm:
        print("  RunningObsNorm enabled")

    # ── Policy ──
    if args.arch == "resmlp":
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, eps=1e-5,
                            weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: {total_params:,} params ({total_params * 4 / 1e6:.1f} MB)")
    action_meta = {
        "action_allocation_bins": int(config.action_allocation_bins),
        "action_level_bins": int(config.action_level_bins),
        "action_max_offset_bps": float(config.action_max_offset_bps),
    }
    resume_state = ResumeState()
    if args.resume_from:
        resume_state = _load_resume_checkpoint(
            args.resume_from,
            policy=policy,
            optimizer=optimizer,
            device=device,
            disable_shorts=bool(args.disable_shorts),
            action_meta=action_meta,
        )
        print(
            "  Resumed from {} (update={} global_step={} best_return={:+.4f})".format(
                args.resume_from,
                resume_state.update,
                resume_state.global_step,
                resume_state.best_return,
            )
        )

    # ── Estimate GPU memory ──
    rollout_mem = num_envs * args.rollout_len * (obs_size * 4 + 4 * 4) / 1e6
    print(f"  Estimated rollout buffer: {rollout_mem:.1f} MB")

    # ── Rollout buffers ──
    T = args.rollout_len
    N = num_envs
    buf_obs = torch.zeros((T, N, obs_size), dtype=torch.float32)
    buf_act = torch.zeros((T, N), dtype=torch.long)
    buf_logprob = torch.zeros((T, N), dtype=torch.float32)
    buf_reward = torch.zeros((T, N), dtype=torch.float32)
    buf_done = torch.zeros((T, N), dtype=torch.float32)
    buf_value = torch.zeros((T, N), dtype=torch.float32)

    # ── Training loop ──
    global_step = resume_state.global_step
    num_updates = args.total_timesteps // (T * N)
    start_time = time.time()
    best_return = resume_state.best_return
    start_update = resume_state.update

    print(f"\nTraining: {num_updates} updates, {args.total_timesteps:,} additional steps")
    print(f"  rollout_len={T}, num_envs={N}, batch_size={T * N}")
    print(f"  PPO epochs={args.ppo_epochs}, minibatch_size={args.minibatch_size}")
    print()

    for update in range(start_update + 1, start_update + num_updates + 1):
        local_update = update - start_update
        # ── Learning rate schedule ──
        if args.lr_schedule == "cosine":
            lr_now = cosine_lr_with_warmup(local_update, num_updates, args.lr,
                                           warmup_frac=args.lr_warmup_frac,
                                           min_ratio=args.lr_min_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
        elif args.anneal_lr:
            frac = 1.0 - (local_update - 1) / max(num_updates, 1)
            lr_now = frac * args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        # ── Entropy & clip annealing ──
        ent_coef = linear_anneal(local_update, num_updates,
                                 args.ent_coef, args.ent_coef_end) if args.anneal_ent else args.ent_coef
        clip_eps = linear_anneal(local_update, num_updates,
                                 args.clip_eps, args.clip_eps_end) if args.anneal_clip else args.clip_eps

        # ── Collect rollout ──
        policy.eval()
        for step in range(T):
            raw_obs = obs_buf.copy()
            if obs_norm is not None:
                obs_norm.update(raw_obs)
                raw_obs = obs_norm.normalize(raw_obs)
            obs_tensor = torch.from_numpy(raw_obs).to(device, non_blocking=True)
            buf_obs[step] = obs_tensor

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    obs_tensor,
                    disable_shorts=args.disable_shorts,
                )

            buf_act[step] = action.cpu()
            buf_logprob[step] = logprob.cpu()
            buf_value[step] = value.cpu()

            # Write actions to C buffer and step
            act_buf[:] = action.cpu().numpy().astype(np.int32)
            binding.vec_step(vec_handle)

            buf_reward[step] = torch.from_numpy(rew_buf.copy())
            buf_done[step] = torch.from_numpy(term_buf.copy().astype(np.float32))

            global_step += N

        # ── Compute advantages ──
        with torch.no_grad():
            raw_next = obs_buf.copy()
            if obs_norm is not None:
                raw_next = obs_norm.normalize(raw_next)
            next_obs = torch.from_numpy(raw_next).to(device, non_blocking=True)
            next_value = policy.get_value(next_obs).cpu()

        # Flatten envs for GAE
        advantages = torch.zeros_like(buf_reward)
        last_gae = torch.zeros(N)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = buf_value[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_reward[t] + args.gamma * next_val * not_done - buf_value[t]
            last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + buf_value

        # ── PPO update ──
        policy.train()
        b_obs = buf_obs.reshape(-1, obs_size).to(device, non_blocking=True)
        b_act = buf_act.reshape(-1).to(device, non_blocking=True)
        b_logprob = buf_logprob.reshape(-1).to(device, non_blocking=True)
        b_advantages = advantages.reshape(-1).to(device, non_blocking=True)
        b_returns = returns.reshape(-1).to(device, non_blocking=True)
        b_values = buf_value.reshape(-1).to(device, non_blocking=True)

        # Normalise advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        num_mb = 0

        batch_size = T * N
        mb_size = args.minibatch_size

        for epoch in range(args.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = policy.get_action_and_value(
                    b_obs[mb_idx],
                    b_act[mb_idx],
                    disable_shorts=args.disable_shorts,
                )

                # PPO clipped objective
                log_ratio = new_logprob - b_logprob[mb_idx]
                ratio = log_ratio.exp()
                mb_adv = b_advantages[mb_idx]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping (PPO-style)
                if args.clip_vloss:
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx], -clip_eps, clip_eps
                    )
                    v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += ent_loss.item()
                num_mb += 1

        # ── Logging ──
        log_info = binding.vec_log(vec_handle)
        elapsed = time.time() - start_time
        sps = global_step / elapsed

        avg_pg = total_pg_loss / max(num_mb, 1)
        avg_vl = total_v_loss / max(num_mb, 1)
        avg_ent = total_entropy / max(num_mb, 1)

        if log_info and "total_return" in log_info:
            ep_return = log_info["total_return"]
            ep_annualized = annualize_total_return(
                float(ep_return),
                periods=float(args.max_steps),
                periods_per_year=float(args.periods_per_year),
            )
            ep_sortino = log_info.get("sortino", 0)
            ep_trades = log_info.get("num_trades", 0)
            ep_wr = log_info.get("win_rate", 0)
            n = log_info.get("n", 0)

            if ep_return > best_return:
                best_return = ep_return
                ckpt_path = Path(args.checkpoint_dir) / "best.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _checkpoint_payload(
                        policy,
                        optimizer,
                        update=update,
                        global_step=global_step,
                        best_return=best_return,
                        disable_shorts=bool(args.disable_shorts),
                        action_meta=action_meta,
                    ),
                    ckpt_path,
                )

            print(
                f"[{local_update:4d}/{num_updates}] "
                f"step={global_step:8,d}  sps={sps:.0f}  "
                f"ret={ep_return:+.4f}  ann_ret={ep_annualized:+.2%}  sortino={ep_sortino:.2f}  "
                f"trades={ep_trades:.0f}  wr={ep_wr:.2f}  "
                f"pg={avg_pg:.4f}  vl={avg_vl:.4f}  ent={avg_ent:.3f}  "
                f"n={n:.0f}"
            )
        else:
            print(
                f"[{local_update:4d}/{num_updates}] "
                f"step={global_step:8,d}  sps={sps:.0f}  "
                f"pg={avg_pg:.4f}  vl={avg_vl:.4f}  ent={avg_ent:.3f}"
            )

        # ── Periodic checkpoint ──
        if update % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"update_{update:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _checkpoint_payload(
                    policy,
                    optimizer,
                    update=update,
                    global_step=global_step,
                    best_return=best_return,
                    disable_shorts=bool(args.disable_shorts),
                    action_meta=action_meta,
                ),
                ckpt_path,
            )

    # ── Final save ──
    ckpt_path = Path(args.checkpoint_dir) / "final.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            policy,
            optimizer,
            update=start_update + num_updates,
            global_step=global_step,
            best_return=best_return,
            disable_shorts=bool(args.disable_shorts),
            action_meta=action_meta,
        ),
        ckpt_path,
    )

    binding.vec_close(vec_handle)
    print(f"\nTraining complete. Best return: {best_return:.4f}")
    print(f"Checkpoints saved to {args.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="PPO training for C trading env")
    # Environment
    parser.add_argument("--data-path", default="pufferlib_market/data/market_data.bin")
    parser.add_argument("--max-steps", type=int, default=720, help="Episode length (hours)")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0,
                        help="Annualisation factor for Sortino (8760=hourly, 365=daily, 252=trading days)")
    parser.add_argument("--reward-scale", type=float, default=10.0, help="Reward = ret * scale")
    parser.add_argument("--reward-clip", type=float, default=5.0, help="Clip |reward| to this")
    parser.add_argument(
        "--action-allocation-bins",
        type=int,
        default=1,
        help="Discrete exposure bins per symbol/side (1 keeps legacy full-allocation actions).",
    )
    parser.add_argument(
        "--action-level-bins",
        type=int,
        default=1,
        help="Discrete execution-level bins around close per symbol/side (1 keeps market-at-close behavior).",
    )
    parser.add_argument(
        "--action-max-offset-bps",
        type=float,
        default=0.0,
        help="Max absolute limit-offset in bps for action levels (e.g. 50 => +/-0.50%%).",
    )
    parser.add_argument("--cash-penalty", type=float, default=0.01, help="Per-step flat penalty")
    parser.add_argument("--drawdown-penalty", type=float, default=0.0, help="Drawdown penalty scale")
    parser.add_argument(
        "--downside-penalty",
        type=float,
        default=0.0,
        help="Penalty scale for negative returns: reward -= downside_penalty * ret^2 when ret < 0",
    )
    parser.add_argument(
        "--trade-penalty",
        type=float,
        default=0.0,
        help="Per-trade penalty (counts opens+closes executed by action); discourages churn",
    )
    parser.add_argument(
        "--smooth-downside-penalty",
        type=float,
        default=0.0,
        help="Smooth downside penalty scale using softplus(-ret/temp)^2.",
    )
    parser.add_argument(
        "--smooth-downside-temperature",
        type=float,
        default=0.02,
        help="Temperature for smooth downside shaping (smaller -> sharper downside proxy).",
    )
    parser.add_argument(
        "--fill-slippage-bps",
        type=float,
        default=0.0,
        help="Adverse fill slippage in basis points (realistic: 5-12). Buys fill higher, sells fill lower.",
    )
    parser.add_argument(
        "--fill-probability",
        type=float,
        default=1.0,
        help="Probability an order fills [0-1]. 1.0=always fills. 0.8=20%% of entries randomly rejected (simulates low liquidity).",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0, help="Annual borrow rate applied to open shorts")
    parser.add_argument("--max-hold-hours", type=int, default=0, help="Force close position after N hours (0=disabled)")

    # Policy
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp",
                        help="Architecture: mlp (default) or resmlp (residual+LayerNorm)")
    parser.add_argument("--disable-shorts", action="store_true", help="Mask short actions (long/flat only)")

    # PPO
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anneal-lr", action="store_true", help="Linear LR annealing (legacy)")
    parser.add_argument("--lr-schedule", choices=["none", "cosine", "linear"], default="none",
                        help="LR schedule: cosine (warmup+cosine+floor) or linear (legacy anneal)")
    parser.add_argument("--lr-warmup-frac", type=float, default=0.02, help="Fraction of training for LR warmup")
    parser.add_argument("--lr-min-ratio", type=float, default=0.05, help="Minimum LR as fraction of base (floor)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0.005 recommended)")
    parser.add_argument("--anneal-ent", action="store_true", help="Anneal entropy coefficient over training")
    parser.add_argument("--ent-coef-end", type=float, default=0.02, help="Final entropy coef (with --anneal-ent)")
    parser.add_argument("--anneal-clip", action="store_true", help="Anneal clip epsilon over training")
    parser.add_argument("--clip-eps-end", type=float, default=0.05, help="Final clip eps (with --anneal-clip)")
    parser.add_argument("--clip-vloss", action="store_true", help="PPO-style value function clipping")
    parser.add_argument("--obs-norm", action="store_true", help="Running observation normalization")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume from")

    # Output
    parser.add_argument("--checkpoint-dir", default="pufferlib_market/checkpoints")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
