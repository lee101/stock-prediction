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
import atexit
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

try:
    import wandb
except ImportError:
    wandb = None

# PufferLib imports
import pufferlib
import pufferlib.vector

# Local
from pufferlib_market.environment import TradingEnvConfig, TradingEnv
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.advantage_utils import normalize_advantages
from src.checkpoint_manager import TopKCheckpointManager, prune_periodic_checkpoints


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


# ─── Activation helpers ──────────────────────────────────────────────


def relu_sq(x: torch.Tensor) -> torch.Tensor:
    """ReLU² activation: proven better than ReLU/GELU at small scale (cutellm research)."""
    return torch.relu(x) ** 2


def get_activation(name: str):
    """Return an activation function by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "relu_sq":
        return _ReLUSq()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name!r}. Choose relu, relu_sq, or gelu.")


class _ReLUSq(nn.Module):
    """Module wrapper for relu_sq activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu_sq(x)


# ─── TransformerTradingPolicy ─────────────────────────────────────────


class TransformerTradingPolicy(nn.Module):
    """
    Treats each symbol as a token. Uses multi-head self-attention across symbols
    + a feedforward head. Inspired by cross-symbol correlation reasoning.

    obs shape: (batch, S*16 + 5 + S) where S=num_symbols
    Strategy:
      1. Split obs into per-symbol features (S*16) and portfolio state (5+S)
      2. Reshape per-symbol features to (batch, S, 16)
      3. Project to (batch, S, embed_dim) with a linear layer
      4. Apply multi-head self-attention across symbols
      5. Flatten + concat portfolio state -> MLP head for actor/critic
    """

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256,
                 activation: str = "relu"):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Infer num_symbols from obs_size: obs = S*16 + 5 + S => S*17 + 5 = obs_size
        # S = (obs_size - 5) / 17
        num_symbols = (obs_size - 5) // 17
        self.num_symbols = num_symbols
        self.per_symbol_features = 16
        self.portfolio_size = obs_size - num_symbols * 16  # 5 + S

        # Embedding: project 16-dim per-symbol features -> hidden//4 (embed_dim)
        embed_dim = max(hidden // 4, 32)
        self.embed_dim = embed_dim
        self.symbol_proj = nn.Linear(self.per_symbol_features, embed_dim)

        # Multi-head self-attention (num_heads=4, or fewer if embed_dim is small)
        num_heads = min(4, embed_dim // 8) if embed_dim >= 8 else 1
        num_heads = max(num_heads, 1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # MLP after attention
        attn_out_size = num_symbols * embed_dim + self.portfolio_size
        act = get_activation(activation)
        self.mlp = nn.Sequential(
            nn.Linear(attn_out_size, hidden),
            act,
            nn.Linear(hidden, hidden),
            get_activation(activation),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            get_activation(activation),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            get_activation(activation),
            nn.Linear(hidden // 2, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        # Split: per-symbol part and portfolio state
        sym_flat = x[:, :self.num_symbols * self.per_symbol_features]
        portfolio = x[:, self.num_symbols * self.per_symbol_features:]

        # Reshape to (batch, S, 16) -> project to (batch, S, embed_dim)
        sym_tokens = sym_flat.view(batch, self.num_symbols, self.per_symbol_features)
        sym_emb = self.symbol_proj(sym_tokens)  # (batch, S, embed_dim)

        # Self-attention across symbols
        attn_out, _ = self.attn(sym_emb, sym_emb, sym_emb)
        sym_emb = self.attn_norm(sym_emb + attn_out)  # residual + norm

        # Flatten symbols + concat portfolio
        h = torch.cat([sym_emb.reshape(batch, -1), portfolio], dim=-1)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action=None, *, disable_shorts: bool = False):
        logits, value = self.forward(x)
        if disable_shorts:
            logits = _mask_short_logits(logits, self.num_actions)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor):
        _, value = self.forward(x)
        return value


# ─── GRUTradingPolicy ─────────────────────────────────────────────────


class GRUTradingPolicy(nn.Module):
    """
    GRU-gated policy: uses GRU cells for temporal gating within the feedforward pass.
    The input is processed through GRU cells (treating the hidden layers as a sequence)
    to allow gated information flow inspired by temporal sequence models.

    Concretely: input projection -> 2-layer GRU (seq_len=1, batch=batch) -> actor/critic heads.
    """

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256,
                 activation: str = "relu"):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Input projection
        self.input_proj = nn.Linear(obs_size, hidden)
        self.input_act = get_activation(activation)

        # 2-layer GRU: processes the hidden representation with gating
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=2, batch_first=True)

        self.out_norm = nn.LayerNorm(hidden)

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            get_activation(activation),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            get_activation(activation),
            nn.Linear(hidden // 2, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.input_act(self.input_proj(x))  # (batch, hidden)
        # GRU expects (batch, seq_len, input_size); use seq_len=1
        h_seq = h.unsqueeze(1)  # (batch, 1, hidden)
        gru_out, _ = self.gru(h_seq)  # (batch, 1, hidden)
        h = self.out_norm(gru_out.squeeze(1))  # (batch, hidden)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action=None, *, disable_shorts: bool = False):
        logits, value = self.forward(x)
        if disable_shorts:
            logits = _mask_short_logits(logits, self.num_actions)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor):
        _, value = self.forward(x)
        return value


# ─── DepthRecurrenceTradingPolicy ─────────────────────────────────────


class DepthRecurrenceTradingPolicy(nn.Module):
    """
    N unique MLP blocks executed K times (depth recurrence).
    Uses N=3 blocks run K=2 times = 6 effective layers at 3-block parameter cost.
    Gives more representational capacity without proportionally more parameters.
    From cutellm depth-recurrence research.
    """

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256,
                 num_blocks: int = 3, num_recurrences: int = 2,
                 activation: str = "relu"):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.num_recurrences = num_recurrences

        self.input_proj = nn.Linear(obs_size, hidden)

        # N unique blocks (each is a pre-norm residual: LN -> Linear -> act -> Linear)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden),
                "fc1": nn.Linear(hidden, hidden),
                "fc2": nn.Linear(hidden, hidden),
            })
            for _ in range(num_blocks)
        ])
        self.act_name = activation
        self.out_norm = nn.LayerNorm(hidden)

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            get_activation(activation),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            get_activation(activation),
            nn.Linear(hidden // 2, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _apply_block(self, x: torch.Tensor, block: nn.ModuleDict) -> torch.Tensor:
        """Apply one pre-norm residual block."""
        h = block["norm"](x)
        h = block["fc1"](h)
        h = relu_sq(h) if self.act_name == "relu_sq" else torch.relu(h) if self.act_name == "relu" else torch.nn.functional.gelu(h)
        h = block["fc2"](h)
        return x + h

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        # Run all N blocks K times
        for _ in range(self.num_recurrences):
            for block in self.blocks:
                h = self._apply_block(h, block)
        h = self.out_norm(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action=None, *, disable_shorts: bool = False):
        logits, value = self.forward(x)
        if disable_shorts:
            logits = _mask_short_logits(logits, self.num_actions)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor):
        _, value = self.forward(x)
        return value


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
        smoothness_penalty=args.smoothness_penalty,
        fill_slippage_bps=args.fill_slippage_bps,
        fill_probability=args.fill_probability,
        max_hold_hours=args.max_hold_hours,
        enable_drawdown_profit_early_exit=args.drawdown_profit_early_exit,
        drawdown_profit_early_exit_verbose=args.drawdown_profit_early_exit_verbose,
        drawdown_profit_early_exit_min_steps=args.drawdown_profit_early_exit_min_steps,
        drawdown_profit_early_exit_progress_fraction=args.drawdown_profit_early_exit_progress_fraction,
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
        enable_drawdown_profit_early_exit=config.enable_drawdown_profit_early_exit,
        drawdown_profit_early_exit_verbose=config.drawdown_profit_early_exit_verbose,
        drawdown_profit_early_exit_min_steps=config.drawdown_profit_early_exit_min_steps,
        drawdown_profit_early_exit_progress_fraction=config.drawdown_profit_early_exit_progress_fraction,
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
    elif args.arch == "transformer":
        policy = TransformerTradingPolicy(
            obs_size, num_actions, hidden=args.hidden_size,
            activation=args.activation,
        ).to(device)
    elif args.arch == "gru":
        policy = GRUTradingPolicy(
            obs_size, num_actions, hidden=args.hidden_size,
            activation=args.activation,
        ).to(device)
    elif args.arch == "depth_recurrence":
        policy = DepthRecurrenceTradingPolicy(
            obs_size, num_actions, hidden=args.hidden_size,
            activation=args.activation,
        ).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    if args.optimizer == "muon":
        from pufferlib_market.muon import make_muon_optimizer
        optimizer = make_muon_optimizer(
            policy,
            muon_lr=args.lr,
            muon_momentum=args.muon_momentum,
            adamw_lr=args.muon_adamw_lr,
            adamw_wd=args.weight_decay,
            ns_steps=args.muon_ns_steps,
        )
        print(
            f"  Optimizer: Muon (lr={args.lr}, momentum={args.muon_momentum},"
            f" ns_steps={args.muon_ns_steps}, adamw_lr={args.muon_adamw_lr})"
        )
    else:
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

    # ── Weights & Biases ──
    wandb_run = None
    if args.wandb_project is not None:
        if wandb is None:
            print("WARNING: --wandb-project set but wandb not installed. pip install wandb")
        elif args.wandb_mode == "disabled":
            print("  W&B disabled via --wandb-mode=disabled")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or None,
                group=args.wandb_group or None,
                config=vars(args),
                mode=args.wandb_mode,
            )
            atexit.register(wandb_run.finish)
            print(f"  W&B run: {wandb_run.url or wandb_run.id}")
            wandb_run.log({
                "hyperparams/trade_penalty": args.trade_penalty,
                "hyperparams/fill_slippage_bps": args.fill_slippage_bps,
                "hyperparams/fee_rate": args.fee_rate,
            }, step=0)

    # ── Estimate GPU memory ──
    rollout_mem = num_envs * args.rollout_len * (obs_size * 4 + 4 * 4) / 1e6
    print(f"  Estimated rollout buffer: {rollout_mem:.1f} MB")

    # ── CUDA graph for rollout inference ──
    use_cuda_graph = device.type == "cuda"
    if use_cuda_graph:
        # Static GPU buffers (required for CUDA graph capture)
        _static_obs = torch.zeros(num_envs, obs_size, device=device)
        _static_logits = torch.zeros(num_envs, num_actions, device=device)
        _static_value = torch.zeros(num_envs, device=device)
        # Outputs computed from logits inside graph
        _static_action = torch.zeros(num_envs, dtype=torch.long, device=device)
        _static_logprob = torch.zeros(num_envs, device=device)

        # Short-mask tensor (baked into graph)
        _short_mask_idx = 1 + (num_actions - 1) // 2 if args.disable_shorts else 0

        def _graph_forward():
            """Forward + sample inside CUDA graph (no Categorical, no Python control flow)."""
            logits, value = policy(_static_obs)
            _static_logits.copy_(logits)
            _static_value.copy_(value)
            if _short_mask_idx > 0:
                _static_logits[:, _short_mask_idx:] = torch.finfo(logits.dtype).min
            # Manual multinomial sampling (CUDA-graph-safe)
            probs = torch.softmax(_static_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            _static_action.copy_(action)
            log_probs_all = torch.log_softmax(_static_logits, dim=-1)
            _static_logprob.copy_(log_probs_all.gather(1, action.unsqueeze(1)).squeeze(1))

        # Warmup
        policy.eval()
        with torch.no_grad():
            for _ in range(3):
                _graph_forward()
        torch.cuda.synchronize()

        # Capture
        _inference_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(_inference_graph):
            with torch.no_grad():
                _graph_forward()
        torch.cuda.synchronize()
        print("  CUDA graph captured for rollout inference")

    # Pinned memory for fast CPU↔GPU transfers
    _pinned_obs = torch.zeros(num_envs, obs_size, dtype=torch.float32, pin_memory=True) if use_cuda_graph else None
    _pinned_action = torch.zeros(num_envs, dtype=torch.int32, pin_memory=True) if use_cuda_graph else None

    # ── torch.compile for PPO update ──
    def _ppo_loss_fn(policy, obs, act, old_logprob, advantages, returns, old_values,
                     clip_eps_t, vf_coef_t, ent_coef_t, clip_vloss):
        """PPO loss computation (compilable)."""
        logits, new_value = policy(obs)
        # Manual log_prob and entropy (avoids Categorical Python control flow for fullgraph)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        new_logprob = log_probs_all.gather(1, act.unsqueeze(1)).squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(-1)

        log_ratio = new_logprob - old_logprob
        ratio = log_ratio.exp()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_eps_t, 1 + clip_eps_t)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        if clip_vloss:
            v_clipped = old_values + torch.clamp(new_value - old_values, -clip_eps_t, clip_eps_t)
            v_loss = 0.5 * torch.max(
                (new_value - returns) ** 2,
                (v_clipped - returns) ** 2,
            ).mean()
        else:
            v_loss = 0.5 * ((new_value - returns) ** 2).mean()

        ent_loss = entropy.mean()
        loss = pg_loss + vf_coef_t * v_loss - ent_coef_t * ent_loss
        return loss, pg_loss, v_loss, ent_loss

    if device.type == "cuda":
        try:
            _compiled_ppo_loss = torch.compile(_ppo_loss_fn, mode="reduce-overhead", fullgraph=True)
            # Warmup compile with dummy data
            policy.train()
            _dummy_obs = torch.randn(2048, obs_size, device=device)
            _dummy_act = torch.randint(0, num_actions, (2048,), device=device)
            _dummy_lp = torch.zeros(2048, device=device)
            _dummy_adv = torch.zeros(2048, device=device)
            _dummy_ret = torch.zeros(2048, device=device)
            _dummy_val = torch.zeros(2048, device=device)
            _clip_t = torch.tensor(0.2, device=device)
            _vf_t = torch.tensor(0.5, device=device)
            _ent_t = torch.tensor(0.01, device=device)
            for _ in range(3):
                _l, _, _, _ = _compiled_ppo_loss(policy, _dummy_obs, _dummy_act, _dummy_lp,
                                                  _dummy_adv, _dummy_ret, _dummy_val,
                                                  _clip_t, _vf_t, _ent_t, False)
                _l.backward()
                optimizer.zero_grad()
            torch.cuda.synchronize()
            del _dummy_obs, _dummy_act, _dummy_lp, _dummy_adv, _dummy_ret, _dummy_val, _l
            print("  torch.compile: PPO loss compiled (reduce-overhead, fullgraph=True)")
        except Exception as e:
            print(f"  torch.compile: failed ({e}), using eager PPO")
            _compiled_ppo_loss = _ppo_loss_fn
    else:
        _compiled_ppo_loss = _ppo_loss_fn

    # ── Rollout buffers (on GPU when possible to avoid per-step CPU↔GPU copies) ──
    T = args.rollout_len
    N = num_envs
    if use_cuda_graph:
        buf_obs = torch.zeros((T, N, obs_size), dtype=torch.float32, device=device)
        buf_act = torch.zeros((T, N), dtype=torch.long, device=device)
        buf_logprob = torch.zeros((T, N), dtype=torch.float32, device=device)
        buf_value = torch.zeros((T, N), dtype=torch.float32, device=device)
    else:
        buf_obs = torch.zeros((T, N, obs_size), dtype=torch.float32)
        buf_act = torch.zeros((T, N), dtype=torch.long)
        buf_logprob = torch.zeros((T, N), dtype=torch.float32)
        buf_value = torch.zeros((T, N), dtype=torch.float32)
    buf_reward = torch.zeros((T, N), dtype=torch.float32)
    buf_done = torch.zeros((T, N), dtype=torch.float32)

    # ── Training loop ──
    global_step = resume_state.global_step
    num_updates = args.total_timesteps // (T * N)
    start_time = time.time()
    best_return = resume_state.best_return
    start_update = resume_state.update
    periodic_ckpt_mgr = TopKCheckpointManager(
        Path(args.checkpoint_dir), max_keep=args.max_periodic_checkpoints, mode="max",
    )
    prune_periodic_checkpoints(
        Path(args.checkpoint_dir),
        max_keep_latest=args.max_periodic_checkpoints,
    )

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
            if use_cuda_graph:
                # Fast path: CUDA graph inference (buffers on GPU, only actions go to CPU)
                if obs_norm is not None:
                    obs_norm.update(obs_buf)
                    np.copyto(_pinned_obs.numpy(), obs_norm.normalize(obs_buf))
                else:
                    np.copyto(_pinned_obs.numpy(), obs_buf)
                _static_obs.copy_(_pinned_obs, non_blocking=True)
                _inference_graph.replay()
                # Store on GPU (no .cpu() calls)
                buf_obs[step].copy_(_static_obs)
                buf_act[step].copy_(_static_action)
                buf_logprob[step].copy_(_static_logprob)
                buf_value[step].copy_(_static_value)
                # Only action needs CPU for C env
                act_buf[:] = _static_action.cpu().to(torch.int32).numpy()
            else:
                # Fallback: eager inference (CPU or no CUDA graph)
                raw_obs = obs_buf.copy()
                if obs_norm is not None:
                    obs_norm.update(raw_obs)
                    raw_obs = obs_norm.normalize(raw_obs)
                obs_tensor = torch.from_numpy(raw_obs).to(device, non_blocking=True)
                buf_obs[step] = obs_tensor
                with torch.no_grad():
                    action, logprob, _, value = policy.get_action_and_value(
                        obs_tensor, disable_shorts=args.disable_shorts,
                    )
                buf_act[step] = action.cpu()
                buf_logprob[step] = logprob.cpu()
                buf_value[step] = value.cpu()
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

        # GAE on CPU (sequential loop, small cost)
        values_cpu = buf_value.cpu() if buf_value.is_cuda else buf_value
        advantages = torch.zeros_like(buf_reward)
        last_gae = torch.zeros(N)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values_cpu[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_reward[t] + args.gamma * next_val * not_done - values_cpu[t]
            last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values_cpu

        # ── PPO update ──
        policy.train()
        # buf_obs/act/logprob/value are already on GPU when using CUDA graphs
        b_obs = buf_obs.reshape(-1, obs_size) if buf_obs.is_cuda else buf_obs.reshape(-1, obs_size).to(device, non_blocking=True)
        b_act = buf_act.reshape(-1) if buf_act.is_cuda else buf_act.reshape(-1).to(device, non_blocking=True)
        b_logprob = buf_logprob.reshape(-1) if buf_logprob.is_cuda else buf_logprob.reshape(-1).to(device, non_blocking=True)
        b_returns = returns.reshape(-1).to(device, non_blocking=True)
        b_values = values_cpu.reshape(-1).to(device, non_blocking=True)

        b_advantages = normalize_advantages(
            advantages.to(device, non_blocking=True),
            rewards=buf_reward.to(device, non_blocking=True),
            mode=args.advantage_norm,
            group_relative_size=args.group_relative_size,
            group_relative_mix=args.group_relative_mix,
            group_relative_clip=args.group_relative_clip,
        ).reshape(-1)

        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        num_mb = 0

        batch_size = T * N
        mb_size = args.minibatch_size

        # Scalar tensors for compiled PPO (avoid reallocation)
        clip_eps_t = torch.tensor(clip_eps, device=device)
        vf_coef_t = torch.tensor(args.vf_coef, device=device)
        ent_coef_t = torch.tensor(ent_coef, device=device)

        for epoch in range(args.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_idx = indices[start:end]

                loss, pg_loss, v_loss, ent_loss = _compiled_ppo_loss(
                    policy, b_obs[mb_idx], b_act[mb_idx], b_logprob[mb_idx],
                    b_advantages[mb_idx], b_returns[mb_idx], b_values[mb_idx],
                    clip_eps_t, vf_coef_t, ent_coef_t, args.clip_vloss,
                )

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
                if wandb_run is not None:
                    try:
                        artifact = wandb.Artifact(
                            f"checkpoint-{wandb_run.id}",
                            type="model",
                            metadata={"val_return": best_return, "global_step": global_step},
                        )
                        artifact.add_file(str(ckpt_path))
                        wandb_run.log_artifact(artifact)
                    except Exception:
                        pass  # never crash training due to artifact upload failure

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

        # ── W&B logging (single path for both branches) ──
        if wandb_run is not None:
            wb_metrics = {
                "loss/policy": avg_pg,
                "loss/value": avg_vl,
                "loss/entropy": avg_ent,
                "loss/total": avg_pg + args.vf_coef * avg_vl - ent_coef * avg_ent,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/ent_coef": ent_coef,
                "train/clip_eps": clip_eps,
                "perf/steps_per_sec": sps,
                "global_step": global_step,
            }
            if log_info and "total_return" in log_info:
                wb_metrics.update({
                    "train/return": ep_return,
                    "train/annualized_return": ep_annualized,
                    "train/sortino": ep_sortino,
                    "train/win_rate": ep_wr,
                    "train/num_trades": ep_trades,
                    "train/episode_return_mean": ep_return,
                    "train/episode_return_std": float(log_info.get("return_std", 0.0)),
                    "train/episode_length_mean": float(args.max_steps),
                })
            wandb_run.log(wb_metrics, step=global_step)

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
            metric = ep_return if (log_info and "total_return" in log_info) else best_return
            periodic_ckpt_mgr.register(ckpt_path, metric)

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
        "--smoothness-penalty",
        type=float,
        default=0.0,
        help="Penalty scale for volatile return changes: reward -= smoothness_penalty * (ret_t-ret_t-1)^2",
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
    parser.add_argument(
        "--drawdown-profit-early-exit",
        action="store_true",
        help="Stop an episode once max drawdown exceeds profit after the configured progress threshold.",
    )
    parser.add_argument(
        "--drawdown-profit-early-exit-verbose",
        action="store_true",
        help="Print the drawdown-vs-profit early-exit reason when the rule triggers.",
    )
    parser.add_argument(
        "--drawdown-profit-early-exit-min-steps",
        type=int,
        default=20,
        help="Minimum episode length before drawdown-vs-profit early exit can trigger.",
    )
    parser.add_argument(
        "--drawdown-profit-early-exit-progress-fraction",
        type=float,
        default=0.5,
        help="Episode progress threshold for drawdown-vs-profit early exit.",
    )

    # Policy
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--arch",
        choices=["mlp", "resmlp", "transformer", "gru", "depth_recurrence"],
        default="mlp",
        help=(
            "Architecture: mlp (default), resmlp (residual+LayerNorm), "
            "transformer (symbol-level attention), gru (GRU-gated), "
            "depth_recurrence (N blocks × K passes)"
        ),
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "relu_sq", "gelu"],
        default="relu",
        help="Activation function for new architectures (relu_sq proven better at small scale)",
    )
    parser.add_argument("--disable-shorts", action="store_true", help="Mask short actions (long/flat only)")

    # PPO
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--advantage-norm",
        choices=["global", "per_env", "group_relative"],
        default="global",
        help="Advantage normalization: global PPO default, per-env sequence normalization, or GSPO-like group-relative weighting.",
    )
    parser.add_argument(
        "--group-relative-size",
        type=int,
        default=8,
        help="Group size for group-relative rollout weighting.",
    )
    parser.add_argument(
        "--group-relative-mix",
        type=float,
        default=0.0,
        help="How strongly to scale per-env advantages by group-relative rollout score.",
    )
    parser.add_argument(
        "--group-relative-clip",
        type=float,
        default=2.0,
        help="Clip z-scored rollout ranks before they rescale advantages.",
    )
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
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
        help="Optimizer: adamw (default) or muon (Newton-Schulz orthogonalised SGD for weight matrices)",
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="SGD momentum for Muon (default 0.95)")
    parser.add_argument("--muon-ns-steps", type=int, default=5, help="Newton-Schulz iterations for Muon (default 5)")
    parser.add_argument("--muon-adamw-lr", type=float, default=3e-4, help="AdamW lr for 1D params when using Muon (default 3e-4)")
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
    parser.add_argument(
        "--max-periodic-checkpoints",
        type=int,
        default=int(os.getenv("PUFFERLIB_MAX_PERIODIC_CHECKPOINTS", "3")),
        help="How many periodic update_*.pt checkpoints to retain alongside best/final.",
    )
    parser.add_argument("--cpu", action="store_true")

    # Weights & Biases
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (enables logging when set)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team or username)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B run group (useful for grouping autoresearch trials)")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="W&B run mode")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
