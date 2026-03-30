#!/usr/bin/env python3
"""Evaluate a PPO checkpoint on the *tail window* of an MKTD dataset.

Why this exists:
- The compiled C env randomises episode start offsets, which makes it hard to
  compare "latest 30d" apples-to-apples.
- Here we use the pure-python simulator in ``pufferlib_market.hourly_replay`` on
  a deterministic tail slice (e.g., last 720 hours = ~30 calendar days).

Limitations:
- Currently supports the legacy action space only: action_allocation_bins=1 and
  action_level_bins=1 (actions: flat, long(sym), short(sym)).
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from pufferlib_market.checkpoint_loader import (
    extract_checkpoint_state_dict,
    infer_arch_from_state_dict,
    infer_hidden_size_from_state_dict,
    infer_resmlp_blocks_from_state_dict,
    load_checkpoint_payload,
    load_policy_from_resolved_metadata,
    resolve_checkpoint_action_grid_config,
)
from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return


def _mask_all_shorts(logits: torch.Tensor, *, num_symbols: int) -> torch.Tensor:
    # action layout: 0 flat, 1..S long, S+1..2S short
    masked = logits.clone()
    masked[:, 1 + num_symbols :] = torch.finfo(masked.dtype).min
    return masked


def _mask_disallowed_shorts(
    logits: torch.Tensor,
    *,
    num_symbols: int,
    shortable_mask: torch.Tensor,
) -> torch.Tensor:
    if shortable_mask.numel() != num_symbols:
        raise ValueError("shortable_mask length mismatch")
    if shortable_mask.dtype is not torch.bool:
        shortable_mask = shortable_mask.to(torch.bool)
    if bool(torch.all(shortable_mask)):
        return logits
    masked = logits.clone()
    min_val = torch.finfo(masked.dtype).min
    disallowed = (~shortable_mask).nonzero(as_tuple=False).view(-1).tolist()
    for sym_idx in disallowed:
        masked[:, 1 + num_symbols + int(sym_idx)] = min_val
    return masked


class TradingPolicy(nn.Module):
    """Must match pufferlib_market.train.TradingPolicy exactly."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # encoder_norm added to train.py to prevent BF16 precision collapse.
        # _use_encoder_norm set by load_policy: True only for new ckpts that have this layer.
        self.encoder_norm = nn.LayerNorm(hidden)
        self._use_encoder_norm = False
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        if getattr(self, "_use_encoder_norm", False):
            h = self.encoder_norm(h)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ResidualTradingPolicy(nn.Module):
    """Must match pufferlib_market.train.ResidualTradingPolicy exactly."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(obs_size, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.out_norm(self.blocks(self.input_proj(x)))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


def _infer_num_actions(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    for key in ("actor.2.bias", "actor.2.weight"):
        if key in state_dict:
            t = state_dict[key]
            if t.ndim == 1:
                return int(t.shape[0])
            if t.ndim == 2:
                return int(t.shape[0])
    return int(fallback)


def _build_shortable_mask(symbols: list[str], shortable_csv: str | None) -> torch.Tensor | None:
    if not shortable_csv:
        return None
    requested = {s.strip().upper() for s in shortable_csv.split(",") if s.strip()}
    if not requested:
        return None
    return torch.tensor([sym.upper() in requested for sym in symbols], dtype=torch.bool)


def _slice_tail(data: MktdData, steps: int) -> MktdData:
    if steps < 1:
        raise ValueError("steps must be >=1")
    need = int(steps) + 1
    if data.num_timesteps < need:
        raise ValueError(f"MKTD too short: timesteps={data.num_timesteps} need={need}")
    start = data.num_timesteps - need
    end = data.num_timesteps
    return MktdData(
        version=data.version,
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=None if data.tradable is None else data.tradable[start:end].copy(),
    )


@dataclass(frozen=True)
class LoadedPolicy:
    policy: nn.Module
    arch: str
    hidden_size: int
    action_allocation_bins: int
    action_level_bins: int
    action_max_offset_bps: float


def load_policy(
    *,
    checkpoint_path: str | Path,
    obs_size: int,
    num_symbols: int,
    arch: str,
    hidden_size: int | None,
    device: torch.device,
) -> LoadedPolicy:
    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    state_dict = extract_checkpoint_state_dict(payload)

    action_allocation_bins, action_level_bins, action_max_offset_bps = resolve_checkpoint_action_grid_config(
        payload,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
    )
    if action_allocation_bins != 1 or action_level_bins != 1:
        raise ValueError(
            "evaluate_tail only supports action_allocation_bins=1 and action_level_bins=1 "
            f"(got alloc_bins={action_allocation_bins} level_bins={action_level_bins})"
        )

    fallback_actions = 1 + 2 * num_symbols
    num_actions = _infer_num_actions(state_dict, fallback=fallback_actions)
    if num_actions != fallback_actions:
        raise ValueError(f"Checkpoint num_actions={num_actions} does not match expected={fallback_actions}")

    effective_arch = str(arch)
    if effective_arch == "auto":
        effective_arch = infer_arch_from_state_dict(state_dict)
    effective_hidden = int(hidden_size) if hidden_size is not None else infer_hidden_size_from_state_dict(state_dict, effective_arch)

    policy = load_policy_from_resolved_metadata(
        state_dict=state_dict,
        effective_arch=effective_arch,
        effective_hidden_size=effective_hidden,
        effective_resmlp_blocks=infer_resmlp_blocks_from_state_dict(state_dict) if effective_arch == "resmlp" else None,
        obs_size=obs_size,
        num_actions=num_actions,
        device=device,
        mlp_factory=lambda obs_size, num_actions, hidden: TradingPolicy(obs_size, num_actions, hidden=hidden),
        resmlp_factory=(
            lambda obs_size, num_actions, hidden, num_blocks: ResidualTradingPolicy(
                obs_size,
                num_actions,
                hidden=hidden,
                num_blocks=num_blocks,
            )
        ),
    )
    return LoadedPolicy(
        policy=policy,
        arch=effective_arch,
        hidden_size=effective_hidden,
        action_allocation_bins=int(action_allocation_bins),
        action_level_bins=int(action_level_bins),
        action_max_offset_bps=float(action_max_offset_bps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on tail window (pure python sim)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to MKTD .bin")
    parser.add_argument("--eval-hours", type=int, default=720, help="Number of steps to simulate (default: 720h ~= 30d)")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument(
        "--fill-buffer-bps",
        type=float,
        default=5.0,
        help="Require the daily bar to trade through each limit by this many bps before fill.",
    )
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--arch", choices=["auto", "mlp", "resmlp"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--disable-shorts", action="store_true")
    parser.add_argument("--shortable-symbols", type=str, default=None)
    parser.add_argument(
        "--decision-lag",
        type=int,
        default=0,
        help="Delay actions by N bars (0=execute immediately; 1=use previous bar's decision).",
    )
    parser.add_argument("--deterministic", action="store_true", help="Argmax actions (recommended for eval)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    data = read_mktd(Path(args.data_path))
    tail = _slice_tail(data, steps=int(args.eval_hours))

    num_symbols = tail.num_symbols
    features_per_sym = tail.features.shape[2]
    obs_size = num_symbols * features_per_sym + 5 + num_symbols
    loaded = load_policy(
        checkpoint_path=ckpt_path,
        obs_size=obs_size,
        num_symbols=num_symbols,
        arch=str(args.arch),
        hidden_size=args.hidden_size,
        device=device,
    )
    policy = loaded.policy
    arch = loaded.arch
    hidden = loaded.hidden_size

    shortable_mask = _build_shortable_mask(tail.symbols, args.shortable_symbols)
    if shortable_mask is not None:
        shortable_mask = shortable_mask.to(device=device)

    decision_lag = int(args.decision_lag)
    if decision_lag < 0:
        raise ValueError("--decision-lag must be >= 0")
    pending_actions: collections.deque[int] = collections.deque(maxlen=max(1, decision_lag + 1))

    def _policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
        if args.disable_shorts:
            logits = _mask_all_shorts(logits, num_symbols=num_symbols)
        elif shortable_mask is not None:
            logits = _mask_disallowed_shorts(logits, num_symbols=num_symbols, shortable_mask=shortable_mask)
        if args.deterministic:
            action_now = int(torch.argmax(logits, dim=-1).item())
        else:
            action_now = int(Categorical(logits=logits).sample().item())

        if decision_lag <= 0:
            return action_now

        pending_actions.append(action_now)
        if len(pending_actions) <= decision_lag:
            return 0  # flat until the lag buffer fills
        return pending_actions.popleft()

    result = simulate_daily_policy(
        tail,
        _policy_fn,
        max_steps=int(args.eval_hours),
        fee_rate=float(args.fee_rate),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.max_leverage),
        periods_per_year=float(args.periods_per_year),
        short_borrow_apr=float(args.short_borrow_apr),
    )
    annualized_return = annualize_total_return(
        float(result.total_return),
        periods=float(args.eval_hours),
        periods_per_year=float(args.periods_per_year),
    )

    out = {
        "checkpoint": str(ckpt_path),
        "data_path": str(Path(args.data_path)),
        "symbols": tail.symbols,
        "eval_hours": int(args.eval_hours),
        "decision_lag": int(decision_lag),
        "fee_rate": float(args.fee_rate),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "max_leverage": float(args.max_leverage),
        "short_borrow_apr": float(args.short_borrow_apr),
        "periods_per_year": float(args.periods_per_year),
        "arch": str(arch),
        "hidden_size": int(hidden),
        "action_allocation_bins": int(loaded.action_allocation_bins),
        "action_level_bins": int(loaded.action_level_bins),
        "action_max_offset_bps": float(loaded.action_max_offset_bps),
        "total_return": float(result.total_return),
        "annualized_return": float(annualized_return),
        "sortino": float(result.sortino),
        "max_drawdown": float(result.max_drawdown),
        "num_trades": int(result.num_trades),
        "win_rate": float(result.win_rate),
        "avg_hold_steps": float(result.avg_hold_steps),
        "summary": {
            "total_return": float(result.total_return),
            "annualized_return": float(annualized_return),
            "sortino": float(result.sortino),
            "max_drawdown": float(result.max_drawdown),
            "num_trades": int(result.num_trades),
            "win_rate": float(result.win_rate),
            "avg_hold_steps": float(result.avg_hold_steps),
        },
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
