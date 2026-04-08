#!/usr/bin/env python3
"""Evaluate a PPO checkpoint on the *tail window* of an MKTD dataset.

Why this exists:
- The compiled C env randomises episode start offsets, which makes it hard to
  compare "latest 30d" apples-to-apples.
- Here we use the pure-python simulator in ``pufferlib_market.hourly_replay`` on
  a deterministic tail slice (e.g., last 720 hours = ~30 calendar days).
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
    load_checkpoint_payload,
    load_policy_from_resolved_metadata,
    resolve_checkpoint_action_grid_config,
    resolve_checkpoint_policy_details,
)
from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return


def _mask_all_shorts(logits: torch.Tensor, *, num_symbols: int, per_symbol_actions: int = 1) -> torch.Tensor:
    masked = logits.clone()
    side_block = int(num_symbols) * max(1, int(per_symbol_actions))
    masked[:, 1 + side_block :] = torch.finfo(masked.dtype).min
    return masked


def _mask_disallowed_shorts(
    logits: torch.Tensor,
    *,
    num_symbols: int,
    per_symbol_actions: int,
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
    side_block = int(num_symbols) * max(1, int(per_symbol_actions))
    disallowed = (~shortable_mask).nonzero(as_tuple=False).view(-1).tolist()
    for sym_idx in disallowed:
        start = 1 + side_block + int(sym_idx) * max(1, int(per_symbol_actions))
        end = start + max(1, int(per_symbol_actions))
        masked[:, start:end] = min_val
    return masked


class TradingPolicy(nn.Module):
    """Must match pufferlib_market.train.TradingPolicy exactly."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, activation: str = "relu"):
        super().__init__()
        self._activation_name = activation
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            _get_activation(activation),
            nn.Linear(hidden, hidden),
            _get_activation(activation),
            nn.Linear(hidden, hidden),
            _get_activation(activation),
        )
        # encoder_norm added to train.py to prevent BF16 precision collapse.
        # _use_encoder_norm set by load_policy: True only for new ckpts that have this layer.
        self.encoder_norm = nn.LayerNorm(hidden)
        self._use_encoder_norm = False
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            _get_activation(activation),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            _get_activation(activation),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        if getattr(self, "_use_encoder_norm", False):
            h = self.encoder_norm(h)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class _ReLUSq(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


def _get_activation(name: str) -> nn.Module:
    if name == "relu_sq":
        return _ReLUSq()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


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
    num_actions: int
    per_symbol_actions: int
    action_allocation_bins: int
    action_level_bins: int
    action_max_offset_bps: float


def _infer_action_grid(
    *,
    payload: object,
    state_dict: dict[str, torch.Tensor],
    num_symbols: int,
) -> tuple[int, int, int, int, float]:
    action_allocation_bins, action_level_bins, action_max_offset_bps = resolve_checkpoint_action_grid_config(
        payload,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
    )
    per_symbol_actions = int(action_allocation_bins) * int(action_level_bins)
    expected_actions = 1 + 2 * int(num_symbols) * int(per_symbol_actions)
    num_actions = _infer_num_actions(state_dict, fallback=expected_actions)
    if num_actions != expected_actions:
        denom = 2 * int(num_symbols)
        rem = int(num_actions) - 1
        if rem <= 0 or denom <= 0 or rem % denom != 0:
            raise ValueError(
                f"Checkpoint num_actions={num_actions} incompatible with num_symbols={num_symbols} "
                f"(cannot infer action grid)"
            )
        inferred_per_symbol = rem // denom
        if inferred_per_symbol % int(action_level_bins) == 0:
            action_allocation_bins = inferred_per_symbol // int(action_level_bins)
        else:
            action_level_bins = 1
            action_allocation_bins = inferred_per_symbol
        per_symbol_actions = inferred_per_symbol
    if per_symbol_actions < 1:
        raise ValueError("Invalid action grid (per_symbol_actions < 1)")
    return (
        int(num_actions),
        int(per_symbol_actions),
        int(action_allocation_bins),
        int(action_level_bins),
        float(action_max_offset_bps),
    )


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

    (
        num_actions,
        per_symbol_actions,
        action_allocation_bins,
        action_level_bins,
        action_max_offset_bps,
    ) = _infer_action_grid(
        payload=payload,
        state_dict=state_dict,
        num_symbols=num_symbols,
    )

    resolved = resolve_checkpoint_policy_details(
        payload if isinstance(payload, dict) and "model" in payload else {"model": state_dict},
        arch=(arch if arch != "auto" else "mlp"),
        hidden_size=(hidden_size if hidden_size is not None else 256),
    )
    effective_arch = resolved.effective_arch
    effective_hidden = resolved.effective_hidden_size

    policy = load_policy_from_resolved_metadata(
        state_dict=state_dict,
        effective_arch=effective_arch,
        effective_hidden_size=effective_hidden,
        effective_activation=resolved.effective_activation,
        effective_resmlp_blocks=resolved.effective_resmlp_blocks,
        obs_size=obs_size,
        num_actions=num_actions,
        device=device,
        mlp_factory=lambda obs_size, num_actions, hidden, activation="relu": TradingPolicy(
            obs_size,
            num_actions,
            hidden=hidden,
            activation=activation,
        ),
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
        num_actions=int(num_actions),
        per_symbol_actions=int(per_symbol_actions),
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
    per_symbol_actions = loaded.per_symbol_actions

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
            logits = _mask_all_shorts(logits, num_symbols=num_symbols, per_symbol_actions=per_symbol_actions)
        elif shortable_mask is not None:
            logits = _mask_disallowed_shorts(
                logits,
                num_symbols=num_symbols,
                per_symbol_actions=per_symbol_actions,
                shortable_mask=shortable_mask,
            )
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
        action_allocation_bins=int(loaded.action_allocation_bins),
        action_level_bins=int(loaded.action_level_bins),
        action_max_offset_bps=float(loaded.action_max_offset_bps),
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
