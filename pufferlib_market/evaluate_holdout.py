#!/usr/bin/env python3
"""Holdout evaluation for PPO checkpoints on MKTD datasets.

This samples multiple random fixed-length windows and evaluates a policy
deterministically (argmax) to estimate stability of returns/Sortino.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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


def _act_holdout(name: str) -> nn.Module:
    if name == "relu_sq":
        from pufferlib_market.train import _ReLUSq
        return _ReLUSq()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class TradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, activation: str = "relu"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            _act_holdout(activation),
            nn.Linear(hidden, hidden),
            _act_holdout(activation),
            nn.Linear(hidden, hidden),
            _act_holdout(activation),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            _act_holdout(activation),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            _act_holdout(activation),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ResidualTradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(obs_size, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.out_norm(self.blocks(self.input_proj(x)))
        return self.actor(h), self.critic(h).squeeze(-1)


def _infer_num_actions(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    for key in ("actor.2.bias", "actor.2.weight"):
        if key in state_dict:
            t = state_dict[key]
            if t.ndim == 1:
                return int(t.shape[0])
            if t.ndim == 2:
                return int(t.shape[0])
    return int(fallback)


def _infer_action_grid(
    *,
    payload: object,
    state_dict: dict[str, torch.Tensor],
    num_symbols: int,
) -> tuple[int, int, int, float]:
    alloc_bins = 1
    level_bins = 1
    max_offset_bps = 0.0
    if isinstance(payload, dict):
        try:
            alloc_bins = max(1, int(payload.get("action_allocation_bins", 1)))
        except (TypeError, ValueError):
            alloc_bins = 1
        try:
            level_bins = max(1, int(payload.get("action_level_bins", 1)))
        except (TypeError, ValueError):
            level_bins = 1
        try:
            max_offset_bps = max(0.0, float(payload.get("action_max_offset_bps", 0.0)))
        except (TypeError, ValueError):
            max_offset_bps = 0.0

    per_symbol_actions = int(alloc_bins) * int(level_bins)
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
        if inferred_per_symbol % int(level_bins) == 0:
            alloc_bins = inferred_per_symbol // int(level_bins)
        else:
            level_bins = 1
            alloc_bins = inferred_per_symbol
        per_symbol_actions = inferred_per_symbol
    if per_symbol_actions < 1:
        raise ValueError("Invalid action grid (per_symbol_actions < 1)")
    return int(num_actions), int(alloc_bins), int(level_bins), float(max_offset_bps)


def _infer_arch(state_dict: dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    # GRU must come before resmlp: GRUTradingPolicy also has input_proj.weight
    if any("gru." in k for k in keys):
        return "gru"
    # Transformer: has attn.in_proj_weight or sym_embed (symbol_proj also present)
    if any(k.startswith("attn.") or k.startswith("sym_embed") or k.startswith("symbol_proj") for k in keys):
        return "transformer"
    if "input_proj.weight" in keys:
        return "resmlp"
    if "encoder.0.weight" in keys:
        return "mlp"
    # DepthRecurrence: has blocks.0.net.0.weight and shares block weights
    if any(k.startswith("blocks.") for k in keys):
        return "depth_recurrence"
    for key in state_dict:
        if key.startswith("encoder."):
            return "mlp"
    raise ValueError("Could not infer policy arch from checkpoint state_dict keys")


def _infer_hidden_size(state_dict: dict[str, torch.Tensor], *, arch: str) -> int:
    if arch == "resmlp":
        w = state_dict.get("input_proj.weight")
        if w is None or w.ndim != 2:
            raise ValueError("Checkpoint missing input_proj.weight for resmlp")
        return int(w.shape[0])
    if arch == "transformer":
        # TransformerTradingPolicy: mlp.0.weight shape is (hidden, attn_out_size)
        w = state_dict.get("mlp.0.weight")
        if w is not None and w.ndim == 2:
            return int(w.shape[0])
    if arch == "gru":
        # GRUTradingPolicy: input_proj.weight shape is (hidden, obs_size)
        w = state_dict.get("input_proj.weight")
        if w is not None and w.ndim == 2:
            return int(w.shape[0])
    w = state_dict.get("encoder.0.weight")
    if w is None or w.ndim != 2:
        raise ValueError(f"Checkpoint missing weight key for arch={arch}")
    return int(w.shape[0])


def _infer_resmlp_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    idxs: list[int] = []
    for key in state_dict:
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 2:
            continue
        try:
            idxs.append(int(parts[1]))
        except ValueError:
            continue
    if not idxs:
        return 3
    return int(max(idxs) + 1)


def _build_shortable_mask(symbols: list[str], shortable_csv: Optional[str]) -> Optional[torch.Tensor]:
    if not shortable_csv:
        return None
    requested = {s.strip().upper() for s in shortable_csv.split(",") if s.strip()}
    if not requested:
        return None
    return torch.tensor([sym.upper() in requested for sym in symbols], dtype=torch.bool)


def _slice_window(data: MktdData, *, start: int, steps: int) -> MktdData:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = int(start)
    need = int(steps) + 1
    end = start + need
    if start < 0 or end > int(data.num_timesteps):
        raise ValueError(f"Window out of range: start={start} end={end} timesteps={data.num_timesteps}")
    return MktdData(
        version=data.version,
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=None if data.tradable is None else data.tradable[start:end].copy(),
    )


@dataclass(frozen=True)
class WindowMetric:
    start_idx: int
    total_return: float
    annualized_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-window holdout evaluation for PPO checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to MKTD .bin")
    parser.add_argument("--eval-hours", type=int, default=720, help="Window length in steps (default: 720h)")
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--end-within-hours", type=int, default=None, help="Only sample windows ending within last N hours")
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
    parser.add_argument("--disable-shorts", action="store_true")
    parser.add_argument("--shortable-symbols", type=str, default=None)
    parser.add_argument("--decision-lag", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Argmax actions (recommended)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    steps = int(args.eval_hours)
    if steps < 1:
        raise ValueError("--eval-hours must be >= 1")
    n_windows = int(args.n_windows)
    if n_windows < 1:
        raise ValueError("--n-windows must be >= 1")
    decision_lag = int(args.decision_lag)
    if decision_lag < 0:
        raise ValueError("--decision-lag must be >= 0")

    device = torch.device(args.device)
    payload = torch.load(str(Path(args.checkpoint)), map_location=device, weights_only=False)
    state_dict = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format (expected state_dict or dict with 'model')")

    data = read_mktd(Path(args.data_path))
    num_symbols = data.num_symbols
    features_per_sym = int(data.features.shape[2])
    obs_size = int(num_symbols) * features_per_sym + 5 + int(num_symbols)
    num_actions, alloc_bins, level_bins, max_offset_bps = _infer_action_grid(
        payload=payload,
        state_dict=state_dict,
        num_symbols=int(num_symbols),
    )
    per_symbol_actions = int(alloc_bins) * int(level_bins)

    # Use arch stored in checkpoint (added in newer checkpoints) or infer from state_dict
    stored_arch = payload.get("arch", "") if isinstance(payload, dict) else ""
    arch = stored_arch if stored_arch else _infer_arch(state_dict)
    hidden = _infer_hidden_size(state_dict, arch=arch)
    if arch == "resmlp":
        num_blocks = _infer_resmlp_blocks(state_dict)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=int(hidden), num_blocks=int(num_blocks)).to(device)
    elif arch == "transformer":
        from pufferlib_market.train import TransformerTradingPolicy
        policy = TransformerTradingPolicy(obs_size, num_actions, hidden=int(hidden)).to(device)
    elif arch == "gru":
        from pufferlib_market.train import GRUTradingPolicy
        policy = GRUTradingPolicy(obs_size, num_actions, hidden=int(hidden)).to(device)
    elif arch == "depth_recurrence":
        from pufferlib_market.train import DepthRecurrenceTradingPolicy
        policy = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=int(hidden)).to(device)
    elif arch == "mlp_relu_sq":
        policy = TradingPolicy(obs_size, num_actions, hidden=int(hidden), activation="relu_sq").to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=int(hidden)).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    shortable_mask = _build_shortable_mask(list(data.symbols), args.shortable_symbols)
    if shortable_mask is not None:
        shortable_mask = shortable_mask.to(device=device)

    pending_actions: collections.deque[int] = collections.deque(maxlen=max(1, decision_lag + 1))

    def _policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
        if args.disable_shorts:
            logits = _mask_all_shorts(
                logits,
                num_symbols=int(num_symbols),
                per_symbol_actions=int(per_symbol_actions),
            )
        elif shortable_mask is not None:
            logits = _mask_disallowed_shorts(
                logits,
                num_symbols=int(num_symbols),
                per_symbol_actions=int(per_symbol_actions),
                shortable_mask=shortable_mask.to(device=logits.device),
            )
        if args.deterministic:
            action_now = int(torch.argmax(logits, dim=-1).item())
        else:
            action_now = int(Categorical(logits=logits).sample().item())

        if decision_lag <= 0:
            return action_now
        pending_actions.append(action_now)
        if len(pending_actions) <= decision_lag:
            return 0
        return pending_actions.popleft()

    rng = np.random.default_rng(int(args.seed))
    T = int(data.num_timesteps)
    window_len = steps + 1
    if T < window_len:
        raise ValueError(f"MKTD too short for eval_hours={steps}: timesteps={T}")

    # Candidate end indices (exclusive).
    end_min = window_len
    end_max = T
    if args.end_within_hours is not None:
        within = int(args.end_within_hours)
        if within < 1:
            raise ValueError("--end-within-hours must be >= 1")
        end_min = max(end_min, T - within)

    candidate_starts = np.arange(end_min - window_len, end_max - window_len + 1, dtype=np.int64)
    if candidate_starts.size <= 0:
        raise ValueError("No candidate windows (check --end-within-hours and --eval-hours)")

    replace = bool(candidate_starts.size < n_windows)
    starts = rng.choice(candidate_starts, size=n_windows, replace=replace)

    metrics: list[WindowMetric] = []
    for start_idx in starts.tolist():
        window = _slice_window(data, start=int(start_idx), steps=steps)
        pending_actions.clear()
        result = simulate_daily_policy(
            window,
            _policy_fn,
            max_steps=int(steps),
            fee_rate=float(args.fee_rate),
            fill_buffer_bps=float(args.fill_buffer_bps),
            max_leverage=float(args.max_leverage),
            periods_per_year=float(args.periods_per_year),
            short_borrow_apr=float(args.short_borrow_apr),
            action_allocation_bins=int(alloc_bins),
            action_level_bins=int(level_bins),
            action_max_offset_bps=float(max_offset_bps),
        )
        annualized_return = annualize_total_return(
            float(result.total_return),
            periods=float(steps),
            periods_per_year=float(args.periods_per_year),
        )
        metrics.append(
            WindowMetric(
                start_idx=int(start_idx),
                total_return=float(result.total_return),
                annualized_return=float(annualized_return),
                sortino=float(result.sortino),
                max_drawdown=float(result.max_drawdown),
                num_trades=int(result.num_trades),
                win_rate=float(result.win_rate),
            )
        )

    returns = [m.total_return for m in metrics]
    annualized_returns = [m.annualized_return for m in metrics]
    sortinos = [m.sortino for m in metrics]
    maxdds = [m.max_drawdown for m in metrics]

    out = {
        "checkpoint": str(Path(args.checkpoint)),
        "data_path": str(Path(args.data_path)),
        "symbols": list(data.symbols),
        "arch": str(arch),
        "hidden_size": int(hidden),
        "eval_hours": int(steps),
        "n_windows": int(n_windows),
        "seed": int(args.seed),
        "end_within_hours": int(args.end_within_hours) if args.end_within_hours is not None else None,
        "decision_lag": int(decision_lag),
        "fee_rate": float(args.fee_rate),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "max_leverage": float(args.max_leverage),
        "short_borrow_apr": float(args.short_borrow_apr),
        "action_allocation_bins": int(alloc_bins),
        "action_level_bins": int(level_bins),
        "action_max_offset_bps": float(max_offset_bps),
        "periods_per_year": float(args.periods_per_year),
        "summary": {
            "median_total_return": float(_percentile(returns, 50)),
            "p10_total_return": float(_percentile(returns, 10)),
            "p90_total_return": float(_percentile(returns, 90)),
            "median_annualized_return": float(_percentile(annualized_returns, 50)),
            "p10_annualized_return": float(_percentile(annualized_returns, 10)),
            "p90_annualized_return": float(_percentile(annualized_returns, 90)),
            "median_sortino": float(_percentile(sortinos, 50)),
            "p10_sortino": float(_percentile(sortinos, 10)),
            "p90_sortino": float(_percentile(sortinos, 90)),
            "median_max_drawdown": float(_percentile(maxdds, 50)),
            "p90_max_drawdown": float(_percentile(maxdds, 90)),
        },
        "windows": [asdict(m) for m in metrics],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(json.dumps(out["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
