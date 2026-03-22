#!/usr/bin/env python3
"""Fast vectorized evaluation using the C env with batched GPU inference.

Replaces the slow Python hourly_replay simulator used in evaluate_holdout.py
with the compiled C environment, batched GPU forward passes, and torch.compile.

Key speedups vs evaluate_holdout.py:
  1. C env is ~100x faster per step than Python simulator
  2. Batched GPU inference (N envs per forward pass vs batch_size=1)
  3. torch.compile eliminates Python overhead in policy forward pass
  4. Aggressive early-exit skips clearly-bad checkpoints

Usage:
  python -m pufferlib_market.evaluate_fast \
    --checkpoint path/to/best.pt \
    --data-path path/to/val.bin \
    --n-windows 20 --eval-hours 720 --deterministic

  # As a library (from autoresearch_rl.py):
  from pufferlib_market.evaluate_fast import fast_holdout_eval
  result = fast_holdout_eval(checkpoint, data_path, n_windows=20, ...)
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Policy definitions (must match train.py exactly)
# ---------------------------------------------------------------------------

class TradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
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


# ---------------------------------------------------------------------------
# Checkpoint introspection helpers
# ---------------------------------------------------------------------------

def _infer_num_actions(state_dict: dict, fallback: int) -> int:
    for key in ("actor.2.bias", "actor.2.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return int(fallback)


def _infer_arch(state_dict: dict) -> str:
    if "input_proj.weight" in state_dict:
        return "resmlp"
    if "encoder.0.weight" in state_dict:
        return "mlp"
    for key in state_dict:
        if key.startswith(("input_proj.", "blocks.")):
            return "resmlp"
        if key.startswith("encoder."):
            return "mlp"
    raise ValueError("Could not infer arch from state_dict")


def _infer_hidden_size(state_dict: dict, arch: str) -> int:
    if arch == "resmlp":
        return int(state_dict["input_proj.weight"].shape[0])
    return int(state_dict["encoder.0.weight"].shape[0])


def _infer_resmlp_blocks(state_dict: dict) -> int:
    idxs = [int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.") and k.split(".")[1].isdigit()]
    return max(idxs) + 1 if idxs else 3


# ---------------------------------------------------------------------------
# Dataclass for per-window results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowResult:
    start_idx: int
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_hold_hours: float


# ---------------------------------------------------------------------------
# Core: fast vectorized evaluation using C env
# ---------------------------------------------------------------------------

def fast_holdout_eval(
    checkpoint_path: str,
    data_path: str,
    *,
    n_windows: int = 20,
    eval_hours: int = 720,
    seed: int = 1337,
    fee_rate: float = 0.001,
    fill_slippage_bps: float = 8.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    deterministic: bool = True,
    disable_shorts: bool = False,
    arch: str = "auto",
    hidden_size: Optional[int] = None,
    device_str: str = "auto",
    use_compile: bool = True,
    early_exit_after: int = 5,
    early_exit_threshold: float = -0.15,
    verbose: bool = False,
) -> dict:
    """Run fast holdout evaluation using the C env with batched GPU inference.

    Returns a dict compatible with evaluate_holdout.py's JSON output format.
    """
    t0 = time.monotonic()

    # Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Read data header
    with open(data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])

    obs_size = num_symbols * 16 + 5 + num_symbols

    # Load checkpoint
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload

    # Infer action grid
    alloc_bins = int(payload.get("action_allocation_bins", 1)) if isinstance(payload, dict) else 1
    level_bins = int(payload.get("action_level_bins", 1)) if isinstance(payload, dict) else 1
    per_sym_actions = max(1, alloc_bins) * max(1, level_bins)
    num_actions = 1 + 2 * num_symbols * per_sym_actions

    # Infer architecture
    if arch == "auto":
        arch = _infer_arch(state_dict)
    hidden = hidden_size if hidden_size is not None else _infer_hidden_size(state_dict, arch)

    if arch == "resmlp":
        num_blocks = _infer_resmlp_blocks(state_dict)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden, num_blocks=num_blocks).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # torch.compile for faster inference
    if use_compile and device.type == "cuda":
        try:
            compiled_policy = torch.compile(policy, mode="reduce-overhead", fullgraph=True)
            # Warmup
            dummy = torch.randn(n_windows, obs_size, device=device)
            with torch.inference_mode():
                compiled_policy(dummy)
                compiled_policy(dummy)
            if verbose:
                print(f"  torch.compile: OK (reduce-overhead)")
        except Exception as e:
            if verbose:
                print(f"  torch.compile: failed ({e}), using eager mode")
            compiled_policy = policy
    else:
        compiled_policy = policy

    # Short masking setup
    side_block = num_symbols * per_sym_actions

    # Load shared market data
    import pufferlib_market.binding as binding
    binding.shared(data_path=str(Path(data_path).resolve()))

    # Generate window start indices (reproducible with seed)
    rng = np.random.default_rng(seed)
    window_len = eval_hours + 1
    max_offset = num_timesteps - window_len
    if max_offset < 0:
        raise ValueError(f"Data too short: {num_timesteps} timesteps, need {window_len}")
    starts = rng.choice(np.arange(max_offset + 1), size=n_windows, replace=(max_offset + 1 < n_windows))

    # Allocate shared buffers
    obs_bufs = np.zeros((n_windows, obs_size), dtype=np.float32)
    act_bufs = np.zeros((n_windows,), dtype=np.int32)
    rew_bufs = np.zeros((n_windows,), dtype=np.float32)
    term_bufs = np.zeros((n_windows,), dtype=np.uint8)
    trunc_bufs = np.zeros((n_windows,), dtype=np.uint8)

    # Initialize vectorized env (all envs with forced_offset=-1 initially)
    vec_handle = binding.vec_init(
        obs_bufs, act_bufs, rew_bufs, term_bufs, trunc_bufs,
        n_windows, seed,
        max_steps=eval_hours,
        fee_rate=fee_rate,
        max_leverage=max_leverage,
        short_borrow_apr=short_borrow_apr,
        periods_per_year=periods_per_year,
        fill_slippage_bps=fill_slippage_bps,
        forced_offset=-1,
        action_allocation_bins=alloc_bins,
        action_level_bins=level_bins,
        enable_drawdown_profit_early_exit=True,
        drawdown_profit_early_exit_min_steps=20,
        drawdown_profit_early_exit_progress_fraction=0.5,
    )

    # Set per-env forced offsets using vec_set_offsets
    binding.vec_set_offsets(vec_handle, starts.astype(np.int32))

    # Reset with forced offsets in place
    binding.vec_reset(vec_handle, seed)

    # Track per-window results
    completed = [None] * n_windows  # type: list[Optional[WindowResult]]
    active = np.ones(n_windows, dtype=bool)
    n_completed = 0
    early_exited = False

    # Run episodes - all windows step in lockstep
    for step in range(eval_hours + 10):  # +10 safety margin
        if not active.any():
            break

        # Batched GPU inference - THE key speedup
        obs_tensor = torch.from_numpy(obs_bufs).to(device, non_blocking=True)
        with torch.inference_mode():
            logits, _ = compiled_policy(obs_tensor)

        # Mask shorts if needed
        if disable_shorts:
            logits[:, 1 + side_block:] = torch.finfo(logits.dtype).min

        # Deterministic or sampled actions
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()

        act_bufs[:] = actions.cpu().numpy().astype(np.int32)

        # Step all envs
        binding.vec_step(vec_handle)

        # Check for completed episodes via terminal flag
        for i in range(n_windows):
            if not active[i]:
                continue
            if term_bufs[i]:
                # Read per-env log data before vec_log zeros it
                env_handle = binding.vec_env_at(vec_handle, i)
                env_data = binding.env_get(env_handle)
                completed[i] = WindowResult(
                    start_idx=int(starts[i]),
                    total_return=float(env_data.get("total_return", 0.0)),
                    sortino=float(env_data.get("sortino", 0.0)),
                    max_drawdown=float(env_data.get("max_drawdown", 0.0)),
                    num_trades=int(env_data.get("num_trades", 0)),
                    win_rate=float(env_data.get("win_rate", 0.0)),
                    avg_hold_hours=float(env_data.get("avg_hold_hours", 0.0)),
                )
                active[i] = False
                n_completed += 1

                if verbose:
                    r = completed[i]
                    print(f"  Window {i}: ret={r.total_return:+.4f} sortino={r.sortino:.2f} "
                          f"dd={r.max_drawdown:.4f} trades={r.num_trades}")

        # Aggressive early-exit: skip clearly bad checkpoints
        if early_exit_after > 0 and n_completed >= early_exit_after and not early_exited:
            done_returns = [c.total_return for c in completed if c is not None]
            median_ret = float(np.median(done_returns))
            if median_ret < early_exit_threshold:
                if verbose:
                    print(f"  EARLY EXIT: median return {median_ret:+.4f} < {early_exit_threshold} "
                          f"after {n_completed}/{n_windows} windows")
                early_exited = True
                break

    binding.vec_close(vec_handle)

    elapsed = time.monotonic() - t0

    # Collect results
    valid_results = [c for c in completed if c is not None]
    if not valid_results:
        return {"error": "no windows completed", "elapsed_s": elapsed}

    returns = [r.total_return for r in valid_results]
    sortinos = [r.sortino for r in valid_results]
    maxdds = [r.max_drawdown for r in valid_results]

    def _pct(vals, q):
        return float(np.percentile(vals, q)) if vals else 0.0

    summary = {
        "median_total_return": _pct(returns, 50),
        "p10_total_return": _pct(returns, 10),
        "p90_total_return": _pct(returns, 90),
        "median_sortino": _pct(sortinos, 50),
        "p10_sortino": _pct(sortinos, 10),
        "p90_sortino": _pct(sortinos, 90),
        "median_max_drawdown": _pct(maxdds, 50),
        "p90_max_drawdown": _pct(maxdds, 90),
    }

    out = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "eval_hours": eval_hours,
        "n_windows": n_windows,
        "n_completed": len(valid_results),
        "seed": seed,
        "fee_rate": fee_rate,
        "fill_slippage_bps": fill_slippage_bps,
        "max_leverage": max_leverage,
        "periods_per_year": periods_per_year,
        "elapsed_s": elapsed,
        "early_exit": early_exited,
        "summary": summary,
        "windows": [asdict(r) for r in valid_results],
    }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fast vectorized holdout evaluation (C env + batched GPU)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--eval-hours", type=int, default=720)
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-slippage-bps", type=float, default=8.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-shorts", action="store_true")
    parser.add_argument("--arch", choices=["auto", "mlp", "resmlp"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--early-exit-after", type=int, default=5,
                        help="Early-exit after N windows if median return < threshold (0=disable)")
    parser.add_argument("--early-exit-threshold", type=float, default=-0.15,
                        help="Median return threshold for early-exit (default: -15%%)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = fast_holdout_eval(
        args.checkpoint,
        args.data_path,
        n_windows=args.n_windows,
        eval_hours=args.eval_hours,
        seed=args.seed,
        fee_rate=args.fee_rate,
        fill_slippage_bps=args.fill_slippage_bps,
        max_leverage=args.max_leverage,
        periods_per_year=args.periods_per_year,
        short_borrow_apr=args.short_borrow_apr,
        deterministic=args.deterministic,
        disable_shorts=args.disable_shorts,
        arch=args.arch,
        hidden_size=args.hidden_size,
        device_str=args.device,
        use_compile=not args.no_compile,
        early_exit_after=args.early_exit_after,
        early_exit_threshold=args.early_exit_threshold,
        verbose=args.verbose,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result.get("summary", result), indent=2))
    if "elapsed_s" in result:
        print(f"\nElapsed: {result['elapsed_s']:.2f}s ({result.get('n_completed', 0)}/{result.get('n_windows', 0)} windows)")


if __name__ == "__main__":
    main()
