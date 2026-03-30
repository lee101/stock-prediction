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

import numpy as np
import torch
from torch import nn

from pufferlib_market.checkpoint_loader import (
    extract_checkpoint_state_dict,
    infer_arch_from_state_dict,
    infer_hidden_size_from_state_dict,
    infer_resmlp_blocks_from_state_dict,
    load_checkpoint_payload,
    resolve_checkpoint_action_grid_config,
)


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


@dataclass(frozen=True)
class LoadedPolicy:
    policy: nn.Module
    arch: str
    hidden_size: int
    action_allocation_bins: int
    action_level_bins: int
    action_max_offset_bps: float


def _load_policy_for_eval(
    *,
    payload: object,
    obs_size: int,
    num_symbols: int,
    arch: str,
    hidden_size: int | None,
    device: torch.device,
) -> tuple[LoadedPolicy, dict[str, torch.Tensor]]:
    state_dict = extract_checkpoint_state_dict(payload)

    alloc_bins, level_bins, max_offset_bps = resolve_checkpoint_action_grid_config(
        payload,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
    )
    per_sym_actions = max(1, alloc_bins) * max(1, level_bins)
    num_actions = 1 + 2 * num_symbols * per_sym_actions

    if arch == "auto":
        arch = infer_arch_from_state_dict(state_dict)
    hidden = hidden_size if hidden_size is not None else infer_hidden_size_from_state_dict(state_dict, arch)

    if arch == "resmlp":
        num_blocks = infer_resmlp_blocks_from_state_dict(state_dict)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden, num_blocks=num_blocks).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden).to(device)

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    ignored = {"obs_mean", "obs_std", "encoder_norm.weight", "encoder_norm.bias"}
    bad_missing = [k for k in missing if k not in ignored]
    bad_unexpected = [k for k in unexpected if k not in ignored]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch — missing: {bad_missing}, unexpected: {bad_unexpected}"
        )

    policy.eval()
    return (
        LoadedPolicy(
            policy=policy,
            arch=str(arch),
            hidden_size=int(hidden),
            action_allocation_bins=int(alloc_bins),
            action_level_bins=int(level_bins),
            action_max_offset_bps=float(max_offset_bps),
        ),
        state_dict,
    )


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


def _summarize_window_results(valid_results: list[WindowResult]) -> dict[str, object]:
    returns = [r.total_return for r in valid_results]
    sortinos = [r.sortino for r in valid_results]
    maxdds = [r.max_drawdown for r in valid_results]

    def _pct(vals: list[float], q: int) -> float:
        return float(np.percentile(vals, q)) if vals else 0.0

    best_window = max(valid_results, key=lambda r: r.total_return)
    worst_window = min(valid_results, key=lambda r: r.total_return)
    positive_window_count = sum(r.total_return > 0.0 for r in valid_results)

    return {
        "median_total_return": _pct(returns, 50),
        "p10_total_return": _pct(returns, 10),
        "p90_total_return": _pct(returns, 90),
        "median_sortino": _pct(sortinos, 50),
        "p10_sortino": _pct(sortinos, 10),
        "p90_sortino": _pct(sortinos, 90),
        "median_max_drawdown": _pct(maxdds, 50),
        "p90_max_drawdown": _pct(maxdds, 90),
        "positive_window_count": positive_window_count,
        "positive_window_ratio": positive_window_count / len(valid_results),
        "best_window": asdict(best_window),
        "worst_window": asdict(worst_window),
    }


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
    hidden_size: int | None = None,
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
    _, _, num_symbols, num_timesteps, features_per_sym, _ = struct.unpack("<4sIIIII", header[:24])
    if features_per_sym == 0:
        features_per_sym = 16  # v1/v2 backwards compat

    obs_size = num_symbols * features_per_sym + 5 + num_symbols

    # Load checkpoint
    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    loaded, _state_dict = _load_policy_for_eval(
        payload=payload,
        obs_size=obs_size,
        num_symbols=num_symbols,
        arch=arch,
        hidden_size=hidden_size,
        device=device,
    )
    policy = loaded.policy

    alloc_bins = loaded.action_allocation_bins
    level_bins = loaded.action_level_bins
    per_sym_actions = max(1, alloc_bins) * max(1, level_bins)

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
                print("  torch.compile: OK (reduce-overhead)")
        except Exception as e:
            if verbose:
                print(f"  torch.compile: failed ({e}), using eager mode")
            compiled_policy = policy
    else:
        compiled_policy = policy

    # Short masking setup
    side_block = num_symbols * per_sym_actions

    # Load shared market data
    from pufferlib_market import binding  # noqa: PLC0415

    binding.shared(data_path=str(Path(data_path).resolve()))

    # Generate window start indices (reproducible with seed)
    rng = np.random.default_rng(seed)
    window_len = eval_hours + 1
    max_offset = num_timesteps - window_len
    if max_offset < 0:
        raise ValueError(f"Data too short: {num_timesteps} timesteps, need {window_len}")
    starts = rng.choice(max_offset + 1, size=n_windows, replace=(max_offset + 1 < n_windows))

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

    try:
        # Set per-env forced offsets using vec_set_offsets
        binding.vec_set_offsets(vec_handle, starts.astype(np.int32))

        # Reset with forced offsets in place
        binding.vec_reset(vec_handle, seed)

        # Track per-window results
        completed = [None] * n_windows  # type: list[Optional[WindowResult]]
        active = np.ones(n_windows, dtype=bool)
        n_completed = 0
        early_exited = False

        # Static CUDA obs buffer; _obs_cpu is a zero-copy view of the numpy obs_bufs array
        _obs_cuda = torch.zeros(n_windows, obs_size, dtype=torch.float32, device=device)
        _obs_cpu = torch.from_numpy(obs_bufs)

        # Run episodes - all windows step in lockstep
        for _step in range(eval_hours + 10):  # +10 safety margin
            if not active.any():
                break

            # Batched GPU inference - THE key speedup
            _obs_cuda.copy_(_obs_cpu, non_blocking=True)
            with torch.inference_mode():
                logits, _ = compiled_policy(_obs_cuda)

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
    finally:
        binding.vec_close(vec_handle)

    elapsed = time.monotonic() - t0

    # Collect results
    valid_results = [c for c in completed if c is not None]
    if not valid_results:
        return {"error": "no windows completed", "elapsed_s": elapsed}

    summary = _summarize_window_results(valid_results)

    out = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "arch": loaded.arch,
        "hidden_size": loaded.hidden_size,
        "eval_hours": eval_hours,
        "n_windows": n_windows,
        "n_completed": len(valid_results),
        "seed": seed,
        "fee_rate": fee_rate,
        "fill_slippage_bps": fill_slippage_bps,
        "max_leverage": max_leverage,
        "action_allocation_bins": loaded.action_allocation_bins,
        "action_level_bins": loaded.action_level_bins,
        "action_max_offset_bps": loaded.action_max_offset_bps,
        "periods_per_year": periods_per_year,
        "elapsed_s": elapsed,
        "early_exit": early_exited,
        "summary": summary,
        "windows": [asdict(r) for r in valid_results],
    }
    return out


# ---------------------------------------------------------------------------
# Multi-period evaluation
# ---------------------------------------------------------------------------

# Weights for smoothness_score: shorter windows penalise single-spike wins more.
_SMOOTHNESS_WEIGHTS: dict[int, int] = {5: 3, 15: 2, 30: 2, 60: 1, 90: 1}


def multi_period_eval(
    checkpoint_path: str,
    data_path: str,
    *,
    window_sizes: tuple[int, ...] = (5, 15, 30, 60, 90),
    n_windows_per_size: int = 8,
    fee_rate: float = 0.001,
    fill_slippage_bps: float = 8.0,
    periods_per_year: float = 252.0,
    max_leverage: float = 1.0,
    short_borrow_apr: float = 0.0,
    deterministic: bool = True,
    arch: str = "auto",
    hidden_size: int | None = None,
    device_str: str = "auto",
    seed: int = 1337,
) -> dict:
    """Evaluate across multiple horizons. Returns per-size metrics + composite smoothness_score.

    smoothness_score: weighted average of p10_sortino across window sizes.
    Shorter windows get more weight (penalises single-spike wins).
    Weights: 5d=3, 15d=2, 30d=2, 60d=1, 90d=1 (total weight=9).
    """

    per_size: dict[int, dict] = {}
    for ws in window_sizes:
        result = fast_holdout_eval(
            checkpoint_path,
            data_path,
            n_windows=n_windows_per_size,
            eval_hours=ws,
            seed=seed,
            fee_rate=fee_rate,
            fill_slippage_bps=fill_slippage_bps,
            max_leverage=max_leverage,
            periods_per_year=periods_per_year,
            short_borrow_apr=short_borrow_apr,
            deterministic=deterministic,
            arch=arch,
            hidden_size=hidden_size,
            device_str=device_str,
            # Disable early exit for multi-period: each window is small, we need all results
            early_exit_after=0,
            use_compile=True,
        )
        summary = result.get("summary", {})
        per_size[ws] = {
            "p10_sortino": float(summary.get("p10_sortino", 0.0)),
            "median_total_return": float(summary.get("median_total_return", 0.0)),
            "p10_total_return": float(summary.get("p10_total_return", 0.0)),
            "median_sortino": float(summary.get("median_sortino", 0.0)),
        }

    total_weight = sum(_SMOOTHNESS_WEIGHTS.get(d, 1) for d in window_sizes)
    smoothness_score = sum(
        _SMOOTHNESS_WEIGHTS.get(d, 1) * max(min(per_size[d]["p10_sortino"], 5.0), -5.0)
        for d in window_sizes
    ) / total_weight

    return {
        "smoothness_score": float(smoothness_score),
        "per_size": per_size,
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "window_sizes": list(window_sizes),
        "n_windows_per_size": n_windows_per_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli_summary_payload(result: dict[str, object]) -> dict[str, object]:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return result

    summary_context = {
        key: result[key]
        for key in (
            "checkpoint",
            "data_path",
            "arch",
            "hidden_size",
            "action_allocation_bins",
            "action_level_bins",
            "action_max_offset_bps",
            "eval_hours",
            "n_windows",
            "n_completed",
            "early_exit",
        )
        if key in result
    }
    return {**summary_context, **summary}


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
    parser.add_argument("--multi-windows", default=None,
        help="Comma-separated window sizes in trading days for multi-period eval, e.g. '5,15,30,60,90'")
    parser.add_argument("--n-windows-per-size", type=int, default=8,
        help="Number of random windows per size in multi-period eval (default: 8)")
    args = parser.parse_args()

    if args.multi_windows is not None:
        window_sizes = tuple(int(x.strip()) for x in args.multi_windows.split(",") if x.strip())
        result = multi_period_eval(
            args.checkpoint,
            args.data_path,
            window_sizes=window_sizes,
            n_windows_per_size=args.n_windows_per_size,
            fee_rate=args.fee_rate,
            fill_slippage_bps=args.fill_slippage_bps,
            periods_per_year=args.periods_per_year,
            max_leverage=args.max_leverage,
            short_borrow_apr=args.short_borrow_apr,
            deterministic=args.deterministic,
            arch=args.arch,
            hidden_size=args.hidden_size,
            device_str=args.device,
            seed=args.seed,
        )
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

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

    print(json.dumps(_build_cli_summary_payload(result), indent=2))
    if "elapsed_s" in result:
        print(f"\nElapsed: {result['elapsed_s']:.2f}s ({result.get('n_completed', 0)}/{result.get('n_windows', 0)} windows)")


if __name__ == "__main__":
    main()
