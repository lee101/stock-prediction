#!/usr/bin/env python3
"""
Test-time compute scaling for RL trading: K-rollout search.

At each decision point, instead of one greedy action, simulate K parallel
futures per candidate action and pick the one with the highest mean discounted
cumulative return.  The C env is fast enough (~<5 ms for 640 steps) that this
is practical even at inference time.

Usage (single decision):
    python -m pufferlib_market.inference_tts \
        --checkpoint PATH --data-path PATH \
        --tts-k 32 --horizon 20 --timestep 0

Benchmark mode (measures ms/decision for various K):
    python -m pufferlib_market.inference_tts \
        --checkpoint PATH --data-path PATH --benchmark
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Policy loading helpers (shared with evaluate_fast.py conventions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedPolicy:
    policy: torch.nn.Module
    arch: str
    hidden_size: int
    action_allocation_bins: int
    action_level_bins: int
    action_max_offset_bps: float
    n_actions: int


def _load_policy(checkpoint_path: str, obs_size: int, num_symbols: int, device: torch.device) -> LoadedPolicy:
    """Load a policy from a checkpoint, auto-detecting architecture and action grid."""
    from pufferlib_market.checkpoint_loader import (
        extract_checkpoint_state_dict,
        infer_arch_from_state_dict,
        infer_hidden_size_from_state_dict,
        infer_resmlp_blocks_from_state_dict,
        load_checkpoint_payload,
        resolve_checkpoint_action_grid_config,
    )
    from pufferlib_market.evaluate_fast import ResidualTradingPolicy, TradingPolicy

    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    state_dict = extract_checkpoint_state_dict(payload)

    arch = infer_arch_from_state_dict(state_dict)
    hidden = infer_hidden_size_from_state_dict(state_dict, arch)
    alloc_bins, level_bins, max_offset_bps = resolve_checkpoint_action_grid_config(
        payload,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
    )
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)
    n_actions = 1 + 2 * int(num_symbols) * per_symbol_actions

    if arch == "resmlp":
        policy = ResidualTradingPolicy(
            obs_size,
            n_actions,
            hidden=hidden,
            num_blocks=infer_resmlp_blocks_from_state_dict(state_dict),
        )
    else:
        policy = TradingPolicy(obs_size, n_actions, hidden=hidden)

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    ignored = {"obs_mean", "obs_std", "encoder_norm.weight", "encoder_norm.bias"}
    bad_missing = [key for key in missing if key not in ignored]
    bad_unexpected = [key for key in unexpected if key not in ignored]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint architecture mismatch — missing: {bad_missing}, unexpected: {bad_unexpected}")

    policy.to(device)
    policy.eval()
    return LoadedPolicy(
        policy=policy,
        arch=str(arch),
        hidden_size=int(hidden),
        action_allocation_bins=int(alloc_bins),
        action_level_bins=int(level_bins),
        action_max_offset_bps=float(max_offset_bps),
        n_actions=int(n_actions),
    )


def _read_data_metadata(data_path: str) -> tuple[list[str], int, int]:
    """Read MKTD header + symbol table. Returns (symbols, num_timesteps, features_per_sym)."""
    with open(data_path, "rb") as f:
        header = f.read(64)
        if len(header) != 64:
            raise ValueError(f"Short MKTD header: {data_path}")
        magic, _, num_symbols, num_timesteps, features_per_sym, _ = struct.unpack("<4sIIIII", header[:24])
        if magic != b"MKTD":
            raise ValueError(f"Bad MKTD magic in {data_path}: {magic!r}")
        if num_symbols <= 0 or num_timesteps <= 0:
            raise ValueError(f"Invalid MKTD dimensions in {data_path}")
        if features_per_sym == 0:
            features_per_sym = 16  # v1/v2 backwards compat
        raw_symbols = f.read(int(num_symbols) * 16)
        if len(raw_symbols) != int(num_symbols) * 16:
            raise ValueError(f"Short MKTD symbol table: {data_path}")

    symbols = []
    for idx in range(int(num_symbols)):
        raw = raw_symbols[idx * 16 : (idx + 1) * 16]
        symbol = raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip() or f"SYM{idx}"
        symbols.append(symbol)
    return symbols, int(num_timesteps), int(features_per_sym)


def _read_data_header(data_path: str) -> tuple[int, int, int]:
    symbols, num_timesteps, features_per_sym = _read_data_metadata(data_path)
    return len(symbols), num_timesteps, features_per_sym


def _action_level_offset_bps(level_idx: int, level_bins: int, max_offset_bps: float) -> float:
    if level_bins <= 1 or max_offset_bps <= 0.0:
        return 0.0
    frac = float(level_idx) / float(level_bins - 1)
    return (2.0 * frac - 1.0) * float(max_offset_bps)


def _format_action_label(
    action: int,
    *,
    num_symbols: int,
    action_allocation_bins: int,
    action_level_bins: int,
    action_max_offset_bps: float,
    symbols: list[str] | None = None,
) -> str:
    if action <= 0:
        return "flat"

    alloc_bins = max(1, int(action_allocation_bins))
    level_bins = max(1, int(action_level_bins))
    per_symbol_actions = alloc_bins * level_bins
    action_idx = action - 1
    side_block = num_symbols * per_symbol_actions
    is_short = action_idx >= side_block
    if is_short:
        action_idx -= side_block

    sym_idx = action_idx // per_symbol_actions
    bucket = action_idx % per_symbol_actions
    alloc_idx = bucket // level_bins
    level_idx = bucket % level_bins
    alloc_pct = (alloc_idx + 1) / alloc_bins
    symbol_label = symbols[sym_idx] if symbols is not None and 0 <= sym_idx < len(symbols) else f"sym{sym_idx}"
    label = f"{'short' if is_short else 'long'}_{symbol_label} alloc={alloc_pct:.0%}"
    if level_bins > 1 and action_max_offset_bps > 0.0:
        level_bps = _action_level_offset_bps(level_idx, level_bins, action_max_offset_bps)
        label += f" level={level_bps:+g}bps"
    return label


# ---------------------------------------------------------------------------
# Core TTS function
# ---------------------------------------------------------------------------


def best_action_tts(
    policy,
    current_obs: np.ndarray,
    data_path: str,
    current_timestep: int,
    n_actions: int,
    K: int = 32,
    horizon: int = 20,
    gamma: float = 0.99,
    fee_rate: float = 0.001,
    fill_slippage_bps: float = 5.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 252.0,
    device: str = "cuda",
    deterministic_after_first: bool = False,
) -> tuple[int, float, dict]:
    """
    Test-time search: pick the action with the highest mean discounted return
    across K parallel rollouts per candidate action.

    Algorithm:
      1. Create n_actions * K environments, all starting at current_timestep.
      2. For env batch [a*K : (a+1)*K], force action=a on step 0.
      3. Run `horizon` more steps with stochastic (or greedy) policy.
      4. Accumulate discounted rewards per rollout.
      5. Return action with highest mean return.

    Returns:
        (best_action, expected_return, stats)
        stats: dict with per-action mean returns and margin (best - 2nd best).
    """
    # Fast path: K=1 just runs greedy from the policy directly.
    if K <= 1:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        obs_t = torch.from_numpy(current_obs).unsqueeze(0).to(dev)
        with torch.inference_mode():
            logits, _ = policy(obs_t)
        best = int(logits.argmax(dim=-1).item())
        return best, float(logits[0, best].item()), {"margin": 0.0, "action_returns": []}

    import pufferlib_market.binding as binding

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    obs_size = int(current_obs.shape[0])
    total_envs = n_actions * K

    # Allocate shared C-env buffers
    obs_bufs = np.zeros((total_envs, obs_size), dtype=np.float32)
    act_bufs = np.zeros(total_envs, dtype=np.int32)
    rew_bufs = np.zeros(total_envs, dtype=np.float32)
    term_bufs = np.zeros(total_envs, dtype=np.uint8)
    trunc_bufs = np.zeros(total_envs, dtype=np.uint8)

    # Make sure shared market data is loaded
    binding.shared(data_path=str(Path(data_path).resolve()))

    # All envs start at current_timestep; max_steps=horizon so they terminate naturally
    vec_handle = binding.vec_init(
        obs_bufs,
        act_bufs,
        rew_bufs,
        term_bufs,
        trunc_bufs,
        total_envs,
        0,  # seed=0; stochastic sampling handles diversity
        max_steps=horizon,
        fee_rate=fee_rate,
        max_leverage=max_leverage,
        short_borrow_apr=0.0,
        periods_per_year=periods_per_year,
        fill_slippage_bps=fill_slippage_bps,
        forced_offset=current_timestep,
        action_allocation_bins=1,
        action_level_bins=1,
    )

    # Set all envs to exactly current_timestep
    offsets = np.full(total_envs, current_timestep, dtype=np.int32)
    binding.vec_set_offsets(vec_handle, offsets)
    binding.vec_reset(vec_handle, 0)

    # ── Step 0: force each action-block to its candidate action ──
    for a in range(n_actions):
        act_bufs[a * K : (a + 1) * K] = a
    binding.vec_step(vec_handle)

    # Track per-rollout cumulative discounted return (from step 1 onward)
    cumulative = np.zeros(total_envs, dtype=np.float64)
    done = np.zeros(total_envs, dtype=bool)

    # Accumulate step-0 rewards undiscounted (t=0): the forced action's immediate reward
    cumulative += rew_bufs
    done |= term_bufs.astype(bool) | trunc_bufs.astype(bool)

    obs_cpu = torch.from_numpy(obs_bufs)
    obs_device = torch.empty((total_envs, obs_size), dtype=torch.float32, device=dev)

    # ── Steps 1..horizon-1: stochastic/greedy policy ──
    for h in range(1, horizon):
        # Only active (non-terminated) envs need actions; terminated ones are ignored by C env
        if done.all():
            break

        obs_device.copy_(obs_cpu, non_blocking=True)
        with torch.inference_mode():
            logits, _ = policy(obs_device)

        if deterministic_after_first:
            actions = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy().astype(np.int32)

        # Don't override action for terminated envs (C env ignores them, but keep buffers clean)
        act_bufs[:] = actions
        binding.vec_step(vec_handle)

        discount = gamma**h
        cumulative += discount * rew_bufs * (~done)
        done |= term_bufs.astype(bool) | trunc_bufs.astype(bool)

    binding.vec_close(vec_handle)

    # ── Group by action and compute mean return ──
    action_returns = np.zeros(n_actions, dtype=np.float64)
    for a in range(n_actions):
        action_returns[a] = cumulative[a * K : (a + 1) * K].mean()

    best_action = int(np.argmax(action_returns))
    best_return = float(action_returns[best_action])

    # Margin = gap between best and second-best action
    sorted_returns = np.sort(action_returns)[::-1]
    margin = float(sorted_returns[0] - sorted_returns[1]) if n_actions > 1 else 0.0
    second_best = float(sorted_returns[1]) if n_actions > 1 else best_return

    stats = {
        "action_returns": action_returns.tolist(),
        "margin": margin,
        "best_action": best_action,
        "best_return": best_return,
        "second_best_return": second_best,
    }
    return best_action, best_return, stats


# ---------------------------------------------------------------------------
# Integration with PPOTrader: TTS-aware get_signal
# ---------------------------------------------------------------------------


def get_signal_tts(
    trader,
    features: np.ndarray,
    prices: dict,
    data_path: str,
    current_timestep: int,
    tts_k: int = 32,
    horizon: int = 20,
    gamma: float = 0.99,
    fee_rate: float = 0.001,
    fill_slippage_bps: float = 5.0,
) -> tuple:
    """
    Drop-in replacement for PPOTrader.get_signal() that uses TTS search.

    Returns (TradingSignal, stats_dict).
    """
    obs = trader.build_observation(features, prices)

    if tts_k <= 1:
        signal = trader.get_signal(features, prices)
        return signal, {"margin": 0.0, "action_returns": []}

    best_action, _expected_return, stats = best_action_tts(
        policy=trader.policy,
        current_obs=obs,
        data_path=data_path,
        current_timestep=current_timestep,
        n_actions=trader.num_actions,
        K=tts_k,
        horizon=horizon,
        gamma=gamma,
        fee_rate=fee_rate,
        fill_slippage_bps=fill_slippage_bps,
        max_leverage=1.0,
        periods_per_year=trader.max_steps,  # daily: 252, hourly: 8760
        device=str(trader.device),
    )

    # Get confidence and value from a single forward pass, then decode using PPOTrader's method
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(trader.device)
    with torch.inference_mode():
        logits, value = trader.policy(obs_t)
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, best_action].item())
        value_est = float(value.item())

    signal = trader._decode_action(best_action, confidence, value_est)
    return signal, stats


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark_tts(
    checkpoint_path: str,
    data_path: str,
    k_values: list[int] | None = None,
    horizon: int = 20,
    n_reps: int = 5,
    device_str: str = "auto",
):
    """Measure ms/decision for each K value in k_values."""
    if k_values is None:
        k_values = [1, 8, 32, 64, 128]

    import pufferlib_market.binding as binding

    dev_str = device_str
    if dev_str == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)

    symbols, num_timesteps, features_per_sym = _read_data_metadata(data_path)
    num_symbols = len(symbols)
    obs_size = num_symbols * features_per_sym + 5 + num_symbols
    loaded = _load_policy(checkpoint_path, obs_size, num_symbols, device)
    policy = loaded.policy
    n_actions = loaded.n_actions

    # Shared data must be loaded before calling best_action_tts
    binding.shared(data_path=str(Path(data_path).resolve()))

    # Use a fixed observation and timestep for benchmarking
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(obs_size).astype(np.float32) * 0.1
    timestep = min(10, num_timesteps - horizon - 5)

    print(f"\nBenchmark: TTS on {Path(data_path).name}, {num_symbols} symbols, {n_actions} actions")
    print(f"  horizon={horizon}, device={dev_str}")
    print(f"  effective_config: arch={loaded.arch}, hidden_size={loaded.hidden_size}")
    print(
        "  action_grid: "
        f"alloc_bins={loaded.action_allocation_bins}, "
        f"level_bins={loaded.action_level_bins}, "
        f"max_offset_bps={loaded.action_max_offset_bps:g}"
    )
    print(f"  {'K':>6}  {'ms/decision':>12}  {'ms/rollout':>12}  {'best_action':>12}  {'margin':>8}  label")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}  {'-' * 24}")

    results = []
    for K in k_values:
        times = []
        best_action_val = None
        best_action_label = None
        margin_val = None
        for rep in range(n_reps):
            t0 = time.perf_counter()
            # Re-load shared data each call (binding.shared is idempotent if same path)
            best_a, best_r, stats = best_action_tts(
                policy=policy,
                current_obs=obs,
                data_path=data_path,
                current_timestep=timestep,
                n_actions=n_actions,
                K=K,
                horizon=horizon,
                device=dev_str,
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            best_action_val = best_a
            best_action_label = _format_action_label(
                best_a,
                num_symbols=num_symbols,
                action_allocation_bins=loaded.action_allocation_bins,
                action_level_bins=loaded.action_level_bins,
                action_max_offset_bps=loaded.action_max_offset_bps,
                symbols=symbols,
            )
            margin_val = stats.get("margin", 0.0)

        median_ms = float(np.median(times))
        total_rollouts = n_actions * K
        ms_per_rollout = median_ms / max(1, total_rollouts)
        print(
            f"  {K:>6}  {median_ms:>12.2f}  {ms_per_rollout:>12.4f}  {best_action_val:>12}  {margin_val:>8.4f}  {best_action_label}"
        )
        results.append({"K": K, "median_ms": median_ms, "ms_per_rollout": ms_per_rollout})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test-time compute scaling: K-rollout search for best trading action.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to MKTD binary")
    parser.add_argument("--tts-k", type=int, default=32, help="Rollouts per candidate action (default: 32)")
    parser.add_argument("--horizon", type=int, default=20, help="Steps per rollout (default: 20)")
    parser.add_argument("--timestep", type=int, default=0, help="Starting timestep in data (default: 0)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Fee rate (default: 0.001)")
    parser.add_argument("--fill-slippage-bps", type=float, default=5.0, help="Fill slippage bps (default: 5.0)")
    parser.add_argument("--max-leverage", type=float, default=1.0, help="Max leverage (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda (default: auto)")
    parser.add_argument(
        "--deterministic-after-first",
        action="store_true",
        help="Use greedy policy after the forced first action (default: stochastic)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be run without executing (for testing CLI)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark across K=1,8,32,64,128 and report ms/decision"
    )
    parser.add_argument(
        "--benchmark-k-values",
        type=str,
        default="1,8,32,64,128",
        help="Comma-separated K values for benchmark (default: 1,8,32,64,128)",
    )
    parser.add_argument(
        "--benchmark-reps", type=int, default=5, help="Number of repetitions per K for benchmark (default: 5)"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — would execute:")
        print(f"  checkpoint:    {args.checkpoint}")
        print(f"  data_path:     {args.data_path}")
        print(f"  tts_k:         {args.tts_k}")
        print(f"  horizon:       {args.horizon}")
        print(f"  timestep:      {args.timestep}")
        print(f"  gamma:         {args.gamma}")
        print(f"  fee_rate:      {args.fee_rate}")
        print(f"  slippage_bps:  {args.fill_slippage_bps}")
        print(f"  device:        {args.device}")
        print(f"  deterministic_after_first: {args.deterministic_after_first}")
        return

    ckpt = Path(args.checkpoint)
    data = Path(args.data_path)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)
    if not data.exists():
        print(f"ERROR: data file not found: {data}", file=sys.stderr)
        sys.exit(1)

    dev_str = args.device
    if dev_str == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    if args.benchmark:
        k_values = [int(x.strip()) for x in args.benchmark_k_values.split(",")]
        benchmark_tts(
            checkpoint_path=str(ckpt),
            data_path=str(data),
            k_values=k_values,
            horizon=args.horizon,
            n_reps=args.benchmark_reps,
            device_str=dev_str,
        )
        return

    # Single-decision mode
    import pufferlib_market.binding as binding

    symbols, num_timesteps, features_per_sym = _read_data_metadata(str(data))
    num_symbols = len(symbols)
    obs_size = num_symbols * features_per_sym + 5 + num_symbols
    device = torch.device(dev_str)
    loaded = _load_policy(str(ckpt), obs_size, num_symbols, device)
    policy = loaded.policy
    n_actions = loaded.n_actions

    binding.shared(data_path=str(data.resolve()))

    # Build a plausible observation (zeros — replace with real obs for actual use)
    obs = np.zeros(obs_size, dtype=np.float32)

    timestep = min(args.timestep, num_timesteps - args.horizon - 5)
    print(f"Running TTS: K={args.tts_k}, horizon={args.horizon}, timestep={timestep}")
    print(f"  checkpoint={ckpt.name}, data={data.name}")
    print(f"  num_symbols={num_symbols}, n_actions={n_actions}, obs_size={obs_size}")
    print(f"  effective_config: arch={loaded.arch}, hidden_size={loaded.hidden_size}")
    print(
        "  action_grid: "
        f"alloc_bins={loaded.action_allocation_bins}, "
        f"level_bins={loaded.action_level_bins}, "
        f"max_offset_bps={loaded.action_max_offset_bps:g}"
    )
    print(f"  device={dev_str}")

    t0 = time.perf_counter()
    best_action, expected_return, stats = best_action_tts(
        policy=policy,
        current_obs=obs,
        data_path=str(data),
        current_timestep=timestep,
        n_actions=n_actions,
        K=args.tts_k,
        horizon=args.horizon,
        gamma=args.gamma,
        fee_rate=args.fee_rate,
        fill_slippage_bps=args.fill_slippage_bps,
        max_leverage=args.max_leverage,
        periods_per_year=252.0,
        device=dev_str,
        deterministic_after_first=args.deterministic_after_first,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    best_action_label = _format_action_label(
        best_action,
        num_symbols=num_symbols,
        action_allocation_bins=loaded.action_allocation_bins,
        action_level_bins=loaded.action_level_bins,
        action_max_offset_bps=loaded.action_max_offset_bps,
        symbols=symbols,
    )

    print(f"\nResult:")
    print(f"  best_action:     {best_action}")
    print(f"  best_action_label: {best_action_label}")
    print(f"  expected_return: {expected_return:.6f}")
    print(f"  margin (vs 2nd): {stats['margin']:.6f}")
    print(f"  elapsed:         {elapsed_ms:.2f} ms")
    print(f"\nPer-action mean returns:")
    action_labels = [
        _format_action_label(
            action,
            num_symbols=num_symbols,
            action_allocation_bins=loaded.action_allocation_bins,
            action_level_bins=loaded.action_level_bins,
            action_max_offset_bps=loaded.action_max_offset_bps,
            symbols=symbols,
        )
        for action in range(n_actions)
    ]
    for a, (label, ret) in enumerate(zip(action_labels, stats["action_returns"])):
        marker = " <-- BEST" if a == best_action else ""
        print(f"  [{a:3d}] {label:<32} {ret:+.6f}{marker}")


if __name__ == "__main__":
    main()
