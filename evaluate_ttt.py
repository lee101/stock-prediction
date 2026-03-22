#!/usr/bin/env python3
"""Test-time training (TTT) evaluation framework.

For each holdout window, the policy is adapted on-the-fly:
  1. Run the base policy for ttt_warmup_steps steps, collecting transitions.
  2. Do a mini PPO update (policy gradient loss) on collected transitions,
     updating only the last layers (actor head + last encoder layer).
  3. Run the adapted policy for the remaining steps.
  4. Restore original weights before the next window (per-window TTT, not cumulative).

Also implements meta checkpoint selection: given multiple checkpoints, use
the first selection_steps of each window to pick the best one, then continue
with the winner.

Usage:
  python evaluate_ttt.py \\
    --checkpoint path/to/best.pt \\
    --data-path path/to/val.bin \\
    --n-windows 20 --eval-steps 90 \\
    --ttt-warmup-steps 20 --ttt-update-steps 50 --ttt-lr 1e-5 --ttt-epochs 2 \\
    --compare-baseline

  # Meta checkpoint selection:
  python evaluate_ttt.py \\
    --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \\
    --data-path path/to/val.bin \\
    --mode meta-select --selection-steps 10

CLI output: JSON summary + optional comparison table when --compare-baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Policy definitions — kept in sync with pufferlib_market/evaluate_fast.py
# ---------------------------------------------------------------------------

from pufferlib_market.train import (
    TradingPolicy,
    ResidualTradingPolicy,
)
from pufferlib_market.evaluate_fast import (
    _infer_arch,
    _infer_hidden_size,
    _infer_resmlp_blocks,
)


def _load_policy(checkpoint_path: str, obs_size: int, num_actions: int,
                 device: torch.device) -> tuple[nn.Module, dict]:
    """Load a policy from a checkpoint file. Returns (policy, payload).

    num_actions is derived from the state_dict actor head shape so it matches
    exactly what was used at training time, regardless of the data file's symbol count.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload

    arch = _infer_arch(state_dict)
    hidden = _infer_hidden_size(state_dict, arch)

    # Infer num_actions from actor output layer in checkpoint (ignore caller's value)
    for key in ("actor.2.bias", "actor.2.weight"):
        if key in state_dict:
            num_actions = int(state_dict[key].shape[0])
            break

    # Infer obs_size from first weight layer in checkpoint
    if arch == "resmlp" and "input_proj.weight" in state_dict:
        obs_size = int(state_dict["input_proj.weight"].shape[1])
    elif "encoder.0.weight" in state_dict:
        obs_size = int(state_dict["encoder.0.weight"].shape[1])

    if arch == "resmlp":
        num_blocks = _infer_resmlp_blocks(state_dict)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden,
                                       num_blocks=num_blocks).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden).to(device)

    policy.load_state_dict(state_dict)
    return policy, payload


def _read_data_header(data_path: str) -> tuple[int, int]:
    """Return (num_symbols, num_timesteps) from binary file header."""
    with open(data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])
    return int(num_symbols), int(num_timesteps)


def _compute_obs_size(num_symbols: int) -> int:
    return num_symbols * 16 + 5 + num_symbols


# ---------------------------------------------------------------------------
# TTT: identify which parameters to fine-tune (last layers only)
# ---------------------------------------------------------------------------

def _get_ttt_params(policy: nn.Module) -> list[nn.Parameter]:
    """Return parameters to update during TTT: actor head + last encoder layer.

    Freezes the feature extractor backbone to prevent catastrophic forgetting.
    Only the actor head and the final encoder layer are updated.
    """
    ttt_params = []

    # For TradingPolicy: update actor and last encoder layer (encoder[4] and encoder[5])
    if hasattr(policy, "actor") and hasattr(policy, "encoder"):
        # actor head — always update
        for p in policy.actor.parameters():
            ttt_params.append(p)
        # last linear in encoder (index 4 of the Sequential = Linear(h,h))
        encoder = policy.encoder
        if isinstance(encoder, nn.Sequential):
            layers = list(encoder)
            # Find the last Linear layer in the encoder
            last_linear_idx = max(
                (i for i, l in enumerate(layers) if isinstance(l, nn.Linear)),
                default=None,
            )
            if last_linear_idx is not None:
                for p in layers[last_linear_idx].parameters():
                    ttt_params.append(p)

    # For ResidualTradingPolicy: update actor + last residual block + out_norm
    elif hasattr(policy, "actor") and hasattr(policy, "blocks"):
        for p in policy.actor.parameters():
            ttt_params.append(p)
        blocks = list(policy.blocks)
        if blocks:
            for p in blocks[-1].parameters():
                ttt_params.append(p)
        if hasattr(policy, "out_norm"):
            for p in policy.out_norm.parameters():
                ttt_params.append(p)

    # Fallback: just the actor head
    else:
        if hasattr(policy, "actor"):
            for p in policy.actor.parameters():
                ttt_params.append(p)
        else:
            ttt_params = list(policy.parameters())

    return ttt_params


# ---------------------------------------------------------------------------
# Shared small helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _sample_window_starts(num_timesteps: int, eval_steps: int,
                          n_windows: int, seed: int) -> np.ndarray:
    window_len = eval_steps + 1
    max_offset = num_timesteps - window_len
    if max_offset < 0:
        raise ValueError(f"Data too short: {num_timesteps} timesteps, need {window_len}")
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.arange(max_offset + 1),
        size=n_windows,
        replace=(max_offset + 1 < n_windows),
    )


def _summarize_windows(results: list) -> dict:
    if not results:
        return {}
    returns = [r.total_return for r in results]
    sortinos = [r.sortino for r in results]
    drawdowns = [r.max_drawdown for r in results]
    return {
        "median_total_return": float(np.percentile(returns, 50)),
        "p10_total_return": float(np.percentile(returns, 10)),
        "p90_total_return": float(np.percentile(returns, 90)),
        "mean_total_return": float(np.mean(returns)),
        "pct_profitable": float(np.mean([r > 0 for r in returns])),
        "median_sortino": float(np.percentile(sortinos, 50)),
        "median_max_drawdown": float(np.percentile(drawdowns, 50)),
        "n_completed": len(results),
    }


def _make_env_buffers(obs_size: int) -> tuple[np.ndarray, ...]:
    obs_buf = np.zeros((1, obs_size), dtype=np.float32)
    act_buf = np.zeros((1,), dtype=np.int32)
    rew_buf = np.zeros((1,), dtype=np.float32)
    term_buf = np.zeros((1,), dtype=np.uint8)
    trunc_buf = np.zeros((1,), dtype=np.uint8)
    return obs_buf, act_buf, rew_buf, term_buf, trunc_buf


# ---------------------------------------------------------------------------
# Core per-window evaluation helpers
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    start_idx: int
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    warmup_steps_used: int
    ttt_applied: bool


@dataclass
class MetaWindowResult:
    start_idx: int
    selected_checkpoint: str
    selected_idx: int
    selection_returns: list
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int


def _run_single_window(
    *,
    policy: nn.Module,
    binding,
    vec_handle,
    obs_buf: np.ndarray,
    act_buf: np.ndarray,
    rew_buf: np.ndarray,
    term_buf: np.ndarray,
    trunc_buf: np.ndarray,
    eval_steps: int,
    deterministic: bool,
    device: torch.device,
    start_idx: int,
    ttt_warmup_steps: int = 0,
    ttt_update_steps: int = 0,
    ttt_lr: float = 1e-5,
    ttt_epochs: int = 2,
    kl_coef: float = 0.1,
    apply_ttt: bool = False,
) -> Optional[WindowResult]:
    """Run one window with optional TTT adaptation.

    Sets the env to start_idx via forced offset, then steps through eval_steps.
    If apply_ttt=True, collects warmup_steps transitions, updates the last
    layers with a policy-gradient loss, then continues evaluation.

    Returns None if the episode didn't terminate naturally.
    """
    # Set forced offset and reset
    binding.vec_set_offsets(vec_handle, np.array([start_idx], dtype=np.int32))
    binding.vec_reset(vec_handle, start_idx)

    # Snapshot env log BEFORE the episode so we can compute per-window deltas.
    # env->log is cumulative (never reset by c_reset), so without this, window 2+
    # would return cumulative totals instead of per-window values.
    env_handle = binding.vec_env_at(vec_handle, 0)
    log_before = dict(binding.env_get(env_handle))

    # Save original weights before TTT so we can restore them after
    original_state = None
    if apply_ttt:
        original_state = copy.deepcopy(policy.state_dict())

    ttt_params = _get_ttt_params(policy) if apply_ttt else []

    # Warmup phase: collect transitions for TTT
    warmup_obs: list[torch.Tensor] = []
    warmup_actions: list[torch.Tensor] = []
    warmup_rewards: list[float] = []
    warmup_log_probs: list[torch.Tensor] = []
    warmup_values: list[torch.Tensor] = []

    episode_done = False
    result_data = None
    steps_taken = 0

    policy.eval()

    for step in range(eval_steps + 10):  # +10 safety margin
        if episode_done:
            break
        if steps_taken >= eval_steps:
            break

        obs_tensor = torch.from_numpy(obs_buf[:1].copy()).to(device)

        # During warmup: collect (obs, action, logprob, value)
        in_warmup = apply_ttt and step < ttt_warmup_steps
        if in_warmup:
            with torch.no_grad():
                logits, value = policy(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample() if not deterministic else logits.argmax(dim=-1)
            log_prob = dist.log_prob(action)
            warmup_obs.append(obs_tensor.detach())
            warmup_actions.append(action.detach())
            warmup_rewards.append(0.0)  # filled below after stepping
            warmup_log_probs.append(log_prob.detach())
            warmup_values.append(value.detach())
        else:
            with torch.no_grad():
                logits, _ = policy(obs_tensor)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = Categorical(logits=logits).sample()

        act_buf[0] = int(action.item())
        binding.vec_step(vec_handle)
        steps_taken += 1

        # Capture reward into last warmup slot
        if in_warmup and warmup_rewards:
            warmup_rewards[-1] = float(rew_buf[0])

        # TTT update: right after warmup ends
        if apply_ttt and step == ttt_warmup_steps - 1 and len(warmup_obs) > 0:
            _do_ttt_update(
                policy=policy,
                ttt_params=ttt_params,
                obs_list=warmup_obs,
                actions_list=warmup_actions,
                rewards_list=warmup_rewards,
                old_log_probs_list=warmup_log_probs,
                old_values_list=warmup_values,
                ttt_update_steps=ttt_update_steps,
                ttt_epochs=ttt_epochs,
                ttt_lr=ttt_lr,
                kl_coef=kl_coef,
                device=device,
            )
            policy.eval()

        if term_buf[0]:
            env_data_after = dict(binding.env_get(env_handle))
            # Compute per-window deltas to avoid cumulative log contamination
            _DELTA_KEYS = ("total_return", "sortino", "max_drawdown", "num_trades", "win_rate")
            result_data = {k: env_data_after[k] - log_before[k] for k in _DELTA_KEYS
                          if k in env_data_after and k in log_before}
            # Fall back to raw values for any keys not in the snapshot
            for k in env_data_after:
                if k not in result_data:
                    result_data[k] = env_data_after[k]
            episode_done = True

    # Restore original weights (TTT is per-window only)
    if apply_ttt and original_state is not None:
        policy.load_state_dict(original_state)

    if result_data is None:
        return None

    return WindowResult(
        start_idx=start_idx,
        total_return=float(result_data.get("total_return", 0.0)),
        sortino=float(result_data.get("sortino", 0.0)),
        max_drawdown=float(result_data.get("max_drawdown", 0.0)),
        num_trades=int(result_data.get("num_trades", 0)),
        win_rate=float(result_data.get("win_rate", 0.0)),
        warmup_steps_used=min(ttt_warmup_steps, steps_taken) if apply_ttt else 0,
        ttt_applied=apply_ttt,
    )


def _do_ttt_update(
    *,
    policy: nn.Module,
    ttt_params: list[nn.Parameter],
    obs_list: list[torch.Tensor],
    actions_list: list[torch.Tensor],
    rewards_list: list[float],
    old_log_probs_list: list[torch.Tensor],
    old_values_list: list[torch.Tensor],
    ttt_update_steps: int,
    ttt_epochs: int,
    ttt_lr: float,
    kl_coef: float,
    device: torch.device,
) -> None:
    """Mini PPO update on collected warmup transitions.

    Uses policy gradient loss with KL penalty to prevent large updates.
    Only updates the parameters in ttt_params (last layers).
    """
    if not obs_list or not ttt_params:
        return

    obs_t = torch.cat(obs_list, dim=0).to(device)           # (T, obs_size)
    acts_t = torch.cat(actions_list, dim=0).to(device)       # (T,)
    old_lp_t = torch.cat(old_log_probs_list, dim=0).to(device)  # (T,)
    old_val_t = torch.cat(old_values_list, dim=0).to(device)    # (T,)

    # Compute simple Monte-Carlo returns (no GAE for simplicity with small batches)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    T = len(rewards)
    gamma = 0.99
    returns = torch.zeros(T, device=device)
    running = 0.0
    for t in reversed(range(T)):
        running = float(rewards[t]) + gamma * running
        returns[t] = running

    # Normalize advantages
    advantages = returns - old_val_t.detach()
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.clamp(-5.0, 5.0)

    optimizer = torch.optim.Adam(ttt_params, lr=ttt_lr)

    # Determine minibatch size: use full batch if small, else chunk
    mb_size = min(ttt_update_steps, T) if ttt_update_steps > 0 else T

    policy.train()
    for _ in range(ttt_epochs):
        # Random shuffle for minibatches
        perm = torch.randperm(T, device=device)
        for start in range(0, T, mb_size):
            idx = perm[start: start + mb_size]
            if len(idx) == 0:
                continue

            mb_obs = obs_t[idx]
            mb_acts = acts_t[idx]
            mb_adv = advantages[idx]
            mb_old_lp = old_lp_t[idx]

            logits, _ = policy(mb_obs)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(mb_acts)

            # Policy gradient loss
            pg_loss = -(new_lp * mb_adv).mean()

            # KL penalty: KL(new || old) ~= (new_lp - old_lp).exp() - (new_lp - old_lp) - 1
            log_ratio = new_lp - mb_old_lp.detach()
            kl = (log_ratio.exp() - log_ratio - 1.0).mean()
            kl_loss = kl_coef * kl

            loss = pg_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ttt_params, max_norm=1.0)
            optimizer.step()


# ---------------------------------------------------------------------------
# Main TTT holdout evaluation
# ---------------------------------------------------------------------------

def ttt_holdout_eval(
    checkpoint_path: str,
    data_path: str,
    *,
    n_windows: int = 20,
    eval_steps: int = 90,
    ttt_warmup_steps: int = 20,
    ttt_update_steps: int = 50,
    ttt_lr: float = 1e-5,
    ttt_epochs: int = 2,
    kl_coef: float = 0.1,
    fee_rate: float = 0.001,
    fill_slippage_bps: float = 5.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    seed: int = 42,
    deterministic: bool = True,
    device_str: str = "auto",
    compare_baseline: bool = True,
    verbose: bool = False,
) -> dict:
    """Run TTT holdout evaluation.

    For each window:
      1. Warm up for ttt_warmup_steps (collect transitions).
      2. Do mini PPO update of last layers only with lr=ttt_lr.
      3. Continue evaluating for remaining (eval_steps - ttt_warmup_steps) steps.
      4. Restore original weights before next window.

    Returns dict with per-window results and aggregate summary.
    """
    t0 = time.monotonic()

    device = _resolve_device(device_str)

    num_symbols, num_timesteps = _read_data_header(data_path)
    obs_size_data = _compute_obs_size(num_symbols)

    policy, payload = _load_policy(checkpoint_path, obs_size_data,
                                   1 + 2 * num_symbols, device)
    obs_size = policy.obs_size if hasattr(policy, "obs_size") else obs_size_data

    alloc_bins = int(payload.get("action_allocation_bins", 1)) if isinstance(payload, dict) else 1
    level_bins = int(payload.get("action_level_bins", 1)) if isinstance(payload, dict) else 1

    import pufferlib_market.binding as binding
    binding.shared(data_path=str(Path(data_path).resolve()))

    starts = _sample_window_starts(num_timesteps, eval_steps, n_windows, seed)
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf = _make_env_buffers(obs_size)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        1, seed,
        max_steps=eval_steps,
        fee_rate=fee_rate,
        max_leverage=max_leverage,
        short_borrow_apr=short_borrow_apr,
        periods_per_year=periods_per_year,
        fill_slippage_bps=fill_slippage_bps,
        forced_offset=-1,
        action_allocation_bins=alloc_bins,
        action_level_bins=level_bins,
        enable_drawdown_profit_early_exit=False,
        drawdown_profit_early_exit_min_steps=20,
        drawdown_profit_early_exit_progress_fraction=0.5,
    )

    ttt_results: list[WindowResult] = []
    baseline_results: list[WindowResult] = []

    for i, start_idx in enumerate(starts):
        if verbose:
            print(f"  Window {i+1}/{n_windows} start={start_idx} ...", end="", flush=True)

        # TTT pass
        r_ttt = _run_single_window(
            policy=policy,
            binding=binding,
            vec_handle=vec_handle,
            obs_buf=obs_buf,
            act_buf=act_buf,
            rew_buf=rew_buf,
            term_buf=term_buf,
            trunc_buf=trunc_buf,
            eval_steps=eval_steps,
            deterministic=deterministic,
            device=device,
            start_idx=int(start_idx),
            ttt_warmup_steps=ttt_warmup_steps,
            ttt_update_steps=ttt_update_steps,
            ttt_lr=ttt_lr,
            ttt_epochs=ttt_epochs,
            kl_coef=kl_coef,
            apply_ttt=True,
        )
        if r_ttt is not None:
            ttt_results.append(r_ttt)

        # Baseline pass (same window, no TTT)
        if compare_baseline:
            r_base = _run_single_window(
                policy=policy,
                binding=binding,
                vec_handle=vec_handle,
                obs_buf=obs_buf,
                act_buf=act_buf,
                rew_buf=rew_buf,
                term_buf=term_buf,
                trunc_buf=trunc_buf,
                eval_steps=eval_steps,
                deterministic=deterministic,
                device=device,
                start_idx=int(start_idx),
                apply_ttt=False,
            )
            if r_base is not None:
                baseline_results.append(r_base)

        if verbose:
            ttt_ret = r_ttt.total_return if r_ttt else float("nan")
            base_ret = baseline_results[-1].total_return if (compare_baseline and baseline_results) else float("nan")
            print(f" ttt={ttt_ret:+.4f}  base={base_ret:+.4f}")

    binding.vec_close(vec_handle)
    elapsed = time.monotonic() - t0

    out: dict = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "eval_steps": eval_steps,
        "n_windows": n_windows,
        "ttt_warmup_steps": ttt_warmup_steps,
        "ttt_update_steps": ttt_update_steps,
        "ttt_lr": ttt_lr,
        "ttt_epochs": ttt_epochs,
        "kl_coef": kl_coef,
        "seed": seed,
        "fee_rate": fee_rate,
        "fill_slippage_bps": fill_slippage_bps,
        "elapsed_s": elapsed,
        "ttt_summary": _summarize_windows(ttt_results),
        "ttt_windows": [asdict(r) for r in ttt_results],
    }

    if compare_baseline:
        out["baseline_summary"] = _summarize_windows(baseline_results)
        out["baseline_windows"] = [asdict(r) for r in baseline_results]

    return out


# ---------------------------------------------------------------------------
# Meta checkpoint selection
# ---------------------------------------------------------------------------

def meta_select_holdout_eval(
    checkpoint_paths: list[str],
    data_path: str,
    *,
    n_windows: int = 20,
    eval_steps: int = 90,
    selection_steps: int = 10,
    fee_rate: float = 0.001,
    fill_slippage_bps: float = 5.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    seed: int = 42,
    deterministic: bool = True,
    device_str: str = "auto",
    verbose: bool = False,
) -> dict:
    """Meta checkpoint selection evaluation.

    For each window:
      1. Run all checkpoints for selection_steps steps.
      2. Pick the checkpoint with highest cumulative return so far.
      3. Continue with the selected checkpoint for remaining steps.

    Returns per-window results including which checkpoint was selected.
    """
    t0 = time.monotonic()

    device = _resolve_device(device_str)

    num_symbols, num_timesteps = _read_data_header(data_path)
    obs_size_data = _compute_obs_size(num_symbols)

    policies = []
    for ckpt_path in checkpoint_paths:
        p, payload = _load_policy(ckpt_path, obs_size_data, 1 + 2 * num_symbols, device)
        p.eval()
        policies.append((ckpt_path, p, payload))

    if not policies:
        raise ValueError("No valid checkpoints provided")

    _, _, first_payload = policies[0]
    alloc_bins = int(first_payload.get("action_allocation_bins", 1)) if isinstance(first_payload, dict) else 1
    level_bins = int(first_payload.get("action_level_bins", 1)) if isinstance(first_payload, dict) else 1

    # Validate that all checkpoints share the same action grid before creating the env.
    # Mixing checkpoints with different grids silently handicaps the ones that don't match.
    for ckpt_path, _, payload in policies[1:]:
        p_alloc = int(payload.get("action_allocation_bins", 1)) if isinstance(payload, dict) else 1
        p_level = int(payload.get("action_level_bins", 1)) if isinstance(payload, dict) else 1
        if p_alloc != alloc_bins or p_level != level_bins:
            raise ValueError(
                f"Checkpoint {ckpt_path} has incompatible action grid: "
                f"alloc_bins={p_alloc} (expected {alloc_bins}), "
                f"level_bins={p_level} (expected {level_bins})"
            )

    import pufferlib_market.binding as binding
    binding.shared(data_path=str(Path(data_path).resolve()))

    starts = _sample_window_starts(num_timesteps, eval_steps, n_windows, seed)
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf = _make_env_buffers(obs_size_data)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        1, seed,
        max_steps=eval_steps,
        fee_rate=fee_rate,
        max_leverage=max_leverage,
        short_borrow_apr=short_borrow_apr,
        periods_per_year=periods_per_year,
        fill_slippage_bps=fill_slippage_bps,
        forced_offset=-1,
        action_allocation_bins=alloc_bins,
        action_level_bins=level_bins,
        enable_drawdown_profit_early_exit=False,
        drawdown_profit_early_exit_min_steps=20,
        drawdown_profit_early_exit_progress_fraction=0.5,
    )

    meta_results: list[MetaWindowResult] = []

    for win_i, start_idx in enumerate(starts):
        if verbose:
            print(f"  Meta window {win_i+1}/{n_windows} start={start_idx} selecting from "
                  f"{len(policies)} checkpoints ...", end="", flush=True)

        # Selection phase: run each policy for selection_steps and measure actual trading return.
        # We use env_get() total_return delta rather than shaped reward to avoid ranking
        # on scaled/clipped/penalty-adjusted reward signals that don't reflect true PnL.
        selection_returns = []
        for _ckpt_path, sel_policy, _ in policies:
            sel_policy.eval()
            binding.vec_set_offsets(vec_handle, np.array([int(start_idx)], dtype=np.int32))
            binding.vec_reset(vec_handle, int(start_idx))

            # Snapshot log before selection episode
            sel_env_handle = binding.vec_env_at(vec_handle, 0)
            sel_log_before = dict(binding.env_get(sel_env_handle))

            sel_terminated = False
            for _step in range(selection_steps):
                obs_tensor = torch.from_numpy(obs_buf[:1].copy()).to(device)
                with torch.no_grad():
                    logits, _ = sel_policy(obs_tensor)
                action = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
                act_buf[0] = int(action.item())
                binding.vec_step(vec_handle)
                if term_buf[0]:
                    sel_terminated = True
                    break

            # Use actual total_return delta as selection criterion
            sel_log_after = dict(binding.env_get(sel_env_handle))
            if sel_terminated and "total_return" in sel_log_after and "total_return" in sel_log_before:
                sel_return = sel_log_after["total_return"] - sel_log_before["total_return"]
            else:
                sel_return = sel_log_after.get("total_return", 0.0) - sel_log_before.get("total_return", 0.0)
            selection_returns.append(sel_return)

        # Pick best checkpoint
        best_idx = int(np.argmax(selection_returns))
        best_path, best_policy, _ = policies[best_idx]

        if verbose:
            print(f" -> selected ckpt {best_idx} (cum_rew={selection_returns[best_idx]:.4f})")

        # Run the selected policy for the full remaining window starting from scratch
        # (we rerun from start so the evaluation is consistent)
        best_policy.eval()
        binding.vec_set_offsets(vec_handle, np.array([int(start_idx)], dtype=np.int32))
        binding.vec_reset(vec_handle, int(start_idx))

        # Snapshot log before this evaluation episode for delta computation
        eval_env_handle = binding.vec_env_at(vec_handle, 0)
        eval_log_before = dict(binding.env_get(eval_env_handle))

        result_data = None
        for _step in range(eval_steps + 10):
            obs_tensor = torch.from_numpy(obs_buf[:1].copy()).to(device)
            with torch.no_grad():
                logits, _ = best_policy(obs_tensor)
            action = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
            act_buf[0] = int(action.item())
            binding.vec_step(vec_handle)
            if term_buf[0]:
                eval_log_after = dict(binding.env_get(eval_env_handle))
                _DELTA_KEYS = ("total_return", "sortino", "max_drawdown", "num_trades", "win_rate")
                result_data = {k: eval_log_after[k] - eval_log_before[k] for k in _DELTA_KEYS
                               if k in eval_log_after and k in eval_log_before}
                for k in eval_log_after:
                    if k not in result_data:
                        result_data[k] = eval_log_after[k]
                break

        if result_data is not None:
            meta_results.append(MetaWindowResult(
                start_idx=int(start_idx),
                selected_checkpoint=str(best_path),
                selected_idx=best_idx,
                selection_returns=[float(r) for r in selection_returns],
                total_return=float(result_data.get("total_return", 0.0)),
                sortino=float(result_data.get("sortino", 0.0)),
                max_drawdown=float(result_data.get("max_drawdown", 0.0)),
                num_trades=int(result_data.get("num_trades", 0)),
            ))

    binding.vec_close(vec_handle)
    elapsed = time.monotonic() - t0

    if not meta_results:
        return {"error": "no windows completed", "elapsed_s": elapsed}

    returns = [r.total_return for r in meta_results]
    sortinos = [r.sortino for r in meta_results]
    selection_counts = {}
    for r in meta_results:
        k = str(r.selected_checkpoint)
        selection_counts[k] = selection_counts.get(k, 0) + 1

    summary = {
        "median_total_return": float(np.percentile(returns, 50)),
        "p10_total_return": float(np.percentile(returns, 10)),
        "p90_total_return": float(np.percentile(returns, 90)),
        "mean_total_return": float(np.mean(returns)),
        "pct_profitable": float(np.mean([r > 0 for r in returns])),
        "median_sortino": float(np.percentile(sortinos, 50)),
        "selection_counts": selection_counts,
        "n_completed": len(meta_results),
    }

    return {
        "checkpoints": [str(p) for p in checkpoint_paths],
        "data_path": str(data_path),
        "eval_steps": eval_steps,
        "selection_steps": selection_steps,
        "n_windows": n_windows,
        "seed": seed,
        "elapsed_s": elapsed,
        "summary": summary,
        "windows": [asdict(r) for r in meta_results],
    }


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def _print_comparison_table(ttt_windows: list[dict], baseline_windows: list[dict]) -> None:
    """Print a side-by-side comparison table of TTT vs baseline returns."""
    if not ttt_windows or not baseline_windows:
        return
    n = min(len(ttt_windows), len(baseline_windows))
    print(f"\n{'Win':>4}  {'Baseline':>12}  {'TTT':>12}  {'Delta':>10}")
    print("-" * 44)
    total_delta = 0.0
    for i in range(n):
        base_ret = baseline_windows[i]["total_return"]
        ttt_ret = ttt_windows[i]["total_return"]
        delta = ttt_ret - base_ret
        total_delta += delta
        marker = " *" if delta > 0 else "  "
        print(f"{i+1:>4}  {base_ret:>+12.4f}  {ttt_ret:>+12.4f}  {delta:>+10.4f}{marker}")
    print("-" * 44)
    avg_delta = total_delta / n
    ttt_median = float(np.median([w["total_return"] for w in ttt_windows[:n]]))
    base_median = float(np.median([w["total_return"] for w in baseline_windows[:n]]))
    print(f"{'Median':>4}  {base_median:>+12.4f}  {ttt_median:>+12.4f}  {avg_delta:>+10.4f}")
    winner = "TTT" if avg_delta > 0 else "Baseline"
    print(f"\n{winner} wins by avg {abs(avg_delta)*100:.2f}% per window")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-time training (TTT) evaluation for trading policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")
    subparsers.required = False

    # ---- shared args ----
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-path", required=True, help="Path to binary market data")
        p.add_argument("--n-windows", type=int, default=20, help="Number of evaluation windows")
        p.add_argument("--eval-steps", type=int, default=90, help="Steps per evaluation window")
        p.add_argument("--fee-rate", type=float, default=0.001)
        p.add_argument("--fill-slippage-bps", type=float, default=5.0)
        p.add_argument("--max-leverage", type=float, default=1.0)
        p.add_argument("--periods-per-year", type=float, default=8760.0)
        p.add_argument("--short-borrow-apr", type=float, default=0.0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--deterministic", action="store_true", default=True,
                       help="Deterministic action selection (argmax)")
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--out", type=str, default=None, help="JSON output path")
        p.add_argument("--verbose", action="store_true")

    # ---- TTT mode (default) ----
    ttt_parser = subparsers.add_parser(
        "ttt", help="Test-time training with partial layer fine-tuning"
    )
    _add_common(ttt_parser)
    ttt_parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ttt_parser.add_argument("--ttt-warmup-steps", type=int, default=20,
                            help="Steps to collect before TTT update")
    ttt_parser.add_argument("--ttt-update-steps", type=int, default=50,
                            help="Minibatch size for TTT gradient steps")
    ttt_parser.add_argument("--ttt-lr", type=float, default=1e-5,
                            help="Learning rate for TTT Adam optimizer")
    ttt_parser.add_argument("--ttt-epochs", type=int, default=2,
                            help="Number of gradient epochs during TTT update")
    ttt_parser.add_argument("--kl-coef", type=float, default=0.1,
                            help="KL divergence penalty coefficient")
    ttt_parser.add_argument("--compare-baseline", action="store_true", default=True,
                            help="Also run baseline (no TTT) for comparison")
    ttt_parser.add_argument("--no-compare-baseline", action="store_false",
                            dest="compare_baseline")

    # ---- Meta checkpoint selection mode ----
    meta_parser = subparsers.add_parser(
        "meta-select", help="Meta checkpoint selection (pick best ckpt per window)"
    )
    _add_common(meta_parser)
    meta_parser.add_argument("--checkpoints", nargs="+", required=True,
                             help="Space-separated list of checkpoint paths")
    meta_parser.add_argument("--selection-steps", type=int, default=10,
                             help="Steps used to evaluate each checkpoint before picking")

    # ---- Backward-compat: allow top-level --checkpoint without subcommand ----
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (shortcut for ttt mode)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Checkpoint paths (shortcut for meta-select mode)")
    parser.add_argument("--data-path", default=None, help="Data path (shortcut)")
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=90)
    parser.add_argument("--ttt-warmup-steps", type=int, default=20)
    parser.add_argument("--ttt-update-steps", type=int, default=50)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--ttt-epochs", type=int, default=2)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--compare-baseline", action="store_true", default=True)
    parser.add_argument("--no-compare-baseline", action="store_false",
                        dest="compare_baseline")
    parser.add_argument("--selection-steps", type=int, default=10)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Determine mode from explicit subcommand or presence of --checkpoints
    mode = args.mode
    if mode is None:
        if args.checkpoints and not args.checkpoint:
            mode = "meta-select"
        else:
            mode = "ttt"

    # Validate required args
    data_path = args.data_path
    if data_path is None:
        parser.error("--data-path is required")

    if mode == "meta-select":
        checkpoints = args.checkpoints
        if not checkpoints:
            parser.error("--checkpoints required for meta-select mode")
        result = meta_select_holdout_eval(
            checkpoint_paths=checkpoints,
            data_path=data_path,
            n_windows=args.n_windows,
            eval_steps=args.eval_steps,
            selection_steps=args.selection_steps,
            fee_rate=args.fee_rate,
            fill_slippage_bps=args.fill_slippage_bps,
            max_leverage=args.max_leverage,
            periods_per_year=args.periods_per_year,
            short_borrow_apr=args.short_borrow_apr,
            seed=args.seed,
            deterministic=args.deterministic,
            device_str=args.device,
            verbose=args.verbose,
        )
    else:
        # TTT mode
        checkpoint = args.checkpoint
        if checkpoint is None:
            parser.error("--checkpoint is required for ttt mode")
        result = ttt_holdout_eval(
            checkpoint_path=checkpoint,
            data_path=data_path,
            n_windows=args.n_windows,
            eval_steps=args.eval_steps,
            ttt_warmup_steps=args.ttt_warmup_steps,
            ttt_update_steps=args.ttt_update_steps,
            ttt_lr=args.ttt_lr,
            ttt_epochs=args.ttt_epochs,
            kl_coef=args.kl_coef,
            fee_rate=args.fee_rate,
            fill_slippage_bps=args.fill_slippage_bps,
            max_leverage=args.max_leverage,
            periods_per_year=args.periods_per_year,
            short_borrow_apr=args.short_borrow_apr,
            seed=args.seed,
            deterministic=args.deterministic,
            device_str=args.device,
            compare_baseline=args.compare_baseline,
            verbose=args.verbose,
        )

        # Print comparison table
        if args.compare_baseline and "ttt_windows" in result and "baseline_windows" in result:
            _print_comparison_table(result["ttt_windows"], result["baseline_windows"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        if args.verbose:
            print(f"\nSaved to {args.out}")

    summary_key = "ttt_summary" if mode == "ttt" else "summary"
    print(json.dumps(result.get(summary_key, result), indent=2))
    print(f"\nElapsed: {result.get('elapsed_s', 0):.2f}s")


if __name__ == "__main__":
    main()
