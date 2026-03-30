#!/usr/bin/env python3
"""Evaluate a PPO checkpoint across multiple time periods (1d, 7d, 30d).

Uses the pure-python simulator from ``pufferlib_market.hourly_replay`` on
deterministic tail slices of MKTD validation data.

Usage:
    python -m pufferlib_market.evaluate_multiperiod \
        --checkpoint path/to/best.pt \
        --data-path path/to/crypto6_val.bin \
        --deterministic --disable-shorts

    # Compare multiple checkpoints:
    python -m pufferlib_market.evaluate_multiperiod \
        --checkpoints ckpt1.pt,ckpt2.pt \
        --data-path path/to/val.bin \
        --deterministic
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from pufferlib_market.checkpoint_loader import (
    build_action_grid_summary_line,
    extract_checkpoint_state_dict,
    infer_arch_from_state_dict,
    infer_hidden_size_from_state_dict,
    infer_resmlp_blocks_from_state_dict,
    load_checkpoint_payload,
    resolve_checkpoint_action_grid_config,
)
from pufferlib_market.evaluate_tail import (
    ResidualTradingPolicy,
    TradingPolicy,
    _build_shortable_mask,
    _infer_num_actions,
    _mask_all_shorts,
    _mask_disallowed_shorts,
    _slice_tail,
)
from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return


PERIODS = {
    "1d": 24,
    "7d": 168,
    "30d": 720,
}


@dataclass(frozen=True)
class LoadedPolicy:
    policy: nn.Module
    arch: str
    hidden_size: int
    action_allocation_bins: int
    action_level_bins: int
    action_max_offset_bps: float
    num_actions: int


def load_policy(
    checkpoint_path: str,
    num_symbols: int,
    *,
    arch: str = "auto",
    hidden_size: int | None = None,
    device: torch.device = torch.device("cpu"),
    features_per_sym: int = 16,
) -> LoadedPolicy:
    """Load a checkpoint and build the policy network."""
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
            f"Only supports alloc_bins=1 level_bins=1 (got {action_allocation_bins}, {action_level_bins})"
        )

    obs_size = num_symbols * features_per_sym + 5 + num_symbols
    fallback_actions = 1 + 2 * num_symbols
    num_actions = _infer_num_actions(state_dict, fallback=fallback_actions)

    if arch == "auto":
        arch = infer_arch_from_state_dict(state_dict)
    hidden = hidden_size if hidden_size is not None else infer_hidden_size_from_state_dict(state_dict, arch)

    if arch == "resmlp":
        num_blocks = infer_resmlp_blocks_from_state_dict(state_dict)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden, num_blocks=num_blocks).to(device)
    elif arch == "mlp":
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if hasattr(policy, "_use_encoder_norm"):
        # Prefer the stored flag (from _checkpoint_payload "use_encoder_norm" key).
        # Fall back to inferring from missing keys for old checkpoints that don't store it.
        if isinstance(payload, dict) and "use_encoder_norm" in payload:
            policy._use_encoder_norm = bool(payload["use_encoder_norm"])
        else:
            policy._use_encoder_norm = "encoder_norm.weight" not in missing
    ignored = {"obs_mean", "obs_std", "encoder_norm.weight", "encoder_norm.bias"}
    bad_missing = [k for k in missing if k not in ignored]
    bad_unexpected = [k for k in unexpected if k not in ignored]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch — missing: {bad_missing}, unexpected: {bad_unexpected}"
        )
    policy.eval()
    return LoadedPolicy(
        policy=policy,
        arch=str(arch),
        hidden_size=int(hidden),
        action_allocation_bins=int(action_allocation_bins),
        action_level_bins=int(action_level_bins),
        action_max_offset_bps=float(action_max_offset_bps),
        num_actions=int(num_actions),
    )


def make_policy_fn(
    policy: nn.Module,
    *,
    num_symbols: int,
    disable_shorts: bool = False,
    shortable_mask: torch.Tensor | None = None,
    deterministic: bool = True,
    decision_lag: int = 0,
    device: torch.device = torch.device("cpu"),
):
    """Create a policy_fn closure for simulate_daily_policy."""
    pending_actions: collections.deque[int] = collections.deque(maxlen=max(1, decision_lag + 1))

    def _policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
        if disable_shorts:
            logits = _mask_all_shorts(logits, num_symbols=num_symbols)
        elif shortable_mask is not None:
            logits = _mask_disallowed_shorts(logits, num_symbols=num_symbols, shortable_mask=shortable_mask)
        if deterministic:
            action_now = int(torch.argmax(logits, dim=-1).item())
        else:
            action_now = int(Categorical(logits=logits).sample().item())

        if decision_lag <= 0:
            return action_now
        pending_actions.append(action_now)
        if len(pending_actions) <= decision_lag:
            return 0
        return pending_actions.popleft()

    return _policy_fn


def evaluate_period(
    policy: nn.Module,
    data: MktdData,
    eval_hours: int,
    *,
    num_symbols: int,
    fee_rate: float = 0.001,
    fill_buffer_bps: float = 5.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    disable_shorts: bool = False,
    shortable_mask: torch.Tensor | None = None,
    deterministic: bool = True,
    decision_lag: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Evaluate a single time period on the tail of data."""
    if data.num_timesteps < eval_hours + 1:
        return {
            "eval_hours": eval_hours,
            "error": f"Data too short ({data.num_timesteps} timesteps, need {eval_hours + 1})",
        }

    tail = _slice_tail(data, steps=eval_hours)
    policy_fn = make_policy_fn(
        policy,
        num_symbols=num_symbols,
        disable_shorts=disable_shorts,
        shortable_mask=shortable_mask,
        deterministic=deterministic,
        decision_lag=decision_lag,
        device=device,
    )

    result = simulate_daily_policy(
        tail,
        policy_fn,
        max_steps=eval_hours,
        fee_rate=fee_rate,
        fill_buffer_bps=fill_buffer_bps,
        max_leverage=max_leverage,
        periods_per_year=periods_per_year,
        short_borrow_apr=short_borrow_apr,
    )

    ann_ret = annualize_total_return(
        float(result.total_return),
        periods=float(eval_hours),
        periods_per_year=periods_per_year,
    )

    return {
        "eval_hours": eval_hours,
        "total_return": float(result.total_return),
        "annualized_return": float(ann_ret),
        "sortino": float(result.sortino),
        "max_drawdown": float(result.max_drawdown),
        "num_trades": int(result.num_trades),
        "win_rate": float(result.win_rate),
        "avg_hold_steps": float(result.avg_hold_steps),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    data_path: str,
    periods: dict[str, int],
    *,
    arch: str = "auto",
    hidden_size: int | None = None,
    fee_rate: float = 0.001,
    fill_buffer_bps: float = 5.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    disable_shorts: bool = False,
    shortable_symbols: str | None = None,
    deterministic: bool = True,
    decision_lag: int = 0,
    device_str: str = "cpu",
) -> list[dict]:
    """Load checkpoint once and evaluate across all periods."""
    device = torch.device(device_str)
    data = read_mktd(Path(data_path))
    num_symbols = data.num_symbols
    features_per_sym = data.features.shape[2]

    loaded = load_policy(
        checkpoint_path, num_symbols, arch=arch, hidden_size=hidden_size, device=device,
        features_per_sym=features_per_sym,
    )
    policy = loaded.policy

    shortable_mask = _build_shortable_mask(data.symbols, shortable_symbols)
    if shortable_mask is not None:
        shortable_mask = shortable_mask.to(device=device)

    results = []
    for period_name, hours in sorted(periods.items(), key=lambda x: x[1]):
        r = evaluate_period(
            policy, data, hours,
            num_symbols=num_symbols,
            fee_rate=fee_rate,
            fill_buffer_bps=fill_buffer_bps,
            max_leverage=max_leverage,
            periods_per_year=periods_per_year,
            short_borrow_apr=short_borrow_apr,
            disable_shorts=disable_shorts,
            shortable_mask=shortable_mask,
            deterministic=deterministic,
            decision_lag=decision_lag,
            device=device,
        )
        r["period"] = period_name
        r["checkpoint"] = checkpoint_path
        r["data_path"] = data_path
        r["arch"] = loaded.arch
        r["hidden_size"] = loaded.hidden_size
        r["action_allocation_bins"] = loaded.action_allocation_bins
        r["action_level_bins"] = loaded.action_level_bins
        r["action_max_offset_bps"] = loaded.action_max_offset_bps
        results.append(r)

    return results


def format_table(all_results: list[list[dict]], checkpoint_names: list[str]) -> str:
    """Format results as an aligned text table."""
    lines = []

    for results, name in zip(all_results, checkpoint_names):
        if len(all_results) > 1:
            lines.append(f"\n=== {name} ===")
        else:
            lines.append(f"Checkpoint: {name}")
        if results:
            config = results[0]
            lines.append(f"Effective config: arch={config['arch']}, hidden_size={config['hidden_size']}")
            lines.append(
                build_action_grid_summary_line(
                    action_allocation_bins=config["action_allocation_bins"],
                    action_level_bins=config["action_level_bins"],
                    action_max_offset_bps=config["action_max_offset_bps"],
                )
            )
        lines.append(
            f"{'Period':<8} {'Return%':>9} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7} {'WinRate':>8} {'AvgHold':>8}"
        )
        lines.append("-" * 62)
        successful_results = []
        for r in results:
            if "error" in r:
                lines.append(f"{r['period']:<8} {r['error']}")
                continue
            successful_results.append(r)
            ret_pct = r["total_return"] * 100
            dd_pct = r["max_drawdown"] * 100
            wr_pct = r["win_rate"] * 100
            hold_h = r["avg_hold_steps"]
            lines.append(
                f"{r['period']:<8} {ret_pct:>+8.2f}% {r['sortino']:>8.2f} {dd_pct:>6.2f}% "
                f"{r['num_trades']:>7d} {wr_pct:>7.1f}% {hold_h:>7.1f}h"
            )
        if successful_results:
            best_result = max(successful_results, key=lambda result: float(result["total_return"]))
            worst_result = min(successful_results, key=lambda result: float(result["total_return"]))
            positive_period_count = sum(float(result["total_return"]) > 0.0 for result in successful_results)
            lines.append(
                "Best period: "
                f"{best_result['period']} ({float(best_result['total_return']) * 100:+.2f}%)"
            )
            lines.append(
                "Worst period: "
                f"{worst_result['period']} ({float(worst_result['total_return']) * 100:+.2f}%)"
            )
            lines.append(
                f"Positive periods: {positive_period_count}/{len(successful_results)}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-period PPO evaluation on tail windows")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Single checkpoint path")
    group.add_argument("--checkpoints", help="Comma-separated checkpoint paths for comparison")
    parser.add_argument("--data-path", required=True, help="Path to MKTD .bin")
    parser.add_argument("--periods", default="1d,7d,30d", help="Comma-separated periods (default: 1d,7d,30d)")
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
    parser.add_argument("--decision-lag", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    # Parse periods
    periods = {}
    for period_spec_raw in args.periods.split(","):
        period_spec = period_spec_raw.strip()
        if period_spec in PERIODS:
            period_hours = PERIODS[period_spec]
        elif period_spec.endswith("d") and period_spec[:-1].isdigit():
            period_hours = int(period_spec[:-1]) * 24
        elif period_spec.endswith("h") and period_spec[:-1].isdigit():
            period_hours = int(period_spec[:-1])
        else:
            parser.error(
                f"Unknown period: {period_spec}. Use Nd (days) or Nh (hours), e.g. 1d, 7d, 30d, 168h"
            )
        if period_hours < 1:
            parser.error(f"Period must be at least 1 hour: {period_spec}")
        periods[period_spec] = period_hours

    # Get checkpoint list
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
        if not ckpt_paths:
            parser.error("--checkpoints must include at least one checkpoint path")

    all_results = []
    names = []
    for ckpt in ckpt_paths:
        name = Path(ckpt).parent.name + "/" + Path(ckpt).name
        names.append(name)
        eval_stdout = sys.stderr if args.json else sys.stdout
        with redirect_stdout(eval_stdout):
            results = evaluate_checkpoint(
                ckpt, args.data_path, periods,
                arch=args.arch,
                hidden_size=args.hidden_size,
                fee_rate=args.fee_rate,
                fill_buffer_bps=args.fill_buffer_bps,
                max_leverage=args.max_leverage,
                periods_per_year=args.periods_per_year,
                short_borrow_apr=args.short_borrow_apr,
                disable_shorts=args.disable_shorts,
                shortable_symbols=args.shortable_symbols,
                deterministic=args.deterministic,
                decision_lag=args.decision_lag,
                device_str=args.device,
            )
        all_results.append(results)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print(format_table(all_results, names))


if __name__ == "__main__":
    main()
