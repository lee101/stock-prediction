"""Test-time training evaluation with LoRA adaptation.

Before evaluating each episode, quickly fine-tune LoRA adapters on the
first ``calibration_steps`` observations of that episode (calibration phase),
then trade the rest using the adapted policy.

This adapts to market regimes:
- High volatility period: LoRA adjusts position sizing
- Trending vs mean-reverting: LoRA adjusts entry/exit timing
- Correlation structure changes: LoRA adjusts relative weighting

Usage:
  python -m pufferlib_market.evaluate_ttt \\
    --checkpoint checkpoints/best.pt \\
    --data-path pufferlib_market/data/mixed23_daily_val.bin \\
    --lora-rank 4 \\
    --calibration-steps 30 \\
    --lora-lr 0.01 \\
    --deterministic
"""

from __future__ import annotations

import argparse
import importlib
import struct
import time
from pathlib import Path

import numpy as np
import torch

from pufferlib_market.checkpoint_loader import (
    build_action_grid_summary_line,
    format_action_grid_override_note,
    build_checkpoint_summary_lines,
    build_cli_policy_config_line,
    build_runtime_summary_line,
    load_checkpoint_payload,
    load_policy_from_checkpoint as _load_base_policy_from_checkpoint_impl,
    resolve_checkpoint_action_grid_config,
)
from pufferlib_market.train import TradingPolicy, ResidualTradingPolicy
from pufferlib_market.lora import LoRAPolicy, reset_adam_state
from pufferlib_market.metrics import annualize_total_return


# ---------------------------------------------------------------------------
# Single-episode TTT evaluation helpers
# ---------------------------------------------------------------------------


def _run_episode_ttt(
    *,
    lora_policy: LoRAPolicy,
    lora_optimizer: torch.optim.Adam,
    binding,
    vec_handle,
    obs_buf: np.ndarray,
    act_buf: np.ndarray,
    rew_buf: np.ndarray,
    term_buf: np.ndarray,
    trunc_buf: np.ndarray,
    calibration_steps: int,
    lora_grad_steps: int,
    deterministic: bool,
    disable_shorts: bool,
    device: torch.device,
) -> dict | None:
    """Run one TTT episode: calibrate on first K steps, then trade the rest.

    Returns the log dict from binding.vec_log or None if the episode did not
    complete.
    """
    # Reset LoRA to identity and clear Adam momentum for a fresh start
    lora_policy.reset_lora()
    reset_adam_state(lora_optimizer)

    # --- Calibration phase: collect K observations and do gradient steps ---
    calib_obs: list[torch.Tensor] = []
    calib_actions: list[torch.Tensor] = []

    step = 0
    episode_done = False
    log_result: dict | None = None

    while not episode_done:
        obs_tensor = torch.from_numpy(obs_buf[0:1].copy()).to(device)  # (1, obs_size)

        if step < calibration_steps:
            # During calibration we still need to step the environment, but we
            # collect observations and the actions we would take, then do a
            # supervised-style gradient update on the LoRA params using the
            # policy's own predictions as pseudo-labels (self-distillation).
            with torch.no_grad():
                action = lora_policy.get_action(
                    obs_tensor, deterministic=deterministic, disable_shorts=disable_shorts
                )
            calib_obs.append(obs_tensor.detach())
            calib_actions.append(action.detach())

        if step == calibration_steps and len(calib_obs) > 0:
            # Perform gradient steps on the collected calibration batch
            calib_obs_t = torch.cat(calib_obs, dim=0)       # (K, obs_size)
            calib_act_t = torch.cat(calib_actions, dim=0)   # (K,)

            for _ in range(lora_grad_steps):
                lora_optimizer.zero_grad()
                _, log_probs, entropy, _ = lora_policy.get_action_and_value(
                    calib_obs_t, calib_act_t, disable_shorts=disable_shorts
                )
                # Behaviour-cloning loss: maximise log-prob of collected actions
                # (negative log-prob = cross-entropy with self as teacher)
                bc_loss = -log_probs.mean() - 0.01 * entropy.mean()
                bc_loss.backward()
                # Gradient clipping to avoid overshooting with small batches
                torch.nn.utils.clip_grad_norm_(lora_policy.lora_parameters(), 1.0)
                lora_optimizer.step()

        # Step environment with current (possibly adapted) policy
        if step >= calibration_steps:
            with torch.no_grad():
                action = lora_policy.get_action(
                    obs_tensor, deterministic=deterministic, disable_shorts=disable_shorts
                )

        act_buf[0] = int(action.item())
        binding.vec_step(vec_handle)
        step += 1

        # Check for episode completion
        log_info = binding.vec_log(vec_handle)
        if log_info and "total_return" in log_info:
            log_result = log_info
            episode_done = True

        # Safety: bail if env never terminates (shouldn't happen)
        if step > 100_000:
            break

    return log_result


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_ttt(args, lora_policy: LoRAPolicy, base_policy, binding, obs_size: int,
                 num_actions: int, device: torch.device):
    """Run TTT evaluation collecting per-episode stats.

    Also runs a baseline (no-TTT) pass for comparison.
    """
    num_envs = 1  # TTT requires sequential per-episode processing

    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    def _make_vec(seed_offset: int = 0):
        return binding.vec_init(
            obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
            num_envs, args.seed + seed_offset,
            max_steps=args.max_steps,
            fee_rate=args.fee_rate,
            max_leverage=args.max_leverage,
            short_borrow_apr=args.short_borrow_apr,
            periods_per_year=args.periods_per_year,
            action_allocation_bins=args.action_allocation_bins,
            action_level_bins=args.action_level_bins,
            action_max_offset_bps=args.action_max_offset_bps,
            fill_slippage_bps=args.fill_slippage_bps,
            max_hold_hours=args.max_hold_hours,
            enable_drawdown_profit_early_exit=False,
            drawdown_profit_early_exit_verbose=False,
            drawdown_profit_early_exit_min_steps=20,
            drawdown_profit_early_exit_progress_fraction=0.5,
        )

    # -----------------------------------------------------------------------
    # TTT pass
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TTT EVALUATION (rank={args.lora_rank}, calib={args.calibration_steps}, "
          f"lr={args.lora_lr}, grad_steps={args.lora_grad_steps})")
    print(f"{'='*60}")

    ttt_returns: list[float] = []
    ttt_trades: list[float] = []
    ttt_win_rates: list[float] = []
    ttt_sortinos: list[float] = []

    lora_optimizer = torch.optim.Adam(lora_policy.lora_parameters(), lr=args.lora_lr)

    vec_handle = _make_vec(seed_offset=0)
    binding.vec_reset(vec_handle, args.seed)
    t0 = time.time()

    for ep in range(args.num_episodes):
        log = _run_episode_ttt(
            lora_policy=lora_policy,
            lora_optimizer=lora_optimizer,
            binding=binding,
            vec_handle=vec_handle,
            obs_buf=obs_buf,
            act_buf=act_buf,
            rew_buf=rew_buf,
            term_buf=term_buf,
            trunc_buf=trunc_buf,
            calibration_steps=args.calibration_steps,
            lora_grad_steps=args.lora_grad_steps,
            deterministic=args.deterministic,
            disable_shorts=args.disable_shorts,
            device=device,
        )
        if log is not None:
            ttt_returns.append(float(log["total_return"]))
            ttt_trades.append(float(log.get("num_trades", 0)))
            ttt_win_rates.append(float(log.get("win_rate", 0)))
            ttt_sortinos.append(float(log.get("sortino", 0)))
            if (ep + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  TTT ep {ep+1}/{args.num_episodes}  "
                      f"return={ttt_returns[-1]:+.4f}  "
                      f"elapsed={elapsed:.1f}s")

    binding.vec_close(vec_handle)

    # -----------------------------------------------------------------------
    # Baseline pass (no TTT) — same seeds for fair comparison
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION (no TTT)")
    print(f"{'='*60}")

    base_returns: list[float] = []
    base_trades: list[float] = []
    base_win_rates: list[float] = []
    base_sortinos: list[float] = []

    base_obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    base_act_buf = np.zeros((num_envs,), dtype=np.int32)
    base_rew_buf = np.zeros((num_envs,), dtype=np.float32)
    base_term_buf = np.zeros((num_envs,), dtype=np.uint8)
    base_trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    # Reuse _make_vec but with baseline buffers via a thin wrapper
    vec_handle_base = binding.vec_init(
        base_obs_buf, base_act_buf, base_rew_buf, base_term_buf, base_trunc_buf,
        num_envs, args.seed,
        max_steps=args.max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.periods_per_year,
        action_allocation_bins=args.action_allocation_bins,
        action_level_bins=args.action_level_bins,
        action_max_offset_bps=args.action_max_offset_bps,
        fill_slippage_bps=args.fill_slippage_bps,
        max_hold_hours=args.max_hold_hours,
        enable_drawdown_profit_early_exit=False,
        drawdown_profit_early_exit_verbose=False,
        drawdown_profit_early_exit_min_steps=20,
        drawdown_profit_early_exit_progress_fraction=0.5,
    )
    binding.vec_reset(vec_handle_base, args.seed)

    base_episodes_done = 0
    base_step = 0
    while base_episodes_done < args.num_episodes and base_step < 100_000_000:
        obs_t = torch.from_numpy(base_obs_buf.copy()).to(device)
        with torch.no_grad():
            actions = base_policy.get_action(
                obs_t, deterministic=args.deterministic, disable_shorts=args.disable_shorts
            )
        base_act_buf[:] = actions.cpu().numpy().astype(np.int32)
        binding.vec_step(vec_handle_base)
        base_step += num_envs

        log_info = binding.vec_log(vec_handle_base)
        if log_info and "total_return" in log_info:
            base_returns.append(float(log_info["total_return"]))
            base_trades.append(float(log_info.get("num_trades", 0)))
            base_win_rates.append(float(log_info.get("win_rate", 0)))
            base_sortinos.append(float(log_info.get("sortino", 0)))
            base_episodes_done += 1

    binding.vec_close(vec_handle_base)

    return (
        np.array(ttt_returns), np.array(ttt_trades),
        np.array(ttt_win_rates), np.array(ttt_sortinos),
        np.array(base_returns), np.array(base_trades),
        np.array(base_win_rates), np.array(base_sortinos),
    )


def _print_stats(label: str, returns: np.ndarray, trades: np.ndarray,
                 win_rates: np.ndarray, sortinos: np.ndarray,
                 max_steps: int, periods_per_year: float) -> None:
    if len(returns) == 0:
        print(f"{label}: no episodes completed")
        return
    print(f"\n{label} ({len(returns)} episodes)")
    print(f"  Return:   mean={returns.mean():+.4f}  median={np.median(returns):+.4f}  "
          f"std={returns.std():.4f}")
    print(f"            min={returns.min():+.4f}  max={returns.max():+.4f}  "
          f">0: {(returns > 0).sum()}/{len(returns)} ({(returns > 0).mean()*100:.1f}%)")
    print(f"  Trades:   mean={trades.mean():.1f}")
    print(f"  Win rate: mean={win_rates.mean():.4f}")
    print(f"  Sortino:  mean={sortinos.mean():.2f}  std={sortinos.std():.2f}")

    cum_mult = float(np.prod(1.0 + returns))
    total_steps = max_steps * len(returns)
    ann = annualize_total_return(
        cum_mult - 1.0, periods=float(total_steps), periods_per_year=float(periods_per_year)
    )
    print(f"  Cum mult: {cum_mult:.4f}x   Est. annualized: {ann*100:+.1f}%")


def _load_base_policy_from_checkpoint(
    *,
    ckpt,
    obs_size: int,
    num_actions: int,
    arch: str,
    hidden_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str, int]:
    return _load_base_policy_from_checkpoint_impl(
        ckpt=ckpt,
        obs_size=obs_size,
        num_actions=num_actions,
        arch=arch,
        hidden_size=hidden_size,
        device=device,
        mlp_factory=lambda obs, acts, hidden: TradingPolicy(obs, acts, hidden=hidden),
        resmlp_factory=lambda obs, acts, hidden, num_blocks: ResidualTradingPolicy(
            obs, acts, hidden=hidden, num_blocks=num_blocks
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Test-time LoRA adaptation evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to binary data file")
    parser.add_argument("--max-steps", type=int, default=720, help="Episode length")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0)
    parser.add_argument("--action-allocation-bins", type=int, default=1)
    parser.add_argument("--action-level-bins", type=int, default=1)
    parser.add_argument("--action-max-offset-bps", type=float, default=0.0)
    parser.add_argument("--fill-slippage-bps", type=float, default=0.0)
    parser.add_argument("--max-hold-hours", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-shorts", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--cpu", action="store_true")
    # TTT-specific args
    parser.add_argument("--lora-rank", type=int, default=4,
                        help="LoRA rank (4-8 recommended to avoid overfitting)")
    parser.add_argument("--lora-num-layers", type=int, default=2,
                        help="Number of last linear layers to apply LoRA to")
    parser.add_argument("--calibration-steps", type=int, default=30,
                        help="Number of bars used for TTT calibration before trading")
    parser.add_argument("--lora-lr", type=float, default=0.01,
                        help="Learning rate for LoRA Adam optimizer")
    parser.add_argument("--lora-grad-steps", type=int, default=3,
                        help="Number of gradient steps during calibration phase")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Read binary header
    with open(args.data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, features_per_sym, _ = struct.unpack("<4sIIIII", header[:24])
    if features_per_sym == 0:
        features_per_sym = 16  # v1/v2 backwards compat

    obs_size = num_symbols * features_per_sym + 5 + num_symbols

    ckpt = load_checkpoint_payload(args.checkpoint, map_location=device)
    requested_action_allocation_bins = args.action_allocation_bins
    requested_action_level_bins = args.action_level_bins
    requested_action_max_offset_bps = args.action_max_offset_bps
    (
        args.action_allocation_bins,
        args.action_level_bins,
        args.action_max_offset_bps,
    ) = resolve_checkpoint_action_grid_config(
        ckpt,
        action_allocation_bins=requested_action_allocation_bins,
        action_level_bins=requested_action_level_bins,
        action_max_offset_bps=requested_action_max_offset_bps,
    )

    per_symbol_actions = max(1, int(args.action_allocation_bins)) * max(1, int(args.action_level_bins))
    num_actions = 1 + 2 * num_symbols * per_symbol_actions

    base_policy, effective_arch, effective_hidden_size = _load_base_policy_from_checkpoint(
        ckpt=ckpt,
        obs_size=obs_size,
        num_actions=num_actions,
        arch=args.arch,
        hidden_size=args.hidden_size,
        device=device,
    )

    print(f"Data: {num_symbols} symbols, {num_timesteps} timesteps")
    print(f"obs_size={obs_size}, num_actions={num_actions}")
    print(build_cli_policy_config_line(arch=args.arch, hidden_size=args.hidden_size))
    print(
        build_action_grid_summary_line(
            action_allocation_bins=args.action_allocation_bins,
            action_level_bins=args.action_level_bins,
            action_max_offset_bps=args.action_max_offset_bps,
        )
    )
    action_grid_override_note = format_action_grid_override_note(
        requested_allocation_bins=requested_action_allocation_bins,
        requested_level_bins=requested_action_level_bins,
        requested_max_offset_bps=requested_action_max_offset_bps,
        effective_allocation_bins=args.action_allocation_bins,
        effective_level_bins=args.action_level_bins,
        effective_max_offset_bps=args.action_max_offset_bps,
    )
    if action_grid_override_note is not None:
        print(action_grid_override_note)
    print(build_runtime_summary_line(device=device, deterministic=args.deterministic))
    for line in build_checkpoint_summary_lines(
        ckpt=ckpt,
        requested_arch=args.arch,
        requested_hidden_size=args.hidden_size,
        effective_arch=effective_arch,
        effective_hidden_size=effective_hidden_size,
        checkpoint_path=args.checkpoint,
    ):
        print(line)
    # Wrap with LoRA for TTT
    lora_policy = LoRAPolicy(base_policy, rank=args.lora_rank, num_layers=args.lora_num_layers).to(device)
    lora_policy.eval()
    n_lora_params = sum(p.numel() for p in lora_policy.lora_parameters())
    n_base_params = sum(p.numel() for p in base_policy.parameters())
    print(f"LoRA: rank={args.lora_rank}, layers={args.lora_num_layers}, "
          f"lora_params={n_lora_params:,} ({n_lora_params/n_base_params*100:.2f}% of base)")

    # Load shared data via importlib so tests can replace the binding module
    # through sys.modules without depending on package attribute caching.
    binding = importlib.import_module("pufferlib_market.binding")
    data_path = str(Path(args.data_path).resolve())
    binding.shared(data_path=data_path)

    (ttt_ret, ttt_tr, ttt_wr, ttt_so,
     base_ret, base_tr, base_wr, base_so) = evaluate_ttt(
        args, lora_policy, base_policy, binding, obs_size, num_actions, device
    )

    periods_per_year = args.periods_per_year if args.periods_per_year > 0 else 8760.0

    _print_stats("TTT (with LoRA adaptation)", ttt_ret, ttt_tr, ttt_wr, ttt_so,
                 args.max_steps, periods_per_year)
    _print_stats("Baseline (no adaptation)",  base_ret, base_tr, base_wr, base_so,
                 args.max_steps, periods_per_year)

    # Delta summary
    if len(ttt_ret) > 0 and len(base_ret) > 0:
        ttt_mean = float(ttt_ret.mean())
        base_mean = float(base_ret.mean())
        delta = ttt_mean - base_mean
        print(f"\nDelta (TTT - Baseline): mean_return {delta:+.4f} "
              f"({'TTT wins' if delta > 0 else 'Baseline wins'})")


if __name__ == "__main__":
    main()
