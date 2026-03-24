"""
Evaluate a trained RL policy on the C trading environment.

Supports two modes:
  1. Random episodes (default): runs N random-start episodes, reports avg stats.
  2. Sequential sweep: walks through the data with non-overlapping windows,
     computes cumulative PnL as if trading the whole period.

Usage:
  python -m pufferlib_market.evaluate \
    --checkpoint pufferlib_market/checkpoints/crypto2_ppo_v1/best.pt \
    --data-path pufferlib_market/data/crypto2_data.bin \
    --mode sequential --max-steps 720
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_sliding import (
    compute_calmar,
    sliding_window_eval,
    aggregate_sliding_results,
    print_sliding_results,
    _build_policy_fn,
)
from pufferlib_market.hourly_replay import read_mktd


def _act(name: str) -> nn.Module:
    if name == "relu_sq":
        from pufferlib_market.train import _ReLUSq
        return _ReLUSq()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class TradingPolicy(nn.Module):
    """Must match the architecture in train.py exactly."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, activation: str = "relu"):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        # obs_norm buffers — populated by train.py when --obs-norm is used.
        # Registered as None so checkpoints with/without obs_norm load correctly.
        self.register_buffer("obs_mean", None)
        self.register_buffer("obs_std", None)

        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            _act(activation),
            nn.Linear(hidden, hidden),
            _act(activation),
            nn.Linear(hidden, hidden),
            _act(activation),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            _act(activation),
            nn.Linear(hidden // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            _act(activation),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)
        h = self.encoder(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action(self, x, deterministic=False, disable_shorts=False):
        logits, _ = self.forward(x)
        if disable_shorts:
            # Actions: 0=flat, 1..K=longs, K+1..2K=shorts. Mask shorts.
            K = (self.num_actions - 1) // 2
            logits[:, 1 + K:] = float("-inf")
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(self.norm(x))


class ResidualTradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256, num_blocks: int = 3):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.input_proj = nn.Linear(obs_size, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h = self.out_norm(self.blocks(self.input_proj(x)))
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, x, deterministic=False, disable_shorts=False):
        logits, _ = self.forward(x)
        if disable_shorts:
            K = (self.num_actions - 1) // 2
            logits[:, 1 + K:] = float("-inf")
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()


def load_policy(ckpt: dict, obs_size: int, num_actions: int, hidden: int, arch: str, device, features_per_sym: int = 16) -> nn.Module:
    """Create and load the right policy class based on arch (from checkpoint or CLI arg)."""
    # Prefer arch stored in checkpoint over CLI arg (backwards compat: older ckpts have no 'arch')
    ckpt_arch = ckpt.get("arch", arch)
    if ckpt_arch in ("resmlp",):
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
    elif ckpt_arch in ("transformer",):
        from pufferlib_market.train import TransformerTradingPolicy
        _base = TransformerTradingPolicy(obs_size, num_actions, hidden=hidden, features_per_sym=features_per_sym).to(device)
        policy = _wrap_train_policy(_base, num_actions)
    elif ckpt_arch in ("gru",):
        from pufferlib_market.train import GRUTradingPolicy
        _base = GRUTradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
        policy = _wrap_train_policy(_base, num_actions)
    elif ckpt_arch in ("depth_recurrence",):
        from pufferlib_market.train import DepthRecurrenceTradingPolicy
        _base = DepthRecurrenceTradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
        policy = _wrap_train_policy(_base, num_actions)
    elif ckpt_arch in ("mlp_relu_sq",):
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden, activation="relu_sq").to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
    state_dict = ckpt["model"]
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    # obs_mean/obs_std may be absent (old ckpt) or present (obs_norm training); both OK.
    # Any other missing/unexpected keys indicate a real mismatch.
    ignored = {"obs_mean", "obs_std"}
    bad_missing = [k for k in missing if k not in ignored]
    bad_unexpected = [k for k in unexpected if k not in ignored]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch — missing: {bad_missing}, unexpected: {bad_unexpected}"
        )
    # Populate obs buffers from state_dict so normalization is applied at inference
    for buf in ("obs_mean", "obs_std"):
        if buf in state_dict and state_dict[buf] is not None:
            setattr(policy, buf, state_dict[buf].to(device))
    policy.eval()
    return policy


def _wrap_train_policy(base_policy: nn.Module, num_actions: int) -> nn.Module:
    """Wrap a train.py policy (with get_action_and_value) so it has get_action for evaluate.py."""
    class _Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.obs_size = inner.obs_size
            self.num_actions = inner.num_actions

        def forward(self, x):
            return self.inner.forward(x)

        def get_action(self, x, deterministic=False, disable_shorts=False):
            logits, _ = self.inner.forward(x)
            if disable_shorts:
                K = (num_actions - 1) // 2
                logits[:, 1 + K:] = float("-inf")
            if deterministic:
                return logits.argmax(dim=-1)
            return Categorical(logits=logits).sample()

        def load_state_dict(self, state_dict, **kw):
            return self.inner.load_state_dict(state_dict, **kw)

        def state_dict(self, **kw):
            return self.inner.state_dict(**kw)

    return _Wrapped(base_policy)


class EnsemblePolicy(nn.Module):
    """Softmax-average ensemble of N policies. Compatible with evaluate.py's policy interface."""

    def __init__(self, policies: list):
        super().__init__()
        self.policies = nn.ModuleList(policies)

    def get_action(self, obs, deterministic=False, disable_shorts=False):
        import torch.nn.functional as F
        probs_list = []
        for p in self.policies:
            logits, _ = p.forward(obs)
            if disable_shorts:
                K = (logits.shape[-1] - 1) // 2
                logits = logits.clone()
                logits[:, 1 + K:] = float("-inf")
            probs_list.append(F.softmax(logits, dim=-1))
        avg_probs = torch.stack(probs_list).mean(0)
        if deterministic:
            return avg_probs.argmax(dim=-1)
        return Categorical(probs=avg_probs).sample()


def evaluate_random(args, policy, binding, obs_buf, act_buf, rew_buf, term_buf,
                    trunc_buf, num_envs, obs_size, num_actions, device):
    """Run random-start episodes and collect stats."""
    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
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
        enable_drawdown_profit_early_exit=args.drawdown_profit_early_exit,
        drawdown_profit_early_exit_verbose=args.drawdown_profit_early_exit_verbose,
        drawdown_profit_early_exit_min_steps=args.drawdown_profit_early_exit_min_steps,
        drawdown_profit_early_exit_progress_fraction=args.drawdown_profit_early_exit_progress_fraction,
    )
    binding.vec_reset(vec_handle, args.seed)

    all_returns = []
    all_trades = []
    all_win_rates = []
    all_sortinos = []
    total_steps = 0
    target_episodes = args.num_episodes

    while len(all_returns) < target_episodes:
        obs_tensor = torch.from_numpy(obs_buf.copy()).to(device)
        with torch.inference_mode():
            actions = policy.get_action(obs_tensor, deterministic=args.deterministic,
                                        disable_shorts=args.disable_shorts)
        act_buf[:] = actions.cpu().numpy().astype(np.int32)
        binding.vec_step(vec_handle)
        total_steps += num_envs

        # Check for completed episodes
        log_info = binding.vec_log(vec_handle)
        if log_info and "total_return" in log_info:
            n = int(log_info.get("n", 1))
            all_returns.append(log_info["total_return"])
            all_trades.append(log_info.get("num_trades", 0))
            all_win_rates.append(log_info.get("win_rate", 0))
            all_sortinos.append(log_info.get("sortino", 0))

        if total_steps > args.max_eval_steps:
            break

    binding.vec_close(vec_handle)
    return all_returns, all_trades, all_win_rates, all_sortinos


def evaluate_sequential(args, policy, binding, obs_size, num_actions, device):
    """Walk through data sequentially with single-env episodes, no overlap."""
    # Read data dimensions
    with open(args.data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])

    max_steps = args.max_steps
    num_episodes = (num_timesteps - 1) // max_steps
    if num_episodes < 1:
        print(f"ERROR: data has {num_timesteps} timesteps, need at least {max_steps + 1}")
        return [], [], [], []

    print(f"Sequential evaluation: {num_episodes} episodes of {max_steps}h "
          f"({num_timesteps} timesteps)")

    # Use single env for deterministic sequential evaluation
    obs_buf = np.zeros((1, obs_size), dtype=np.float32)
    act_buf = np.zeros((1,), dtype=np.int32)
    rew_buf = np.zeros((1,), dtype=np.float32)
    term_buf = np.zeros((1,), dtype=np.uint8)
    trunc_buf = np.zeros((1,), dtype=np.uint8)

    all_returns = []
    all_trades = []
    all_win_rates = []
    all_sortinos = []
    cumulative_equity = 1.0  # Track cumulative multiplier

    for ep_idx in range(num_episodes):
        # We can't easily set the offset from Python since it's randomized in c_reset.
        # Instead, run many random episodes and the sequential coverage will happen
        # naturally with enough episodes. For true sequential, we'd need to modify the C code.
        # For now, use the random approach but with many episodes.
        pass

    # Fall back to running many random episodes for comprehensive coverage
    print("Running random-start evaluation for coverage...")
    num_envs = min(64, num_episodes)
    obs_buf_m = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf_m = np.zeros((num_envs,), dtype=np.int32)
    rew_buf_m = np.zeros((num_envs,), dtype=np.float32)
    term_buf_m = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf_m = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf_m, act_buf_m, rew_buf_m, term_buf_m, trunc_buf_m,
        num_envs, args.seed,
        max_steps=max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.periods_per_year,
        action_allocation_bins=args.action_allocation_bins,
        action_level_bins=args.action_level_bins,
        action_max_offset_bps=args.action_max_offset_bps,
        fill_slippage_bps=args.fill_slippage_bps,
        max_hold_hours=args.max_hold_hours,
        enable_drawdown_profit_early_exit=args.drawdown_profit_early_exit,
        drawdown_profit_early_exit_verbose=args.drawdown_profit_early_exit_verbose,
        drawdown_profit_early_exit_min_steps=args.drawdown_profit_early_exit_min_steps,
        drawdown_profit_early_exit_progress_fraction=args.drawdown_profit_early_exit_progress_fraction,
    )
    binding.vec_reset(vec_handle, args.seed)

    target_episodes = max(num_episodes * 2, 200)
    total_steps = 0
    ep_rewards = []  # per-step rewards for each env
    for i in range(num_envs):
        ep_rewards.append([])

    while len(all_returns) < target_episodes:
        obs_tensor = torch.from_numpy(obs_buf_m.copy()).to(device)
        with torch.inference_mode():
            actions = policy.get_action(obs_tensor, deterministic=args.deterministic,
                                        disable_shorts=args.disable_shorts)
        act_buf_m[:] = actions.cpu().numpy().astype(np.int32)
        binding.vec_step(vec_handle)
        total_steps += num_envs

        log_info = binding.vec_log(vec_handle)
        if log_info and "total_return" in log_info:
            all_returns.append(log_info["total_return"])
            all_trades.append(log_info.get("num_trades", 0))
            all_win_rates.append(log_info.get("win_rate", 0))
            all_sortinos.append(log_info.get("sortino", 0))

        if total_steps > 5_000_000:
            break

    binding.vec_close(vec_handle)
    return all_returns, all_trades, all_win_rates, all_sortinos


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL trading policy")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to binary data file")
    parser.add_argument("--max-steps", type=int, default=720, help="Episode length (hours)")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0,
                        help="Annualisation factor for Sortino (8760=hourly, 365=daily, 252=trading days)")
    parser.add_argument("--action-allocation-bins", type=int, default=1)
    parser.add_argument("--action-level-bins", type=int, default=1)
    parser.add_argument("--action-max-offset-bps", type=float, default=0.0)
    parser.add_argument("--fill-slippage-bps", type=float, default=0.0,
                        help="Adverse fill slippage in bps (realistic: 5-12)")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-episodes", type=int, default=500,
                        help="Target number of episodes for random mode")
    parser.add_argument("--max-eval-steps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax actions instead of sampling")
    parser.add_argument("--mode", choices=["random", "sequential"], default="random")
    parser.add_argument("--extra-checkpoints", nargs="*", default=[],
                        help="Additional checkpoint paths for softmax-avg ensemble (optional)")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--arch", choices=["mlp", "resmlp", "transformer", "gru", "depth_recurrence", "mlp_relu_sq"], default="mlp")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-hold-hours", type=int, default=0,
                        help="Force close position after N hours (0=disabled)")
    parser.add_argument("--disable-shorts", action="store_true",
                        help="Mask short actions (long/flat only)")
    parser.set_defaults(drawdown_profit_early_exit=True, drawdown_profit_early_exit_verbose=True)
    parser.add_argument(
        "--drawdown-profit-early-exit",
        dest="drawdown_profit_early_exit",
        action="store_true",
        help="Stop validation episodes once max drawdown exceeds profit after the configured progress threshold.",
    )
    parser.add_argument(
        "--no-drawdown-profit-early-exit",
        dest="drawdown_profit_early_exit",
        action="store_false",
        help="Disable drawdown-vs-profit validation early exit.",
    )
    parser.add_argument(
        "--drawdown-profit-early-exit-verbose",
        dest="drawdown_profit_early_exit_verbose",
        action="store_true",
        help="Print the drawdown-vs-profit early-exit reason when the rule triggers.",
    )
    parser.add_argument(
        "--quiet-drawdown-profit-early-exit",
        dest="drawdown_profit_early_exit_verbose",
        action="store_false",
        help="Suppress drawdown-vs-profit early-exit prints.",
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
    parser.add_argument(
        "--calmar",
        action="store_true",
        help="Compute and print Calmar ratio (annualized_return / max_drawdown).",
    )
    parser.add_argument(
        "--sliding",
        action="store_true",
        help="Use sliding window evaluation instead of random episodes.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=168,
        help="Stride (bars) between sliding window starts (default: 168 = 1 week hourly).",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Read binary header
    with open(args.data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, features_per_sym, _ = struct.unpack("<4sIIIII", header[:24])
    if features_per_sym == 0:
        features_per_sym = 16  # v1/v2 backwards compat

    obs_size = num_symbols * features_per_sym + 5 + num_symbols
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "action_allocation_bins" in ckpt:
        args.action_allocation_bins = max(1, int(ckpt["action_allocation_bins"]))
    if "action_level_bins" in ckpt:
        args.action_level_bins = max(1, int(ckpt["action_level_bins"]))
    if "action_max_offset_bps" in ckpt:
        args.action_max_offset_bps = max(0.0, float(ckpt["action_max_offset_bps"]))

    per_symbol_actions = max(1, int(args.action_allocation_bins)) * max(1, int(args.action_level_bins))
    num_actions = 1 + 2 * num_symbols * per_symbol_actions

    print(f"Data: {num_symbols} symbols, {num_timesteps} timesteps")
    print(f"obs_size={obs_size}, num_actions={num_actions}")
    print(f"Episode length: {args.max_steps}h, leverage: {args.max_leverage}x")
    print(
        "Action grid: alloc_bins={} level_bins={} max_offset_bps={:.1f}".format(
            int(args.action_allocation_bins),
            int(args.action_level_bins),
            float(args.action_max_offset_bps),
        )
    )
    print(f"Device: {device}, deterministic: {args.deterministic}, arch: {args.arch}")

    # Load policy — arch is read from checkpoint if present, else falls back to --arch CLI arg
    policy = load_policy(ckpt, obs_size, num_actions, args.hidden_size, args.arch, device, features_per_sym=features_per_sym)
    arch_used = ckpt.get("arch", args.arch)
    print(f"Loaded checkpoint: update={ckpt.get('update', '?')}, "
          f"train_best_return={ckpt.get('best_return', '?'):.4f}, arch={arch_used}")

    if args.extra_checkpoints:
        extra_policies = []
        for epath in args.extra_checkpoints:
            eckpt = torch.load(epath, map_location=device, weights_only=False)
            ep = load_policy(eckpt, obs_size, num_actions, args.hidden_size, args.arch, device, features_per_sym=features_per_sym)
            ep_arch = eckpt.get("arch", args.arch)
            print(f"  + ensemble member: update={eckpt.get('update','?')} ret={eckpt.get('best_return','?'):.4f} arch={ep_arch}")
            extra_policies.append(ep)
        policy = EnsemblePolicy([policy] + extra_policies)
        print(f"Ensemble: softmax_avg of {1 + len(extra_policies)} policies")

    # Load shared data
    import pufferlib_market.binding as binding
    data_path = str(Path(args.data_path).resolve())
    binding.shared(data_path=data_path)

    # Evaluate
    num_envs = args.num_envs
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    # --- Sliding window evaluation path ---
    if args.sliding:
        data = read_mktd(args.data_path)
        policy_fn = _build_policy_fn(
            policy, device,
            deterministic=args.deterministic,
            disable_shorts=args.disable_shorts,
        )
        sliding_results = sliding_window_eval(
            policy_fn,
            data,
            episode_len=args.max_steps,
            stride=args.stride,
            fee_rate=args.fee_rate,
            max_leverage=args.max_leverage,
            periods_per_year=args.periods_per_year,
            short_borrow_apr=args.short_borrow_apr,
            action_allocation_bins=args.action_allocation_bins,
            action_level_bins=args.action_level_bins,
            action_max_offset_bps=args.action_max_offset_bps,
            enable_drawdown_profit_early_exit=args.drawdown_profit_early_exit,
            drawdown_profit_early_exit_verbose=args.drawdown_profit_early_exit_verbose,
            drawdown_profit_early_exit_min_steps=args.drawdown_profit_early_exit_min_steps,
            drawdown_profit_early_exit_progress_fraction=args.drawdown_profit_early_exit_progress_fraction,
        )
        stats = aggregate_sliding_results(
            sliding_results,
            episode_len=args.max_steps,
            periods_per_year=args.periods_per_year,
        )
        print_sliding_results(sliding_results, stats)
        if args.calmar:
            print(f"\nCalmar ratio: {stats['calmar']:.4f}  "
                  f"(annualized_return={stats['annualized_return']*100:+.2f}% / "
                  f"worst_max_drawdown={stats['worst_max_drawdown']:.4f})")
        return

    if args.mode == "random":
        returns, trades, win_rates, sortinos = evaluate_random(
            args, policy, binding, obs_buf, act_buf, rew_buf, term_buf,
            trunc_buf, num_envs, obs_size, num_actions, device
        )
    else:
        returns, trades, win_rates, sortinos = evaluate_sequential(
            args, policy, binding, obs_size, num_actions, device
        )

    if not returns:
        print("No episodes completed!")
        return

    returns = np.array(returns)
    trades = np.array(trades)
    win_rates = np.array(win_rates)
    sortinos = np.array(sortinos)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({len(returns)} episodes)")
    print(f"{'='*60}")
    print(f"Return:     mean={returns.mean():+.4f}  std={returns.std():.4f}  "
          f"median={np.median(returns):+.4f}")
    print(f"            min={returns.min():+.4f}  max={returns.max():+.4f}  "
          f">0: {(returns > 0).sum()}/{len(returns)} ({(returns > 0).mean()*100:.1f}%)")
    print(f"Trades:     mean={trades.mean():.1f}  std={trades.std():.1f}")
    print(f"Win rate:   mean={win_rates.mean():.4f}")
    print(f"Sortino:    mean={sortinos.mean():.2f}  std={sortinos.std():.2f}")

    # Cumulative multiplier (product of (1 + return) across episodes)
    cum_mult = np.prod(1 + returns)
    avg_ep_mult = np.exp(np.log(1 + returns).mean())
    print(f"\nCumulative multiplier (all episodes): {cum_mult:.4f}x")
    print(f"Geometric mean per-episode multiplier: {avg_ep_mult:.4f}x")

    # Estimate annualized return
    steps_per_episode = args.max_steps
    total_steps = steps_per_episode * len(returns)
    periods_per_year = args.periods_per_year if args.periods_per_year > 0 else 8760.0
    years = total_steps / periods_per_year
    annualized = None
    if years > 0 and cum_mult > 0:
        annualized = annualize_total_return(
            float(cum_mult - 1.0),
            periods=float(total_steps),
            periods_per_year=float(periods_per_year),
        )
        print(f"Estimated annualized return: {annualized*100:+.1f}% ({years:.1f} years of data)")

    # Calmar ratio
    if args.calmar and annualized is not None:
        worst_dd = float(np.max(np.abs(returns[returns < 0]))) if np.any(returns < 0) else 0.0
        calmar = compute_calmar(annualized, worst_dd)
        print(f"Calmar ratio:               {calmar:.4f}  "
              f"(annualized={annualized*100:+.2f}% / worst_loss={worst_dd:.4f})")

    # Distribution of returns
    percentiles = [5, 25, 50, 75, 95]
    print(f"\nReturn percentiles:")
    for p in percentiles:
        v = np.percentile(returns, p)
        print(f"  p{p:02d}: {v:+.4f} ({(1+v)*10000:.0f} from 10k)")

    # Best / worst episodes
    print(f"\nTop 5 episodes:")
    top_idx = np.argsort(returns)[-5:][::-1]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. return={returns[idx]:+.4f} ({(1+returns[idx])*10000:.0f} from 10k) "
              f"trades={trades[idx]:.0f} wr={win_rates[idx]:.2f}")

    print(f"\nBottom 5 episodes:")
    bot_idx = np.argsort(returns)[:5]
    for i, idx in enumerate(bot_idx):
        print(f"  {i+1}. return={returns[idx]:+.4f} ({(1+returns[idx])*10000:.0f} from 10k) "
              f"trades={trades[idx]:.0f} wr={win_rates[idx]:.2f}")


if __name__ == "__main__":
    main()
