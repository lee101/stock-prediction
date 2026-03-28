"""Sliding window evaluation for trading policies.

Instead of independent random-start episodes, evaluates the policy over
all overlapping windows of the validation data. This gives a more complete
picture of policy performance across different market regimes.

Usage:
  python -m pufferlib_market.evaluate_sliding \\
    --checkpoint checkpoints/best.pt \\
    --data-path pufferlib_market/data/mixed23_daily_val.bin \\
    --episode-len 720 --stride 168 --deterministic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

def compute_calmar(annualized_return: float, max_drawdown: float) -> float:
    """Calmar ratio = annualized_return / max_drawdown. Higher is better."""
    if max_drawdown == 0:
        return float("inf") if annualized_return > 0 else 0.0
    return annualized_return / abs(max_drawdown)


# ---------------------------------------------------------------------------
# Per-window result
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_idx: int
    start_offset: int
    total_return: float
    max_drawdown: float
    sortino: float
    num_trades: int
    win_rate: float
    stopped_early: bool


# ---------------------------------------------------------------------------
# Policy classes (mirror evaluate.py; no shared module exists to avoid circular imports)
# ---------------------------------------------------------------------------

class TradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h = self.encoder(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, x, deterministic=False, disable_shorts=False):
        logits, _ = self.forward(x)
        if disable_shorts:
            K = (self.num_actions - 1) // 2
            logits[:, 1 + K:] = float("-inf")
        if deterministic:
            return logits.argmax(dim=-1)
        from torch.distributions import Categorical
        return Categorical(logits=logits).sample()


class _ResidualBlock(nn.Module):
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
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden) for _ in range(num_blocks)])
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
        from torch.distributions import Categorical
        return Categorical(logits=logits).sample()


# ---------------------------------------------------------------------------
# Core sliding window evaluation
# ---------------------------------------------------------------------------

def sliding_window_eval(
    policy_fn: Callable[[np.ndarray], int],
    data: MktdData,
    episode_len: int,
    stride: int,
    *,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    action_allocation_bins: int = 1,
    action_level_bins: int = 1,
    action_max_offset_bps: float = 0.0,
    enable_drawdown_profit_early_exit: bool = False,
    drawdown_profit_early_exit_verbose: bool = False,
    drawdown_profit_early_exit_min_steps: int = 20,
    drawdown_profit_early_exit_progress_fraction: float = 0.5,
) -> list[WindowResult]:
    """Run the policy over all overlapping windows and return per-window results.

    Parameters
    ----------
    policy_fn:
        A callable that takes an obs array (shape [obs_size]) and returns an
        integer action. Wraps a torch policy via a lambda.
    data:
        Full MKTD dataset. Windows are sliced from it.
    episode_len:
        Number of timesteps per window (e.g. 720 = 30 days hourly).
    stride:
        How many bars to advance between window starts (e.g. 168 = 1 week).
    """
    T = data.num_timesteps
    # Need episode_len + 1 bars (simulate_daily_policy requires max_steps < T)
    if episode_len + 1 > T:
        raise ValueError(
            f"episode_len={episode_len} + 1 exceeds data length T={T}. "
            "Use a shorter episode_len or more data."
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    results: list[WindowResult] = []

    # Generate all valid start offsets
    starts = list(range(0, T - episode_len, stride))
    if not starts:
        raise ValueError(
            f"No windows fit: T={T}, episode_len={episode_len}, stride={stride}"
        )

    for idx, start in enumerate(starts):
        end = start + episode_len + 1  # +1 so simulate_daily_policy has room
        window_data = MktdData(
            version=data.version,
            symbols=data.symbols,
            features=data.features[start:end],
            prices=data.prices[start:end],
            tradable=data.tradable[start:end] if data.tradable is not None else None,
        )

        result = simulate_daily_policy(
            window_data,
            policy_fn,
            max_steps=episode_len,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            periods_per_year=periods_per_year,
            short_borrow_apr=short_borrow_apr,
            action_allocation_bins=action_allocation_bins,
            action_level_bins=action_level_bins,
            action_max_offset_bps=action_max_offset_bps,
            enable_drawdown_profit_early_exit=enable_drawdown_profit_early_exit,
            drawdown_profit_early_exit_verbose=drawdown_profit_early_exit_verbose,
            drawdown_profit_early_exit_min_steps=drawdown_profit_early_exit_min_steps,
            drawdown_profit_early_exit_progress_fraction=drawdown_profit_early_exit_progress_fraction,
        )

        results.append(WindowResult(
            window_idx=idx,
            start_offset=start,
            total_return=float(result.total_return),
            max_drawdown=float(result.max_drawdown),
            sortino=float(result.sortino),
            num_trades=int(result.num_trades),
            win_rate=float(result.win_rate),
            stopped_early=bool(result.stopped_early),
        ))

    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def aggregate_sliding_results(
    results: list[WindowResult],
    *,
    episode_len: int,
    periods_per_year: float = 8760.0,
) -> dict:
    """Compute aggregate stats from a list of WindowResult."""
    if not results:
        return {}

    returns = np.array([r.total_return for r in results], dtype=np.float64)
    max_dds = np.array([r.max_drawdown for r in results], dtype=np.float64)
    sortinos = np.array([r.sortino for r in results], dtype=np.float64)
    num_trades = np.array([r.num_trades for r in results], dtype=np.float64)
    win_rates = np.array([r.win_rate for r in results], dtype=np.float64)

    # Cumulative return (chain windows end-to-end)
    cum_mult = float(np.prod(1.0 + returns))
    total_steps = episode_len * len(results)
    ann_return = annualize_total_return(
        float(cum_mult - 1.0),
        periods=float(total_steps),
        periods_per_year=float(periods_per_year),
    )

    mean_dd = float(np.mean(max_dds))
    worst_dd = float(np.max(max_dds))
    calmar = compute_calmar(ann_return, worst_dd)

    return {
        "num_windows": len(results),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "pct_profitable": float((returns > 0).mean()),
        "cum_mult": cum_mult,
        "annualized_return": ann_return,
        "mean_sortino": float(np.mean(sortinos)),
        "mean_max_drawdown": mean_dd,
        "worst_max_drawdown": worst_dd,
        "calmar": calmar,
        "mean_trades": float(np.mean(num_trades)),
        "mean_win_rate": float(np.mean(win_rates)),
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_sliding_results(
    results: list[WindowResult],
    stats: dict,
    *,
    top_n: int = 5,
) -> None:
    n = stats["num_windows"]
    print(f"\n{'=' * 62}")
    print(f"SLIDING WINDOW EVALUATION ({n} windows)")
    print(f"{'=' * 62}")
    print(f"Return:     mean={stats['mean_return']:+.4f}  std={stats['std_return']:.4f}  "
          f"median={stats['median_return']:+.4f}")
    print(f"            min={stats['min_return']:+.4f}  max={stats['max_return']:+.4f}  "
          f"profitable={stats['pct_profitable']*100:.1f}%")
    print(f"Sortino:    mean={stats['mean_sortino']:.3f}")
    print(f"MaxDD:      mean={stats['mean_max_drawdown']:.4f}  worst={stats['worst_max_drawdown']:.4f}")
    print(f"Calmar:     {stats['calmar']:.3f}  (annualized_return / worst_max_drawdown)")
    print(f"Annualized: {stats['annualized_return'] * 100:+.2f}%  (cumulative {stats['cum_mult']:.4f}x)")
    print(f"Trades:     mean={stats['mean_trades']:.1f}  win_rate={stats['mean_win_rate']:.4f}")

    returns = np.array([r.total_return for r in results])
    print(f"\nReturn percentiles:")
    for p in [5, 25, 50, 75, 95]:
        v = float(np.percentile(returns, p))
        print(f"  p{p:02d}: {v:+.4f}")

    sorted_results = sorted(results, key=lambda r: r.total_return)
    print(f"\nTop {top_n} windows:")
    for r in reversed(sorted_results[-top_n:]):
        early = " [early-exit]" if r.stopped_early else ""
        print(f"  window={r.window_idx:3d} start={r.start_offset:5d}  "
              f"return={r.total_return:+.4f}  dd={r.max_drawdown:.4f}  "
              f"trades={r.num_trades:3d}  wr={r.win_rate:.2f}{early}")

    print(f"\nBottom {top_n} windows:")
    for r in sorted_results[:top_n]:
        early = " [early-exit]" if r.stopped_early else ""
        print(f"  window={r.window_idx:3d} start={r.start_offset:5d}  "
              f"return={r.total_return:+.4f}  dd={r.max_drawdown:.4f}  "
              f"trades={r.num_trades:3d}  wr={r.win_rate:.2f}{early}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_policy_fn(
    policy: nn.Module,
    device: torch.device,
    *,
    deterministic: bool = True,
    disable_shorts: bool = False,
) -> Callable[[np.ndarray], int]:
    """Wrap a torch policy module as a policy_fn(obs) -> int callable."""
    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.get_action(obs_t, deterministic=deterministic, disable_shorts=disable_shorts)
        return int(action.item())
    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="Sliding window evaluation of a trained RL trading policy")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to MKTD binary data file")
    parser.add_argument("--episode-len", type=int, default=720,
                        help="Window length in bars (default: 720 = 30 days hourly)")
    parser.add_argument("--stride", type=int, default=168,
                        help="Slide step in bars (default: 168 = 1 week hourly)")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=0.0,
                        help="Adverse fill slippage in bps (realistic: 5-12)")
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=8760.0,
                        help="Annualisation factor (8760=hourly, 365=daily, 252=trading days)")
    parser.add_argument("--action-allocation-bins", type=int, default=1)
    parser.add_argument("--action-level-bins", type=int, default=1)
    parser.add_argument("--action-max-offset-bps", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax actions instead of sampling")
    parser.add_argument("--disable-shorts", action="store_true",
                        help="Mask short actions (long/flat only)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = read_mktd(args.data_path)
    num_symbols = data.num_symbols
    print(f"Data: {num_symbols} symbols, {data.num_timesteps} timesteps")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "action_allocation_bins" in ckpt:
        args.action_allocation_bins = max(1, int(ckpt["action_allocation_bins"]))
    if "action_level_bins" in ckpt:
        args.action_level_bins = max(1, int(ckpt["action_level_bins"]))
    if "action_max_offset_bps" in ckpt:
        args.action_max_offset_bps = max(0.0, float(ckpt["action_max_offset_bps"]))

    features_per_sym = data.features.shape[2]
    obs_size = num_symbols * features_per_sym + 5 + num_symbols
    per_symbol_actions = max(1, int(args.action_allocation_bins)) * max(1, int(args.action_level_bins))
    num_actions = 1 + 2 * num_symbols * per_symbol_actions

    print(f"obs_size={obs_size}, num_actions={num_actions}, arch={args.arch}")
    print(f"episode_len={args.episode_len}, stride={args.stride}, deterministic={args.deterministic}")

    if args.arch == "resmlp":
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"Loaded checkpoint: update={ckpt.get('update', '?')}, "
          f"train_best_return={ckpt.get('best_return', '?'):.4f}")

    policy_fn = _build_policy_fn(
        policy, device,
        deterministic=args.deterministic,
        disable_shorts=args.disable_shorts,
    )

    results = sliding_window_eval(
        policy_fn,
        data,
        episode_len=args.episode_len,
        stride=args.stride,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        max_leverage=args.max_leverage,
        periods_per_year=args.periods_per_year,
        short_borrow_apr=args.short_borrow_apr,
        action_allocation_bins=args.action_allocation_bins,
        action_level_bins=args.action_level_bins,
        action_max_offset_bps=args.action_max_offset_bps,
    )

    stats = aggregate_sliding_results(
        results,
        episode_len=args.episode_len,
        periods_per_year=args.periods_per_year,
    )
    print_sliding_results(results, stats)

    if args.calmar:
        print(f"\nCalmar ratio (annualized_return / worst_max_drawdown): {stats['calmar']:.4f}")


if __name__ == "__main__":
    main()
