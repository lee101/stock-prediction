#!/usr/bin/env python3
"""Compare ensemble methods for crypto30: softmax avg vs majority vote vs max-confidence."""
from __future__ import annotations
import collections
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import (
    read_mktd, MktdData, INITIAL_CASH, P_CLOSE,
    simulate_daily_policy, DailySimResult,
)
from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts

VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"
PROD = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]


def metrics(eq):
    ret = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    return ret, sort, dd


def run_ensemble_method(data, policies, S, method, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    """Run ensemble with specified combination method."""
    T = data.num_timesteps
    max_steps = T - 1
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            all_logits_raw = []
            all_logits_masked = []
            all_probs_raw = []
            all_probs_masked = []
            for p in policies:
                lg, _ = p(obs_t)
                all_logits_raw.append(lg)
                all_probs_raw.append(torch.softmax(lg, dim=-1))
                lg_m = _mask_all_shorts(lg, num_symbols=S, per_symbol_actions=1)
                all_logits_masked.append(lg_m)
                all_probs_masked.append(torch.softmax(lg_m, dim=-1))

            if method == "softmax_avg":
                # Original approach: softmax BEFORE masking, then mask, then argmax
                avg_probs = torch.stack(all_probs_raw).mean(dim=0)
                logits_from_avg = torch.log(avg_probs + 1e-8)
                logits_from_avg = _mask_all_shorts(logits_from_avg, num_symbols=S, per_symbol_actions=1)
                action_now = int(torch.argmax(logits_from_avg, dim=-1).item())

            elif method == "softmax_avg_masked":
                # Alternative: mask BEFORE softmax, then average
                avg_probs = torch.stack(all_probs_masked).mean(dim=0)
                action_now = int(torch.argmax(avg_probs, dim=-1).item())

            elif method == "majority_vote":
                # Each model votes for its argmax action (masked)
                votes = [int(torch.argmax(lg, dim=-1).item()) for lg in all_logits_masked]
                counter = collections.Counter(votes)
                action_now = counter.most_common(1)[0][0]

            elif method == "max_confidence":
                # Pick the action from the most confident model
                max_conf = -1
                action_now = 0
                for lg, pr in zip(all_logits_masked, all_probs_masked):
                    a = int(torch.argmax(lg, dim=-1).item())
                    conf = float(pr[0, a].item())
                    if conf > max_conf:
                        max_conf = conf
                        action_now = a

            elif method == "geometric_avg":
                # Geometric mean of probabilities (more conservative, masked)
                log_probs = torch.stack([torch.log(p + 1e-8) for p in all_probs_masked])
                avg_log = log_probs.mean(dim=0)
                action_now = int(torch.argmax(avg_log, dim=-1).item())

            elif method == "rank_vote":
                # Each model ranks actions, sum ranks, pick lowest total rank
                n_actions = all_logits_masked[0].shape[1]
                rank_sum = torch.zeros(n_actions)
                for lg in all_logits_masked:
                    # Rank: higher logit = lower rank (better)
                    ranks = torch.argsort(torch.argsort(lg[0], descending=True))
                    rank_sum += ranks.float()
                action_now = int(torch.argmin(rank_sum).item())

            else:
                raise ValueError(f"Unknown method: {method}")

        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    result = simulate_daily_policy(
        data, policy_fn,
        max_steps=max_steps,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        fill_buffer_bps=5.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        enable_drawdown_profit_early_exit=False,
    )

    eq = np.array(result.equity_curve) if result.equity_curve is not None else np.ones(max_steps + 1) * INITIAL_CASH
    return eq


def main():
    data = read_mktd(str(VAL_BIN))
    S = data.num_symbols
    print(f"data: {data.num_timesteps}d x {S}sym\n")

    policies = []
    for cp in PROD:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        policies.append(loaded.policy)

    methods = ["softmax_avg", "softmax_avg_masked", "majority_vote", "max_confidence", "geometric_avg", "rank_vote"]

    print(f"{'Method':<20} {'Ret%':>10} {'Sort':>8} {'MaxDD%':>8}")
    print("-" * 50)

    for method in methods:
        eq = run_ensemble_method(data, policies, S, method)
        ret, sort, dd = metrics(eq)
        print(f"{method:<20} {ret*100:>+10.2f}% {sort:>8.2f} {dd*100:>8.2f}%")

    # Slippage sweep on best methods
    print(f"\nSlippage sweep:")
    print(f"{'Slip':>6} {'softmax':>12} {'smax_mask':>12} {'majority':>12} {'geometric':>12}")
    print("-" * 58)
    for slip in [0, 5, 10, 20]:
        results = {}
        for method in ["softmax_avg", "softmax_avg_masked", "majority_vote", "geometric_avg"]:
            eq = run_ensemble_method(data, policies, S, method, slippage_bps=slip)
            ret, _, _ = metrics(eq)
            results[method] = ret
        print(f"{slip:>4}bp {results['softmax_avg']*100:>+12.2f}% {results['softmax_avg_masked']*100:>+12.2f}% {results['majority_vote']*100:>+12.2f}% {results['geometric_avg']*100:>+12.2f}%")


if __name__ == "__main__":
    main()
