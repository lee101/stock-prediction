"""Compare softmax_avg vs logit_avg vs tie-break ensemble modes for flat rate."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import _mask_all_shorts, _slice_window, load_policy  # noqa: E402
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def collect_stacked_logits(policies, obs_list, num_symbols):
    all_stacked = []  # [T, N, A]
    with torch.inference_mode():
        for obs in obs_list:
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).view(1, -1)
            logits_by_policy = []
            for p in policies:
                logits, _ = p(obs_t)
                logits = _mask_all_shorts(logits, num_symbols=num_symbols)
                logits_by_policy.append(logits.squeeze(0))  # [A]
            all_stacked.append(torch.stack(logits_by_policy, dim=0))  # [N, A]
    return torch.stack(all_stacked, dim=0)  # [T, N, A]


def mode_softmax_avg(stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(stacked, dim=-1).mean(dim=1)  # [T, A]
    argmax = torch.argmax(probs, dim=-1)
    return argmax, probs


def mode_logit_avg(stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = stacked.mean(dim=1)
    probs = F.softmax(logits, dim=-1)
    argmax = torch.argmax(logits, dim=-1)
    return argmax, probs


def mode_majority_vote(stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    per_policy_argmax = torch.argmax(stacked, dim=-1)  # [T, N]
    T = stacked.shape[0]
    A = stacked.shape[-1]
    results = []
    probs_out = []
    for t in range(T):
        votes = per_policy_argmax[t].tolist()
        c = Counter(votes)
        winner, _ = c.most_common(1)[0]
        results.append(winner)
        p_row = torch.zeros(A)
        for a, n in c.items():
            p_row[a] = n / len(votes)
        probs_out.append(p_row)
    return torch.tensor(results, dtype=torch.long), torch.stack(probs_out, dim=0)


def mode_softmax_avg_noflat_tiebreak(stacked: torch.Tensor, margin: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """If flat wins argmax but the best non-flat is within `margin` of flat, pick best non-flat."""
    probs = F.softmax(stacked, dim=-1).mean(dim=1)  # [T, A]
    argmax = torch.argmax(probs, dim=-1)
    for t in range(argmax.shape[0]):
        if int(argmax[t].item()) == 0:
            non_flat = probs[t].clone()
            non_flat[0] = -1
            best_nf = int(torch.argmax(non_flat).item())
            if probs[t, 0].item() - probs[t, best_nf].item() < margin:
                argmax[t] = best_nf
    return argmax, probs


def summarize(argmax_t: torch.Tensor, probs_t: torch.Tensor, name: str) -> dict:
    argmax = argmax_t.tolist()
    flat_count = sum(1 for a in argmax if a == 0)
    T = len(argmax)
    top = Counter(argmax).most_common(5)
    # margin between top1 and flat
    top1_probs = [float(probs_t[t, int(argmax_t[t])].item()) for t in range(T)]
    flat_probs = [float(probs_t[t, 0].item()) for t in range(T)]
    return {
        "mode": name,
        "flat_fraction": flat_count / max(1, T),
        "flat_count": flat_count,
        "n_steps": T,
        "mean_top1_prob": float(np.mean(top1_probs)),
        "mean_flat_prob": float(np.mean(flat_probs)),
        "top_actions": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--num-windows", type=int, default=3)
    args = parser.parse_args()

    data = read_mktd(Path(args.val_data).resolve())
    num_symbols = data.num_symbols
    T = data.num_timesteps

    starts = []
    for i in range(args.num_windows):
        s = T - (args.window_days + 1) * (i + 1)
        if s < 0:
            break
        starts.append(s)
    starts.sort()

    obs_list: list[np.ndarray] = []
    for s in starts:
        w = _slice_window(data, start=s, steps=args.window_days)

        def _zero(obs: np.ndarray) -> int:
            obs_list.append(obs.copy())
            return 0

        simulate_daily_policy(
            w, _zero, max_steps=args.window_days,
            fee_rate=0.001, slippage_bps=5.0, fill_buffer_bps=5.0, max_leverage=1.0,
        )
    print(f"collected {len(obs_list)} obs")

    ckpts = [DEFAULT_CHECKPOINT] + list(DEFAULT_EXTRA_CHECKPOINTS)
    policies = [load_policy(c, num_symbols=num_symbols, device=torch.device("cpu")).policy for c in ckpts]

    stacked = collect_stacked_logits(policies, obs_list, num_symbols=num_symbols)  # [T, N, A]
    print(f"stacked shape: {tuple(stacked.shape)}")

    modes = [
        ("softmax_avg", *mode_softmax_avg(stacked)),
        ("logit_avg",   *mode_logit_avg(stacked)),
        ("majority_vote", *mode_majority_vote(stacked)),
        ("softmax_avg_tiebreak_2pct", *mode_softmax_avg_noflat_tiebreak(stacked, margin=0.02)),
        ("softmax_avg_tiebreak_5pct", *mode_softmax_avg_noflat_tiebreak(stacked, margin=0.05)),
    ]

    print(f"\n{'mode':<32}  flat_frac  n_flat  mean_top1  mean_flat  top5")
    for name, argmax, probs in modes:
        s = summarize(argmax, probs, name)
        top = ", ".join(f"a{a}:{c}" for a, c in s["top_actions"])
        print(f"{name:<32}  {s['flat_fraction']:>8.1%}  {s['flat_count']:>6}  "
              f"{s['mean_top1_prob']:>8.3f}  {s['mean_flat_prob']:>8.3f}  {top}")


if __name__ == "__main__":
    main()
