#!/usr/bin/env python3
"""Analyze ensemble confidence distribution across walk-forward windows."""
from __future__ import annotations
import collections
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, MktdData, INITIAL_CASH, P_CLOSE, simulate_daily_policy
from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts

VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"
PROD = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]


def run_window_with_confidence(data, policies, S, min_conf=0.0, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))
    step_counter = [0]
    confidences = []

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1
        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
            avg = torch.stack(probs).mean(dim=0)
            masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
            action_now = int(torch.argmax(masked, dim=-1).item())
            conf = float(avg[0, action_now].item())
            confidences.append((action_now, conf))
            if conf < min_conf and action_now != 0:
                action_now = 0

        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    result = simulate_daily_policy(
        data, policy_fn, max_steps=T - 1,
        fee_rate=fee_rate, slippage_bps=slippage_bps,
        fill_buffer_bps=5.0, max_leverage=1.0,
        periods_per_year=365.0, enable_drawdown_profit_early_exit=False,
    )
    eq = np.array(result.equity_curve) if result.equity_curve is not None and len(result.equity_curve) > 0 else np.ones(T) * INITIAL_CASH
    ret = eq[-1] / eq[0] - 1
    return ret, confidences


def main():
    data_full = read_mktd(str(VAL_BIN))
    S = data_full.num_symbols
    T = data_full.num_timesteps
    print(f"data: {T}d x {S}sym\n")

    policies = []
    for cp in PROD:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        policies.append(loaded.policy)

    window = 30
    stride = 5

    # First pass: get confidence distribution
    all_confs = []
    for w_start in range(0, T - window + 1, stride):
        w_end = w_start + window
        data_w = MktdData(
            version=data_full.version,
            symbols=list(data_full.symbols),
            features=data_full.features[w_start:w_end].copy(),
            prices=data_full.prices[w_start:w_end].copy(),
            tradable=data_full.tradable[w_start:w_end].copy() if data_full.tradable is not None else None,
        )
        _, confs = run_window_with_confidence(data_w, policies, S, min_conf=0.0)
        for action, conf in confs:
            all_confs.append((action, conf))

    non_flat = [(a, c) for a, c in all_confs if a != 0]
    flat = [(a, c) for a, c in all_confs if a == 0]
    print(f"Total signals: {len(all_confs)} (flat={len(flat)}, non-flat={len(non_flat)})")
    if non_flat:
        confs = [c for _, c in non_flat]
        print(f"Non-flat confidence: mean={np.mean(confs):.4f} med={np.median(confs):.4f} "
              f"p10={np.percentile(confs, 10):.4f} p25={np.percentile(confs, 25):.4f} "
              f"p75={np.percentile(confs, 75):.4f} p90={np.percentile(confs, 90):.4f}")

    # Second pass: sweep confidence gates
    print(f"\n{'MinConf':<10} {'Mean%':>10} {'Med%':>10} {'%Pos':>8} {'Min%':>10} {'Max%':>10}")
    print("-" * 60)
    for min_conf in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        rets = []
        for w_start in range(0, T - window + 1, stride):
            w_end = w_start + window
            data_w = MktdData(
                version=data_full.version,
                symbols=list(data_full.symbols),
                features=data_full.features[w_start:w_end].copy(),
                prices=data_full.prices[w_start:w_end].copy(),
                tradable=data_full.tradable[w_start:w_end].copy() if data_full.tradable is not None else None,
            )
            ret, _ = run_window_with_confidence(data_w, policies, S, min_conf=min_conf)
            rets.append(ret)
        rets = np.array(rets)
        print(f"{min_conf:<10.2f} {rets.mean()*100:>+10.2f}% {np.median(rets)*100:>+10.2f}% "
              f"{100*(rets>0).mean():>7.0f}% {rets.min()*100:>+10.2f}% {rets.max()*100:>+10.2f}%")


if __name__ == "__main__":
    main()
