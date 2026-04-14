#!/usr/bin/env python3
"""Test BTC regime filter + confidence gate combined on walk-forward."""
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


def run_window(data, policies, S, btc_close_global, w_start, regime_ma=0, min_conf=0.0,
               decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))
    step_counter = [0]

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1
        global_t = w_start + t

        # BTC regime filter
        if regime_ma > 0 and global_t >= regime_ma:
            ma_val = float(np.mean(btc_close_global[global_t - regime_ma + 1:global_t + 1]))
            if btc_close_global[global_t] <= ma_val:
                action_now = 0
                if decision_lag <= 0:
                    return action_now
                pending.append(action_now)
                if len(pending) <= decision_lag:
                    return 0
                return pending.popleft()

        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
            avg = torch.stack(probs).mean(dim=0)
            masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
            action_now = int(torch.argmax(masked, dim=-1).item())
            conf = float(avg[0, action_now].item())

        # Confidence gate
        if min_conf > 0 and action_now != 0 and conf < min_conf:
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
    return eq[-1] / eq[0] - 1


def main():
    data_full = read_mktd(str(VAL_BIN))
    S = data_full.num_symbols
    T = data_full.num_timesteps
    print(f"data: {T}d x {S}sym\n")

    btc_idx = next(i for i, s in enumerate(data_full.symbols) if "BTC" in s)
    btc_close = data_full.prices[:, btc_idx, P_CLOSE].astype(np.float64)

    policies = []
    for cp in PROD:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        policies.append(loaded.policy)

    window = 30
    stride = 5

    configs = [
        ("NoFilter", 0, 0.0),
        ("MA15 only", 15, 0.0),
        ("Conf0.20 only", 0, 0.20),
        ("MA15+Conf0.20", 15, 0.20),
        ("MA15+Conf0.25", 15, 0.25),
        ("MA20 only", 20, 0.0),
        ("MA20+Conf0.20", 20, 0.20),
    ]

    print(f"{'Config':<20} {'Mean%':>10} {'Med%':>10} {'%Pos':>8} {'Min%':>10} {'Max%':>10}")
    print("-" * 70)

    for label, regime_ma, min_conf in configs:
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
            ret = run_window(data_w, policies, S, btc_close, w_start,
                             regime_ma=regime_ma, min_conf=min_conf)
            rets.append(ret)
        rets = np.array(rets)
        print(f"{label:<20} {rets.mean()*100:>+10.2f}% {np.median(rets)*100:>+10.2f}% "
              f"{100*(rets>0).mean():>7.0f}% {rets.min()*100:>+10.2f}% {rets.max()*100:>+10.2f}%")


if __name__ == "__main__":
    main()
