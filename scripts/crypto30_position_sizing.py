#!/usr/bin/env python3
"""Test position sizing based on BTC distance from MA15.
Instead of binary (trade/no-trade), scale allocation by how far above MA BTC is.
"""
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


def run_full_period(data, policies, S, btc_close, strategy="binary_ma15",
                    decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    """Run full period with different regime strategies.

    Strategies:
    - "none": no filter
    - "binary_ma15": flat when BTC < MA15, full when BTC > MA15
    - "scaled_ma15": scale allocation by (BTC/MA15 - 1), clamp [0, 1]
    - "scaled_ma15_aggressive": scale by 5*(BTC/MA15 - 1), clamp [0, 1]
    - "ewma_trend": BTC EMA8/EMA21 crossover
    """
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))
    step_counter = [0]

    # Precompute EMA for ewma_trend
    if "ewma" in strategy:
        ema8 = np.zeros(T)
        ema21 = np.zeros(T)
        ema8[0] = btc_close[0]
        ema21[0] = btc_close[0]
        a8, a21 = 2/9, 2/22
        for t in range(1, T):
            ema8[t] = a8 * btc_close[t] + (1-a8) * ema8[t-1]
            ema21[t] = a21 * btc_close[t] + (1-a21) * ema21[t-1]

    def get_regime_scale(t):
        if strategy == "none":
            return 1.0
        elif strategy == "binary_ma15":
            if t < 15:
                return 1.0
            ma = float(np.mean(btc_close[t-14:t+1]))
            return 1.0 if btc_close[t] > ma else 0.0
        elif strategy == "scaled_ma15":
            if t < 15:
                return 1.0
            ma = float(np.mean(btc_close[t-14:t+1]))
            ratio = btc_close[t] / ma - 1.0
            return float(np.clip(ratio / 0.05, 0.0, 1.0))  # full at +5% above MA
        elif strategy == "scaled_ma15_aggressive":
            if t < 15:
                return 1.0
            ma = float(np.mean(btc_close[t-14:t+1]))
            ratio = btc_close[t] / ma - 1.0
            return float(np.clip(ratio / 0.02, 0.0, 1.0))  # full at +2% above MA
        elif strategy == "ewma_trend":
            if t < 21:
                return 1.0
            return 1.0 if ema8[t] > ema21[t] else 0.0
        return 1.0

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1
        scale = get_regime_scale(t)
        if scale <= 0:
            action_now = 0
        else:
            obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
            with torch.no_grad():
                probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
                avg = torch.stack(probs).mean(dim=0)
                masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
                action_now = int(torch.argmax(masked, dim=-1).item())
            # For scaled strategies with partial allocation, we can't easily
            # reduce position size in the sim (it uses discrete actions).
            # So for scale < 1, use probabilistic entry: trade with prob=scale
            if 0 < scale < 1 and action_now != 0:
                if np.random.random() > scale:
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
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    return ret, sort, dd


def main():
    data = read_mktd(str(VAL_BIN))
    S = data.num_symbols
    T = data.num_timesteps
    print(f"data: {T}d x {S}sym\n")

    btc_idx = next(i for i, s in enumerate(data.symbols) if "BTC" in s)
    btc_close = data.prices[:, btc_idx, P_CLOSE].astype(np.float64)

    policies = []
    for cp in PROD:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        policies.append(loaded.policy)

    strategies = ["none", "binary_ma15", "scaled_ma15", "scaled_ma15_aggressive", "ewma_trend"]

    print(f"{'Strategy':<30} {'Ret%':>10} {'Sort':>8} {'DD%':>8}")
    print("-" * 60)

    # Run 5 seeds for scaled strategies (which use randomness)
    for strat in strategies:
        if "scaled" in strat:
            rets, sorts, dds = [], [], []
            for seed in range(5):
                np.random.seed(seed)
                r, s, d = run_full_period(data, policies, S, btc_close, strategy=strat)
                rets.append(r)
                sorts.append(s)
                dds.append(d)
            print(f"{strat:<30} {np.mean(rets)*100:>+10.2f}% {np.mean(sorts):>8.2f} {np.mean(dds)*100:>8.2f}% (5-seed avg)")
        else:
            r, s, d = run_full_period(data, policies, S, btc_close, strategy=strat)
            print(f"{strat:<30} {r*100:>+10.2f}% {s:>8.2f} {d*100:>8.2f}%")


if __name__ == "__main__":
    main()
