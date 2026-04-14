#!/usr/bin/env python3
"""Test BTC MA15 regime filter at different slippage levels."""
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


def eval_full_period(data, policies, S, btc_close, regime_ma=0, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=3)
    step_counter = [0]

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1
        if regime_ma > 0 and t >= regime_ma:
            ma_val = float(np.mean(btc_close[t - regime_ma + 1:t + 1]))
            if btc_close[t] <= ma_val:
                action_now = 0
                pending.append(action_now)
                if len(pending) <= 2:
                    return 0
                return pending.popleft()

        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
            avg = torch.stack(probs).mean(dim=0)
            masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
            action_now = int(torch.argmax(masked, dim=-1).item())

        pending.append(action_now)
        if len(pending) <= 2:
            return 0
        return pending.popleft()

    result = simulate_daily_policy(
        data, policy_fn, max_steps=T - 1,
        fee_rate=0.001, slippage_bps=slippage_bps,
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

    print(f"{'Slip(bps)':<12} {'NoFilter Ret%':>15} {'NoFilter Sort':>14} {'NoFilter DD%':>13} | {'MA15 Ret%':>12} {'MA15 Sort':>10} {'MA15 DD%':>10}")
    print("-" * 100)

    for slip in [0, 5, 10, 20, 50]:
        r0, s0, d0 = eval_full_period(data, policies, S, btc_close, regime_ma=0, slippage_bps=slip)
        r1, s1, d1 = eval_full_period(data, policies, S, btc_close, regime_ma=15, slippage_bps=slip)
        print(f"{slip:<12} {r0*100:>+14.2f}% {s0:>14.2f} {d0*100:>12.2f}% | {r1*100:>+11.2f}% {s1:>10.2f} {d1*100:>10.2f}%")


if __name__ == "__main__":
    main()
