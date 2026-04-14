#!/usr/bin/env python3
"""Walk-forward analysis of crypto30 prod ensemble on val data.

Slides a 30-day window across 192-day val set to test robustness.
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


def run_ensemble_window(data, policies, S, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
            avg = torch.stack(probs).mean(dim=0)
            masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
            action_now = int(torch.argmax(masked, dim=-1).item())
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
    return eq


def main():
    data_full = read_mktd(str(VAL_BIN))
    S = data_full.num_symbols
    print(f"data: {data_full.num_timesteps}d x {S}sym\n")

    policies = []
    for cp in PROD:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        policies.append(loaded.policy)

    window = 30
    stride = 5
    n_windows = (data_full.num_timesteps - window) // stride + 1

    print(f"Walk-forward: {window}d windows, {stride}d stride, {n_windows} windows")
    print(f"{'Win':>4} {'Start':>6} {'End':>6} {'Ret%':>10} {'Sort':>8} {'MaxDD%':>8}")
    print("-" * 50)

    rets = []
    sorts = []
    dds = []

    for w in range(n_windows):
        start = w * stride
        end = start + window
        if end > data_full.num_timesteps:
            break

        data_w = MktdData(
            version=data_full.version,
            symbols=list(data_full.symbols),
            features=data_full.features[start:end].copy(),
            prices=data_full.prices[start:end].copy(),
            tradable=data_full.tradable[start:end].copy() if data_full.tradable is not None else None,
        )

        eq = run_ensemble_window(data_w, policies, S)
        ret = eq[-1] / eq[0] - 1
        dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
        neg = dr[dr < 0]
        sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
        peak = np.maximum.accumulate(eq)
        dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))

        rets.append(ret)
        sorts.append(sort)
        dds.append(dd)
        print(f"{w:>4} d{start:>4}-{end:<4} {ret*100:>+10.2f}% {sort:>8.2f} {dd*100:>8.2f}%")

    rets = np.array(rets)
    sorts = np.array(sorts)
    dds = np.array(dds)

    print(f"\n{'Stat':>12} {'Return':>10} {'Sortino':>10} {'MaxDD':>10}")
    print("-" * 45)
    print(f"{'Mean':>12} {rets.mean()*100:>+10.2f}% {sorts.mean():>10.2f} {dds.mean()*100:>10.2f}%")
    print(f"{'Median':>12} {np.median(rets)*100:>+10.2f}% {np.median(sorts):>10.2f} {np.median(dds)*100:>10.2f}%")
    print(f"{'Std':>12} {rets.std()*100:>10.2f}% {sorts.std():>10.2f} {dds.std()*100:>10.2f}%")
    print(f"{'Min':>12} {rets.min()*100:>+10.2f}% {sorts.min():>10.2f} {dds.max()*100:>10.2f}%")
    print(f"{'Max':>12} {rets.max()*100:>+10.2f}% {sorts.max():>10.2f} {dds.min()*100:>10.2f}%")
    print(f"{'% positive':>12} {(rets > 0).mean()*100:>10.0f}%")
    print(f"{'% sort>0':>12} {(sorts > 0).mean()*100:>10.0f}%")

    # Slippage sensitivity
    print(f"\nSlippage sensitivity (30d median return):")
    for slip in [0, 5, 10, 20]:
        slip_rets = []
        for w in range(n_windows):
            start = w * stride
            end = start + window
            if end > data_full.num_timesteps:
                break
            data_w = MktdData(
                version=data_full.version,
                symbols=list(data_full.symbols),
                features=data_full.features[start:end].copy(),
                prices=data_full.prices[start:end].copy(),
                tradable=data_full.tradable[start:end].copy() if data_full.tradable is not None else None,
            )
            eq = run_ensemble_window(data_w, policies, S, slippage_bps=slip)
            slip_rets.append(eq[-1] / eq[0] - 1)
        slip_rets = np.array(slip_rets)
        print(f"  {slip:>4}bp: med={np.median(slip_rets)*100:+.2f}% mean={slip_rets.mean()*100:+.2f}% %pos={100*(slip_rets>0).mean():.0f}%")


if __name__ == "__main__":
    main()
