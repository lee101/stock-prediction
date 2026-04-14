#!/usr/bin/env python3
"""Test BTC regime filter for crypto30 ensemble.

Idea: only take long positions when BTC > 20d MA (bull regime).
When bear, stay flat (action=0).
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


def metrics(eq):
    ret = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    neg = dr[dr < 0]
    sort = float(np.mean(dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(365)) if len(neg) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / np.maximum(peak, 1e-8)))
    return ret, sort, dd


def run_with_regime_filter(data, policies, S, btc_idx, ma_period=20,
                           decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    # Precompute BTC MA
    btc_close = data.prices[:, btc_idx, P_CLOSE].astype(np.float64)
    btc_ma = np.full(T, np.nan)
    for t in range(ma_period - 1, T):
        btc_ma[t] = btc_close[t - ma_period + 1:t + 1].mean()

    regime_bull = np.zeros(T, dtype=bool)
    for t in range(T):
        if not np.isnan(btc_ma[t]):
            regime_bull[t] = btc_close[t] > btc_ma[t]

    step_counter = [0]

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1

        if not regime_bull[min(t, T - 1)]:
            action_now = 0  # flat in bear
        else:
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
    bull_pct = regime_bull.mean() * 100
    return eq, bull_pct


def run_no_filter(data, policies, S, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
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
    data = read_mktd(str(VAL_BIN))
    S = data.num_symbols
    print(f"data: {data.num_timesteps}d x {S}sym")

    # Find BTC index
    btc_idx = None
    for i, sym in enumerate(data.symbols):
        if "BTC" in sym:
            btc_idx = i
            break
    if btc_idx is None:
        print("ERROR: BTC not found in symbols")
        return
    print(f"BTC index: {btc_idx} ({data.symbols[btc_idx]})\n")

    policies = []
    for cp in PROD:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device("cpu"))
        policies.append(loaded.policy)

    # Full period comparison
    print("=== Full period (192d) ===")
    eq_base = run_no_filter(data, policies, S)
    ret_base, sort_base, dd_base = metrics(eq_base)
    print(f"  No filter:  ret={ret_base*100:+.2f}% sort={sort_base:.2f} dd={dd_base*100:.2f}%")

    for ma in [10, 15, 20, 30, 50]:
        eq, bull_pct = run_with_regime_filter(data, policies, S, btc_idx, ma_period=ma)
        ret, sort, dd = metrics(eq)
        print(f"  BTC MA{ma:<3}: ret={ret*100:+.2f}% sort={sort:.2f} dd={dd*100:.2f}% (bull={bull_pct:.0f}%)")

    # Walk-forward with regime filter
    print(f"\n=== Walk-forward (30d windows) ===")
    window = 30
    stride = 5

    for ma in [0, 20]:
        rets = []
        for w_start in range(0, data.num_timesteps - window + 1, stride):
            w_end = w_start + window
            data_w = MktdData(
                version=data.version,
                symbols=list(data.symbols),
                features=data.features[w_start:w_end].copy(),
                prices=data.prices[w_start:w_end].copy(),
                tradable=data.tradable[w_start:w_end].copy() if data.tradable is not None else None,
            )
            if ma == 0:
                eq = run_no_filter(data_w, policies, S)
            else:
                eq, _ = run_with_regime_filter(data_w, policies, S, btc_idx, ma_period=ma)
            rets.append(eq[-1] / eq[0] - 1)

        rets = np.array(rets)
        label = f"MA{ma}" if ma > 0 else "NoFilter"
        print(f"  {label:<10}: mean={rets.mean()*100:+.2f}% med={np.median(rets)*100:+.2f}% %pos={100*(rets>0).mean():.0f}% min={rets.min()*100:+.2f}% max={rets.max()*100:+.2f}%")

    # Slippage test with best regime filter
    print(f"\n=== Slippage: NoFilter vs BTC MA20 ===")
    for slip in [0, 5, 10, 20]:
        eq_base = run_no_filter(data, policies, S, slippage_bps=slip)
        ret_base, _, _ = metrics(eq_base)
        eq_filt, _ = run_with_regime_filter(data, policies, S, btc_idx, ma_period=20, slippage_bps=slip)
        ret_filt, _, _ = metrics(eq_filt)
        print(f"  {slip:>4}bp: NoFilter={ret_base*100:+.2f}% BTC_MA20={ret_filt*100:+.2f}%")


if __name__ == "__main__":
    main()
