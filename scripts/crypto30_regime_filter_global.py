#!/usr/bin/env python3
"""Test BTC regime filter with GLOBAL MA (computed on full data, not per window).

Previous test computed MA within each 30d window which starved short windows
of lookback. This version computes the MA across the entire dataset first.
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


def run_window(data, policies, S, regime_mask, decision_lag=2, fee_rate=0.001, slippage_bps=5.0):
    T = data.num_timesteps
    pending = collections.deque(maxlen=max(1, decision_lag + 1))
    step_counter = [0]

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1
        if regime_mask is not None and not regime_mask[min(t, len(regime_mask) - 1)]:
            action_now = 0
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

    # Compute GLOBAL BTC MAs once
    for ma_period in [0, 10, 15, 20, 30]:
        if ma_period == 0:
            global_regime = None
            label = "NoFilter"
        else:
            global_regime = np.zeros(T, dtype=bool)
            btc_ma = np.full(T, np.nan)
            for t in range(ma_period - 1, T):
                btc_ma[t] = btc_close[t - ma_period + 1:t + 1].mean()
            for t in range(T):
                if not np.isnan(btc_ma[t]):
                    global_regime[t] = btc_close[t] > btc_ma[t]
            bull_pct = global_regime.mean() * 100
            label = f"BTC_MA{ma_period} (bull={bull_pct:.0f}%)"

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
            # Pass window-specific slice of GLOBAL regime mask
            regime_slice = global_regime[w_start:w_end] if global_regime is not None else None
            ret = run_window(data_w, policies, S, regime_slice)
            rets.append(ret)

        rets = np.array(rets)
        print(f"  {label:<30}: mean={rets.mean()*100:+.2f}% med={np.median(rets)*100:+.2f}% %pos={100*(rets>0).mean():.0f}% min={rets.min()*100:+.2f}% max={rets.max()*100:+.2f}%")


if __name__ == "__main__":
    main()
