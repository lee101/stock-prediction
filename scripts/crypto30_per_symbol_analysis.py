#!/usr/bin/env python3
"""Analyze which symbols the crypto30 ensemble trades and their P&L contribution."""
from __future__ import annotations
import collections
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd, INITIAL_CASH, P_CLOSE, simulate_daily_policy
from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts

VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"
PROD = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]


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

    # Track which actions the ensemble takes
    action_counts = collections.Counter()
    pending = collections.deque(maxlen=3)
    step_counter = [0]
    actions_per_step = []

    # Track position-level P&L
    position_log = []  # (entry_day, exit_day, symbol, entry_price, exit_price, return)
    current_pos = {"symbol": None, "entry_day": 0, "entry_price": 0.0}

    def policy_fn(obs):
        t = step_counter[0]
        step_counter[0] += 1

        # BTC regime filter (MA15)
        if t >= 15:
            ma = float(np.mean(btc_close[t-14:t+1]))
            if btc_close[t] <= ma:
                action_now = 0
                pending.append(action_now)
                if len(pending) <= 2:
                    return 0
                executed = pending.popleft()
                actions_per_step.append((t, action_now, executed))
                return executed

        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
        with torch.no_grad():
            probs = [torch.softmax(p(obs_t)[0], dim=-1) for p in policies]
            avg = torch.stack(probs).mean(dim=0)
            masked = _mask_all_shorts(torch.log(avg + 1e-8), num_symbols=S, per_symbol_actions=1)
            action_now = int(torch.argmax(masked, dim=-1).item())

        pending.append(action_now)
        if len(pending) <= 2:
            return 0
        executed = pending.popleft()
        actions_per_step.append((t, action_now, executed))
        action_counts[executed] += 1
        return executed

    result = simulate_daily_policy(
        data, policy_fn, max_steps=T - 1,
        fee_rate=0.001, slippage_bps=5.0,
        fill_buffer_bps=5.0, max_leverage=1.0,
        periods_per_year=365.0, enable_drawdown_profit_early_exit=False,
    )
    eq = np.array(result.equity_curve) if result.equity_curve is not None and len(result.equity_curve) > 0 else np.ones(T) * INITIAL_CASH
    ret = eq[-1] / eq[0] - 1
    print(f"Full period with MA15: ret={ret*100:+.2f}%\n")

    # Analyze action distribution
    print("=== Action distribution ===")
    total = sum(action_counts.values())
    flat_count = action_counts[0]
    print(f"Flat (action=0): {flat_count}/{total} ({100*flat_count/total:.1f}%)")
    sym_actions = collections.Counter()
    for a, count in action_counts.items():
        if a == 0:
            continue
        if a <= S:
            sym = data.symbols[a - 1]
            sym_actions[sym] += count

    print(f"\nTop symbols by selection frequency:")
    for sym, count in sym_actions.most_common(15):
        pct = 100 * count / total
        print(f"  {sym:<12}: {count:>4} ({pct:>5.1f}%)")

    # Analyze per-symbol returns by tracking transitions in actions_per_step
    print(f"\n=== Per-symbol P&L analysis ===")
    symbol_pnl = collections.defaultdict(list)  # sym -> list of per-trade returns

    current_sym = None
    entry_day = 0
    entry_price = 0.0

    for step_data in actions_per_step:
        t, _, executed = step_data
        if executed == 0:
            new_sym = None
        elif executed <= S:
            new_sym = data.symbols[executed - 1]
        else:
            new_sym = None

        # Check for position change
        if new_sym != current_sym:
            if current_sym is not None and t < T:
                # Close old position
                sym_idx = list(data.symbols).index(current_sym)
                exit_price = float(data.prices[min(t, T-1), sym_idx, P_CLOSE])
                if entry_price > 0:
                    trade_ret = exit_price / entry_price - 1
                    symbol_pnl[current_sym].append(trade_ret)

            if new_sym is not None and t < T:
                # Open new position
                sym_idx = list(data.symbols).index(new_sym)
                entry_price = float(data.prices[min(t, T-1), sym_idx, P_CLOSE])
                entry_day = t

            current_sym = new_sym

    print(f"{'Symbol':<12} {'Trades':>8} {'MeanRet%':>10} {'WinRate%':>10} {'TotalRet%':>12}")
    print("-" * 55)
    for sym in sorted(symbol_pnl.keys(), key=lambda s: sum(symbol_pnl[s]), reverse=True):
        trades = symbol_pnl[sym]
        if not trades:
            continue
        mean_ret = np.mean(trades)
        win_rate = sum(1 for r in trades if r > 0) / len(trades)
        total_ret = np.prod([1 + r for r in trades]) - 1
        print(f"{sym:<12} {len(trades):>8} {mean_ret*100:>+10.2f}% {win_rate*100:>9.0f}% {total_ret*100:>+11.2f}%")


if __name__ == "__main__":
    main()
