from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch

from rl_trading.config import EnvConfig
from rl_trading.data_loader import load_market_data
from rl_trading.env import TradingEnv
from rl_trading.policy import TradingPolicy


def run_eval(policy_path: str, env_config: EnvConfig | None = None, device: str = "cpu",
             hidden_dim: int = 256):
    cfg = env_config or EnvConfig()
    market_data = load_market_data(cfg.symbols, cfg.data_root, cfg.validation_days)
    val_bars = market_data["n_bars"] - market_data["val_start"]

    eval_cfg = EnvConfig(
        symbols=cfg.symbols,
        data_root=cfg.data_root,
        initial_cash=cfg.initial_cash,
        fee_rate=cfg.fee_rate,
        max_hold_bars=cfg.max_hold_bars,
        episode_length=val_bars + 10,
        validation_days=cfg.validation_days,
    )

    env = TradingEnv(
        num_envs=1,
        env_config=eval_cfg,
        market_data=market_data,
        use_val=True,
        seed=0,
    )

    policy = TradingPolicy(env, hidden_dim=hidden_dim)
    state_dict = torch.load(policy_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)

    obs, _ = env.reset(seed=42)

    equity = cfg.initial_cash
    equities = [equity]
    actions_taken = []
    n_resets = 0

    for step in range(val_bars - 1):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            logits, _ = policy(obs_t)
            action = logits.argmax(dim=-1).cpu().numpy()
        obs, rewards, terminals, truncations, info = env.step(action)
        actions_taken.append(int(action[0]))

        rew = float(rewards[0])
        equity *= (1.0 + rew / 100.0)
        equities.append(equity)

        if terminals[0] or truncations[0]:
            n_resets += 1

    env.close()

    eq = pd.Series(equities)
    total_return = eq.iloc[-1] / cfg.initial_cash
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    returns = eq.pct_change().dropna()
    neg = returns[returns < 0]
    downside_std = neg.std() if len(neg) > 0 else 1e-8
    sortino = (returns.mean() / downside_std) * np.sqrt(24 * 365) if downside_std > 0 else 0

    from collections import Counter
    act_counts = Counter(actions_taken)
    n_sym = len(cfg.symbols)
    print(f"Val period: {val_bars} bars ({val_bars/24:.0f} days)")
    print(f"Total return: {total_return:.4f}x")
    print(f"Final equity: ${eq.iloc[-1]:.2f}")
    print(f"Max drawdown: {dd:.4f}")
    print(f"Sortino: {sortino:.2f}")
    print(f"Resets: {n_resets}")
    print(f"Actions: hold={act_counts.get(0,0)}", end="")
    for i in range(n_sym):
        print(f" buy_{cfg.symbols[i]}={act_counts.get(i+1,0)}", end="")
    for i in range(n_sym):
        print(f" sell_{cfg.symbols[i]}={act_counts.get(n_sym+i+1,0)}", end="")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--max-hold-bars", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()

    cfg = EnvConfig(fee_rate=args.fee_rate, max_hold_bars=args.max_hold_bars)
    run_eval(args.checkpoint, cfg, args.device, args.hidden_dim)


if __name__ == "__main__":
    main()
