"""
Evaluation helpers shared across GymRL scripts.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from gymrl.portfolio_env import PortfolioEnv


def evaluate_trained_policy(model, env: PortfolioEnv) -> Dict[str, float]:
    """
    Roll out ``model`` deterministically on ``env`` and collect portfolio stats.
    """

    obs, _ = env.reset(options={"start_index": env.start_index})
    rewards = []
    turnovers = []
    trading_costs = []
    drawdowns = []
    net_returns = []
    portfolio_values = [env.portfolio_value]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        turnovers.append(info["turnover"])
        trading_costs.append(info["trading_cost"])
        drawdowns.append(info["drawdown"])
        net_returns.append(info["net_return"])
        portfolio_values.append(info.get("portfolio", info.get("portfolio_value", env.portfolio_value)))
        if terminated or truncated:
            break

    portfolio_values = np.asarray(portfolio_values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    turnovers = np.asarray(turnovers, dtype=np.float32)
    trading_costs = np.asarray(trading_costs, dtype=np.float32)
    drawdowns = np.asarray(drawdowns, dtype=np.float32)
    net_returns = np.asarray(net_returns, dtype=np.float32)

    final_value = float(portfolio_values[-1])
    cumulative_return = float(final_value - 1.0)
    avg_turnover = float(turnovers.mean()) if turnovers.size else 0.0
    avg_cost = float(trading_costs.mean()) if trading_costs.size else 0.0
    max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
    avg_reward = float(rewards.mean()) if rewards.size else 0.0
    total_steps = rewards.size

    return {
        "final_portfolio_value": final_value,
        "cumulative_return": cumulative_return,
        "average_turnover": avg_turnover,
        "average_trading_cost": avg_cost,
        "max_drawdown": max_drawdown,
        "average_log_reward": avg_reward,
        "total_steps": total_steps,
    }


__all__ = ["evaluate_trained_policy"]
