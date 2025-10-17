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
    crypto_net_returns = []
    non_crypto_net_returns = []
    crypto_values = [1.0]
    non_crypto_values = [1.0]
    crypto_weights = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        turnovers.append(info["turnover"])
        trading_costs.append(info["trading_cost"])
        drawdowns.append(info["drawdown"])
        net_return = info["net_return"]
        net_returns.append(net_return)
        portfolio_values.append(info.get("portfolio", info.get("portfolio_value", env.portfolio_value)))

        crypto_net = float(info.get("net_return_crypto", 0.0))
        non_crypto_net = float(info.get("net_return_non_crypto", net_return - crypto_net))
        crypto_net_returns.append(crypto_net)
        non_crypto_net_returns.append(non_crypto_net)

        last_crypto_value = crypto_values[-1]
        last_non_crypto_value = non_crypto_values[-1]
        crypto_values.append(float(last_crypto_value * max(1e-6, 1.0 + crypto_net)))
        non_crypto_values.append(float(last_non_crypto_value * max(1e-6, 1.0 + non_crypto_net)))

        crypto_weights.append(float(info.get("weight_crypto", 0.0)))

        if terminated or truncated:
            break

    portfolio_values = np.asarray(portfolio_values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    turnovers = np.asarray(turnovers, dtype=np.float32)
    trading_costs = np.asarray(trading_costs, dtype=np.float32)
    drawdowns = np.asarray(drawdowns, dtype=np.float32)
    net_returns = np.asarray(net_returns, dtype=np.float32)
    crypto_net_returns = np.asarray(crypto_net_returns, dtype=np.float32) if crypto_net_returns else np.array([], dtype=np.float32)
    non_crypto_net_returns = np.asarray(non_crypto_net_returns, dtype=np.float32) if non_crypto_net_returns else np.array([], dtype=np.float32)
    crypto_values = np.asarray(crypto_values, dtype=np.float32)
    non_crypto_values = np.asarray(non_crypto_values, dtype=np.float32)

    final_value = float(portfolio_values[-1])
    cumulative_return = float(final_value - 1.0)
    avg_turnover = float(turnovers.mean()) if turnovers.size else 0.0
    avg_cost = float(trading_costs.mean()) if trading_costs.size else 0.0
    max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
    avg_reward = float(rewards.mean()) if rewards.size else 0.0
    total_steps = rewards.size

    final_crypto_value = float(crypto_values[-1])
    final_non_crypto_value = float(non_crypto_values[-1])
    avg_crypto_weight = float(np.mean(crypto_weights)) if crypto_weights else 0.0
    avg_crypto_net = float(crypto_net_returns.mean()) if crypto_net_returns.size else 0.0
    avg_non_crypto_net = float(non_crypto_net_returns.mean()) if non_crypto_net_returns.size else 0.0

    return {
        "final_portfolio_value": final_value,
        "cumulative_return": cumulative_return,
        "average_turnover": avg_turnover,
        "average_trading_cost": avg_cost,
        "max_drawdown": max_drawdown,
        "average_log_reward": avg_reward,
        "total_steps": total_steps,
        "final_portfolio_value_crypto_only": final_crypto_value,
        "cumulative_return_crypto_only": float(final_crypto_value - 1.0),
        "final_portfolio_value_non_crypto": final_non_crypto_value,
        "cumulative_return_non_crypto": float(final_non_crypto_value - 1.0),
        "average_net_return_crypto": avg_crypto_net,
        "average_net_return_non_crypto": avg_non_crypto_net,
        "average_crypto_weight": avg_crypto_weight,
    }


__all__ = ["evaluate_trained_policy"]
