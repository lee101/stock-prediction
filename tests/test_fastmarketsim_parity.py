from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fastmarketsim import FastMarketEnv
from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig


def _load_price_tensor(symbol: str, data_root: str):
    frame = pd.read_csv(f"{data_root}/{symbol}.csv")
    frame.columns = [str(c).lower() for c in frame.columns]
    cols = [
        col
        for col in frame.columns
        if col in {"open", "high", "low", "close"} or pd.api.types.is_numeric_dtype(frame[col])
    ]
    values = frame[cols].to_numpy(dtype=np.float32)
    return torch.from_numpy(values), tuple(cols)


def test_fast_env_matches_python_env():
    prices, columns = _load_price_tensor("AAPL", "trainingdata")
    cfg = MarketEnvConfig(context_len=64, horizon=1, device="cpu")

    py_env = MarketEnv(prices=prices, price_columns=columns, cfg=cfg)
    fast_env = FastMarketEnv(prices=prices, price_columns=columns, cfg=cfg, device="cpu")

    rng = np.random.default_rng(1234)
    actions = rng.uniform(-1.0, 1.0, size=256).astype(np.float32)

    py_obs, _ = py_env.reset()
    fast_obs, _ = fast_env.reset()
    np.testing.assert_allclose(py_obs, fast_obs, rtol=1e-5, atol=1e-6)

    py_metrics = {"reward": [], "gross": [], "trading_cost": [], "financing_cost": [], "equity": []}
    fast_metrics = {key: [] for key in py_metrics}

    for action in actions:
        py_obs, py_reward, py_done, py_truncated, py_info = py_env.step(action)
        fast_obs, fast_reward, fast_done, fast_truncated, fast_info = fast_env.step(action)

        np.testing.assert_allclose(py_obs, fast_obs, rtol=1e-5, atol=1e-6)

        py_metrics["reward"].append(py_reward)
        fast_metrics["reward"].append(fast_reward)
        py_metrics["gross"].append(py_info.get("gross_pnl", 0.0))
        fast_metrics["gross"].append(fast_info.get("gross_pnl", 0.0))

        py_metrics["trading_cost"].append(py_info.get("trading_cost", 0.0))
        fast_trade_cost = fast_info.get("trading_cost", 0.0) + fast_info.get("deleverage_cost", 0.0)
        fast_metrics["trading_cost"].append(fast_trade_cost)

        py_metrics["financing_cost"].append(py_info.get("financing_cost", 0.0))
        fast_metrics["financing_cost"].append(fast_info.get("financing_cost", 0.0))

        py_equity = float(py_env.equity.detach().cpu().item())
        py_metrics["equity"].append(py_info.get("equity", py_equity))
        fast_metrics["equity"].append(fast_info.get("equity", 0.0))

        if py_done or py_truncated or fast_done or fast_truncated:
            break

    for key, py_values in py_metrics.items():
        fast_values = fast_metrics[key]
        np.testing.assert_allclose(py_values, fast_values, rtol=1e-4, atol=1e-5, err_msg=f"mismatch in {key}")

    py_env.close()
    fast_env.close()
