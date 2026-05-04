#!/usr/bin/env python3
"""Sanity-check: does PortfolioBracketEnv grow equity under zero / random
actions on real val prices? If yes, the env has an accounting bug. If no,
the training is learning a real (or exploit) strategy.
"""
from __future__ import annotations

import sys

import torch


try:
    from scripts._gpu_env_bootstrap import ensure_gpu_trading_env
except ModuleNotFoundError:
    from _gpu_env_bootstrap import ensure_gpu_trading_env

GPU_ENV_PYTHON = ensure_gpu_trading_env(sys_module=sys)

import gpu_trading_env


def summarize(name, env, eq_before, eq_after):
    ret = (eq_after - eq_before) / eq_before.clamp_min(1e-6)
    ret = ret.clamp(-1.0, 10.0)
    print(f"[{name}] B={env.B} 20-step return: "
          f"mean={ret.mean().item():+.4f} "
          f"med={ret.median().item():+.4f} "
          f"p10={ret.quantile(0.1).item():+.4f} "
          f"p90={ret.quantile(0.9).item():+.4f} "
          f"final_eq_mean=${eq_after.mean().item():,.0f}")


def main():
    data = gpu_trading_env.load_bin_full("pufferlib_market/data/screened32_full_val.bin")
    prices = data["prices"]
    tradable = data["tradable"]
    print(f"[tape] T={data['T']} S={data['S']}")

    for policy_name in ("zero", "random_buy_only", "random_all", "max_buy_sell"):
        env = gpu_trading_env.make_portfolio_bracket(
            B=512, prices=prices, tradable_tape=tradable,
            params={
                "episode_len": 25,
                "fee_bps": gpu_trading_env.PRODUCTION_FEE_BPS,
                "fill_buffer_bps": gpu_trading_env.PRODUCTION_FILL_BUFFER_BPS,
                "max_leverage": 1.5,
            },
        )
        eq_before = env.state["equity"].clone()
        g = torch.Generator(device=env.prices.device).manual_seed(0)
        for t in range(20):
            if policy_name == "zero":
                a = torch.zeros(env.B, env.S, 4, device=env.prices.device, dtype=torch.float32)
            elif policy_name == "random_buy_only":
                a = torch.zeros(env.B, env.S, 4, device=env.prices.device, dtype=torch.float32)
                a[..., 2] = torch.rand(env.B, env.S, device=env.prices.device, generator=g) * 0.1
            elif policy_name == "random_all":
                a = torch.randn(env.B, env.S, 4, generator=g, device=env.prices.device) * 0.1
                a[..., 2].clamp_min_(0)
                a[..., 3].clamp_min_(0)
            elif policy_name == "max_buy_sell":
                # Both buy and sell at 0.5 → the "cycling" exploit if any.
                a = torch.zeros(env.B, env.S, 4, device=env.prices.device, dtype=torch.float32)
                a[..., 2] = 0.5
                a[..., 3] = 0.5
            env.step(a)
        eq_after = env.state["equity"]
        summarize(policy_name, env, eq_before, eq_after)


if __name__ == "__main__":
    main()
