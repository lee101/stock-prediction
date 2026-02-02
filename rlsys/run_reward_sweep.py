from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rlsys.config import DataConfig, MarketConfig, PolicyConfig, TrainingConfig
from rlsys.data import prepare_features
from rlsys.market_environment import MarketEnvironment
from rlsys.policy import ActorCriticPolicy
from rlsys.training import PPOTrainer


def _parse_modes(raw: str) -> List[str]:
    modes = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            modes.append(token)
    return modes


def run_sweep(args: argparse.Namespace) -> Dict[str, object]:
    df = pd.read_csv(args.csv)
    df.columns = [col.lower() for col in df.columns]
    data_cfg = DataConfig()
    prepared = prepare_features(df, data_cfg)

    results: Dict[str, object] = {
        "csv": args.csv,
        "reward_modes": [],
        "config": {
            "data": asdict(data_cfg),
            "policy": asdict(PolicyConfig()),
            "training": asdict(TrainingConfig(total_timesteps=args.total_timesteps, rollout_steps=args.rollout_steps, num_epochs=args.num_epochs, minibatch_size=args.minibatch_size, device=args.device)),
        },
    }

    modes = _parse_modes(args.reward_modes)
    for mode in modes:
        market_cfg = MarketConfig(
            initial_capital=args.initial_capital,
            max_leverage=args.max_leverage,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            market_impact=args.market_impact,
            risk_aversion=args.risk_aversion,
            reward_mode=mode,
            drawdown_penalty=args.drawdown_penalty,
            volatility_penalty=args.volatility_penalty,
            sharpe_clip=args.sharpe_clip,
            sharpe_eps=args.sharpe_eps,
        )
        env = MarketEnvironment(
            prices=prepared.targets.numpy(),
            features=prepared.features.numpy(),
            config=market_cfg,
        )
        policy = ActorCriticPolicy(env.observation_space.shape[0], PolicyConfig())
        trainer = PPOTrainer(
            env=env,
            policy=policy,
            training_config=TrainingConfig(
                total_timesteps=args.total_timesteps,
                rollout_steps=args.rollout_steps,
                num_epochs=args.num_epochs,
                minibatch_size=args.minibatch_size,
                device=args.device,
            ),
            market_config=market_cfg,
        )
        final_log: Dict[str, float] = {}
        for log in trainer.train():
            final_log = log
        results["reward_modes"].append(
            {
                "mode": mode,
                "market_config": asdict(market_cfg),
                "metrics": final_log,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO sweeps for different reward modes.")
    parser.add_argument("--csv", default="trainingdatahourlybinance/SOLUSDT.csv")
    parser.add_argument("--reward-modes", default="raw,risk_adjusted,sharpe_like")
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--transaction-cost", type=float, default=0.0002)
    parser.add_argument("--slippage", type=float, default=0.0001)
    parser.add_argument("--market-impact", type=float, default=0.0)
    parser.add_argument("--risk-aversion", type=float, default=0.01)
    parser.add_argument("--drawdown-penalty", type=float, default=0.2)
    parser.add_argument("--volatility-penalty", type=float, default=0.0)
    parser.add_argument("--sharpe-clip", type=float, default=0.1)
    parser.add_argument("--sharpe-eps", type=float, default=1e-8)
    parser.add_argument("--output", default="rlsys/results/reward_sweep.json")
    args = parser.parse_args()

    results = run_sweep(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved sweep results: {output_path}")


if __name__ == "__main__":
    main()
