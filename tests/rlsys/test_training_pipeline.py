import math

import numpy as np
import pandas as pd

from rlsys.config import DataConfig, MarketConfig, PolicyConfig, TrainingConfig
from rlsys.data import prepare_features
from rlsys.market_environment import MarketEnvironment
from rlsys.policy import ActorCriticPolicy
from rlsys.training import PPOTrainer


def _make_dataframe(length: int = 256) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="H")
    base_price = 100 + np.sin(np.linspace(0, 20, length)) * 2
    data = {
        "open": base_price + np.random.normal(0, 0.1, size=length),
        "high": base_price + 0.5,
        "low": base_price - 0.5,
        "close": base_price + np.random.normal(0, 0.1, size=length),
        "volume": np.random.uniform(1_000, 5_000, size=length),
    }
    return pd.DataFrame(data, index=index)


def test_trainer_produces_finite_metrics():
    df = _make_dataframe(160)
    data_config = DataConfig(window_size=16)
    prepared = prepare_features(df, data_config)
    prices = prepared.targets.numpy()
    features = prepared.features.numpy()

    market_config = MarketConfig(initial_capital=50_000.0, max_leverage=1.5, risk_aversion=0.01)
    env = MarketEnvironment(prices=prices, features=features, config=market_config)

    policy_config = PolicyConfig(hidden_sizes=(64, 64), dropout=0.0)
    policy = ActorCriticPolicy(observation_dim=env.observation_space.shape[0], config=policy_config)

    training_config = TrainingConfig(
        total_timesteps=64,
        rollout_steps=32,
        num_epochs=2,
        minibatch_size=8,
        gamma=0.98,
        gae_lambda=0.9,
        use_amp=False,
        seed=7,
    )

    trainer = PPOTrainer(env, policy, training_config)
    logs = next(trainer.train())

    assert all(math.isfinite(value) for value in logs.values()), logs
    assert "loss_policy" in logs
    assert "episode_reward" in logs
    assert "episode_sortino" in logs

    eval_metrics = trainer.evaluate(num_episodes=2)
    assert set(eval_metrics.keys()) == {"eval_return_mean", "eval_return_std", "eval_sharpe_mean"}
    assert all(math.isfinite(value) for value in eval_metrics.values())


def test_linear_lr_schedule_updates_learning_rate():
    df = _make_dataframe(120)
    data_config = DataConfig(window_size=16)
    prepared = prepare_features(df, data_config)
    prices = prepared.targets.numpy()
    features = prepared.features.numpy()

    market_config = MarketConfig(initial_capital=25_000.0, max_leverage=1.0, risk_aversion=0.0)
    env = MarketEnvironment(prices=prices, features=features, config=market_config)

    policy_config = PolicyConfig(hidden_sizes=(32, 32), dropout=0.0)
    policy = ActorCriticPolicy(observation_dim=env.observation_space.shape[0], config=policy_config)

    training_config = TrainingConfig(
        total_timesteps=64,
        rollout_steps=32,
        num_epochs=1,
        minibatch_size=8,
        use_amp=False,
        seed=3,
        lr_schedule="linear",
    )

    trainer = PPOTrainer(env, policy, training_config)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    logs = next(trainer.train())
    assert logs["learning_rate"] < initial_lr


def test_trainer_can_disable_observation_normalization():
    df = _make_dataframe(80)
    prepared = prepare_features(df, DataConfig(window_size=8))
    prices = prepared.targets.numpy()
    features = prepared.features.numpy()

    env = MarketEnvironment(
        prices=prices,
        features=features,
        config=MarketConfig(initial_capital=10_000.0, max_leverage=1.0, risk_aversion=0.0),
    )
    policy = ActorCriticPolicy(
        observation_dim=env.observation_space.shape[0],
        config=PolicyConfig(hidden_sizes=(16, 16), dropout=0.0),
    )
    training_config = TrainingConfig(
        total_timesteps=32,
        rollout_steps=16,
        minibatch_size=8,
        num_epochs=1,
        use_amp=False,
        normalize_observations=False,
    )
    trainer = PPOTrainer(env, policy, training_config)
    assert trainer._normalizer is None
