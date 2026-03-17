from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from compare_hf_pufferlib_marketsim import evaluate_hf_checkpoint, evaluate_puffer_checkpoint
from hftraining.data_utils import StockDataProcessor
from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel
from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig
from pufferlibtraining3.models import MarketPolicy, PolicyConfig


def _write_symbol_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=24, freq="D"),
            "open": [100.0 + i for i in range(24)],
            "high": [101.0 + i for i in range(24)],
            "low": [99.0 + i for i in range(24)],
            "close": [100.5 + i for i in range(24)],
            "volume": [1_000_000 + i for i in range(24)],
        }
    )
    csv_path = path / "AAPL.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def _zero_module_params(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.data.zero_()


def test_evaluate_hf_checkpoint_zero_allocation_returns_flat(tmp_path: Path) -> None:
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    csv_path = _write_symbol_csv(data_root)

    frame = pd.read_csv(csv_path)
    processor = StockDataProcessor(sequence_length=5, prediction_horizon=1, use_toto_forecasts=False)
    features = processor.prepare_features(frame, symbol="AAPL")
    processor.fit_scalers(features)

    checkpoint_dir = tmp_path / "hf_run"
    checkpoint_dir.mkdir()
    processor_path = checkpoint_dir / "data_processor.pkl"
    processor.save_scalers(str(processor_path))

    config = HFTrainingConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        dropout_rate=0.0,
        sequence_length=5,
        prediction_horizon=1,
        use_gradient_checkpointing=False,
        output_dir=str(checkpoint_dir),
        logging_dir=str(checkpoint_dir / "logs"),
    )
    model = TransformerTradingModel(config, input_dim=features.shape[1])
    _zero_module_params(model)

    checkpoint_path = checkpoint_dir / "final_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "input_dim": features.shape[1],
        },
        checkpoint_path,
    )

    result = evaluate_hf_checkpoint(
        checkpoint_path=checkpoint_path,
        processor_path=processor_path,
        symbol="AAPL",
        data_root=data_root,
        mode="open_close",
        device="cpu",
    )

    assert result.framework == "hf"
    assert result.steps > 0
    assert result.total_return == 0.0
    assert result.num_trade_steps == 0
    assert result.fill_rate == 0.0


def test_evaluate_puffer_checkpoint_zero_policy_returns_flat(tmp_path: Path) -> None:
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    csv_path = _write_symbol_csv(data_root)

    frame = pd.read_csv(csv_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date")

    env_cfg = MarketEnvConfig(
        context_len=5,
        mode="open_close",
        data_root=str(data_root),
        symbol="AAPL",
        trading_fee=0.0005,
        slip_bps=5.0,
        intraday_leverage_max=4.0,
        overnight_leverage_max=2.0,
        device="cpu",
    )
    env = MarketEnv(
        prices=torch.tensor(frame[["open", "high", "low", "close", "volume"]].to_numpy(), dtype=torch.float32),
        exog=None,
        price_columns=("open", "high", "low", "close", "volume"),
        cfg=env_cfg,
    )
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space

    policy_cfg = PolicyConfig(hidden_size=16, actor_layers=(16,), critic_layers=(16,), dropout_p=0.0, layer_norm=False)
    policy = MarketPolicy(env, policy_cfg)
    _zero_module_params(policy)

    checkpoint_path = tmp_path / "puffer_policy.pt"
    torch.save(policy.state_dict(), checkpoint_path)

    summary_path = tmp_path / "puffer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_path": str(checkpoint_path),
                "env_config": asdict(env_cfg),
                "policy_config": asdict(policy_cfg),
            },
            indent=2,
        )
    )

    result = evaluate_puffer_checkpoint(
        summary_json=summary_path,
        symbol="AAPL",
        data_root=data_root,
        device="cpu",
    )

    assert result.framework == "pufferlib"
    assert result.steps > 0
    assert result.total_return == 0.0
    assert result.num_trade_steps == 0
    assert result.fill_rate == 0.0
