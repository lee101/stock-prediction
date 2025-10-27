from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch

from hftraining.data_utils import StockDataProcessor
from hftraining.portfolio_rl_trainer import PortfolioAllocationModel, PortfolioRLConfig

from pufferlibinference.config import InferenceDataConfig, PufferInferenceConfig
from pufferlibinference.engine import PortfolioRLInferenceEngine


def _make_synthetic_frame(symbol: str, periods: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(hash(symbol) & 0xFFFF)
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    base_price = 50 + rng.normal(0, 0.5)
    drift = 0.001 if symbol.endswith("A") else -0.0005
    close = base_price * np.cumprod(1 + drift + rng.normal(0, 0.01, size=periods))
    open_price = close * (1 + rng.normal(0, 0.002, size=periods))
    high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.002, size=periods)))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.002, size=periods)))
    volume = rng.integers(low=500_000, high=1_500_000, size=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.mark.parametrize("sequence_length", [16])
def test_pufferlib_inference_end_to_end(tmp_path: Path, sequence_length: int) -> None:
    symbols = ["TESTA", "TESTB"]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    symbol_frames = {sym: _make_synthetic_frame(sym) for sym in symbols}
    for sym, frame in symbol_frames.items():
        frame.to_csv(data_dir / f"{sym}.csv", index=False)

    processor_path = tmp_path / "data_processor.pkl"
    processor = StockDataProcessor(sequence_length=sequence_length, prediction_horizon=1)
    feature_mats = []
    for sym, frame in symbol_frames.items():
        feats = processor.prepare_features(frame, symbol=sym)
        feature_mats.append(feats)
    processor.fit_scalers(np.vstack(feature_mats))
    processor.save_scalers(processor_path)

    feature_dim = processor.transform(feature_mats[0]).shape[1]
    assert feature_dim > 0

    input_dim = feature_dim * len(symbols)
    rl_config = PortfolioRLConfig(hidden_size=64, num_layers=2, num_heads=4, dropout=0.1)
    torch.manual_seed(1234)
    model = PortfolioAllocationModel(input_dim=input_dim, config=rl_config, num_assets=len(symbols))
    checkpoint_path = tmp_path / "allocator.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": rl_config,
            "symbols": symbols,
            "metrics": {},
            "best_epoch": -1,
            "best_val_profit": 0.0,
        },
        checkpoint_path,
    )

    data_cfg = InferenceDataConfig(symbols=symbols, data_dir=data_dir)
    inference_cfg = PufferInferenceConfig(
        checkpoint_path=checkpoint_path,
        processor_path=processor_path,
        transaction_cost_bps=5.0,
        leverage_limit=1.5,
    )

    engine = PortfolioRLInferenceEngine(inference_cfg, data_cfg)
    result = engine.simulate(initial_value=1.0)

    assert len(result.decisions) > 0
    assert result.equity_curve.size == len(result.decisions) + 1
    assert set(result.summary.keys()) == {
        "annualised_sharpe",
        "average_turnover",
        "cumulative_return",
        "final_value",
        "initial_value",
        "max_drawdown",
    }
    first_decision = result.decisions[0]
    assert set(first_decision.weights.keys()) == set(symbols)
    assert math.isfinite(result.summary["final_value"])


if __name__ == "__main__":  # pragma: no cover
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="pufferlib_test_"))
    try:
        test_pufferlib_inference_end_to_end(tmp_dir, sequence_length=16)
        print("Manual test run completed successfully.")
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
