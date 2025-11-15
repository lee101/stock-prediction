import json
from pathlib import Path

import pandas as pd
import torch

from neuralpricingstrategy.data import build_pricing_dataset, split_dataset_by_date
from neuralpricingstrategy.runtime import NeuralPricingAdjuster
from neuralpricingstrategy.trainer import PricingTrainingConfig, train_pricing_model


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-09-01", periods=4, freq="D", tz="UTC")
    base_low = pd.Series([100.0, 101.0, 102.0, 103.0])
    base_high = pd.Series([110.0, 111.0, 112.0, 113.0])
    return pd.DataFrame(
        {
            "symbol": ["NNN"] * len(timestamps),
            "timestamp": timestamps,
            "maxdiffalwayson_low_price": base_low,
            "maxdiffalwayson_high_price": base_high,
            "maxdiffprofit_low_price": base_low * 1.01,
            "maxdiffprofit_high_price": base_high * 1.02,
            "maxdiff_return": 0.01,
            "maxdiffalwayson_return": 0.0,
            "close": 100.0,
            "predicted_close": 101.0,
        }
    )


def _save_run_artifacts(run_dir: Path, dataset, result) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), run_dir / "pricing_model.pt")
    (run_dir / "feature_spec.json").write_text(json.dumps(dataset.feature_spec.to_dict()))
    (run_dir / "run_config.json").write_text(json.dumps({"clamp_pct": dataset.clamp_pct}))


def test_neural_pricing_adjuster_infers_prices(tmp_path):
    dataset = build_pricing_dataset(_frame())
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.25)
    config = PricingTrainingConfig(epochs=3, batch_size=2, learning_rate=1e-2)
    result = train_pricing_model(train_ds, validation_dataset=val_ds, config=config)
    run_dir = tmp_path / "neuralpricing" / "run"
    _save_run_artifacts(run_dir, dataset, result)

    adjuster = NeuralPricingAdjuster(run_dir=run_dir, device="cpu")
    sample_payload = dataset.frame.iloc[0].to_dict()
    adjustment = adjuster.adjust(sample_payload, symbol="NNN")
    assert adjustment is not None
    assert adjustment.low_price > 0
    assert adjustment.high_price > 0
