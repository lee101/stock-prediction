import pandas as pd
import torch

from neuralpricingstrategy.data import build_pricing_dataset, split_dataset_by_date
from neuralpricingstrategy.models import PricingAdjustmentModel
import neuralpricingstrategy.trainer as trainer_module
from neuralpricingstrategy.trainer import PricingTrainingConfig, train_pricing_model


def _random_frame(rows: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2025-10-01", periods=rows, freq="D", tz="UTC")
    base_low = 100.0 + pd.Series(range(rows), dtype=float)
    base_high = 110.0 + pd.Series(range(rows), dtype=float)
    return pd.DataFrame(
        {
            "symbol": ["ZZZ"] * rows,
            "timestamp": timestamps,
            "maxdiffalwayson_low_price": base_low,
            "maxdiffalwayson_high_price": base_high,
            "maxdiffprofit_low_price": base_low * 1.01,
            "maxdiffprofit_high_price": base_high * 1.015,
            "maxdiff_return": 0.01,
            "maxdiffalwayson_return": 0.0,
            "close": 100.0,
            "predicted_close": 101.0,
        }
    )


def test_train_pricing_model_runs():
    dataset = build_pricing_dataset(_random_frame())
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.25)
    config = PricingTrainingConfig(epochs=5, batch_size=2, learning_rate=5e-3, device="cpu")
    result = train_pricing_model(train_ds, validation_dataset=val_ds, config=config)

    assert len(result.history) == config.epochs
    assert "train/loss" in result.final_metrics
    assert result.final_metrics["train/loss"] >= 0.0


def test_train_pricing_model_falls_back_to_cpu_on_auto_cuda_oom(monkeypatch):
    dataset = build_pricing_dataset(_random_frame())
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.25)
    config = PricingTrainingConfig(epochs=1, batch_size=2, learning_rate=5e-3)

    original_to = PricingAdjustmentModel.to
    calls: list[str] = []

    def flaky_to(self, device, *args, **kwargs):
        resolved = torch.device(device)
        calls.append(resolved.type)
        if resolved.type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(PricingAdjustmentModel, "to", flaky_to)

    result = train_pricing_model(train_ds, validation_dataset=val_ds, config=config)

    assert calls[:2] == ["cuda", "cpu"]
    assert next(result.model.parameters()).device.type == "cpu"
