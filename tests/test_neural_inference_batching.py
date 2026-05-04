import numpy as np
import pandas as pd
import torch

from binanceneural.data import FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame


class _DeterministicPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, features):
        self.forward_calls += 1
        return {"signal": features[..., :1].sum(dim=-1)}

    def decode_actions(self, outputs, *, reference_close, chronos_high, chronos_low):
        signal = outputs["signal"]
        buy_amount = torch.sigmoid(signal)
        sell_amount = 1.0 - buy_amount
        return {
            "buy_price": reference_close - 0.01 * signal,
            "sell_price": reference_close + 0.01 * signal,
            "buy_amount": buy_amount * 100.0,
            "sell_amount": sell_amount * 100.0,
            "trade_amount": buy_amount * 100.0,
        }


def test_generate_actions_from_frame_batches_model_forwards():
    timestamps = pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SOLUSD"] * len(timestamps),
            "reference_close": np.linspace(100.0, 108.0, len(timestamps), dtype=np.float32),
            "predicted_high_p50_h1": np.linspace(101.0, 109.0, len(timestamps), dtype=np.float32),
            "predicted_low_p50_h1": np.linspace(99.0, 107.0, len(timestamps), dtype=np.float32),
            "f1": np.linspace(0.0, 0.8, len(timestamps), dtype=np.float32),
            "f2": np.linspace(1.0, 1.8, len(timestamps), dtype=np.float32),
        }
    )
    normalizer = FeatureNormalizer(mean=np.zeros(2, dtype=np.float32), std=np.ones(2, dtype=np.float32))
    serial_model = _DeterministicPolicy()
    batched_model = _DeterministicPolicy()

    serial = generate_actions_from_frame(
        model=serial_model,
        frame=frame,
        feature_columns=["f1", "f2"],
        normalizer=normalizer,
        sequence_length=3,
        horizon=1,
        device=torch.device("cpu"),
        batch_size=1,
    )
    batched = generate_actions_from_frame(
        model=batched_model,
        frame=frame,
        feature_columns=["f1", "f2"],
        normalizer=normalizer,
        sequence_length=3,
        horizon=1,
        device=torch.device("cpu"),
        batch_size=4,
    )

    pd.testing.assert_frame_equal(serial, batched)
    assert serial_model.forward_calls == len(frame) - 3 + 1
    assert batched_model.forward_calls == 2
