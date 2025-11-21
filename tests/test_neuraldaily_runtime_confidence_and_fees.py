import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from neuraldailymarketsimulator.simulator import NeuralDailyMarketSimulator
from neuraldailytraining.config import DailyDatasetConfig
from neuraldailytraining.data import FeatureNormalizer
from neuraldailytraining.runtime import DailyTradingRuntime


class _DummyBuilder:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def build(self, symbol: str) -> pd.DataFrame:
        return self.frame


def test_simulator_uses_default_crypto_fee() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 11.0, 12.0],
            "high": [13.0, 13.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.0, 11.0, 12.0],
        }
    )

    class DummyRuntime:
        def __init__(self, builder: _DummyBuilder) -> None:
            self._builder = builder
            self.non_tradable: set[str] = set()

        def plan_batch(self, symbols, **_) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(symbol=symbol, buy_price=10.0, sell_price=12.0, trade_amount=1.0) for symbol in symbols
            ]

    runtime = DummyRuntime(_DummyBuilder(frame))

    simulator = NeuralDailyMarketSimulator(runtime, ("BTCUSD",), initial_cash=100.0)
    _, summary = simulator.run(days=1)

    expected_pnl = (12.0 * (1 - 0.0008)) - (10.0 * (1 + 0.0008))
    assert math.isclose(summary["pnl"], expected_pnl, rel_tol=0, abs_tol=1e-6)


def test_runtime_applies_confidence_gate_and_group_mask(tmp_path) -> None:
    dates = pd.date_range("2024-02-01", periods=2, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "date": dates,
            "feat": [1.0, 1.0],
            "reference_close": [10.0, 10.5],
            "chronos_high": [10.2, 10.7],
            "chronos_low": [9.8, 10.3],
            "asset_class": [1.0, 1.0],
        }
    )

    dataset_cfg = DailyDatasetConfig(
        symbols=("AAA", "BBB"),
        data_root=tmp_path,
        forecast_cache_dir=tmp_path,
        sequence_length=2,
    )

    runtime = DailyTradingRuntime.__new__(DailyTradingRuntime)
    runtime.dataset_config = dataset_cfg
    runtime.sequence_length = 2
    runtime.feature_columns = ("feat",)
    runtime.normalizer = FeatureNormalizer(mean=np.zeros(1, dtype=np.float32), std=np.ones(1, dtype=np.float32))
    runtime.device = torch.device("cpu")
    runtime.symbol_to_group_id = {}
    runtime.non_tradable = set()
    runtime._builder = _DummyBuilder(frame)
    runtime.risk_threshold = 1.0

    class FakeConfidenceModel:
        def __init__(self, base_trade: float, confidence: float) -> None:
            self.base_trade = base_trade
            self.confidence = confidence
            self.last_group_mask = None
            self.blocks = True  # trigger group mask path in runtime

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, batch_feats, group_mask=None):
            self.last_group_mask = group_mask
            logits = torch.full((*batch_feats.shape[:2], 1), self.confidence)
            return {"confidence_logits": logits}

        def decode_actions(
            self,
            outputs,
            *,
            reference_close,
            chronos_high,
            chronos_low,
            asset_class,
        ):
            trade = torch.full_like(reference_close, self.base_trade)
            confidence = torch.full_like(reference_close, self.confidence)
            return {
                "buy_price": reference_close * 0.99,
                "sell_price": reference_close * 1.01,
                "trade_amount": trade,
                "confidence": confidence,
            }

    runtime.model = FakeConfidenceModel(base_trade=0.8, confidence=0.25)

    plans = runtime.plan_batch(["AAA", "BBB"], as_of=dates[-1])

    assert runtime.model.last_group_mask is not None
    assert runtime.model.last_group_mask.shape == (2, 2)
    assert all(math.isclose(plan.trade_amount, 0.2, rel_tol=0, abs_tol=1e-6) for plan in plans)
