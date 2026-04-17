from __future__ import annotations

import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import torch

import src.autoresearch_binance.prepare as binance_prepare
import src.autoresearch_crypto.prepare as crypto_prepare
import src.autoresearch_stock.prepare as stock_prepare
import src.autoresearch_stock.train as stock_train
import src.parallel_analysis as parallel_analysis


class _StockPrepareModel(torch.nn.Module):
    def forward(self, features: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        del symbol_ids
        return torch.full((features.shape[0], 3), 2.0, dtype=torch.float32)


class _StockTrainModel(torch.nn.Module):
    weak_action_scale = 1.0

    def predict_with_plan(
        self,
        features: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        batch_size = features.shape[0]
        del symbol_ids
        return (
            torch.full((batch_size, 3), 1.0, dtype=torch.float32),
            torch.full((batch_size, 5), 2.0, dtype=torch.float32),
            torch.full((batch_size, 1), 3.0, dtype=torch.float32),
            torch.full((batch_size, 3), 4.0, dtype=torch.float32),
        )


class _CryptoScenarioModel(torch.nn.Module):
    trade_amount_scale = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        return torch.zeros((features.shape[0], 1), dtype=torch.float32)

    def decode_actions(
        self,
        outputs: torch.Tensor,
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        del outputs, chronos_high, chronos_low
        seq_len = reference_close.shape[1]
        shape = (reference_close.shape[0], seq_len)
        return {
            "buy_price": torch.full(shape, 100.0, dtype=torch.float32),
            "sell_price": torch.full(shape, 101.0, dtype=torch.float32),
            "buy_amount": torch.full(shape, 0.5, dtype=torch.float32),
            "sell_amount": torch.zeros(shape, dtype=torch.float32),
        }


class _WarmupPipeline:
    def __call__(
        self,
        *,
        context: torch.Tensor,
        prediction_length: int,
        num_samples: int,
    ) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        assert context.ndim == 3
        assert prediction_length == 96
        assert num_samples == 2
        return torch.zeros((1,), dtype=torch.float32)


class _BinancePolicy(torch.nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 1024) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.hidden = hidden
        self.obs_size = obs_size

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D401 - torch API
        del state_dict, strict
        return None

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        batch_size = obs.shape[0]
        return (
            torch.zeros((batch_size, self.num_actions), dtype=torch.float32),
            torch.zeros((batch_size,), dtype=torch.float32),
        )


class _FakeBinanceBinding:
    def __init__(self) -> None:
        self.rew_buf: np.ndarray | None = None
        self.term_buf: np.ndarray | None = None
        self.trunc_buf: np.ndarray | None = None

    def shared(self, *, data_path: str) -> None:
        self.data_path = data_path

    def vec_init(self, obs_buf, act_buf, rew_buf, term_buf, trunc_buf, *args, **kwargs):
        del obs_buf, act_buf, args, kwargs
        self.rew_buf = rew_buf
        self.term_buf = term_buf
        self.trunc_buf = trunc_buf
        return 1

    def vec_reset(self, handle, seed):
        del handle, seed

    def vec_step(self, handle):
        del handle
        assert self.rew_buf is not None
        assert self.term_buf is not None
        assert self.trunc_buf is not None
        self.rew_buf[:] = 1.0
        self.term_buf[:] = 1
        self.trunc_buf[:] = 0

    def vec_close(self, handle):
        del handle


def test_stock_prepare_predict_in_batches_uses_inference_mode() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    symbol_ids = np.array([0, 1], dtype=np.int64)

    predictions = stock_prepare._predict_in_batches(
        _StockPrepareModel(),
        features=features,
        symbol_ids=symbol_ids,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert predictions.shape == (2, 3)
    assert predictions.dtype == np.float32


def test_stock_train_predict_ranked_batches_uses_inference_mode() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    symbol_ids = np.array([0, 1], dtype=np.int64)

    predictions, scores, budget_logits = stock_train._predict_ranked_batches(
        _StockTrainModel(),
        features=features,
        symbol_ids=symbol_ids,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert predictions.shape == (2, 3)
    assert scores.shape == (2,)
    assert budget_logits.shape == (2, 3)
    assert predictions.dtype == np.float32


def test_crypto_prepare_run_scenario_uses_inference_mode(monkeypatch) -> None:
    scenario = crypto_prepare.ScenarioData(
        name="BTCUSD_1h",
        symbol="BTCUSD",
        window_hours=1,
        features=np.array(
            [
                [1.0, 2.0],
                [1.5, 2.5],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        ),
        opens=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        highs=np.array([101.0, 102.0, 103.0], dtype=np.float32),
        lows=np.array([99.0, 100.0, 101.0], dtype=np.float32),
        closes=np.array([100.0, 101.0, 110.0], dtype=np.float32),
        reference_close=np.array([100.0, 100.5, 101.0], dtype=np.float32),
        chronos_high=np.array([101.0, 101.5, 102.0], dtype=np.float32),
        chronos_low=np.array([99.0, 99.5, 100.0], dtype=np.float32),
    )
    config = crypto_prepare.CryptoTaskConfig(
        sequence_length=2,
        initial_cash=100.0,
        can_short=False,
    )

    monkeypatch.setattr(
        crypto_prepare,
        "simulate_hourly_trades_binary",
        lambda **kwargs: SimpleNamespace(
            portfolio_values=torch.tensor([[100.0, 110.0]], dtype=torch.float32),
            executed_buys=torch.tensor([[1.0]], dtype=torch.float32),
            executed_sells=torch.tensor([[0.0]], dtype=torch.float32),
        ),
    )

    result = crypto_prepare._run_scenario(
        _CryptoScenarioModel(),
        scenario,
        config,
        torch.device("cpu"),
    )

    assert result["name"] == scenario.name
    assert result["symbol"] == scenario.symbol
    assert result["window_hours"] == scenario.window_hours
    assert result["return_pct"] > 0.0
    assert result["trade_count"] == 1


def test_parallel_analysis_warmup_models_uses_inference_mode() -> None:
    parallel_analysis.warmup_models(lambda: _WarmupPipeline(), load_kronos=None, toto_context_length=4)


def test_binance_prepare_evaluate_checkpoint_uses_inference_mode(monkeypatch, tmp_path) -> None:
    fake_binding = _FakeBinanceBinding()
    fake_package = ModuleType("pufferlib_market")
    fake_package.binding = fake_binding
    monkeypatch.setitem(sys.modules, "pufferlib_market", fake_package)
    monkeypatch.setattr(binance_prepare, "_read_data_header", lambda data_path: (2, 1))
    monkeypatch.setattr(
        binance_prepare.torch,
        "load",
        lambda *args, **kwargs: {
            "model": {
                "encoder.weight": torch.zeros((4, 39), dtype=torch.float32),
                "actor.weight": torch.zeros((2, 4), dtype=torch.float32),
            },
            "hidden_size": 4,
        },
    )

    result = binance_prepare.evaluate_checkpoint(
        tmp_path / "fake.pt",
        _BinancePolicy,
        binance_prepare.BinanceTaskConfig(data_path=tmp_path / "data.bin"),
        device=torch.device("cpu"),
    )

    assert result["robust_score"] > 0.0
    assert result["return_7d"] == 1.0
