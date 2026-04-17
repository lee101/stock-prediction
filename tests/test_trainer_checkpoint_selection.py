from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
from binanceneural.trainer import BinanceHourlyTrainer


class _TinyDataModule:
    def __init__(self) -> None:
        self.feature_columns = [f"feat_{idx}" for idx in range(4)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )

    def train_dataloader(self, batch_size: int, num_workers: int = 0):
        return [None]

    def val_dataloader(self, batch_size: int, num_workers: int = 0):
        return [None]


class _SingleBatchDataModule:
    def __init__(self) -> None:
        self.feature_columns = [f"feat_{idx}" for idx in range(4)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        seq_len = 6
        batch = {
            "features": torch.zeros(1, seq_len, 4),
            "open": torch.full((1, seq_len), 100.0),
            "high": torch.full((1, seq_len), 101.0),
            "low": torch.full((1, seq_len), 99.0),
            "close": torch.full((1, seq_len), 100.0),
            "reference_close": torch.full((1, seq_len), 100.0),
            "chronos_high": torch.full((1, seq_len), 101.0),
            "chronos_low": torch.full((1, seq_len), 99.0),
            "can_long": torch.tensor([1.0]),
            "can_short": torch.tensor([0.0]),
        }
        self._loader = [batch]

    def train_dataloader(self, batch_size: int, num_workers: int = 0):
        return self._loader

    def val_dataloader(self, batch_size: int, num_workers: int = 0):
        return self._loader


class _DummyPolicy(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = (*features.shape[:2], 1)
        zeros = torch.zeros(shape, device=features.device, dtype=features.dtype)
        return {
            "buy_price_logits": zeros,
            "sell_price_logits": zeros,
            "buy_amount_logits": zeros,
            "sell_amount_logits": zeros,
        }

    def decode_actions(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "buy_price": reference_close * 0.999,
            "sell_price": reference_close * 1.001,
            "buy_amount": torch.full_like(reference_close, 50.0),
            "sell_amount": torch.full_like(reference_close, 50.0),
            "trade_amount": torch.full_like(reference_close, 50.0),
        }


def test_trainer_prefers_robust_checkpoint_metric(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    epoch_metrics = iter(
        [
            {"loss": -1.0, "score": 120.0, "sortino": 110.0, "return": 8.0},
            {"loss": -1.0, "score": 100.0, "sortino": 95.0, "return": 6.0},
            {"loss": -1.0, "score": 105.0, "sortino": 100.0, "return": 7.0},
            {"loss": -1.0, "score": 95.0, "sortino": 94.0, "return": 5.0},
        ]
    )

    def _fake_run_epoch(self, model, loader, optimizer, *, train, global_step, current_epoch=1):
        return dict(next(epoch_metrics)), global_step

    def _fake_save_checkpoint(self, model, epoch: int, metrics: dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        path.write_text(json.dumps({"epoch": epoch, "metrics": metrics}))
        return path

    monkeypatch.setattr(BinanceHourlyTrainer, "_run_epoch", _fake_run_epoch)
    monkeypatch.setattr(BinanceHourlyTrainer, "_save_checkpoint", _fake_save_checkpoint)

    with caplog.at_level(logging.INFO, logger="binanceneural.trainer"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainingConfig(
                epochs=2,
                batch_size=2,
                sequence_length=8,
                transformer_dim=16,
                transformer_layers=1,
                transformer_heads=4,
                transformer_dropout=0.0,
                use_compile=False,
                use_amp=False,
                checkpoint_root=Path(tmpdir) / "ckpts",
                run_name="robust_ckpt_test",
                checkpoint_metric="robust_score",
                checkpoint_gap_penalty=1.0,
                top_k_checkpoints=2,
            )
            trainer = BinanceHourlyTrainer(cfg, _TinyDataModule())
            artifacts = trainer.train()

            expected_best = trainer.checkpoint_dir / "epoch_002.pt"
            assert artifacts.best_checkpoint == expected_best

            alias_path = trainer.checkpoint_dir / "best.pt"
            assert alias_path.exists()
            if alias_path.is_symlink():
                assert alias_path.resolve() == expected_best.resolve()
            else:
                assert alias_path.read_text() == expected_best.read_text()

            manifest = json.loads((trainer.checkpoint_dir / ".topk_manifest.json").read_text())
            metrics_by_epoch = {int(row["epoch"]): float(row["metric"]) for row in manifest}
            assert metrics_by_epoch[1] == 80.0
            assert metrics_by_epoch[2] == 85.0

            progress = json.loads((trainer.checkpoint_dir / "training_progress.json").read_text())
            assert progress["trainer_backend"] == "torch"
            assert progress["checkpoint_metric_name"] == "robust_score"
            assert progress["checkpoint_metric"] == 85.0
            assert progress["generalization_gap"] == 10.0
            assert progress["best_checkpoint"].endswith("epoch_002.pt")

    assert "Skipping initial CUDA forward probe because the first training batch was None." in caplog.text


def test_validation_aggregates_binary_fill_metrics_across_lag_range_with_minimax(monkeypatch) -> None:
    lag_calls: list[int] = []

    class _FakeSimResult:
        def __init__(self, lag: int) -> None:
            self.returns = torch.full((1, 4), float(lag))

    def _fake_binary_sim(**kwargs):
        lag = int(kwargs["decision_lag_bars"])
        lag_calls.append(lag)
        return _FakeSimResult(lag)

    def _fake_compute_loss_by_type(returns, *_args, **_kwargs):
        lag_score = float(returns.mean().item())
        score = torch.tensor(lag_score)
        loss = torch.tensor(-lag_score)
        sortino = torch.tensor(lag_score + 10.0)
        annual_return = torch.tensor(lag_score + 20.0)
        return loss, score, sortino, annual_return

    monkeypatch.setattr("binanceneural.trainer.simulate_hourly_trades_binary", _fake_binary_sim)
    monkeypatch.setattr("binanceneural.trainer.compute_loss_by_type", _fake_compute_loss_by_type)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            epochs=1,
            batch_size=1,
            sequence_length=6,
            transformer_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            transformer_dropout=0.0,
            use_compile=False,
            use_amp=False,
            use_tf32=False,
            use_flash_attention=False,
            checkpoint_root=Path(tmpdir) / "ckpts",
            run_name="validation_lag_minimax",
            decision_lag_bars=0,
            decision_lag_range="0,1,2",
            validation_use_binary_fills=True,
            validation_lag_aggregation="minimax",
        )
        trainer = BinanceHourlyTrainer(cfg, _SingleBatchDataModule())
        metrics, _ = trainer._run_epoch(
            _DummyPolicy(),
            trainer.data.val_dataloader(batch_size=1),
            optimizer=None,
            train=False,
            global_step=0,
            current_epoch=1,
        )

    assert lag_calls == [0, 1, 2]
    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["score"] == pytest.approx(0.0)
    assert metrics["sortino"] == pytest.approx(10.0)
    assert metrics["return"] == pytest.approx(20.0)


def test_training_uses_compiled_sim_loss_when_supported(monkeypatch) -> None:
    lag_calls: list[int] = []

    def _fake_compiled_sim_and_loss(**kwargs):
        lag = int(kwargs["decision_lag_bars"])
        lag_calls.append(lag)
        score = torch.tensor(float(lag))
        loss = -score.mean()
        sortino = score + 10.0
        annual_return = score + 20.0
        return loss, score, sortino, annual_return

    def _unexpected_sim(**_kwargs):
        raise AssertionError("standard simulation path should not run when compiled sim+loss is selected")

    monkeypatch.setattr("binanceneural.trainer._compiled_sim_loss_supported", lambda *_args, **_kwargs: (True, 0.0, 8760.0))
    monkeypatch.setattr("binanceneural.trainer.compiled_sim_and_loss", _fake_compiled_sim_and_loss)
    monkeypatch.setattr("binanceneural.trainer.simulate_hourly_trades", _unexpected_sim)
    monkeypatch.setattr("binanceneural.trainer.simulate_hourly_trades_fast", _unexpected_sim)
    monkeypatch.setattr("binanceneural.trainer.simulate_hourly_trades_triton", _unexpected_sim)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            epochs=1,
            batch_size=1,
            sequence_length=6,
            transformer_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            transformer_dropout=0.0,
            use_compile=False,
            use_amp=False,
            use_tf32=False,
            use_flash_attention=False,
            checkpoint_root=Path(tmpdir) / "ckpts",
            run_name="compiled_sim_loss_train",
            decision_lag_bars=0,
            decision_lag_range="0,1,2",
            loss_type="sortino",
            use_compiled_sim_loss=True,
        )
        trainer = BinanceHourlyTrainer(cfg, _SingleBatchDataModule())
        metrics, _ = trainer._run_epoch(
            _DummyPolicy(),
            trainer.data.train_dataloader(batch_size=1),
            optimizer=None,
            train=True,
            global_step=0,
            current_epoch=1,
        )

    assert lag_calls == [0, 1, 2]
    assert metrics["loss"] == pytest.approx(-1.0)
    assert metrics["score"] == pytest.approx(1.0)
    assert metrics["sortino"] == pytest.approx(11.0)
    assert metrics["return"] == pytest.approx(21.0)
