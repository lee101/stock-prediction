import math
from pathlib import Path

import numpy as np
import pytest

from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel
from hftraining.train_hf import HFTrainer, StockDataset


def _make_trainer(tmp_path: Path, *, max_steps: int = 2) -> HFTrainer:
    seq_len = 8
    horizon = 2
    data = np.random.randn(64, 6).astype(np.float32)
    dataset = StockDataset(data, sequence_length=seq_len, prediction_horizon=horizon)

    config = HFTrainingConfig()
    config.hidden_size = 32
    config.num_layers = 2
    config.num_heads = 4
    config.dropout = 0.1
    config.learning_rate = 1e-3
    config.warmup_steps = 0
    config.max_steps = max_steps
    config.gradient_accumulation_steps = 1
    config.max_grad_norm = 1.0
    config.optimizer_name = "adamw"
    config.weight_decay = 0.0
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_epsilon = 1e-8
    config.batch_size = 4
    config.eval_steps = max_steps + 10
    config.save_steps = max_steps + 20
    config.logging_steps = 1
    config.sequence_length = seq_len
    config.prediction_horizon = horizon
    config.use_mixed_precision = False
    config.precision = "fp32"
    config.use_gradient_checkpointing = False
    config.use_data_parallel = False
    config.use_compile = False
    config.use_fused_optimizer = False
    config.use_wandb = False
    config.dataloader_num_workers = 0
    config.persistent_workers = False
    config.prefetch_factor = 2
    config.enable_benchmark_metrics = True
    config.benchmark_step_window = 16
    config.output_dir = str(tmp_path / "out")
    config.logging_dir = str(tmp_path / "logs")
    config.cache_dir = str(tmp_path / "cache")

    model = TransformerTradingModel(config, input_dim=data.shape[1])
    return HFTrainer(model=model, config=config, train_dataset=dataset, eval_dataset=None)


def test_cpu_training_records_step_time(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, max_steps=2)
    trainer.train()

    assert trainer.last_step_time is not None
    assert trainer.last_step_time > 0.0
    assert len(trainer._step_durations) > 0


def test_drain_step_events_handles_pending(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, max_steps=1)

    class FakeEvent:
        def __init__(self, duration_ms: float, ready: bool = True) -> None:
            self.duration_ms = duration_ms
            self._ready = ready

        def query(self) -> bool:
            return self._ready

        def synchronize(self) -> None:
            self._ready = True

        def elapsed_time(self, other: "FakeEvent") -> float:
            return other.duration_ms

    trainer._step_event_queue.clear()
    trainer._step_event_queue.append((FakeEvent(0.0), FakeEvent(12.5)))
    durations = trainer._drain_step_events()
    assert pytest.approx(durations[0], rel=1e-5) == 0.0125

    trainer._step_event_queue.append((FakeEvent(0.0, ready=False), FakeEvent(5.0, ready=False)))
    assert trainer._drain_step_events() == []

    trainer._step_event_queue.append((FakeEvent(0.0, ready=False), FakeEvent(8.0, ready=False)))
    durations = trainer._drain_step_events(wait_for_one=True)
    assert len(durations) == 1
    assert math.isclose(durations[0], 0.005, rel_tol=1e-5)

    # The remaining event should still be queued until it reports ready.
    assert len(trainer._step_event_queue) == 1
    trainer._step_event_queue[0][1].synchronize()
    drained = trainer._drain_step_events()
    assert len(drained) == 1
    assert math.isclose(drained[0], 0.008, rel_tol=1e-5)
