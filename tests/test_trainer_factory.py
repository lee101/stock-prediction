from __future__ import annotations

import pytest

from binanceneural.config import TrainingConfig
from binanceneural.trainer_factory import build_trainer


def test_training_config_defaults_to_torch_backend() -> None:
    cfg = TrainingConfig()
    assert cfg.trainer_backend == "torch"


def test_training_config_normalizes_trainer_backend_aliases() -> None:
    torch_cfg = TrainingConfig(trainer_backend="pytorch")
    jax_cfg = TrainingConfig(trainer_backend="jax")
    assert torch_cfg.trainer_backend == "torch"
    assert jax_cfg.trainer_backend == "jax_classic"


def test_training_config_rejects_unknown_trainer_backend() -> None:
    with pytest.raises(ValueError, match="trainer_backend"):
        TrainingConfig(trainer_backend="mystery")


def test_build_trainer_selects_torch_backend(monkeypatch) -> None:
    calls: list[tuple[TrainingConfig, object]] = []

    class _DummyTorchTrainer:
        def __init__(self, config: TrainingConfig, data_module: object) -> None:
            calls.append((config, data_module))

    monkeypatch.setattr("binanceneural.trainer_factory.BinanceHourlyTrainer", _DummyTorchTrainer)

    cfg = TrainingConfig(trainer_backend="torch")
    data_module = object()
    trainer = build_trainer(cfg, data_module)

    assert isinstance(trainer, _DummyTorchTrainer)
    assert calls == [(cfg, data_module)]


def test_build_trainer_selects_jax_backend(monkeypatch) -> None:
    import binanceneural.jax_trainer as jax_trainer

    calls: list[tuple[TrainingConfig, object]] = []

    class _DummyJaxTrainer:
        def __init__(self, config: TrainingConfig, data_module: object) -> None:
            calls.append((config, data_module))

    monkeypatch.setattr(jax_trainer, "JaxClassicTrainer", _DummyJaxTrainer)

    cfg = TrainingConfig(trainer_backend="jax_classic", model_arch="classic")
    data_module = object()
    trainer = build_trainer(cfg, data_module)

    assert isinstance(trainer, _DummyJaxTrainer)
    assert calls == [(cfg, data_module)]


def test_build_trainer_rejects_non_classic_jax_backend() -> None:
    cfg = TrainingConfig(trainer_backend="jax_classic", model_arch="nano")
    with pytest.raises(ValueError, match="model_arch='classic'"):
        build_trainer(cfg, object())
