from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from hftraining.portfolio_rl_trainer import (
    DifferentiablePortfolioTrainer,
    PortfolioAllocationModel,
    PortfolioRLConfig,
)


class _DeterministicPortfolioDataset(Dataset):
    def __init__(self, *, length: int = 6, seq_len: int = 4, input_dim: int = 6, num_assets: int = 2):
        self.length = length
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_assets = num_assets

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = torch.linspace(0.0, 1.0, steps=self.seq_len * self.input_dim, dtype=torch.float32)
        inputs = (base.view(self.seq_len, self.input_dim) + idx * 0.001).contiguous()
        future_returns = torch.linspace(0.005, 0.005 * self.num_assets, steps=self.num_assets, dtype=torch.float32)
        per_asset_fees = torch.full((self.num_assets,), 0.0001, dtype=torch.float32)
        asset_class_ids = torch.zeros(self.num_assets, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.float32)
        return {
            "input_ids": inputs,
            "future_returns": future_returns,
            "per_asset_fees": per_asset_fees,
            "asset_class_ids": asset_class_ids,
            "attention_mask": attention_mask,
        }


class _DummyMetricsLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, float]]] = []

    def log(self, metrics: dict[str, float], *, step: int, commit: bool = False) -> None:
        self.records.append((step, dict(metrics)))

    def finish(self) -> None:
        pass


def test_portfolio_trainer_emits_epoch_timing(tmp_path) -> None:
    torch.set_num_threads(1)
    dataset = _DeterministicPortfolioDataset(length=6, seq_len=4, input_dim=6, num_assets=2)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    config = PortfolioRLConfig(
        epochs=2,
        batch_size=3,
        device="cpu",
        compile=False,
        use_wandb=False,
        logging_dir=str(tmp_path / "logs"),
        wandb_mode="disabled",
        warmup_steps=0,
        grad_clip=0.0,
    )
    model = PortfolioAllocationModel(input_dim=dataset.input_dim, config=config, num_assets=dataset.num_assets)
    logger = _DummyMetricsLogger()
    trainer = DifferentiablePortfolioTrainer(model, config, loader, metrics_logger=logger)
    metrics = trainer.train()

    assert len(trainer._epoch_timings) == config.epochs  # pylint: disable=protected-access
    assert len(logger.records) >= config.epochs
    for epoch in range(config.epochs):
        assert metrics[f"timing/epoch_seconds_{epoch}"] >= 0.0
        assert metrics[f"timing/steps_per_sec_{epoch}"] > 0.0
        assert metrics[f"timing/samples_per_sec_{epoch}"] > 0.0

    assert metrics["timing/epoch_seconds_mean"] >= metrics["timing/epoch_seconds_min"] >= 0.0
    assert metrics["timing/samples_per_sec_mean"] > 0.0
