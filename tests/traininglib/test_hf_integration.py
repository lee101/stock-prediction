import pytest

pytest.importorskip("transformers")

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from traininglib.hf_integration import build_hf_optimizers


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 64, input_dim: int = 8, num_classes: int = 3):
        generator = torch.Generator().manual_seed(2020)
        self.features = torch.randn(num_samples, input_dim, generator=generator)
        self.labels = torch.randint(
            0, num_classes, (num_samples,), generator=generator, dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"input_ids": self.features[idx], "labels": self.labels[idx]}


class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, labels=None):
        logits = self.linear(input_ids.float())
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def evaluate_loss(model: nn.Module, dataset: Dataset) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for item in dataset:
            output = model(
                input_ids=item["input_ids"].unsqueeze(0),
                labels=item["labels"].unsqueeze(0),
            )
            losses.append(output["loss"].item())
    return float(torch.tensor(losses).mean().item())


def test_shampoo_optimizer_with_trainer(tmp_path) -> None:
    dataset = DummyDataset()
    model = DummyModel()
    base_loss = evaluate_loss(model, dataset)

    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer-out"),
        per_device_train_batch_size=16,
        learning_rate=0.01,
        max_steps=12,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    optimizer, scheduler = build_hf_optimizers(model, "shampoo", lr=0.05)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()
    final_loss = evaluate_loss(model, dataset)
    assert final_loss < base_loss
