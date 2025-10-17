from collections import namedtuple

import torch

from traininglib.ema import EMA
from traininglib.losses import huber_loss, heteroscedastic_gaussian_nll, pinball_loss
from traininglib.prefetch import CudaPrefetcher


def test_cuda_prefetcher_cpu_roundtrip():
    data = [torch.tensor([idx], dtype=torch.float32) for idx in range(6)]
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    prefetcher = CudaPrefetcher(loader, device="cpu")

    baseline = list(loader)
    fetched = list(iter(prefetcher))

    assert len(baseline) == len(fetched)
    for expected, actual in zip(baseline, fetched):
        assert torch.equal(expected, actual)


def test_cuda_prefetcher_namedtuple_roundtrip():
    Batch = namedtuple(
        "Batch",
        ["series", "padding_mask", "id_mask", "timestamp_seconds", "time_interval_seconds"],
    )

    def generate(idx: int) -> Batch:
        base = torch.arange(idx, idx + 4, dtype=torch.float32).view(1, -1)
        return Batch(
            series=base.clone(),
            padding_mask=torch.ones_like(base, dtype=torch.bool),
            id_mask=torch.zeros_like(base, dtype=torch.int64),
            timestamp_seconds=torch.arange(base.numel(), dtype=torch.int64),
            time_interval_seconds=torch.full_like(base, 60, dtype=torch.int64),
        )

    data = [generate(idx) for idx in range(0, 12, 4)]
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    prefetcher = CudaPrefetcher(loader, device="cpu")

    baseline = list(loader)
    fetched = list(iter(prefetcher))

    assert len(baseline) == len(fetched)
    for expected, actual in zip(baseline, fetched):
        assert isinstance(actual, Batch)
        for e_field, a_field in zip(expected, actual):
            assert torch.equal(e_field, a_field)


def test_ema_apply_restore_cycle():
    model = torch.nn.Linear(4, 2, bias=False)
    ema = EMA(model, decay=0.5)

    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    ema.update(model)
    updated = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.apply_to(model)
    for name, param in model.named_parameters():
        assert torch.allclose(param, ema.shadow[name])

    ema.restore(model)
    for name, param in model.named_parameters():
        assert torch.allclose(param, updated[name])


def test_losses_behave_expected():
    pred = torch.tensor([0.0, 0.02])
    target = torch.tensor([0.0, 0.0])
    huber = huber_loss(pred, target, delta=0.01)
    expected_huber = (0.5 * (0.01 ** 2) + 0.01 * (0.02 - 0.01)) / 2
    assert torch.isclose(huber, torch.tensor(expected_huber))

    mean = torch.tensor([0.0, 1.0])
    log_sigma = torch.log(torch.tensor([1.0, 2.0]))
    target_val = torch.tensor([0.0, 0.0])
    hetero = heteroscedastic_gaussian_nll(mean, log_sigma, target_val)
    manual = 0.5 * ((target_val - mean) ** 2 / (torch.exp(log_sigma) ** 2) + 2 * log_sigma)
    assert torch.isclose(hetero, manual.mean())

    quant = pinball_loss(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 2.0]), 0.7)
    manual_pinball = (0.7 * (2.0 - 1.0) + (0.7 - 1) * (2.0 - 3.0)) / 2
    assert torch.isclose(quant, torch.tensor(manual_pinball))
