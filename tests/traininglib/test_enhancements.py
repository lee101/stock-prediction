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
    sigma = torch.exp(log_sigma)
    manual = 0.5 * ((target_val - mean) ** 2 / (sigma**2) + 2 * torch.log(sigma))
    assert torch.isclose(hetero, manual.mean())

    quant = pinball_loss(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 2.0]), 0.7)
    manual_pinball = (0.7 * (2.0 - 1.0) + (0.7 - 1) * (2.0 - 3.0)) / 2
    assert torch.isclose(quant, torch.tensor(manual_pinball))


def test_heteroscedastic_nll_clamp_matches_floor():
    mean = torch.tensor([0.0])
    target = torch.tensor([0.0])
    min_sigma = 1e-4
    # Force the clamp to engage by providing a very small log_sigma.
    log_sigma = torch.tensor([-20.0], requires_grad=True)
    loss = heteroscedastic_gaussian_nll(mean, log_sigma, target, reduction="none", min_sigma=min_sigma)
    expected_sigma = torch.tensor([min_sigma], dtype=mean.dtype)
    expected = 0.5 * ((target - mean) ** 2 / (expected_sigma**2) + 2 * torch.log(expected_sigma))
    assert torch.allclose(loss, expected)
    loss.sum().backward()
    assert log_sigma.grad is not None
    assert torch.all(torch.isfinite(log_sigma.grad))
