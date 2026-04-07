import math
import torch

from fp4.layers import NVFP4MLP


def _make_data(n=512):
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = torch.sin(x)
    return x, y


def _train(model, steps, lr):
    x, y = _make_data()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_nvfp4_mlp_fits_sin():
    torch.manual_seed(0)
    nvfp4_model = NVFP4MLP(1, 64, 1)
    nvfp4_losses = _train(nvfp4_model, steps=2000, lr=5e-3)

    torch.manual_seed(0)
    bf16_model = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1),
    )
    bf16_losses = _train(bf16_model, steps=2000, lr=5e-3)

    final_nvfp4 = sum(nvfp4_losses[-50:]) / 50
    final_bf16 = sum(bf16_losses[-50:]) / 50

    assert math.isfinite(final_nvfp4)
    # NVFP4 final loss within 2x of BF16 baseline.
    assert final_nvfp4 < max(2.0 * final_bf16, 0.05), \
        f"nvfp4 final loss {final_nvfp4} vs bf16 {final_bf16}"
    # And it should clearly have learned something (well below variance of sin).
    assert final_nvfp4 < 0.1, f"nvfp4 did not fit sin: {final_nvfp4}"
