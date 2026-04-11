import torch

from fp4.linear import NVFP4Linear


def test_nvfp4_linear_matches_bf16_reference():
    torch.manual_seed(0)
    in_f, out_f, batch = 256, 128, 64
    layer = NVFP4Linear(in_f, out_f, bias=True)
    ref = torch.nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        ref.weight.copy_(layer.weight)
        ref.bias.copy_(layer.bias)

    x = torch.randn(batch, in_f, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y = layer(x)
    y_ref = ref(x_ref)

    # Forward relative error
    num = (y - y_ref).pow(2).mean().sqrt()
    den = y_ref.pow(2).mean().sqrt().clamp(min=1e-8)
    fwd_rel = (num / den).item()
    assert fwd_rel < 0.15, f"forward relative error too high: {fwd_rel}"

    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)

    def rel(a, b):
        n = (a - b).pow(2).mean().sqrt()
        d = b.pow(2).mean().sqrt().clamp(min=1e-8)
        return (n / d).item()

    rx = rel(x.grad, x_ref.grad)
    rw = rel(layer.weight.grad, ref.weight.grad)
    rb = rel(layer.bias.grad, ref.bias.grad)
    # Tolerance: SR adds variance, so allow up to ~25% relative.
    assert rx < 0.25, f"grad_x rel {rx}"
    assert rw < 0.25, f"grad_w rel {rw}"
    assert rb < 0.25, f"grad_b rel {rb}"
