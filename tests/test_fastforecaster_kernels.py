import torch

from FastForecaster import kernels


def test_mae_loss_fallback_matches_torch():
    pred = torch.tensor([[1.0, 3.0], [2.0, -2.0]])
    target = torch.tensor([[0.0, 4.0], [4.0, -1.0]])

    got = kernels.mae_loss(pred, target, use_cpp=False, build_extension=False)
    expected = torch.mean(torch.abs(pred - target))
    assert torch.isclose(got, expected)


def test_mae_loss_uses_extension_when_available(monkeypatch):
    class _DummyExt:
        @staticmethod
        def fast_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.tensor(42.0, dtype=prediction.dtype)

    monkeypatch.setattr(kernels, "load_fastforecaster_extension", lambda **_: _DummyExt())
    pred = torch.tensor([[1.0]])
    target = torch.tensor([[2.0]])

    got = kernels.mae_loss(pred, target, use_cpp=True, build_extension=False)
    assert float(got.item()) == 42.0


def test_weighted_mae_loss_fallback_matches_torch():
    pred = torch.tensor([[1.0, 3.0], [2.0, -2.0]])
    target = torch.tensor([[0.0, 4.0], [4.0, -1.0]])
    weights = torch.tensor([2.0, 0.5])

    got = kernels.weighted_mae_loss(pred, target, weights, use_cpp=False, build_extension=False)
    expected = torch.mean(torch.abs(pred - target) * weights.view(1, -1))
    assert torch.isclose(got, expected)


def test_weighted_mae_loss_uses_extension_when_available(monkeypatch):
    class _DummyExt:
        @staticmethod
        def fast_weighted_mae(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            return torch.tensor(7.0, dtype=prediction.dtype)

    monkeypatch.setattr(kernels, "load_fastforecaster_extension", lambda **_: _DummyExt())
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[2.0, 3.0]])
    weights = torch.tensor([1.0, 1.0])

    got = kernels.weighted_mae_loss(pred, target, weights, use_cpp=True, build_extension=False)
    assert float(got.item()) == 7.0


def test_mae_loss_backward_matches_closed_form():
    pred = torch.tensor([[1.0, 3.0], [2.0, -2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 4.0], [4.0, -1.0]])

    loss = kernels.mae_loss(pred, target, use_cpp=False, build_extension=False)
    loss.backward()

    expected_grad = torch.sign(pred.detach() - target) / pred.numel()
    assert torch.allclose(pred.grad, expected_grad)


def test_weighted_mae_loss_backward_matches_closed_form():
    pred = torch.tensor([[1.0, 3.0], [2.0, -2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 4.0], [4.0, -1.0]])
    weights = torch.tensor([2.0, 0.5])

    loss = kernels.weighted_mae_loss(pred, target, weights, use_cpp=False, build_extension=False)
    loss.backward()

    expected = torch.sign(pred.detach() - target) * weights.view(1, -1) / pred.numel()
    assert torch.allclose(pred.grad, expected)


def test_mae_loss_native_autograd_used_when_cpp_disabled(monkeypatch):
    def _unexpected_apply(*args, **kwargs):
        raise AssertionError("custom autograd apply should not be used when use_cpp=False")

    monkeypatch.setattr(kernels._FastMaeAutograd, "apply", _unexpected_apply)
    pred = torch.tensor([[1.0, 3.0]], requires_grad=True)
    target = torch.tensor([[0.0, 4.0]])

    loss = kernels.mae_loss(pred, target, use_cpp=False, build_extension=False)
    loss.backward()

    expected_grad = torch.sign(pred.detach() - target) / pred.numel()
    assert torch.allclose(pred.grad, expected_grad)


def test_weighted_mae_native_autograd_used_when_cpp_disabled(monkeypatch):
    def _unexpected_apply(*args, **kwargs):
        raise AssertionError("custom autograd apply should not be used when use_cpp=False")

    monkeypatch.setattr(kernels._FastWeightedMaeAutograd, "apply", _unexpected_apply)
    pred = torch.tensor([[1.0, 3.0]], requires_grad=True)
    target = torch.tensor([[0.0, 4.0]])
    weights = torch.tensor([2.0, 0.5])

    loss = kernels.weighted_mae_loss(pred, target, weights, use_cpp=False, build_extension=False)
    loss.backward()

    expected_grad = torch.sign(pred.detach() - target) * weights.view(1, -1) / pred.numel()
    assert torch.allclose(pred.grad, expected_grad)
