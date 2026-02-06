import torch

from differentiable_loss_utils import compute_hourly_objective


def test_compute_hourly_objective_supports_tensor_periods_per_year() -> None:
    hourly_returns = torch.tensor(
        [
            [0.01, -0.02, 0.03],
            [0.01, -0.02, 0.03],
        ],
        dtype=torch.float32,
    )
    periods = torch.tensor([8760.0, 1764.0], dtype=torch.float32)

    score, sortino, annual_return = compute_hourly_objective(
        hourly_returns,
        periods_per_year=periods,
        return_weight=0.05,
    )

    assert score.shape == (2,)
    assert sortino.shape == (2,)
    assert annual_return.shape == (2,)

    score0, sortino0, annual0 = compute_hourly_objective(
        hourly_returns[:1],
        periods_per_year=float(periods[0].item()),
        return_weight=0.05,
    )
    score1, sortino1, annual1 = compute_hourly_objective(
        hourly_returns[1:2],
        periods_per_year=float(periods[1].item()),
        return_weight=0.05,
    )

    assert torch.allclose(score[0], score0[0], rtol=1e-6, atol=1e-6)
    assert torch.allclose(sortino[0], sortino0[0], rtol=1e-6, atol=1e-6)
    assert torch.allclose(annual_return[0], annual0[0], rtol=1e-6, atol=1e-6)

    assert torch.allclose(score[1], score1[0], rtol=1e-6, atol=1e-6)
    assert torch.allclose(sortino[1], sortino1[0], rtol=1e-6, atol=1e-6)
    assert torch.allclose(annual_return[1], annual1[0], rtol=1e-6, atol=1e-6)

