import math
import torch

from fp4.hadamard import hadamard_matrix, RandomHadamard


def test_hadamard_orthonormal():
    for n in (16, 32, 128):
        H = hadamard_matrix(n)
        I = H @ H.T
        assert torch.allclose(I, torch.eye(n), atol=1e-5)


def test_rht_round_trip_and_energy():
    torch.manual_seed(0)
    rht = RandomHadamard(n=16, seed=42)
    x = torch.randn(8, 64)
    y, pad = rht.forward(x)
    x_back = rht.inverse(y, pad)
    assert torch.allclose(x, x_back, atol=1e-5)
    # Orthonormal -> energy preserved.
    assert math.isclose(x.pow(2).sum().item(), y.pow(2).sum().item(), rel_tol=1e-4)


def test_rht_outlier_kurtosis_drops():
    torch.manual_seed(0)
    # Create a heavy-tailed signal: a few huge spikes.
    x = torch.randn(1, 16 * 64) * 0.1
    x[0, ::16] = 20.0  # one big outlier per block
    rht = RandomHadamard(n=16, seed=1)
    y, _ = rht.forward(x)

    def kurt(t):
        t = t.flatten() - t.mean()
        return (t.pow(4).mean() / t.pow(2).mean().pow(2)).item()

    assert kurt(y) < kurt(x), f"RHT did not reduce kurtosis: {kurt(x)} -> {kurt(y)}"
