import torch

from rlsys.utils import ObservationNormalizer


def test_observation_normalizer_centers_data():
    normalizer = ObservationNormalizer(size=2)
    normalizer.update(torch.tensor([1.0, 2.0]))
    normalizer.update(torch.tensor([2.0, 3.0]))
    normalized = normalizer.normalize(torch.tensor([1.5, 2.5]))
    assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-5)
