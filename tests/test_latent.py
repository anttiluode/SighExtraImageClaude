import torch

from sighextraimage.latent import sample_prior


def test_sample_prior_is_seed_deterministic_and_in_range():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_prior(4, generator=g1)
    b = sample_prior(4, generator=g2)
    assert torch.allclose(a.theta, b.theta)
    assert torch.allclose(a.rgb, b.rgb)
    assert a.rgb.shape == (4, 3)
    assert torch.all((a.theta >= 0.08) & (a.theta <= 1.22))
    assert torch.all((a.rgb >= 0.05) & (a.rgb <= 0.95))
