import torch

from sighextraimage.metrics import contraction_ratio, rgb_error, scalar_error


def test_scalar_and_rgb_errors_are_zero_at_truth():
    assert scalar_error(0.4, 0.4) == 0.0
    assert rgb_error(torch.tensor([0.2,0.3,0.4]), torch.tensor([0.2,0.3,0.4])) == 0.0
    assert scalar_error(0.5, 0.4) > 0


def test_contraction_ratio_is_finite_and_interpretable():
    assert contraction_ratio(0.2, 0.4) == 0.5
    assert contraction_ratio(0.0, 0.4) == 0.0
    assert torch.isfinite(torch.tensor(contraction_ratio(0.2, 0.0)))
