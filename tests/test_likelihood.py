import torch

from sighextraimage.likelihood import physics_residual


def profile():
    x = torch.linspace(0, 1, 48)
    return torch.stack([x, x**2, torch.sin(x*2.0).abs()], dim=1) + 0.2


def test_l2_zero_at_identity():
    y = profile()
    assert float(physics_residual(y, y, "l2")) < 1e-10


def test_affine_is_invariant_to_shared_gain_and_channel_offsets():
    yhat = profile()
    offsets = torch.tensor([0.3, -0.1, 0.2])
    y = 1.7 * yhat + offsets
    assert float(physics_residual(y, yhat, "affine")) < 1e-8


def test_affine_penalizes_wrong_profile_shape():
    y = profile()
    wrong = torch.flip(y, dims=[0])
    assert float(physics_residual(y, wrong, "affine")) > 0.2


def test_affine_per_channel_is_more_color_blind_than_shared_affine():
    yhat = profile()
    y = yhat * torch.tensor([0.4, 1.6, 2.4])
    per_channel = physics_residual(y, yhat, "affine_per_channel")
    shared = physics_residual(y, yhat, "affine")
    assert per_channel < shared * 0.1
