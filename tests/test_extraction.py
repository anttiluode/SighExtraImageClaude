import torch

from sighextraimage.extraction import extract_boundary_signal, profile_correlation
from sighextraimage.latent import PhysicalLatent
from sighextraimage.scene import SceneConfig, render_visible_with_leakage
from sighextraimage.transport import CornerTransport, TransportConfig


def make_latent(theta=0.45, rgb=(0.85,0.15,0.1)):
    return PhysicalLatent.from_scalars(theta=theta, width=0.14, height=0.45, rgb=rgb, brightness=1.0, shape_code=0)


def test_extracted_profile_tracks_true_synthetic_leakage():
    scene = SceneConfig(height=72, visible_width=96, hidden_width=64)
    tr = CornerTransport(TransportConfig(n_measure=64, n_angle=64, ambient=0.0, gain=0.18))
    obs = render_visible_with_leakage(make_latent(), scene, tr, noise_std=0.0, texture_seed=4)
    yhat = extract_boundary_signal(obs.visible_srgb, obs.region, n_measure=64, smooth_sigma=1.5)
    corr = profile_correlation(yhat, obs.y_true[0])
    assert corr > 0.65


def test_extraction_changes_with_hidden_position_and_color():
    scene = SceneConfig(height=72, visible_width=96, hidden_width=64)
    tr = CornerTransport(TransportConfig(n_measure=64, n_angle=64, ambient=0.0, gain=0.2))
    a = render_visible_with_leakage(make_latent(theta=0.25, rgb=(0.9,0.1,0.1)), scene, tr, noise_std=0.0, texture_seed=2)
    b = render_visible_with_leakage(make_latent(theta=0.95, rgb=(0.1,0.1,0.9)), scene, tr, noise_std=0.0, texture_seed=2)
    ya = extract_boundary_signal(a.visible_srgb, a.region, n_measure=64, smooth_sigma=1.0)
    yb = extract_boundary_signal(b.visible_srgb, b.region, n_measure=64, smooth_sigma=1.0)
    assert torch.linalg.vector_norm(ya - yb) > 1e-3
    assert ya[:,0].mean() > ya[:,2].mean()
    assert yb[:,2].mean() > yb[:,0].mean()
