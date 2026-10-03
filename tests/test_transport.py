import torch

from sighextraimage.latent import PhysicalLatent
from sighextraimage.scene import SceneConfig
from sighextraimage.transport import CornerTransport, NoOccluderTransport, TransportConfig


def latent(theta=0.35, rgb=(0.8,0.2,0.1), brightness=1.0):
    return PhysicalLatent.from_scalars(theta=theta, width=0.12, height=0.45, rgb=rgb, brightness=brightness, shape_code=0)


def test_zero_brightness_has_zero_object_leakage():
    scene = SceneConfig(height=48, visible_width=64, hidden_width=48)
    tr = CornerTransport(TransportConfig(n_measure=48, n_angle=48, ambient=0.0))
    y = tr.forward_latent(latent(brightness=0.0), scene)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-8)


def test_color_changes_matching_measurement_channel():
    scene = SceneConfig(height=48, visible_width=64, hidden_width=48)
    tr = CornerTransport(TransportConfig(n_measure=48, n_angle=48, ambient=0.0))
    yr = tr.forward_latent(latent(rgb=(0.9,0.1,0.1)), scene)
    yb = tr.forward_latent(latent(rgb=(0.1,0.1,0.9)), scene)
    assert yr[..., 0].mean() > yr[..., 2].mean()
    assert yb[..., 2].mean() > yb[..., 0].mean()


def test_angular_position_changes_profile_shape():
    scene = SceneConfig(height=48, visible_width=64, hidden_width=48)
    tr = CornerTransport(TransportConfig(n_measure=48, n_angle=48, ambient=0.0))
    ya = tr.forward_latent(latent(theta=0.25), scene)
    yb = tr.forward_latent(latent(theta=0.95), scene)
    assert torch.linalg.vector_norm(ya - yb) > 1e-3


def test_no_occluder_is_more_position_insensitive():
    scene = SceneConfig(height=48, visible_width=64, hidden_width=48)
    cfg = TransportConfig(n_measure=48, n_angle=48, ambient=0.0)
    corner = CornerTransport(cfg)
    no_occ = NoOccluderTransport(cfg)
    a, b = latent(theta=0.25), latent(theta=0.95)
    corner_delta = torch.linalg.vector_norm(corner.forward_latent(a, scene)-corner.forward_latent(b, scene))
    no_occ_delta = torch.linalg.vector_norm(no_occ.forward_latent(a, scene)-no_occ.forward_latent(b, scene))
    assert corner_delta > no_occ_delta * 1.5


def test_transport_is_deterministic_and_spectrum_is_finite():
    tr = CornerTransport(TransportConfig(n_measure=32, n_angle=32))
    scene = SceneConfig(height=32, visible_width=48, hidden_width=32)
    z = latent()
    assert torch.allclose(tr.forward_latent(z, scene), tr.forward_latent(z, scene))
    s = tr.condition_spectrum()
    assert torch.isfinite(s).all()
    assert s[0] >= s[-1]
