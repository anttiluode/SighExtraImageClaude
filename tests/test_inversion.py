import torch

from sighextraimage.inversion import profile_centroid, profile_excess_color, tv_invert
from sighextraimage.latent import PhysicalLatent
from sighextraimage.scene import SceneConfig
from sighextraimage.transport import CornerTransport, NoOccluderTransport, TransportConfig


def hidden(theta=0.82, rgb=(0.85,0.15,0.1)):
    return PhysicalLatent.from_scalars(theta=theta, width=0.10, height=0.5, rgb=rgb, brightness=1.0, shape_code=0)


def test_tv_inversion_recovers_coarse_corner_angle_better_than_no_occluder():
    scene = SceneConfig(height=48, visible_width=64, hidden_width=48)
    cfg = TransportConfig(n_measure=48, n_angle=48, ambient=0.0, gain=1.0)
    corner = CornerTransport(cfg)
    no_occ = NoOccluderTransport(cfg)
    z = hidden(theta=0.82)
    y = corner.forward_latent(z, scene)[0]
    L_corner = tv_invert(corner, y, lambda_tv=0.01, iters=300, lr=0.08, residual_mode="l2")
    L_no = tv_invert(no_occ, y, lambda_tv=0.01, iters=300, lr=0.08, residual_mode="l2")
    err_corner = abs(profile_centroid(L_corner, corner.theta) - 0.82)
    err_no = abs(profile_centroid(L_no, no_occ.theta) - 0.82)
    assert torch.isfinite(L_corner).all() and (L_corner >= 0).all()
    assert err_corner < 0.25
    assert err_corner < err_no


def test_excess_color_tracks_dominant_channel():
    theta = torch.linspace(0, 1, 32)
    L = torch.zeros(32, 3)
    L[:, 0] = torch.exp(-0.5*((theta-0.6)/0.1)**2)
    L[:, 2] = 0.1 * L[:,0]
    c = profile_excess_color(L)
    assert c[0] > c[2]
    assert torch.allclose(c.sum(), torch.tensor(1.0), atol=1e-5)
