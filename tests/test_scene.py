import torch

from sighextraimage.latent import PhysicalLatent
from sighextraimage.scene import SceneConfig, render_hidden


def test_hidden_render_stays_in_hidden_canvas():
    cfg = SceneConfig(height=48, visible_width=64, hidden_width=32)
    lat = PhysicalLatent.from_scalars(theta=0.6, width=0.15, height=0.4, rgb=(0.9,0.1,0.1), brightness=1.0, shape_code=0)
    hidden = render_hidden(lat, cfg)
    assert hidden.shape == (1, 3, 48, 32)
    assert hidden.min() >= 0
    assert hidden.max() <= 1.0
