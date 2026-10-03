import numpy as np
import pytest
import torch

from sighextraimage import outpainting as op
from sighextraimage.extraction import srgb_to_linear
from sighextraimage.transport import CornerTransport, NoOccluderTransport, TransportConfig
from sighextraimage.photo import PhotoConfig


@pytest.mark.parametrize('edge,shape,known', [
    ('right', (40,100,3), (slice(None),slice(0,80))),
    ('left', (40,100,3), (slice(None),slice(20,None))),
    ('top', (50,80,3), (slice(10,None),slice(None))),
    ('bottom', (50,80,3), (slice(0,40),slice(None))),
])
def test_extending_each_edge_preserves_every_original_pixel(edge, shape, known):
    image = np.random.default_rng(4).integers(0,256,(40,80,3),dtype=np.uint8)
    prepared = op.prepare_canvas(image, op.OutpaintConfig(edge=edge, extension_fraction=0.25))
    generated = torch.full_like(prepared.canvas, 0.25)
    result = op.restore_outpaint(generated, prepared)
    assert result.shape == shape
    np.testing.assert_array_equal(result[known], image)
    assert prepared.canvas.shape[-1] % 8 == 0
    assert prepared.canvas.shape[-2] % 8 == 0
    assert not prepared.mask[:,:prepared.visible_height,:prepared.visible_width].any()
    assert prepared.mask[:,:,prepared.visible_width:].all()


def test_working_photo_keeps_aspect_ratio_and_restores_original_resolution():
    image = np.zeros((151,303,3), dtype=np.uint8)
    prepared = op.prepare_canvas(image, op.OutpaintConfig(max_side=128, extension_fraction=0.5))
    assert abs(prepared.visible_width / prepared.visible_height - 303/151) < 0.025
    result = op.restore_outpaint(torch.ones_like(prepared.canvas), prepared)
    assert result.shape == (151,455,3)
    np.testing.assert_array_equal(result[:,:303], image)


@pytest.mark.parametrize('kwargs', [
    {'edge':'diagonal'}, {'extension_fraction':0}, {'extension_fraction':3},
    {'steps':0}, {'max_side':0}, {'physics_strength':float('nan')},
    {'prompt':'   '},
])
def test_invalid_generation_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        op.OutpaintConfig(**kwargs)


def test_flat_and_nonfinite_measurements_disable_guidance():
    transport = CornerTransport(TransportConfig(n_measure=32, n_angle=32))
    for y in (torch.zeros(32,3), torch.full((32,3),0.2), torch.full((32,3),float('nan'))):
        constraint = op.CoarseLightConstraint.from_measurement(y, transport)
        assert constraint.allowed is False
        assert constraint.reason


def test_coarse_guidance_ignores_unobservable_transport_modes():
    transport = CornerTransport(TransportConfig(n_measure=32, n_angle=32))
    hidden = torch.linspace(0.1,0.9,48).expand(3,24,48).clone()
    y = transport.forward_hidden(srgb_to_linear(hidden))[0]
    constraint = op.CoarseLightConstraint.from_measurement(y, transport)
    assert constraint.allowed
    assert 1 < constraint.retained_modes < 32
    hidden.requires_grad_()
    loss = constraint.loss(hidden)
    assert loss.item() < 1e-6
    assert torch.isfinite(torch.autograd.grad(loss,hidden)[0]).all()
    no_corner = NoOccluderTransport(TransportConfig(n_measure=32,n_angle=32))
    assert not op.CoarseLightConstraint.from_measurement(y,no_corner).allowed


def test_light_fit_cannot_explain_signal_with_negative_exposure():
    transport = CornerTransport(TransportConfig(n_measure=32,n_angle=32))
    hidden = torch.linspace(0.1,0.9,48).expand(3,24,48).clone()
    y = transport.forward_hidden(srgb_to_linear(hidden))[0]
    constraint = op.CoarseLightConstraint.from_measurement(y,transport)
    inverted = 2*y.mean(0,keepdim=True)-y
    assert constraint.residual(inverted).item() > 0.99


class ConstantEngine:
    def __init__(self, fail_guidance=False):
        self.fail_guidance = fail_guidance
        self.last_run = {}

    def generate(self,prepared,config,*,constraint=None,progress=None):
        if constraint is not None and self.fail_guidance:
            raise RuntimeError('guided run failed')
        self.last_run = {'guidance_steps':2 if constraint is not None else 0}
        return torch.full_like(prepared.canvas,0.7 if constraint is not None else 0.2)

    def generate_with_diagnostics(self,prepared,config,*,constraint=None,progress=None):
        image = self.generate(prepared,config,constraint=constraint,progress=progress)
        return image,dict(self.last_run)


def test_flat_evidence_still_produces_an_enlarged_prior_image():
    image = np.full((40,80,3),128,dtype=np.uint8)
    result = op.generate_outpaintings(image,PhotoConfig(),op.OutpaintConfig(light_guidance=True),engine=ConstantEngine())
    assert result.prior.shape == (40,116,3)
    assert result.guidance_status == 'skipped'
    np.testing.assert_array_equal(result.prior[:,:80],image)
    np.testing.assert_array_equal(result.prior,result.guided)
    assert 'weak' in ' '.join(result.warnings).lower()


@pytest.mark.parametrize('fail_guidance',[False,True])
def test_usable_measurement_changes_comparison_or_preserves_prior_on_failure(fail_guidance):
    image = np.broadcast_to(np.linspace(40,210,80).astype(np.uint8)[None,:,None],(40,80,3)).copy()
    result = op.generate_outpaintings(image,PhotoConfig(),op.OutpaintConfig(light_guidance=True),
        engine=ConstantEngine(fail_guidance))
    assert result.prior.shape == (40,116,3)
    np.testing.assert_array_equal(result.prior[:,:80],image)
    if fail_guidance:
        assert result.guidance_status == 'failed'
        np.testing.assert_array_equal(result.prior,result.guided)
        assert 'guided run failed' in ' '.join(result.warnings)
    else:
        assert result.guidance_status == 'applied'
        assert not np.array_equal(result.prior[:,80:],result.guided[:,80:])
        np.testing.assert_array_equal(result.guided[:,:80],image)


def test_guidance_status_uses_its_own_run_metadata_when_shared_state_changes():
    class InterleavedEngine(ConstantEngine):
        def generate(self,prepared,config,*,constraint=None,progress=None):
            image = super().generate(prepared,config,constraint=constraint,progress=progress)
            self.last_run = {'guidance_steps':0}
            return image

        def generate_with_diagnostics(self,prepared,config,*,constraint=None,progress=None):
            image = super().generate(prepared,config,constraint=constraint,progress=progress)
            metadata = dict(self.last_run)
            self.last_run = {'guidance_steps':0}
            return image,metadata
    image = np.broadcast_to(np.linspace(40,210,80).astype(np.uint8)[None,:,None],(40,80,3)).copy()
    result = op.generate_outpaintings(image,PhotoConfig(),op.OutpaintConfig(light_guidance=True),engine=InterleavedEngine())
    assert result.guidance_status == 'applied'
    assert result.diagnostics['guidance_steps'] == 2
