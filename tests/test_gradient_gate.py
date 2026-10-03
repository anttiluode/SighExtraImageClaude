import torch

from sighextraimage.gradient_gate import boundary_loss, continuous_vector, evaluate_gate1, latent_from_continuous
from sighextraimage.latent import PhysicalLatent
from sighextraimage.scene import SceneConfig
from sighextraimage.transport import CornerTransport, NoOccluderTransport, WrongCornerTransport, TransportConfig


def z(theta=.82, width=.11, rgb=(.82,.16,.10), brightness=.95):
    return PhysicalLatent.from_scalars(theta=theta, width=width, height=.5, rgb=rgb, brightness=brightness, shape_code=0)


def setup():
    scene=SceneConfig(height=48, visible_width=64, hidden_width=48)
    cfg=TransportConfig(n_measure=48,n_angle=48,ambient=0.,gain=1.)
    return scene, CornerTransport(cfg), WrongCornerTransport(cfg), NoOccluderTransport(cfg)


def test_autograd_matches_finite_difference_on_continuous_coordinates():
    scene, tr, _, _ = setup()
    truth=z(); y=tr.forward_latent(truth, scene)[0].detach()
    v=continuous_vector(z(theta=.55,width=.15,rgb=(.55,.30,.18),brightness=.8)).detach().requires_grad_(True)
    lat=latent_from_continuous(v, shape_code=0)
    loss=boundary_loss(lat,y,tr,scene,"l2")
    grad=torch.autograd.grad(loss,v)[0]
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad) > 0
    eps=1e-3
    for idx in [0,1,3,4,5,6]:
        vp=v.detach().clone(); vm=v.detach().clone(); vp[idx]+=eps; vm[idx]-=eps
        lp=boundary_loss(latent_from_continuous(vp,0),y,tr,scene,"l2")
        lm=boundary_loss(latent_from_continuous(vm,0),y,tr,scene,"l2")
        fd=(lp-lm)/(2*eps)
        assert torch.allclose(grad[idx],fd,rtol=.08,atol=2e-4)


def test_correct_gradient_step_reduces_truth_error_better_than_controls():
    scene, correct, wrong, no_occ = setup()
    truth=z()
    init=z(theta=.48)
    y=correct.forward_latent(truth,scene)[0].detach()
    result=evaluate_gate1(truth,init,y,correct,wrong,no_occ,scene,residual_mode="l2",step_size=.12,random_generator=torch.Generator().manual_seed(4))
    assert result.correct.finite
    assert result.correct.gradient_norm > 0
    assert result.correct.error_after < result.correct.error_before
    assert result.correct.error_reduction > result.wrong.error_reduction
    assert result.correct.error_reduction > result.no_occluder.error_reduction
    assert result.correct.error_reduction > result.random.error_reduction
    assert result.correct.component_alignment[0] > 0  # theta points toward truth
