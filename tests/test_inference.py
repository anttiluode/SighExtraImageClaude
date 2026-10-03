import torch

from sighextraimage.inference import evaluate_gate0, normalize_log_weights
from sighextraimage.latent import PhysicalLatent, sample_prior
from sighextraimage.scene import SceneConfig
from sighextraimage.transport import CornerTransport, NoOccluderTransport, WrongCornerTransport, TransportConfig


def truth():
    return PhysicalLatent.from_scalars(theta=0.84, width=0.11, height=0.50, rgb=(0.82,0.16,0.10), brightness=0.95, shape_code=0)


def test_normalize_log_weights_is_finite_normalized_and_shift_invariant():
    x = torch.tensor([-1000.0, -1001.0, -999.0])
    a = normalize_log_weights(x)
    b = normalize_log_weights(x + 1234.0)
    assert torch.isfinite(a).all()
    assert torch.allclose(a.sum(), torch.tensor(1.0))
    assert torch.allclose(a, b, atol=1e-6)


def test_zero_information_transport_returns_prior_summary():
    scene = SceneConfig(height=32, visible_width=48, hidden_width=32)
    g = torch.Generator().manual_seed(5)
    candidates = sample_prior(128, generator=g)

    class ZeroTransport:
        def forward_latent(self, latent, scene):
            return torch.zeros(latent.batch_size, 16, 3)

    z = truth()
    y = torch.zeros(16, 3)
    res = evaluate_gate0(z, y, candidates, ZeroTransport(), ZeroTransport(), ZeroTransport(), scene, residual_mode="l2", sigma=0.05)
    assert torch.allclose(res.prior.weights, res.correct.weights, atol=1e-7)
    assert torch.allclose(res.prior.weights, res.wrong.weights, atol=1e-7)


def test_correct_physics_contracts_theta_and_beats_controls_on_identifiable_case():
    scene = SceneConfig(height=48, visible_width=64, hidden_width=48)
    cfg = TransportConfig(n_measure=48, n_angle=48, ambient=0.0, gain=1.0)
    correct = CornerTransport(cfg)
    wrong = WrongCornerTransport(cfg)
    no_occ = NoOccluderTransport(cfg)
    z = truth()
    y = correct.forward_latent(z, scene)[0]
    g = torch.Generator().manual_seed(17)
    candidates = sample_prior(1500, generator=g)
    res = evaluate_gate0(z, y, candidates, correct, wrong, no_occ, scene, residual_mode="l2", sigma=0.01)
    prior_err = abs(res.prior.mean_theta - float(z.theta[0]))
    correct_err = abs(res.correct.mean_theta - float(z.theta[0]))
    assert res.correct.std_theta < res.prior.std_theta * 0.75
    assert correct_err < prior_err
    assert correct_err < abs(res.wrong.mean_theta - float(z.theta[0]))
    assert res.correct.std_theta < res.no_occluder.std_theta
    true_rgb = z.rgb[0]
    assert torch.linalg.vector_norm(res.correct.mean_rgb - true_rgb) < torch.linalg.vector_norm(res.prior.mean_rgb - true_rgb)


def test_all_arms_share_exact_candidate_count_and_values():
    scene = SceneConfig(height=32, visible_width=48, hidden_width=32)
    cfg = TransportConfig(n_measure=32, n_angle=32, ambient=0.0)
    correct = CornerTransport(cfg); wrong = WrongCornerTransport(cfg); no_occ = NoOccluderTransport(cfg)
    z = truth(); y = correct.forward_latent(z, scene)[0]
    candidates = sample_prior(64, generator=torch.Generator().manual_seed(2))
    before = candidates.theta.clone()
    res = evaluate_gate0(z, y, candidates, correct, wrong, no_occ, scene, residual_mode="l2", sigma=0.03)
    assert res.prior.weights.numel() == res.correct.weights.numel() == res.wrong.weights.numel() == res.no_occluder.weights.numel() == 64
    assert torch.equal(candidates.theta, before)
