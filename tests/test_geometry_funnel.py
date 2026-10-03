import math

import numpy as np
import pytest
import torch

from sighextraimage.gate2 import SCENE, render_scene, true_geometry
from sighextraimage.geometry_funnel import (
    DEFAULT_HYPOTHESIS, CornerHypothesis, FunnelConfig, background_basis, calibrated_posterior,
    funnel_from_image, hypothesis_survives, nn_misfit, project_out, split_half_noise,
    surviving_geometries,
)
from sighextraimage.latent import PhysicalLatent, sample_prior
from sighextraimage.transport import CornerTransport, TransportConfig, WrongCornerTransport


def test_mirrored_transport_is_a_relabeling_of_theta():
    """WrongCornerTransport(theta) == CornerTransport(1.30 - theta): no data can prefer either."""
    cfg = TransportConfig(n_measure=64, n_angle=64)
    z = sample_prior(200, generator=torch.Generator().manual_seed(1))
    zm = PhysicalLatent(1.30 - z.theta, z.width, z.height, z.rgb, z.brightness, z.shape_code)
    a = WrongCornerTransport(cfg).forward_latent(z, SCENE)
    b = CornerTransport(cfg).forward_latent(zm, SCENE)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-5


def test_background_projection_removes_constant_and_ramp_only():
    q = background_basis(32, 1)
    u = np.linspace(-1, 1, 32)
    flat = np.stack([3.0 + 0.5 * u, -1.0 + 2.0 * u, np.ones(32)], 1)
    assert np.abs(project_out(flat, q)).max() < 1e-12
    bump = np.exp(-((u - 0.3) / 0.1) ** 2)[:, None]
    assert np.linalg.norm(project_out(bump, q)) > 0.5 * np.linalg.norm(bump - bump.mean())


def test_reversed_hypothesis_flips_the_strip():
    h = CornerHypothesis(0.1, 1.1, 0.05, False)
    r = CornerHypothesis(0.1, 1.1, 0.05, True)
    np.testing.assert_allclose(h.funnel_kernel(32, 12)[::-1], r.funnel_kernel(32, 12))
    np.testing.assert_allclose(h.kernel(32, 64)[::-1], r.kernel(32, 64), rtol=1e-6)


def test_nonnegativity_barely_constrains_a_one_dimensional_profile():
    """Characterisation, not a wish: with a ramp removed, positive steps also
    approximate an *inverted* corner signature. This is why the reversed edge is
    rarely refuted from a single 1-D boundary profile."""
    h = CornerHypothesis(0.1, 1.1, 0.05, False)
    K = h.funnel_kernel(64, 16)
    L = np.zeros((16, 3)); L[6] = 1.0
    y = K @ L
    q = background_basis(64)
    sigma = np.full(3, 1e-3 * np.abs(project_out(y, q)).max())
    assert nn_misfit(K, y, sigma, q) < 1e-6
    inverted = nn_misfit(K, -y, sigma, q)
    assert inverted < 50.0          # residual within ~0.1% of the signal per sample


def test_null_scene_reads_no_signal_and_signal_scene_is_detected():
    z, g, obs, image = render_scene(11, null=True)
    null, _ = funnel_from_image(image, obs.region, truth=g)
    assert null.status == "no-signal"
    assert not null.background_refuted
    z, g, obs, image = render_scene(11)
    sig, _ = funnel_from_image(image, obs.region, truth=g)
    assert sig.background_refuted
    assert sig.status in ("undecided-geometry", "decided-geometry")
    assert all(row["truth_still_covered"] for row in sig.ledger)
    assert sig.temperature >= 1.0


def test_split_half_noise_is_positive_and_has_a_quantisation_floor():
    z, g, obs, image = render_scene(7)
    n = split_half_noise(image, obs.region, n_measure=64, smooth_sigma=1.5)
    assert np.all(n.sigma > 0) and n.correlation >= 1.0
    flat = torch.full_like(image, 0.5)
    nf = split_half_noise(flat, obs.region, n_measure=64, smooth_sigma=1.5)
    assert np.all(nf.sigma > 0)


def test_calibrated_posterior_has_exact_normalised_weights():
    z, g, obs, image = render_scene(11)
    funnel, profile = funnel_from_image(image, obs.region, truth=g)
    cand = sample_prior(300, generator=torch.Generator().manual_seed(5))
    post = calibrated_posterior(profile, funnel.noise, SCENE, cand, [g, DEFAULT_HYPOTHESIS],
                                rounds=1, per_round=200, centres=16, seed=0)
    assert post.pool_size == 500
    assert math.isclose(float(post.weights.sum()), 1.0, rel_tol=1e-9)
    assert post.ess >= 1.0
    assert post.temperature >= 1.0
    assert post.interval90[0] <= post.interval68[0] <= post.interval68[1] <= post.interval90[1]
    assert math.isclose(float(post.geometry_weights.sum()), 1.0, rel_tol=1e-9)


def test_surviving_geometries_never_returns_empty_and_truth_survives():
    z, g, obs, image = render_scene(11)
    funnel, profile = funnel_from_image(image, obs.region, truth=g)
    assert hypothesis_survives(profile, funnel.noise, g, funnel)
    keep = surviving_geometries(profile, funnel.noise, [g, CornerHypothesis(0.3, 0.6, 0.15)], funnel)
    assert g in keep and len(keep) >= 1


def test_true_geometry_is_inside_the_funnel_search_box():
    cfg = FunnelConfig()
    for s in range(30):
        h = true_geometry(s)
        assert cfg.phi_start[0] <= h.phi_start <= cfg.phi_start[1]
        assert cfg.phi_end[0] <= h.phi_end <= cfg.phi_end[1]
        assert cfg.log_penumbra[0] <= math.log(h.penumbra) <= cfg.log_penumbra[1]


def test_gate2_criteria_are_applied_as_preregistered():
    from sighextraimage.gate2 import evaluate
    def scene(cov, kept, ess):
        post = {"covers90": cov, "theta_error": 0.1, "ess": ess, "temperature": 1.0}
        return {"truth_survived_all_levels": kept, "prior_theta_error": 0.3,
                "funnel": {"background_refuted": True, "status": "undecided-geometry",
                           "box_refuted_fraction": 0.1, "reversed_refuted": False},
                "v0": {"ess": 1.0, "theta_error": 0.2},
                "posterior_prior_marginal": post, "posterior_default_geometry": post,
                "posterior_oracle_geometry": post}
    nulls = [{"funnel": {"background_refuted": i == 0}} for i in range(16)]
    scenes = [scene(i < 12, i < 15, 25.0) for i in range(16)]
    r = evaluate(scenes, nulls)
    assert r["passed"]
    scenes[11] = scene(False, True, 25.0)
    assert not evaluate(scenes, nulls)["gates"]["G2c_calibrated_interval"]["pass"]
    nulls[1] = {"funnel": {"background_refuted": True}}
    assert not evaluate(scenes, nulls)["gates"]["G2a_no_false_alarms"]["pass"]
