import math

import numpy as np
from scipy.optimize import nnls

from sighextraimage.fan import (
    FanConfig, FanHypothesis, FanLevel, _proj, apply_h, background_basis_2d, distance_to_line,
    fan_kernel, homography_from_points, locate_apex, measure_blocks, render_fan_scene,
)


def test_homography_maps_floor_rays_to_image_lines_through_the_apex():
    """The reason the fan works without calibration: rays from O stay lines through H(O)."""
    sc = render_fan_scene(7)
    apex = sc.apex
    for th in (0.3, 0.8, 1.2):
        pts = apply_h(sc.homography, np.array([[r * math.cos(th), r * math.sin(th)] for r in (0.4, 0.9, 1.7)]))
        v1, v2 = pts[0] - apex, pts[2] - apex
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert cross < 1e-9


def test_dlt_homography_reproduces_point_pairs():
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.3, 0.7]], float)
    H0 = np.array([[2.0, 0.3, 5.0], [-0.1, 1.5, 3.0], [0.02, 0.05, 1.0]])
    dst = apply_h(H0, src)
    H = homography_from_points(src, dst)
    np.testing.assert_allclose(apply_h(H, src), dst, atol=1e-8)


def test_true_fan_explains_a_clean_scene_and_a_shifted_apex_does_not():
    sc = render_fan_scene(7, leak=0.3, texture=0.0, noise_std=0.0, quantize=False)
    m = measure_blocks(sc.image_srgb, 8, sc.floor_mask)
    q = background_basis_2d(m.xy, 256, 2)
    yp = _proj(m.y, q); tot = (yp ** 2).sum()

    def unexplained(h):
        K = _proj(fan_kernel(h, m.xy, 48, 256, 8), q)
        return sum(nnls(K, yp[:, c], maxiter=5000)[1] ** 2 for c in range(3)) / tot

    true = unexplained(FanHypothesis(sc.apex[0], sc.apex[1], sc.hidden_ccw, 0.03))
    flipped = unexplained(FanHypothesis(sc.apex[0], sc.apex[1], not sc.hidden_ccw, 0.03))
    assert true < 0.01
    assert flipped > 20 * true


def test_floor_mask_and_measurement_exclude_the_wall():
    sc = render_fan_scene(3)
    assert 0.2 < sc.floor_mask.mean() < 0.95
    m = measure_blocks(sc.image_srgb, 8, sc.floor_mask)
    nh, nw = m.shape
    assert len(m.y) == len(m.index) < nh * nw
    assert np.all(m.sigma > 0)


def test_null_scene_reads_no_signal_on_a_small_fast_funnel():
    cfg = FanConfig(levels=(FanLevel(16, 12, 48.0, (0.03,), 100), FanLevel(16, 12, 16.0, (0.03,), 100)))
    sc = render_fan_scene(7, null=True, leak=0.3, texture=0.01)
    r = locate_apex(sc.image_srgb, cfg, mask=sc.floor_mask, truth=sc)
    assert r.status == "no-signal"


def test_distance_to_line_is_perpendicular_distance():
    class R:
        line_point = np.array([0.0, 0.0]); line_dir = np.array([1.0, 0.0])
    assert math.isclose(distance_to_line(R, (5.0, 3.0)), 3.0)


def test_gate3_criteria_are_applied_as_preregistered():
    from sighextraimage.gate3 import distinct_edges, evaluate
    assert distinct_edges([{"theta": 0.48}, {"theta": 0.57}, {"theta": 0.66}]) == 1
    assert distinct_edges([{"theta": 0.45}, {"theta": 1.13}, {"theta": 1.24}]) == 2

    def rec(edges, status, err=1.0, line=1.0, side=True, null=False, bg=10.0):
        return {"truth": {"distinct_edges": edges}, "result": {"status": status, "background_delta": bg},
                "apex_error_px": err, "distance_to_line_px": line, "side_correct": side,
                "truth_covered_all_levels": True, "null": null}
    scenes = [rec(2, "located") for _ in range(4)] + [rec(1, "on-a-line", err=200.0, line=5.0) for _ in range(4)]
    nulls = [rec(1, "no-signal", bg=0.5) for _ in range(16)]
    assert evaluate(scenes, nulls)["passed"]
    bad = scenes[:3] + [rec(2, "located", err=40.0)] + scenes[4:]
    r = evaluate(bad, nulls)
    assert not r["gates"]["G3e_no_confident_wrong"]["pass"]
    nulls2 = nulls[:14] + [rec(1, "on-a-line"), rec(1, "located")]
    assert not evaluate(scenes, nulls2)["gates"]["G3a_no_false_alarms"]["pass"]
