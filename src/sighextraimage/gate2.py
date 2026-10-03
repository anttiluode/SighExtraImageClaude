"""Gate 2: corner-geometry funnel + calibrated temperature on synthetic held-out scenes.

Protocol: docs/superpowers/specs/2026-10-03-gate2-geometry-funnel-design.md
"""
from __future__ import annotations

from pathlib import Path
import math
import statistics
import time

import numpy as np
import torch

from .extraction import extract_boundary_signal
from .geometry_funnel import (
    DEFAULT_HYPOTHESIS, CornerHypothesis, calibrated_posterior, funnel_from_image,
    surviving_geometries,
)
from .inference import evaluate_gate0
from .latent import PhysicalLatent, sample_prior
from .receipts import write_receipt
from .scene import SceneConfig, render_visible_with_leakage
from .transport import CornerTransport, NoOccluderTransport, TransportConfig, WrongCornerTransport

HELD_OUT_SEEDS = (41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107)
GEOMETRY_PRIOR_SEEDS = tuple(range(777, 777 + 48))
NOISE_STD = 0.002
SCENE = SceneConfig(height=72, visible_width=96, hidden_width=64)


def geometry_prior_draw(rng: np.random.Generator) -> CornerHypothesis:
    return CornerHypothesis(
        float(rng.uniform(0.02, 0.40)), float(rng.uniform(0.85, 1.30)),
        float(math.exp(rng.uniform(math.log(0.025), math.log(0.12)))), False,
    )


def true_geometry(seed: int) -> CornerHypothesis:
    return geometry_prior_draw(np.random.default_rng(seed + 5000))


def geometry_prior() -> list[CornerHypothesis]:
    return [geometry_prior_draw(np.random.default_rng(s)) for s in GEOMETRY_PRIOR_SEEDS]


def render_scene(seed: int, *, null: bool = False):
    z = sample_prior(1, generator=torch.Generator().manual_seed(seed))
    if null:
        z = PhysicalLatent(z.theta, z.width, z.height, z.rgb, z.brightness * 0, z.shape_code)
    g = true_geometry(seed)
    obs = render_visible_with_leakage(z, SCENE, g.transport(64, 64), noise_std=NOISE_STD, texture_seed=seed + 91)
    image = torch.round(obs.visible_srgb * 255.0) / 255.0
    return z, g, obs, image


def _post(p, theta: float) -> dict:
    d = p.summary()
    d["theta_error"] = abs(p.mean_theta - theta)
    d["covers90"] = p.covers(theta, 90)
    d["covers68"] = p.covers(theta, 68)
    return d


def run_scene(seed: int, *, posteriors: bool = True) -> dict:
    t0 = time.time()
    z, g, obs, image = render_scene(seed)
    theta = float(z.theta[0])
    funnel, profile = funnel_from_image(image, obs.region, truth=g)
    out = {
        "seed": seed,
        "truth": {"theta": theta, "geometry": g.as_dict(), "rgb": z.rgb[0].tolist(),
                  "brightness": float(z.brightness[0]), "shape_code": int(z.shape_code[0])},
        "funnel": funnel.summary(),
        "truth_survived_all_levels": all(r.get("truth_still_covered", False) for r in funnel.ledger),
    }
    # v0 reproduction on the same scene: affine fraction residual / (2 * 0.01^2), default geometry.
    cfg = TransportConfig(n_measure=64, n_angle=64)
    y_v0 = extract_boundary_signal(image, obs.region, n_measure=64, smooth_sigma=1.5)
    cand = sample_prior(2000, generator=torch.Generator().manual_seed(seed + 10000))
    v0 = evaluate_gate0(z, y_v0, cand, CornerTransport(cfg), WrongCornerTransport(cfg),
                        NoOccluderTransport(cfg), SCENE, residual_mode="affine", sigma=0.01)
    out["v0"] = {"ess": v0.correct.ess, "theta_error": abs(v0.correct.mean_theta - theta),
                 "std_theta": v0.correct.std_theta}
    if posteriors:
        prior_set = geometry_prior()
        surv = surviving_geometries(profile, funnel.noise, prior_set, funnel)
        out["geometry_prior_surviving"] = len(surv)
        out["geometry_prior_size"] = len(prior_set)
        out["posterior_prior_marginal"] = _post(
            calibrated_posterior(profile, funnel.noise, SCENE, cand, surv, seed=seed), theta)
        out["posterior_default_geometry"] = _post(
            calibrated_posterior(profile, funnel.noise, SCENE, cand, [DEFAULT_HYPOTHESIS], seed=seed), theta)
        out["posterior_oracle_geometry"] = _post(
            calibrated_posterior(profile, funnel.noise, SCENE, cand, [g], seed=seed), theta)
        prior_mean = float(np.mean([0.08, 1.22]))
        out["prior_theta_error"] = abs(prior_mean - theta)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_null(seed: int) -> dict:
    z, g, obs, image = render_scene(seed, null=True)
    funnel, _ = funnel_from_image(image, obs.region, truth=g)
    return {"seed": seed, "funnel": funnel.summary(),
            "truth_survived_all_levels": all(r.get("truth_still_covered", False) for r in funnel.ledger)}


def evaluate(scenes: list[dict], nulls: list[dict]) -> dict:
    n, m = len(scenes), len(nulls)
    false_alarms = sum(1 for r in nulls if r["funnel"]["background_refuted"])
    truth_kept = sum(1 for r in scenes if r["truth_survived_all_levels"])
    cov = sum(1 for r in scenes if r["posterior_prior_marginal"]["covers90"])
    ess = statistics.median(r["posterior_prior_marginal"]["ess"] for r in scenes)
    gates = {
        "G2a_no_false_alarms": {"observed": f"{false_alarms}/{m}", "required": "<= 1", "pass": false_alarms <= 1},
        "G2b_truth_never_refuted": {"observed": f"{truth_kept}/{n}", "required": f">= {n - 1}", "pass": truth_kept >= n - 1},
        "G2c_calibrated_interval": {"observed": f"{cov}/{n}", "required": ">= 12 of 16", "pass": cov >= math.ceil(0.75 * n)},
        "G2d_no_collapse": {"observed": ess, "required": ">= 20", "pass": ess >= 20},
    }
    med = lambda key, sub: statistics.median(r[key][sub] for r in scenes)
    describe = {
        "detected_signal_scenes": f"{sum(1 for r in scenes if r['funnel']['background_refuted'])}/{n}",
        "status_counts": {s: sum(1 for r in scenes if r["funnel"]["status"] == s)
                          for s in ("refuted-model", "no-signal", "undecided-geometry", "decided-geometry")},
        "median_box_refuted_fraction": statistics.median(r["funnel"]["box_refuted_fraction"] for r in scenes),
        "reversed_refuted": f"{sum(1 for r in scenes if r['funnel']['reversed_refuted'])}/{n}",
        "v0_median_ess": statistics.median(r["v0"]["ess"] for r in scenes),
        "v0_median_theta_error": statistics.median(r["v0"]["theta_error"] for r in scenes),
        "prior_median_theta_error": statistics.median(r["prior_theta_error"] for r in scenes),
        "coverage90": {k: f"{sum(1 for r in scenes if r[k]['covers90'])}/{n}" for k in
                       ("posterior_prior_marginal", "posterior_default_geometry", "posterior_oracle_geometry")},
        "median_theta_error": {k: med(k, "theta_error") for k in
                               ("posterior_prior_marginal", "posterior_default_geometry", "posterior_oracle_geometry")},
        "median_ess": {k: med(k, "ess") for k in
                       ("posterior_prior_marginal", "posterior_default_geometry", "posterior_oracle_geometry")},
        "median_temperature": med("posterior_prior_marginal", "temperature"),
    }
    return {"gates": gates, "passed": all(g["pass"] for g in gates.values()), "descriptive": describe}


def run_gate2(seeds=HELD_OUT_SEEDS, *, output: str | Path = "results/receipts/gate2-v0.json", log=print) -> dict:
    seeds = [int(s) for s in seeds]
    scenes, nulls = [], []
    for s in seeds:
        scenes.append(run_scene(s))
        nulls.append(run_null(s))
        if log:
            r = scenes[-1]; p = r["posterior_prior_marginal"]
            log(f"seed {s}: {r['funnel']['status']}, truth kept {r['truth_survived_all_levels']}, "
                f"null {nulls[-1]['funnel']['status']}, covers90 {p['covers90']}, ESS {p['ess']:.0f}, "
                f"{r['seconds']}s")
    receipt = {
        "schema": "sighextraimage.gate2.v0",
        "protocol": "docs/superpowers/specs/2026-10-03-gate2-geometry-funnel-design.md",
        "seeds": seeds,
        "config": {"noise_std": NOISE_STD, "quantization": "8-bit", "geometry_prior_seeds": list(GEOMETRY_PRIOR_SEEDS),
                   "candidates_initial": 2000},
        "scenes": scenes, "nulls": nulls,
        "result": evaluate(scenes, nulls),
    }
    write_receipt(receipt, output)
    return receipt
