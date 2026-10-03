from __future__ import annotations

from pathlib import Path
import statistics
import torch

from .extraction import extract_boundary_signal, profile_correlation, profile_rmse
from .gradient_gate import evaluate_gate1
from .inference import evaluate_gate0, PosteriorSummary
from .inversion import profile_centroid, profile_excess_color, tv_invert
from .latent import PhysicalLatent, sample_prior
from .metrics import contraction_ratio, rgb_error, scalar_error
from .receipts import write_receipt
from .scene import SceneConfig, render_visible_with_leakage
from .transport import CornerTransport, NoOccluderTransport, WrongCornerTransport, TransportConfig


def _summary_dict(s: PosteriorSummary, truth: PhysicalLatent, prior: PosteriorSummary | None = None) -> dict:
    out = {
        "mean_theta": s.mean_theta,
        "std_theta": s.std_theta,
        "theta_error": scalar_error(s.mean_theta, float(truth.theta[0])),
        "mean_rgb": s.mean_rgb.tolist(),
        "rgb_error": rgb_error(s.mean_rgb, truth.rgb[0]),
        "ess": s.ess,
        "entropy": s.entropy,
    }
    if prior is not None:
        out["theta_contraction"] = contraction_ratio(s.std_theta, prior.std_theta)
    return out


def _gate0_dict(result, truth: PhysicalLatent) -> dict:
    return {
        "prior": _summary_dict(result.prior, truth),
        "correct": _summary_dict(result.correct, truth, result.prior),
        "wrong": _summary_dict(result.wrong, truth, result.prior),
        "no_occluder": _summary_dict(result.no_occluder, truth, result.prior),
    }


def _gate1_step_dict(step) -> dict:
    return {
        "error_before": step.error_before,
        "error_after": step.error_after,
        "error_reduction": step.error_reduction,
        "gradient_norm": step.gradient_norm,
        "finite": step.finite,
        "loss_before": step.loss_before,
        "loss_after": step.loss_after,
        "component_alignment": step.component_alignment.tolist(),
    }


def _offset_theta(truth: PhysicalLatent) -> PhysicalLatent:
    t = float(truth.theta[0])
    shifted = t - 0.28 if t > 0.65 else t + 0.28
    shifted = min(1.18, max(0.12, shifted))
    return PhysicalLatent(
        theta=torch.tensor([shifted]),
        width=truth.width.clone(), height=truth.height.clone(), rgb=truth.rgb.clone(),
        brightness=truth.brightness.clone(), shape_code=truth.shape_code.clone(),
    )


def run_benchmark(
    seeds,
    *,
    candidates: int = 1200,
    sigma: float = 0.01,
    residual_mode: str = "affine",
    output: str | Path = "results/receipts/gates-0-1-v0.json",
    tv_iters: int = 250,
) -> dict:
    seeds = [int(s) for s in seeds]
    scene = SceneConfig(height=72, visible_width=96, hidden_width=64)
    cfg = TransportConfig(n_measure=64, n_angle=64, ambient=0.0, gain=1.0)
    correct = CornerTransport(cfg)
    wrong = WrongCornerTransport(cfg)
    no_occ = NoOccluderTransport(cfg)
    scenes = []

    for seed in seeds:
        truth = sample_prior(1, generator=torch.Generator().manual_seed(seed))
        obs = render_visible_with_leakage(
            truth, scene, correct, noise_std=0.0, texture_seed=seed + 91
        )
        y_oracle = obs.y_true[0].detach()
        y_extracted = extract_boundary_signal(
            obs.visible_srgb, obs.region, n_measure=cfg.n_measure, smooth_sigma=1.5
        ).detach()
        cand = sample_prior(candidates, generator=torch.Generator().manual_seed(seed + 10000))
        gate0_oracle = evaluate_gate0(
            truth, y_oracle, cand, correct, wrong, no_occ, scene,
            residual_mode="l2", sigma=sigma,
        )
        gate0_extracted = evaluate_gate0(
            truth, y_extracted, cand, correct, wrong, no_occ, scene,
            residual_mode=residual_mode, sigma=sigma,
        )
        L = tv_invert(correct, y_oracle, lambda_tv=0.015, iters=tv_iters, lr=0.08, residual_mode="l2")
        init = _offset_theta(truth)
        gate1 = evaluate_gate1(
            truth, init, y_oracle, correct, wrong, no_occ, scene,
            residual_mode="l2", step_size=0.12,
            random_generator=torch.Generator().manual_seed(seed + 20000),
        )
        svals = correct.condition_spectrum()
        no_svals = no_occ.condition_spectrum()
        scenes.append({
            "seed": seed,
            "candidate_count": candidates,
            "truth": {
                "theta": float(truth.theta[0]),
                "width": float(truth.width[0]),
                "height": float(truth.height[0]),
                "rgb": truth.rgb[0].tolist(),
                "brightness": float(truth.brightness[0]),
                "shape_code": int(truth.shape_code[0]),
            },
            "extraction": {
                "profile_correlation": profile_correlation(y_extracted, y_oracle),
                "profile_rmse": profile_rmse(y_extracted, y_oracle),
                "mode": "log_lowpass",
            },
            "conditioning": {
                "corner_condition_number": correct.condition_number(),
                "corner_top_singular": float(svals[0]),
                "corner_tail_singular": float(svals[-1]),
                "no_occluder_top_singular": float(no_svals[0]),
                "no_occluder_tail_singular": float(no_svals[-1]),
            },
            "tv_inversion": {
                "theta_estimate": profile_centroid(L, correct.theta),
                "theta_error": abs(profile_centroid(L, correct.theta) - float(truth.theta[0])),
                "excess_color": profile_excess_color(L).tolist(),
            },
            "gate0_oracle": _gate0_dict(gate0_oracle, truth),
            "gate0_extracted": _gate0_dict(gate0_extracted, truth),
            "gate1": {
                "correct": _gate1_step_dict(gate1.correct),
                "wrong": _gate1_step_dict(gate1.wrong),
                "no_occluder": _gate1_step_dict(gate1.no_occluder),
                "random": _gate1_step_dict(gate1.random),
            },
        })

    receipt = {
        "schema": "sighextraimage.gates01.v0",
        "seeds": seeds,
        "config": {
            "candidates": int(candidates), "sigma": float(sigma),
            "extracted_residual_mode": residual_mode, "tv_iters": int(tv_iters),
        },
        "scenes": scenes,
    }
    if scenes:
        receipt["aggregate"] = {
            "median_oracle_theta_contraction": statistics.median(s["gate0_oracle"]["correct"]["theta_contraction"] for s in scenes),
            "median_oracle_theta_error": statistics.median(s["gate0_oracle"]["correct"]["theta_error"] for s in scenes),
            "median_extraction_correlation": statistics.median(s["extraction"]["profile_correlation"] for s in scenes),
            "mean_gate1_correct_reduction": statistics.mean(s["gate1"]["correct"]["error_reduction"] for s in scenes),
            "mean_gate1_wrong_reduction": statistics.mean(s["gate1"]["wrong"]["error_reduction"] for s in scenes),
            "mean_gate1_no_occluder_reduction": statistics.mean(s["gate1"]["no_occluder"]["error_reduction"] for s in scenes),
        }
    write_receipt(receipt, output)
    return receipt
