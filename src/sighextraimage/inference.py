from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .latent import PhysicalLatent
from .likelihood import physics_residual


@dataclass
class PosteriorSummary:
    weights: torch.Tensor
    mean_theta: float
    std_theta: float
    mean_width: float
    std_width: float
    mean_height: float
    std_height: float
    mean_rgb: torch.Tensor
    std_rgb: torch.Tensor
    mean_brightness: float
    std_brightness: float
    shape_probability: float
    entropy: float
    ess: float


@dataclass
class Gate0Result:
    prior: PosteriorSummary
    correct: PosteriorSummary
    wrong: PosteriorSummary
    no_occluder: PosteriorSummary


def normalize_log_weights(log_w: torch.Tensor) -> torch.Tensor:
    shifted = log_w - torch.max(log_w)
    weights = torch.exp(shifted)
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)


def _weighted_mean_std(x: torch.Tensor, w: torch.Tensor):
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    mean = (w * x).sum(dim=0)
    var = (w * (x - mean) ** 2).sum(dim=0).clamp_min(0.0)
    return mean, torch.sqrt(var)


def summarize(latent: PhysicalLatent, weights: torch.Tensor) -> PosteriorSummary:
    w = weights / weights.sum().clamp_min(1e-12)
    mt, st = _weighted_mean_std(latent.theta, w)
    mw, sw = _weighted_mean_std(latent.width, w)
    mh, sh = _weighted_mean_std(latent.height, w)
    mrgb, srgb = _weighted_mean_std(latent.rgb, w)
    mb, sb = _weighted_mean_std(latent.brightness, w)
    shape_prob = float((w * (latent.shape_code == 1).float()).sum())
    entropy = float(-(w * torch.log(w.clamp_min(1e-12))).sum())
    ess = float(1.0 / (w.square().sum().clamp_min(1e-12)))
    return PosteriorSummary(
        weights=w,
        mean_theta=float(mt), std_theta=float(st),
        mean_width=float(mw), std_width=float(sw),
        mean_height=float(mh), std_height=float(sh),
        mean_rgb=mrgb.detach(), std_rgb=srgb.detach(),
        mean_brightness=float(mb), std_brightness=float(sb),
        shape_probability=shape_prob, entropy=entropy, ess=ess,
    )


def _candidate_residuals(y: torch.Tensor, yhat: torch.Tensor, mode: str) -> torch.Tensor:
    if y.ndim == 3:
        if y.shape[0] != 1:
            raise ValueError("observed y must be [M,3] or [1,M,3]")
        y = y[0]
    if mode == "l2":
        return ((yhat - y.unsqueeze(0)) ** 2).mean(dim=(1,2))
    vals = [physics_residual(y, yhat[i], mode) for i in range(yhat.shape[0])]
    return torch.stack(vals)


def _posterior_weights(y, candidates, transport, scene, residual_mode: str, sigma: float):
    pred = transport.forward_latent(candidates, scene)
    resid = _candidate_residuals(y, pred, residual_mode)
    scale = max(float(sigma), 1e-9)
    return normalize_log_weights(-resid / (2.0 * scale * scale))


def evaluate_gate0(
    z_true: PhysicalLatent,
    observed_y: torch.Tensor,
    candidates: PhysicalLatent,
    correct_transport,
    wrong_transport,
    no_occluder_transport,
    scene,
    *,
    residual_mode: str = "l2",
    sigma: float = 0.05,
) -> Gate0Result:
    n = candidates.batch_size
    uniform = torch.full((n,), 1.0 / n, dtype=candidates.theta.dtype, device=candidates.theta.device)
    correct_w = _posterior_weights(observed_y, candidates, correct_transport, scene, residual_mode, sigma)
    wrong_w = _posterior_weights(observed_y, candidates, wrong_transport, scene, residual_mode, sigma)
    no_w = _posterior_weights(observed_y, candidates, no_occluder_transport, scene, residual_mode, sigma)
    return Gate0Result(
        prior=summarize(candidates, uniform),
        correct=summarize(candidates, correct_w),
        wrong=summarize(candidates, wrong_w),
        no_occluder=summarize(candidates, no_w),
    )
