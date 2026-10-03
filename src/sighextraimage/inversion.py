from __future__ import annotations

import torch

from .likelihood import physics_residual


def _predict_from_radiance(transport, L: torch.Tensor) -> torch.Tensor:
    K = transport.kernel.to(device=L.device, dtype=L.dtype)
    y = K @ L
    if transport.config.ambient:
        y = y + transport.config.ambient
    return y


def tv_invert(
    transport,
    y: torch.Tensor,
    *,
    lambda_tv: float = 0.03,
    iters: int = 600,
    lr: float = 0.05,
    residual_mode: str = "affine",
) -> torch.Tensor:
    J = transport.kernel.shape[1]
    L = torch.full((J, 3), 0.05, dtype=y.dtype, device=y.device, requires_grad=True)
    opt = torch.optim.Adam([L], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        positive = torch.nn.functional.softplus(L)
        yhat = _predict_from_radiance(transport, positive)
        data = physics_residual(y, yhat, residual_mode)
        tv = (positive[1:] - positive[:-1]).abs().mean()
        loss = data + lambda_tv * tv
        loss.backward()
        opt.step()
    return torch.nn.functional.softplus(L.detach())


def profile_centroid(profile: torch.Tensor, theta: torch.Tensor, baseline_q: float = 0.55) -> float:
    s = profile.mean(dim=1) if profile.ndim == 2 else profile
    base = torch.quantile(s, baseline_q)
    w = (s - base).clamp_min(0.0)
    if float(w.sum()) < 1e-8:
        return float(theta.mean())
    if w.numel() != theta.numel():
        w = torch.nn.functional.interpolate(w[None, None], size=theta.numel(), mode="linear", align_corners=True)[0, 0]
    return float((w * theta.to(w)).sum() / w.sum())


def profile_excess_color(profile: torch.Tensor, baseline_q: float = 0.55) -> torch.Tensor:
    s = profile.mean(dim=1)
    base = torch.quantile(s, baseline_q)
    w = (s - base).clamp_min(0.0)
    if float(w.sum()) < 1e-8:
        return torch.full((3,), 1.0 / 3.0, dtype=profile.dtype, device=profile.device)
    c = (w[:, None] * profile).sum(dim=0).clamp_min(0.0)
    return c / c.sum().clamp_min(1e-8)
