from __future__ import annotations

import torch
import torch.nn.functional as F

from .scene import BoundaryRegion


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def _smooth_profile(profile: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return profile
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=profile.dtype, device=profile.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    p = profile.transpose(0, 1).unsqueeze(0)  # [1,3,M]
    p = F.pad(p, (radius, radius), mode="reflect")
    p = F.conv1d(p, k.view(1, 1, -1).repeat(3, 1, 1), groups=3)
    return p[0].transpose(0, 1)


def extract_boundary_signal(
    image: torch.Tensor,
    region: BoundaryRegion,
    *,
    n_measure: int,
    smooth_sigma: float,
    mode: str = "log_lowpass",
) -> torch.Tensor:
    """Estimate the smooth illumination profile along a selected visible strip."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("one image at a time")
        image = image[0]
    if image.shape[0] != 3:
        raise ValueError("image must be [3,H,W]")
    patch = srgb_to_linear(image[:, region.y0:region.y1, region.x0:region.x1])
    if patch.numel() == 0:
        raise ValueError("empty boundary region")
    if mode != "log_lowpass":
        raise ValueError(f"unknown extraction mode: {mode}")
    # Geometric row average suppresses multiplicative texture; preserve x structure.
    profile = torch.exp(torch.log(patch.clamp_min(1e-5)).mean(dim=1)).transpose(0, 1)  # [W,3]
    profile = F.interpolate(profile.transpose(0, 1).unsqueeze(0), size=n_measure, mode="linear", align_corners=True)[0].transpose(0, 1)
    profile = _smooth_profile(profile, smooth_sigma)
    # Remove unknown DC floor/channel albedo while retaining positive leakage shape.
    baseline = torch.quantile(profile, 0.08, dim=0, keepdim=True)
    signal = (profile - baseline).clamp_min(0.0)
    return signal


def extract_boundary_signal_temporal(
    frames: list[torch.Tensor],
    region: BoundaryRegion,
    *,
    n_measure: int,
    smooth_sigma: float,
) -> torch.Tensor:
    if len(frames) < 2:
        raise ValueError("temporal extraction needs at least two frames")
    sigs = torch.stack([
        extract_boundary_signal(f, region, n_measure=n_measure, smooth_sigma=smooth_sigma)
        for f in frames
    ])
    baseline = sigs.median(dim=0).values
    return (sigs - baseline).abs().mean(dim=0)


def profile_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def profile_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)))
