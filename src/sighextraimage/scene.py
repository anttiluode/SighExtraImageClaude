from __future__ import annotations

from dataclasses import dataclass
import torch

from .latent import PhysicalLatent


@dataclass(frozen=True)
class SceneConfig:
    height: int = 96
    visible_width: int = 128
    hidden_width: int = 64


def render_hidden(latent: PhysicalLatent, config: SceneConfig) -> torch.Tensor:
    """Render only the hidden radiance contribution, [N,3,H,W_hidden]."""
    n = latent.batch_size
    device = latent.theta.device
    dtype = latent.theta.dtype
    yy = torch.linspace(0.0, 1.0, config.height, device=device, dtype=dtype)
    xx = torch.linspace(0.0, 1.0, config.hidden_width, device=device, dtype=dtype)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    theta_min, theta_max = 0.0, 1.30
    cx = ((latent.theta - theta_min) / (theta_max - theta_min)).clamp(0.0, 1.0)[:, None, None]
    sx = (latent.width / (theta_max - theta_min)).clamp_min(1e-3)[:, None, None]
    cy = torch.full_like(cx, 0.68)
    sy = (latent.height * 0.36).clamp_min(0.03)[:, None, None]

    disk = torch.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    rect_x = torch.sigmoid((sx - torch.abs(x - cx)) * 80.0)
    rect_y = torch.sigmoid((sy - torch.abs(y - cy)) * 80.0)
    rect = rect_x * rect_y
    choose_rect = latent.shape_code.to(dtype=dtype)[:, None, None]
    occupancy = disk * (1.0 - choose_rect) + rect * choose_rect

    color = latent.rgb[:, :, None, None]
    bright = latent.brightness[:, None, None, None]
    return (occupancy[:, None, :, :] * color * bright).clamp(0.0, 1.0)

@dataclass(frozen=True)
class BoundaryRegion:
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class SyntheticObservation:
    full_truth_srgb: torch.Tensor
    visible_srgb: torch.Tensor
    hidden_srgb: torch.Tensor
    region: BoundaryRegion
    y_true: torch.Tensor
    y_noisy: torch.Tensor


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8).pow(1.0 / 2.4) - 0.055)


def render_visible_with_leakage(
    latent: PhysicalLatent,
    config: SceneConfig,
    transport,
    *,
    noise_std: float = 0.0,
    texture_seed: int = 0,
) -> SyntheticObservation:
    """Bake hidden-light leakage into a visible textured wall/floor crop."""
    if latent.batch_size != 1:
        raise ValueError("synthetic pixel renderer currently expects one latent")
    device = latent.theta.device
    dtype = latent.theta.dtype
    hidden_lin = render_hidden(latent, config)[0]
    y_true = transport.forward_latent(latent, config)  # [1,M,3]

    h, w = config.height, config.visible_width
    gen = torch.Generator(device="cpu").manual_seed(texture_seed)
    yy = torch.linspace(0, 1, h, device=device, dtype=dtype)[:, None]
    base_wall = torch.tensor([0.46, 0.45, 0.43], device=device, dtype=dtype)[:, None, None]
    base_floor = torch.tensor([0.34, 0.33, 0.31], device=device, dtype=dtype)[:, None, None]
    floor_mix = torch.sigmoid((yy - 0.58) * 30.0)[None, :, :]
    albedo = base_wall * (1.0 - floor_mix) + base_floor * floor_mix
    albedo = albedo.expand(3, h, w).clone()
    texture = torch.randn((1, h, w), generator=gen, dtype=dtype).to(device) * 0.012
    texture = torch.nn.functional.avg_pool2d(texture.unsqueeze(0), 5, stride=1, padding=2)[0]
    albedo = (albedo * (1.0 + texture)).clamp(0.05, 0.95)

    illumination = torch.full((3, h, w), 0.55, device=device, dtype=dtype)
    y0, y1 = int(0.64 * h), int(0.93 * h)
    x0, x1 = int(0.44 * w), w
    region = BoundaryRegion(y0, y1, x0, x1)
    sw = x1 - x0
    prof = y_true[0].transpose(0, 1).unsqueeze(0)
    prof = torch.nn.functional.interpolate(prof, size=sw, mode="linear", align_corners=True)[0]
    rows = torch.linspace(0.65, 1.0, y1 - y0, device=device, dtype=dtype)
    vweight = torch.exp(-0.5 * ((rows - 0.82) / 0.22) ** 2)
    patch = prof[:, None, :] * vweight[None, :, None]
    illumination[:, y0:y1, x0:x1] += patch
    visible_lin = (albedo * illumination).clamp_min(0.0)
    if noise_std > 0:
        visible_lin = visible_lin + noise_std * torch.randn_like(visible_lin)
    visible_srgb = _linear_to_srgb(visible_lin)
    hidden_srgb = _linear_to_srgb(hidden_lin)
    full_truth = torch.cat([visible_srgb, hidden_srgb], dim=2)
    y_noisy = y_true.clone()
    if noise_std > 0:
        y_noisy = y_noisy + noise_std * torch.randn_like(y_noisy)
    return SyntheticObservation(full_truth, visible_srgb, hidden_srgb, region, y_true, y_noisy)
