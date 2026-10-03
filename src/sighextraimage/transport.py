from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent import PhysicalLatent
from .scene import SceneConfig, render_hidden


@dataclass(frozen=True)
class TransportConfig:
    n_measure: int = 64
    n_angle: int = 64
    phi_min: float = 0.04
    phi_max: float = 1.25
    theta_min: float = 0.0
    theta_max: float = 1.30
    penumbra: float = 0.055
    r0: float = 1.0
    r_slope: float = 0.35
    hidden_radius: float = 2.0
    falloff_softness: float = 1.5
    ambient: float = 0.0
    gain: float = 1.0
    occluder: bool = True


class CornerTransport(nn.Module):
    def __init__(self, config: TransportConfig):
        super().__init__()
        self.config = config
        g = config
        phi = torch.linspace(g.phi_min, g.phi_max, g.n_measure)
        theta = torch.linspace(g.theta_min, g.theta_max, g.n_angle)
        dtheta = (g.theta_max - g.theta_min) / max(g.n_angle - 1, 1)
        span = max(g.phi_max - g.phi_min, 1e-6)
        r = g.r0 * (1.0 + g.r_slope * (phi - g.phi_min) / span)
        if g.occluder:
            visibility = torch.sigmoid((phi[:, None] - theta[None, :]) / g.penumbra)
        else:
            visibility = torch.ones(g.n_measure, g.n_angle)
        d2 = (
            r[:, None] ** 2 + g.hidden_radius**2
            - 2.0 * r[:, None] * g.hidden_radius * torch.cos(theta[None, :] - phi[:, None])
        )
        falloff = 1.0 / (1.0 + d2 / (g.falloff_softness**2))
        kernel = visibility * falloff * dtheta * g.gain
        self.register_buffer("theta", theta)
        self.register_buffer("kernel", kernel)

    def _angular_radiance(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            hidden = hidden.unsqueeze(0)
        n, c, h, w = hidden.shape
        rows = torch.linspace(0.0, 1.0, h, device=hidden.device, dtype=hidden.dtype)
        weights = torch.exp(-0.5 * ((rows - 0.68) / 0.22) ** 2)
        weights = weights / weights.sum().clamp_min(1e-8)
        cols = (hidden * weights[None, None, :, None]).sum(dim=2)
        if w != self.config.n_angle:
            cols = F.interpolate(cols, size=self.config.n_angle, mode="linear", align_corners=True)
        return cols.transpose(1, 2)

    def forward_hidden(self, hidden_img: torch.Tensor) -> torch.Tensor:
        radiance = self._angular_radiance(hidden_img)
        kernel = self.kernel.to(device=radiance.device, dtype=radiance.dtype)
        y = torch.einsum("mj,njc->nmc", kernel, radiance)
        if self.config.ambient:
            y = y + self.config.ambient
        return y

    def forward_latent(self, latent: PhysicalLatent, scene: SceneConfig) -> torch.Tensor:
        return self.forward_hidden(render_hidden(latent, scene))

    def condition_spectrum(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.kernel.double()).float()

    def condition_number(self) -> float:
        s = self.condition_spectrum()
        return float((s[0] / s[-1].clamp_min(1e-12)).item())


class NoOccluderTransport(CornerTransport):
    def __init__(self, config: TransportConfig):
        cfg = replace(config, occluder=False)
        super().__init__(cfg)
        mean_row = self.kernel.mean(dim=0, keepdim=True)
        self.kernel.copy_(mean_row.repeat(self.config.n_measure, 1))

class WrongCornerTransport(CornerTransport):
    """Deliberately mismatched angular geometry: mirror hidden angles before transport."""
    def __init__(self, config: TransportConfig):
        super().__init__(config)
        self.kernel.copy_(torch.flip(self.kernel, dims=[1]))
