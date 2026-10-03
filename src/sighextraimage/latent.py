from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PhysicalLatent:
    theta: torch.Tensor
    width: torch.Tensor
    height: torch.Tensor
    rgb: torch.Tensor
    brightness: torch.Tensor
    shape_code: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.theta.shape[0])

    @classmethod
    def from_scalars(
        cls,
        *,
        theta: float,
        width: float,
        height: float,
        rgb: tuple[float, float, float],
        brightness: float,
        shape_code: int,
        device: str | torch.device = "cpu",
    ) -> "PhysicalLatent":
        d = torch.device(device)
        return cls(
            theta=torch.tensor([theta], dtype=torch.float32, device=d),
            width=torch.tensor([width], dtype=torch.float32, device=d),
            height=torch.tensor([height], dtype=torch.float32, device=d),
            rgb=torch.tensor([rgb], dtype=torch.float32, device=d),
            brightness=torch.tensor([brightness], dtype=torch.float32, device=d),
            shape_code=torch.tensor([shape_code], dtype=torch.long, device=d),
        )

    def index(self, idx: int | torch.Tensor) -> "PhysicalLatent":
        if isinstance(idx, int):
            sl = slice(idx, idx + 1)
        else:
            sl = idx
        return PhysicalLatent(
            self.theta[sl], self.width[sl], self.height[sl], self.rgb[sl],
            self.brightness[sl], self.shape_code[sl]
        )


PRIOR_RANGES = {
    "theta": (0.08, 1.22),
    "width": (0.07, 0.24),
    "height": (0.20, 0.70),
    "rgb": (0.05, 0.95),
    "brightness": (0.45, 1.15),
}


def _uniform(shape, lo: float, hi: float, generator: torch.Generator, device: str | torch.device):
    return lo + (hi - lo) * torch.rand(shape, generator=generator, device=device)


def sample_prior(
    n: int,
    *,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> PhysicalLatent:
    return PhysicalLatent(
        theta=_uniform((n,), *PRIOR_RANGES["theta"], generator, device),
        width=_uniform((n,), *PRIOR_RANGES["width"], generator, device),
        height=_uniform((n,), *PRIOR_RANGES["height"], generator, device),
        rgb=_uniform((n, 3), *PRIOR_RANGES["rgb"], generator, device),
        brightness=_uniform((n,), *PRIOR_RANGES["brightness"], generator, device),
        shape_code=torch.randint(0, 2, (n,), generator=generator, device=device),
    )
