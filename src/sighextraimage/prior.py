from __future__ import annotations

import torch

from .latent import PhysicalLatent, sample_prior
from .scene import SceneConfig, render_hidden


class ToyScenePrior:
    """Explicit low-dimensional prior used for exact/SIR-style posterior tests."""

    def __init__(self, scene: SceneConfig):
        self.scene = scene

    def sample(self, n: int, generator: torch.Generator) -> PhysicalLatent:
        return sample_prior(n, generator=generator)

    def render(self, latent: PhysicalLatent) -> torch.Tensor:
        return render_hidden(latent, self.scene)
